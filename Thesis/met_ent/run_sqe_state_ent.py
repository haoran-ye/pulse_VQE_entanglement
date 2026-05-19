#!/usr/bin/env python3
"""
run_sqe_state_ent.py

Correct SQE state entanglement scan.

Strategy:
- For each (theta, T), run N_INITS independent seeds in parallel.
- A single multiprocessing Pool is kept alive throughout the scan.
- Each seed calls sqe_optimisation(config) ONCE.
- Entanglement optimisation is handled internally by sqe_optimisation via
  its primal-dual settings.
- Select the successful run with the lowest excess entanglement:

      Delta S_max = max_t [S(t) - S_target]_+

  subject to final infidelity <= FIDELITY_THRESHOLD.
"""

from __future__ import annotations

import sys
import multiprocessing as mp
from pathlib import Path
from dataclasses import replace, fields, is_dataclass
from typing import Any

import numpy as np
from tqdm import tqdm


# ============================================================
# Project imports
# ============================================================

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.sqe_state import sqe_optimisation


# ============================================================
# User settings
# ============================================================

OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_ent/sqe_state_ent/"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

THETAS = np.linspace(0.0 * np.pi, 0.50 * np.pi, 5)
DURATIONS_NS = np.arange(1.0, 500.0, 100)

N_INITS = 4
N_WORKERS = 9
SEED0 = 1234

FIDELITY_THRESHOLD = 1e-5

MAX_N_TIME_STEPS = 50
MIN_SEGMENT_DURATION_NS = 1.0

N_OUTER = 5
INNER_MAXITER = 200
EPS_INFIDELITY = 1e-12
DUAL_LR = 1e10
RHO = 0.0
FD_EPS = 1e-6

ENTANGLEMENT_MARGIN = 0.01
INFIDELITY_FLOOR = 1e-14

J_INIT_MEANS = np.array(
    [0.0] * (N_INITS // 2)
    + [-1.0 + 1e-3] * (N_INITS - N_INITS // 2),
    dtype=float,
)


# ============================================================
# Base config
# ============================================================

BASE_CONFIG_KWARGS: dict[str, Any] = dict(
    zeeman_splittings=np.array([1000.0, 1008.0]),
    initial_state=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128),
    theta_target=0.0,
    n_time_steps=50,
    n_discretisation=1,
    T=200.0,
    max_drive_strength=0.0,
    J_max=10.0,
    J_min=0.0,
    J_init_scale=1e-1,
    J_init_mean=0.0,
    feromagnetic=False,
    gtol=1e-12,
    maxiter=500,
    entanglement_optimisation=True,
    state_ent_measure="von_neumann",
    J_init_seed=None,
    S_max=0.01,
    eps_infidelity=EPS_INFIDELITY,
    rho=RHO,
    dual_lr=DUAL_LR,
    n_outer=N_OUTER,
    inner_maxiter=INNER_MAXITER,
    fd_eps=FD_EPS,
)


# ============================================================
# Config helpers
# ============================================================

def dataclass_field_names(cls: type) -> set[str]:
    if not is_dataclass(cls):
        return set()
    return {f.name for f in fields(cls)}


CONFIG_FIELDS = dataclass_field_names(SimulationConfig)


def filter_kwargs_for_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in CONFIG_FIELDS}


def make_base_config() -> SimulationConfig:
    return SimulationConfig(**filter_kwargs_for_config(BASE_CONFIG_KWARGS))


def safe_replace_config(config: SimulationConfig, **kwargs: Any) -> SimulationConfig:
    return replace(config, **filter_kwargs_for_config(kwargs))


def get_attr_any(obj: Any, names: list[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def as_1d_float_array(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    return np.asarray(x, dtype=float).reshape(-1)


# ============================================================
# Metrics
# ============================================================

def n_time_steps_for_duration(T_ns: float) -> int:
    return max(
        1,
        min(MAX_N_TIME_STEPS, int(np.floor(T_ns / MIN_SEGMENT_DURATION_NS))),
    )


def target_sqe_entropy(theta: float) -> float:
    p = float(np.cos(theta / 2.0) ** 2)
    q = 1.0 - p

    eps = 1e-15
    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))

    return float(-(p * np.log2(p) + q * np.log2(q)))


def compute_entanglement_metrics(
    entropies: np.ndarray,
    target_entropy: float,
) -> dict[str, float]:
    entropies = as_1d_float_array(entropies)

    if entropies.size == 0 or np.all(~np.isfinite(entropies)):
        return dict(
            delta_s_max=np.nan,
            delta_s_avg=np.nan,
            s_max=np.nan,
            s_avg=np.nan,
        )

    excess = np.maximum(entropies - target_entropy, 0.0)

    return dict(
        delta_s_max=float(np.nanmax(excess)),
        delta_s_avg=float(np.nanmean(excess)),
        s_max=float(np.nanmax(entropies)),
        s_avg=float(np.nanmean(entropies)),
    )


def extract_result_arrays(result: Any) -> dict[str, Any]:
    # SQEResults stores the infidelity trajectory in val_final (expectation value
    # of I - |ψ_target><ψ_target| at each time step); val_final[-1] is the final infidelity.
    infidelities = as_1d_float_array(
        get_attr_any(
            result,
            ["infidelities", "infidelity_evolution", "state_infidelities", "val_final"],
        )
    )

    entropies = as_1d_float_array(
        get_attr_any(
            result,
            [
                "entropies",
                "entropy_evolution",
                "von_neumann_entropies",
                "von_neumann_entropy",
                "S_evolution",
                "S_values",
            ],
        )
    )

    # SQEResults has no entropy attribute — compute from the state trajectory.
    if entropies.size == 0:
        final_traj = get_attr_any(result, ["final_traj"])
        if final_traj is not None:
            from pulse_vqe_entanglement.entanglement_measures import von_neumann_entropy_traj
            entropies = von_neumann_entropy_traj(np.asarray(final_traj))

    x_final = get_attr_any(
        result,
        ["x_final", "x_opt", "x", "optimised_x", "optimized_x"],
    )

    x0 = get_attr_any(
        result,
        ["x0", "x_initial", "initial_x"],
    )

    final_infid = get_attr_any(
        result,
        ["final_infidelity", "best_infidelity"],
        default=None,
    )

    if final_infid is None:
        # val_final is the trajectory; take the last (final-time) value
        final_infid = float(infidelities[-1]) if infidelities.size else np.nan

    n_iterations = get_attr_any(
        result,
        ["n_iterations", "nit", "iterations"],
        default=np.nan,
    )

    gradient_norm = get_attr_any(
        result,
        ["gradient_norm", "grad_norm"],
        default=np.nan,
    )

    return dict(
        infidelities=infidelities,
        entropies=entropies,
        x_final=None if x_final is None else np.asarray(x_final, dtype=float),
        x0=None if x0 is None else np.asarray(x0, dtype=float),
        final_infid=float(final_infid),
        n_iterations=n_iterations,
        gradient_norm=gradient_norm,
    )


# ============================================================
# One seed
# ============================================================

def run_one_seed(
    base_config: SimulationConfig,
    theta: float,
    T_ns: float,
    seed: int,
    j_init_mean: float,
) -> dict[str, Any]:

    n_steps = n_time_steps_for_duration(T_ns)
    target_entropy = target_sqe_entropy(theta)

    config = safe_replace_config(
        base_config,
        theta_target=float(theta),
        T=float(T_ns),
        n_time_steps=int(n_steps),
        J_init_seed=int(seed),
        J_init_mean=float(j_init_mean),
        entanglement_optimisation=True,
        S_max=float(target_entropy + ENTANGLEMENT_MARGIN),
        eps_infidelity=float(EPS_INFIDELITY),
        rho=float(RHO),
        dual_lr=float(DUAL_LR),
        n_outer=int(N_OUTER),
        inner_maxiter=int(INNER_MAXITER),
        fd_eps=float(FD_EPS),
    )

    result = sqe_optimisation(config)
    arrays = extract_result_arrays(result)

    final_infid = float(arrays["final_infid"])
    corrected_final_infid = max(final_infid, INFIDELITY_FLOOR)

    metrics = compute_entanglement_metrics(arrays["entropies"], target_entropy)

    success = bool(np.isfinite(final_infid) and final_infid <= FIDELITY_THRESHOLD)

    return dict(
        success=success,
        final_infid=final_infid,
        corrected_final_infid=corrected_final_infid,
        delta_s_max=metrics["delta_s_max"],
        delta_s_avg=metrics["delta_s_avg"],
        s_max=metrics["s_max"],
        s_avg=metrics["s_avg"],
        seed=seed,
        J_init_mean=j_init_mean,
        n_time_steps=n_steps,
        segment_duration_ns=T_ns / n_steps,
        x0=arrays["x0"],
        x_final=arrays["x_final"],
        infidelities=arrays["infidelities"],
        entropies=arrays["entropies"],
        n_iterations=arrays["n_iterations"],
        gradient_norm=arrays["gradient_norm"],
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    base_config = make_base_config()

    n_theta = len(THETAS)
    n_T = len(DURATIONS_NS)
    shape = (n_theta, n_T, N_INITS)

    successes = np.zeros(shape, dtype=bool)
    final_infids = np.full(shape, np.nan)
    corrected_final_infids = np.full(shape, np.nan)

    delta_s_max = np.full(shape, np.nan)
    delta_s_avg = np.full(shape, np.nan)
    s_max = np.full(shape, np.nan)
    s_avg = np.full(shape, np.nan)

    n_iterations = np.full(shape, np.nan)
    gradient_norms = np.full(shape, np.nan)

    seeds = np.full(shape, -1, dtype=np.int64)
    J_init_means = np.full(shape, np.nan)

    n_time_steps_used = np.full((n_theta, n_T), -1, dtype=int)
    segment_duration_ns = np.full((n_theta, n_T), np.nan)
    target_entropies = np.array([target_sqe_entropy(theta) for theta in THETAS])

    best_x0 = np.empty((n_theta, n_T), dtype=object)
    best_x_final = np.empty((n_theta, n_T), dtype=object)
    best_infidelity_traces = np.empty((n_theta, n_T), dtype=object)
    best_entropy_traces = np.empty((n_theta, n_T), dtype=object)
    best_seed_index = np.full((n_theta, n_T), -1, dtype=int)

    rng = np.random.default_rng(SEED0)

    total_points = n_theta * n_T

    with mp.Pool(processes=N_WORKERS) as pool:
        with tqdm(
            total=total_points,
            desc="(theta,T)",
            unit="point",
            dynamic_ncols=True,
        ) as pbar:

            for i, theta in enumerate(THETAS):
                for j, T_ns in enumerate(DURATIONS_NS):

                    n_steps = n_time_steps_for_duration(T_ns)
                    n_time_steps_used[i, j] = n_steps
                    segment_duration_ns[i, j] = T_ns / n_steps

                    seed_list = rng.integers(
                        0,
                        2**31 - 1,
                        size=N_INITS,
                        dtype=np.int64,
                    )

                    args = [
                        (
                            base_config,
                            float(theta),
                            float(T_ns),
                            int(seed_list[k]),
                            float(J_INIT_MEANS[k % len(J_INIT_MEANS)]),
                        )
                        for k in range(N_INITS)
                    ]

                    pbar.set_postfix(
                        theta_pi=f"{theta / np.pi:.3f}",
                        T_ns=f"{T_ns:.1f}",
                        n_steps=n_steps,
                    )

                    results = pool.starmap(run_one_seed, args)

                    point_best_cost = np.inf
                    point_best_k = -1

                    for k, out in enumerate(results):
                        successes[i, j, k] = out["success"]
                        final_infids[i, j, k] = out["final_infid"]
                        corrected_final_infids[i, j, k] = out["corrected_final_infid"]

                        delta_s_max[i, j, k] = out["delta_s_max"]
                        delta_s_avg[i, j, k] = out["delta_s_avg"]
                        s_max[i, j, k] = out["s_max"]
                        s_avg[i, j, k] = out["s_avg"]

                        n_iterations[i, j, k] = out["n_iterations"]
                        gradient_norms[i, j, k] = out["gradient_norm"]

                        seeds[i, j, k] = out["seed"]
                        J_init_means[i, j, k] = out["J_init_mean"]

                        if out["success"] and np.isfinite(out["delta_s_max"]):
                            if out["delta_s_max"] < point_best_cost:
                                point_best_cost = out["delta_s_max"]
                                point_best_k = k

                                best_seed_index[i, j] = k
                                best_x0[i, j] = out["x0"]
                                best_x_final[i, j] = out["x_final"]
                                best_infidelity_traces[i, j] = out["infidelities"]
                                best_entropy_traces[i, j] = out["entropies"]

                    n_success = int(np.sum(successes[i, j]))

                    pbar.set_postfix(
                        theta_pi=f"{theta / np.pi:.3f}",
                        T_ns=f"{T_ns:.1f}",
                        success=f"{n_success}/{N_INITS}",
                        best_dS=f"{point_best_cost:.2e}" if point_best_k >= 0 else "None",
                    )

                    np.savez_compressed(
                        OUTDIR / "sqe_state_entanglement_scan_raw.npz",
                        thetas=THETAS,
                        durations_ns=DURATIONS_NS,
                        target_entropies=target_entropies,
                        successes=successes,
                        final_infids=final_infids,
                        corrected_final_infids=corrected_final_infids,
                        delta_s_max=delta_s_max,
                        delta_s_avg=delta_s_avg,
                        s_max=s_max,
                        s_avg=s_avg,
                        n_iterations=n_iterations,
                        gradient_norms=gradient_norms,
                        seeds=seeds,
                        J_init_means=J_init_means,
                        n_time_steps_used=n_time_steps_used,
                        segment_duration_ns=segment_duration_ns,
                        best_seed_index=best_seed_index,
                        best_x0=best_x0,
                        best_x_final=best_x_final,
                        best_infidelity_traces=best_infidelity_traces,
                        best_entropy_traces=best_entropy_traces,
                        fidelity_threshold=FIDELITY_THRESHOLD,
                        entanglement_margin=ENTANGLEMENT_MARGIN,
                        eps_infidelity=EPS_INFIDELITY,
                        rho=RHO,
                        dual_lr=DUAL_LR,
                        n_outer=N_OUTER,
                        inner_maxiter=INNER_MAXITER,
                        fd_eps=FD_EPS,
                        max_n_time_steps=MAX_N_TIME_STEPS,
                        min_segment_duration_ns=MIN_SEGMENT_DURATION_NS,
                        infidelity_floor=INFIDELITY_FLOOR,
                        n_workers=N_WORKERS,
                    )

                    pbar.update(1)

    successful_delta_s_max = np.where(successes, delta_s_max, np.nan)
    successful_delta_s_avg = np.where(successes, delta_s_avg, np.nan)
    successful_s_max = np.where(successes, s_max, np.nan)
    successful_s_avg = np.where(successes, s_avg, np.nan)
    successful_infids = np.where(successes, corrected_final_infids, np.nan)

    with np.errstate(all="ignore"):
        best_delta_s_max = np.nanmin(successful_delta_s_max, axis=2)
        best_delta_s_avg = np.nanmin(successful_delta_s_avg, axis=2)
        best_s_max = np.nanmin(successful_s_max, axis=2)
        best_s_avg = np.nanmin(successful_s_avg, axis=2)
        best_infid = np.nanmin(successful_infids, axis=2)

    success_rate = np.mean(successes, axis=2)
    any_success = np.any(successes, axis=2)

    np.savez_compressed(
        OUTDIR / "sqe_state_entanglement_scan_processed.npz",
        thetas=THETAS,
        durations_ns=DURATIONS_NS,
        target_entropies=target_entropies,
        best_delta_s_max=best_delta_s_max,
        best_delta_s_avg=best_delta_s_avg,
        best_s_max=best_s_max,
        best_s_avg=best_s_avg,
        best_infid=best_infid,
        success_rate=success_rate,
        any_success=any_success,
        n_time_steps_used=n_time_steps_used,
        segment_duration_ns=segment_duration_ns,
        best_seed_index=best_seed_index,
        fidelity_threshold=FIDELITY_THRESHOLD,
        entanglement_margin=ENTANGLEMENT_MARGIN,
        eps_infidelity=EPS_INFIDELITY,
        rho=RHO,
        dual_lr=DUAL_LR,
        n_outer=N_OUTER,
        inner_maxiter=INNER_MAXITER,
        fd_eps=FD_EPS,
        n_workers=N_WORKERS,
    )

    print(f"\nSaved raw and processed results to:\n  {OUTDIR}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()