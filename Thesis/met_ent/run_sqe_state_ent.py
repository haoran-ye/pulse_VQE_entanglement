#!/usr/bin/env python3
"""
run_sqe_state_ent.py
"""

from __future__ import annotations

import sys
import multiprocessing as mp
from pathlib import Path
from dataclasses import replace, fields, is_dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.sqe_state import sqe_optimisation
from pulse_vqe_entanglement.entanglement_measures import von_neumann_entropy_traj


OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_ent/sqe_state_ent/"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

THETAS = np.linspace(0.0 * np.pi, 0.50 * np.pi, 40)
DURATIONS_NS = np.arange(1.0, 1200.0, 30.0)

N_INITS = 4
N_WORKERS = 9
SEED0 = 1234

INFIDELITY_THRESHOLD = 1e-5
INFIDELITY_FLOOR = 1e-14

MAX_N_TIME_STEPS = 50
MIN_SEGMENT_DURATION_NS = 1.0

N_OUTER = 5
INNER_MAXITER = 200
EPS_INFIDELITY = 1e-12
DUAL_LR = 1e10
RHO = 0.0
FD_EPS = 1e-6

ENTANGLEMENT_MARGIN = 0.01

J_INIT_MEANS = np.array(
    [0.0] * (N_INITS // 2)
    + [0.0 + 1e-3] * (N_INITS - N_INITS // 2),
    dtype=float,
)

#J_INIT_MEANS = np.array(
#    [0.0] * (N_INITS // 2)
#    + [-1.0 + 1e-3] * (N_INITS - N_INITS // 2),
#    dtype=float,
#)

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
    entanglement_optimisation=False,
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


def extract_result_arrays(result: Any) -> dict[str, Any]:
    infidelities = as_1d_float_array(
        get_attr_any(
            result,
            ["infidelities", "infidelity_evolution", "state_infidelities", "val_final"],
        )
    )

    final_infid = get_attr_any(
        result,
        ["final_infidelity", "best_infidelity"],
        default=None,
    )
    if final_infid is None:
        final_infid = float(infidelities[-1]) if infidelities.size else np.nan

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

    if entropies.size == 0:
        final_traj = get_attr_any(result, ["final_traj"])
        if final_traj is not None:
            entropies = von_neumann_entropy_traj(np.asarray(final_traj))

    return dict(
        infidelities=infidelities,
        entropies=entropies,
        x0=get_attr_any(result, ["x0", "x_initial", "initial_x"]),
        x_final=get_attr_any(result, ["x_final", "x_opt", "x", "optimised_x", "optimized_x"]),
        final_infid=float(final_infid),
        n_iterations=get_attr_any(result, ["n_iterations", "nit", "iterations"], default=np.nan),
        gradient_norm=get_attr_any(result, ["gradient_norm", "grad_norm"], default=np.nan),
    )


def max_entropy(entropies: np.ndarray) -> float:
    entropies = as_1d_float_array(entropies)
    if entropies.size == 0 or np.all(~np.isfinite(entropies)):
        return np.nan
    return float(np.nanmax(entropies))


def apply_duration_envelope(z: np.ndarray, prefer: str) -> np.ndarray:
    """
    Duration axis is increasing.

    For prefer='min':
        if a shorter duration has a smaller value, overwrite longer durations.
        Used for infidelity and entanglement costs.
    """
    corrected = z.copy()

    for i_theta in range(corrected.shape[0]):
        running = np.inf

        for i_T in range(corrected.shape[1]):
            val = corrected[i_theta, i_T]
            if not np.isfinite(val):
                continue

            if prefer == "min":
                if val < running:
                    running = val
                else:
                    corrected[i_theta, i_T] = running
            else:
                raise ValueError("Only prefer='min' is implemented.")

    return corrected


def run_one_seed(
    base_config: SimulationConfig,
    theta: float,
    T_ns: float,
    seed: int,
    j_init_mean: float,
) -> dict[str, Any]:

    n_steps = n_time_steps_for_duration(T_ns)
    target_entropy = target_sqe_entropy(theta)

    common_kwargs = dict(
        theta_target=float(theta),
        T=float(T_ns),
        n_time_steps=int(n_steps),
        J_init_seed=int(seed),
        J_init_mean=float(j_init_mean),
        S_max=float(target_entropy + ENTANGLEMENT_MARGIN),
        eps_infidelity=float(EPS_INFIDELITY),
        rho=float(RHO),
        dual_lr=float(DUAL_LR),
        n_outer=int(N_OUTER),
        inner_maxiter=int(INNER_MAXITER),
        fd_eps=float(FD_EPS),
    )

    baseline_config = safe_replace_config(
        base_config,
        **common_kwargs,
        entanglement_optimisation=False,
    )
    baseline_result = sqe_optimisation(baseline_config)
    baseline_arrays = extract_result_arrays(baseline_result)

    baseline_final_infid = float(baseline_arrays["final_infid"])
    baseline_s_max = max_entropy(baseline_arrays["entropies"])

    baseline_success = (
        np.isfinite(baseline_final_infid)
        and baseline_final_infid <= INFIDELITY_THRESHOLD
    )

    if not baseline_success:
        return dict(
            baseline_success=False,
            entropy_skipped=True,
            success=False,
            baseline_final_infid=baseline_final_infid,
            ent_final_infid=np.nan,
            baseline_s_max=baseline_s_max,
            ent_s_max=np.nan,
            seed=seed,
            J_init_mean=j_init_mean,
            n_time_steps=n_steps,
            segment_duration_ns=T_ns / n_steps,
            x0=baseline_arrays["x0"],
            baseline_x_final=baseline_arrays["x_final"],
            ent_x_final=None,
            baseline_infidelities=baseline_arrays["infidelities"],
            ent_infidelities=None,
            baseline_entropies=baseline_arrays["entropies"],
            ent_entropies=None,
            n_iterations=np.nan,
            gradient_norm=np.nan,
        )

    ent_config = safe_replace_config(
        base_config,
        **common_kwargs,
        entanglement_optimisation=True,
    )
    ent_result = sqe_optimisation(ent_config)
    ent_arrays = extract_result_arrays(ent_result)

    ent_final_infid = float(ent_arrays["final_infid"])
    ent_s_max = max_entropy(ent_arrays["entropies"])

    success = (
        np.isfinite(ent_final_infid)
        and ent_final_infid <= INFIDELITY_THRESHOLD
        and np.isfinite(ent_s_max)
    )

    return dict(
        baseline_success=True,
        entropy_skipped=False,
        success=success,
        baseline_final_infid=baseline_final_infid,
        ent_final_infid=ent_final_infid,
        baseline_s_max=baseline_s_max,
        ent_s_max=ent_s_max,
        seed=seed,
        J_init_mean=j_init_mean,
        n_time_steps=n_steps,
        segment_duration_ns=T_ns / n_steps,
        x0=ent_arrays["x0"],
        baseline_x_final=baseline_arrays["x_final"],
        ent_x_final=ent_arrays["x_final"],
        baseline_infidelities=baseline_arrays["infidelities"],
        ent_infidelities=ent_arrays["infidelities"],
        baseline_entropies=baseline_arrays["entropies"],
        ent_entropies=ent_arrays["entropies"],
        n_iterations=ent_arrays["n_iterations"],
        gradient_norm=ent_arrays["gradient_norm"],
    )


def main() -> None:
    base_config = make_base_config()

    n_theta = len(THETAS)
    n_T = len(DURATIONS_NS)
    shape = (n_theta, n_T, N_INITS)

    baseline_successes = np.zeros(shape, dtype=bool)
    entropy_skipped = np.zeros(shape, dtype=bool)
    successes = np.zeros(shape, dtype=bool)

    baseline_final_infids = np.full(shape, np.nan)
    ent_final_infids = np.full(shape, np.nan)

    baseline_s_max = np.full(shape, np.nan)
    ent_s_max = np.full(shape, np.nan)

    n_iterations = np.full(shape, np.nan)
    gradient_norms = np.full(shape, np.nan)
    seeds = np.full(shape, -1, dtype=np.int64)
    J_init_means = np.full(shape, np.nan)

    n_time_steps_used = np.full((n_theta, n_T), -1, dtype=int)
    segment_duration_ns = np.full((n_theta, n_T), np.nan)
    target_entropies = np.array([target_sqe_entropy(theta) for theta in THETAS])

    best_x0 = np.empty((n_theta, n_T), dtype=object)
    best_baseline_x_final = np.empty((n_theta, n_T), dtype=object)
    best_ent_x_final = np.empty((n_theta, n_T), dtype=object)
    best_baseline_infidelity_traces = np.empty((n_theta, n_T), dtype=object)
    best_ent_infidelity_traces = np.empty((n_theta, n_T), dtype=object)
    best_baseline_entropy_traces = np.empty((n_theta, n_T), dtype=object)
    best_ent_entropy_traces = np.empty((n_theta, n_T), dtype=object)
    best_seed_index = np.full((n_theta, n_T), -1, dtype=int)

    rng = np.random.default_rng(SEED0)

    with mp.Pool(processes=N_WORKERS) as pool:
        with tqdm(total=n_theta * n_T, desc="(theta,T)", unit="point", dynamic_ncols=True) as pbar:
            for i, theta in enumerate(THETAS):
                for j, T_ns in enumerate(DURATIONS_NS):
                    n_steps = n_time_steps_for_duration(T_ns)
                    n_time_steps_used[i, j] = n_steps
                    segment_duration_ns[i, j] = T_ns / n_steps

                    seed_list = rng.integers(0, 2**31 - 1, size=N_INITS, dtype=np.int64)

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

                    results = pool.starmap(run_one_seed, args)

                    point_best_s = np.inf
                    point_best_k = -1

                    for k, out in enumerate(results):
                        baseline_successes[i, j, k] = out["baseline_success"]
                        entropy_skipped[i, j, k] = out["entropy_skipped"]
                        successes[i, j, k] = out["success"]

                        baseline_final_infids[i, j, k] = out["baseline_final_infid"]
                        ent_final_infids[i, j, k] = out["ent_final_infid"]

                        baseline_s_max[i, j, k] = out["baseline_s_max"]
                        ent_s_max[i, j, k] = out["ent_s_max"]

                        n_iterations[i, j, k] = out["n_iterations"]
                        gradient_norms[i, j, k] = out["gradient_norm"]
                        seeds[i, j, k] = out["seed"]
                        J_init_means[i, j, k] = out["J_init_mean"]

                        if out["success"] and np.isfinite(out["ent_s_max"]):
                            if out["ent_s_max"] < point_best_s:
                                point_best_s = out["ent_s_max"]
                                point_best_k = k

                                best_seed_index[i, j] = k
                                best_x0[i, j] = out["x0"]
                                best_baseline_x_final[i, j] = out["baseline_x_final"]
                                best_ent_x_final[i, j] = out["ent_x_final"]
                                best_baseline_infidelity_traces[i, j] = out["baseline_infidelities"]
                                best_ent_infidelity_traces[i, j] = out["ent_infidelities"]
                                best_baseline_entropy_traces[i, j] = out["baseline_entropies"]
                                best_ent_entropy_traces[i, j] = out["ent_entropies"]

                    pbar.set_postfix(
                        theta_pi=f"{theta / np.pi:.3f}",
                        T_ns=f"{T_ns:.1f}",
                        baseline=f"{int(np.sum(baseline_successes[i, j]))}/{N_INITS}",
                        ent=f"{int(np.sum(successes[i, j]))}/{N_INITS}",
                        best_S=f"{point_best_s:.3e}" if point_best_k >= 0 else "None",
                    )

                    np.savez_compressed(
                        OUTDIR / "sqe_state_entanglement_scan_raw.npz",
                        thetas=THETAS,
                        durations_ns=DURATIONS_NS,
                        target_entropies=target_entropies,
                        baseline_successes=baseline_successes,
                        entropy_skipped=entropy_skipped,
                        successes=successes,
                        baseline_final_infids=baseline_final_infids,
                        ent_final_infids=ent_final_infids,
                        baseline_s_max=baseline_s_max,
                        ent_s_max=ent_s_max,
                        n_iterations=n_iterations,
                        gradient_norms=gradient_norms,
                        seeds=seeds,
                        J_init_means=J_init_means,
                        n_time_steps_used=n_time_steps_used,
                        segment_duration_ns=segment_duration_ns,
                        best_seed_index=best_seed_index,
                        best_x0=best_x0,
                        best_baseline_x_final=best_baseline_x_final,
                        best_ent_x_final=best_ent_x_final,
                        best_baseline_infidelity_traces=best_baseline_infidelity_traces,
                        best_ent_infidelity_traces=best_ent_infidelity_traces,
                        best_baseline_entropy_traces=best_baseline_entropy_traces,
                        best_ent_entropy_traces=best_ent_entropy_traces,
                        infidelity_threshold=INFIDELITY_THRESHOLD,
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

    best_baseline_infid = np.full((n_theta, n_T), np.nan)
    best_ent_infid = np.full((n_theta, n_T), np.nan)
    best_baseline_s_max = np.full((n_theta, n_T), np.nan)
    best_ent_s_max = np.full((n_theta, n_T), np.nan)

    for i in range(n_theta):
        for j in range(n_T):
            k = best_seed_index[i, j]
            if k >= 0:
                best_baseline_infid[i, j] = max(baseline_final_infids[i, j, k], INFIDELITY_FLOOR)
                best_ent_infid[i, j] = max(ent_final_infids[i, j, k], INFIDELITY_FLOOR)
                best_baseline_s_max[i, j] = baseline_s_max[i, j, k]
                best_ent_s_max[i, j] = ent_s_max[i, j, k]

    corrected_best_baseline_infid = apply_duration_envelope(best_baseline_infid, prefer="min")
    corrected_best_ent_infid = apply_duration_envelope(best_ent_infid, prefer="min")
    corrected_best_baseline_s_max = apply_duration_envelope(best_baseline_s_max, prefer="min")
    corrected_best_ent_s_max = apply_duration_envelope(best_ent_s_max, prefer="min")

    np.savez_compressed(
        OUTDIR / "sqe_state_entanglement_scan_processed.npz",
        thetas=THETAS,
        durations_ns=DURATIONS_NS,
        target_entropies=target_entropies,
        best_baseline_infid=best_baseline_infid,
        best_ent_infid=best_ent_infid,
        best_baseline_s_max=best_baseline_s_max,
        best_ent_s_max=best_ent_s_max,
        corrected_best_baseline_infid=corrected_best_baseline_infid,
        corrected_best_ent_infid=corrected_best_ent_infid,
        corrected_best_baseline_s_max=corrected_best_baseline_s_max,
        corrected_best_ent_s_max=corrected_best_ent_s_max,
        success_rate=np.mean(successes, axis=2),
        baseline_success_rate=np.mean(baseline_successes, axis=2),
        any_success=np.any(successes, axis=2),
        n_time_steps_used=n_time_steps_used,
        segment_duration_ns=segment_duration_ns,
        best_seed_index=best_seed_index,
        infidelity_threshold=INFIDELITY_THRESHOLD,
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