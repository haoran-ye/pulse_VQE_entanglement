#!/usr/bin/env python3
"""
generate_sqe_gate_pulses_raw.py

Generate a reusable SQE-gate pulse database.

This script saves EVERY run/init and does NO selection.

All grid points use the SAME number of time steps:
    N_TIME_STEPS = 50
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
from pulse_vqe_entanglement.sqe_gate import sqe_gate_optimisation


# ============================================================
# Settings
# ============================================================

PULSES_DATABASE = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/PULSES_DATABASE"
)

SYSTEM_NAME = "sqe_gate"

PULSE_FAMILY_LABEL = "positiveJ"

OUTDIR = PULSES_DATABASE
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUTDIR / f"{SYSTEM_NAME}_{PULSE_FAMILY_LABEL}_pulses.npz"

small_thetas = np.linspace(0.0 * np.pi, 0.009 * np.pi, 10)
large_thetas = np.linspace(0.01 * np.pi, 0.50 * np.pi, 40)
THETAS = np.concatenate([small_thetas, large_thetas])
DURATIONS_NS = np.arange(1.0, 1200.0, 20.0)

N_INITS = 2
N_WORKERS = 9
SEED0 = 1234
CHECKPOINT_EVERY = 500

INFIDELITY_THRESHOLD = 1e-5
INFIDELITY_FLOOR = 1e-14

N_TIME_STEPS = 50

N_OUTER = 3
INNER_MAXITER = 200
EPS_INFIDELITY = 1e-12
DUAL_LR = 1e10
RHO = 0.0
FD_EPS = 1e-6

GATE_ENT_MEASURE = "operator_entanglement"
ENTANGLEMENT_MARGIN = 0.01

J_INIT_MEANS = np.array(
    [0.0] * (N_INITS // 2)
    + [0.0] * (N_INITS - N_INITS // 2),
    dtype=float,
)

ZEEMAN = np.array([1000.0, 1008.0])
DUMMY_INITIAL_STATE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

BASE_CONFIG_KWARGS: dict[str, Any] = dict(
    zeeman_splittings=ZEEMAN,
    initial_state=DUMMY_INITIAL_STATE,
    theta_target=0.0,
    n_time_steps=N_TIME_STEPS,
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
    optimisation_target="gate",
    gate_ent_measure=GATE_ENT_MEASURE,
    J_init_seed=None,
    S_max=0.01,
    eps_infidelity=EPS_INFIDELITY,
    rho=RHO,
    dual_lr=DUAL_LR,
    n_outer=N_OUTER,
    inner_maxiter=INNER_MAXITER,
    fd_eps=FD_EPS,
    n_J_channels=1,
)


# ============================================================
# Helpers
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


def fixed_width_x(x: Any, width: int) -> np.ndarray:
    out = np.full(width, np.nan, dtype=float)
    if x is None:
        return out

    arr = np.asarray(x, dtype=float).reshape(-1)
    n = min(width, arr.size)
    out[:n] = arr[:n]
    return out


def average_entanglement(entanglements: np.ndarray) -> float:
    entanglements = as_1d_float_array(entanglements)
    if entanglements.size == 0 or np.all(~np.isfinite(entanglements)):
        return np.nan
    return float(np.nanmean(entanglements))


def max_entanglement(entanglements: np.ndarray) -> float:
    entanglements = as_1d_float_array(entanglements)
    if entanglements.size == 0 or np.all(~np.isfinite(entanglements)):
        return np.nan
    return float(np.nanmax(entanglements))


def extract_result_arrays(result: Any) -> dict[str, Any]:
    infidelities = as_1d_float_array(
        get_attr_any(
            result,
            [
                "infid_final",
                "infidelities",
                "infidelity_evolution",
                "gate_infidelities",
                "val_final",
            ],
        )
    )

    final_infid = get_attr_any(
        result,
        ["final_infidelity", "best_infidelity", "infidelity"],
        default=None,
    )
    if final_infid is None:
        final_infid = float(infidelities[-1]) if infidelities.size else np.nan

    entanglements = as_1d_float_array(
        get_attr_any(
            result,
            [
                "entanglements",
                "gate_entanglements",
                "ent_final",
                "S_final",
                "entropy_final",
                "entanglement_final",
            ],
        )
    )

    return dict(
        infidelities=infidelities,
        entanglements=entanglements,
        x0=get_attr_any(result, ["x0", "x_initial", "initial_x"]),
        x_final=get_attr_any(result, ["x_final", "x_opt", "x", "optimised_x", "optimized_x"]),
        final_infid=float(final_infid),
        n_iterations=get_attr_any(result, ["n_iterations", "nit", "iterations"], default=np.nan),
        gradient_norm=get_attr_any(result, ["gradient_norm", "grad_norm"], default=np.nan),
        optimization_success=bool(get_attr_any(result, ["optimization_success", "success"], default=False)),
    )


# ============================================================
# Worker
# ============================================================

def run_one_seed(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        i_theta,
        i_T,
        k_init,
        theta,
        T_ns,
        seed,
        j_init_mean,
        base_config,
    ) = task

    common_kwargs = dict(
        theta_target=float(theta),
        T=float(T_ns),
        n_time_steps=int(N_TIME_STEPS),
        J_init_seed=int(seed),
        J_init_mean=float(j_init_mean),
        gate_ent_measure=GATE_ENT_MEASURE,
        S_max=float(ENTANGLEMENT_MARGIN),
        eps_infidelity=float(EPS_INFIDELITY),
        rho=float(RHO),
        dual_lr=float(DUAL_LR),
        n_outer=int(N_OUTER),
        inner_maxiter=int(INNER_MAXITER),
        fd_eps=float(FD_EPS),
        optimisation_target="gate",
        n_J_channels=1,
    )

    baseline_config = safe_replace_config(
        base_config,
        **common_kwargs,
        entanglement_optimisation=False,
    )
    baseline_result = sqe_gate_optimisation(baseline_config)
    baseline_arrays = extract_result_arrays(baseline_result)

    ent_config = safe_replace_config(
        base_config,
        **common_kwargs,
        entanglement_optimisation=True,
    )
    ent_result = sqe_gate_optimisation(ent_config)
    ent_arrays = extract_result_arrays(ent_result)

    baseline_final_infid = max(float(baseline_arrays["final_infid"]), INFIDELITY_FLOOR)
    ent_final_infid = max(float(ent_arrays["final_infid"]), INFIDELITY_FLOOR)

    return dict(
        i_theta=i_theta,
        i_T=i_T,
        k_init=k_init,
        seed=seed,
        J_init_mean=j_init_mean,
        n_time_steps=N_TIME_STEPS,
        segment_duration_ns=T_ns / N_TIME_STEPS,

        baseline_success=np.isfinite(baseline_final_infid) and baseline_final_infid <= INFIDELITY_THRESHOLD,
        baseline_optimization_success=baseline_arrays["optimization_success"],
        baseline_final_infid=baseline_final_infid,
        baseline_ent_avg=average_entanglement(baseline_arrays["entanglements"]),
        baseline_ent_max=max_entanglement(baseline_arrays["entanglements"]),
        baseline_x0=fixed_width_x(baseline_arrays["x0"], N_TIME_STEPS),
        baseline_x_final=fixed_width_x(baseline_arrays["x_final"], N_TIME_STEPS),
        baseline_n_iterations=baseline_arrays["n_iterations"],
        baseline_gradient_norm=baseline_arrays["gradient_norm"],

        ent_success=np.isfinite(ent_final_infid) and ent_final_infid <= INFIDELITY_THRESHOLD,
        ent_optimization_success=ent_arrays["optimization_success"],
        ent_final_infid=ent_final_infid,
        ent_ent_avg=average_entanglement(ent_arrays["entanglements"]),
        ent_ent_max=max_entanglement(ent_arrays["entanglements"]),
        ent_x0=fixed_width_x(ent_arrays["x0"], N_TIME_STEPS),
        ent_x_final=fixed_width_x(ent_arrays["x_final"], N_TIME_STEPS),
        ent_n_iterations=ent_arrays["n_iterations"],
        ent_gradient_norm=ent_arrays["gradient_norm"],
    )


# ============================================================
# Saving
# ============================================================

def initialise_arrays() -> dict[str, Any]:
    n_theta = len(THETAS)
    n_T = len(DURATIONS_NS)
    shape = (n_theta, n_T, N_INITS)

    arrays: dict[str, Any] = dict(
        baseline_successes=np.zeros(shape, dtype=bool),
        ent_successes=np.zeros(shape, dtype=bool),
        baseline_optimization_successes=np.zeros(shape, dtype=bool),
        ent_optimization_successes=np.zeros(shape, dtype=bool),

        baseline_final_infids=np.full(shape, np.nan),
        ent_final_infids=np.full(shape, np.nan),

        baseline_ent_avg=np.full(shape, np.nan),
        ent_ent_avg=np.full(shape, np.nan),
        baseline_ent_max=np.full(shape, np.nan),
        ent_ent_max=np.full(shape, np.nan),

        baseline_x0=np.full(shape + (N_TIME_STEPS,), np.nan),
        ent_x0=np.full(shape + (N_TIME_STEPS,), np.nan),
        baseline_x_final=np.full(shape + (N_TIME_STEPS,), np.nan),
        ent_x_final=np.full(shape + (N_TIME_STEPS,), np.nan),

        baseline_n_iterations=np.full(shape, np.nan),
        ent_n_iterations=np.full(shape, np.nan),
        baseline_gradient_norms=np.full(shape, np.nan),
        ent_gradient_norms=np.full(shape, np.nan),

        seeds=np.full(shape, -1, dtype=np.int64),
        J_init_means=np.full(shape, np.nan),

        n_time_steps_used=np.full((n_theta, n_T), N_TIME_STEPS, dtype=int),
        segment_duration_ns=np.full((n_theta, n_T), np.nan),
    )

    return arrays


def save_raw(arrays: dict[str, Any]) -> None:
    np.savez_compressed(
        RAW_OUT,
        thetas=THETAS,
        theta_over_pi=THETAS / np.pi,
        durations_ns=DURATIONS_NS,

        system_name=SYSTEM_NAME,
        pulse_family_label=PULSE_FAMILY_LABEL,
        zeeman=ZEEMAN,
        dummy_initial_state=DUMMY_INITIAL_STATE,

        J_min=BASE_CONFIG_KWARGS["J_min"],
        J_max=BASE_CONFIG_KWARGS["J_max"],
        J_init_scale=BASE_CONFIG_KWARGS["J_init_scale"],
        J_init_means_config=J_INIT_MEANS,
        n_inits=N_INITS,
        seed0=SEED0,

        infidelity_threshold=INFIDELITY_THRESHOLD,
        infidelity_floor=INFIDELITY_FLOOR,
        gate_ent_measure=GATE_ENT_MEASURE,
        entanglement_margin=ENTANGLEMENT_MARGIN,
        eps_infidelity=EPS_INFIDELITY,
        rho=RHO,
        dual_lr=DUAL_LR,
        n_outer=N_OUTER,
        inner_maxiter=INNER_MAXITER,
        fd_eps=FD_EPS,
        n_time_steps=N_TIME_STEPS,
        max_n_time_steps=N_TIME_STEPS,
        n_workers=N_WORKERS,

        **arrays,
    )


def update_arrays(arrays: dict[str, Any], out: dict[str, Any]) -> None:
    i = out["i_theta"]
    j = out["i_T"]
    k = out["k_init"]

    arrays["baseline_successes"][i, j, k] = out["baseline_success"]
    arrays["ent_successes"][i, j, k] = out["ent_success"]
    arrays["baseline_optimization_successes"][i, j, k] = out["baseline_optimization_success"]
    arrays["ent_optimization_successes"][i, j, k] = out["ent_optimization_success"]

    arrays["baseline_final_infids"][i, j, k] = out["baseline_final_infid"]
    arrays["ent_final_infids"][i, j, k] = out["ent_final_infid"]

    arrays["baseline_ent_avg"][i, j, k] = out["baseline_ent_avg"]
    arrays["ent_ent_avg"][i, j, k] = out["ent_ent_avg"]
    arrays["baseline_ent_max"][i, j, k] = out["baseline_ent_max"]
    arrays["ent_ent_max"][i, j, k] = out["ent_ent_max"]

    arrays["baseline_x0"][i, j, k] = out["baseline_x0"]
    arrays["ent_x0"][i, j, k] = out["ent_x0"]
    arrays["baseline_x_final"][i, j, k] = out["baseline_x_final"]
    arrays["ent_x_final"][i, j, k] = out["ent_x_final"]

    arrays["baseline_n_iterations"][i, j, k] = out["baseline_n_iterations"]
    arrays["ent_n_iterations"][i, j, k] = out["ent_n_iterations"]
    arrays["baseline_gradient_norms"][i, j, k] = out["baseline_gradient_norm"]
    arrays["ent_gradient_norms"][i, j, k] = out["ent_gradient_norm"]

    arrays["seeds"][i, j, k] = out["seed"]
    arrays["J_init_means"][i, j, k] = out["J_init_mean"]

    arrays["n_time_steps_used"][i, j] = out["n_time_steps"]
    arrays["segment_duration_ns"][i, j] = out["segment_duration_ns"]


# ============================================================
# Main
# ============================================================

def main() -> None:
    base_config = make_base_config()
    arrays = initialise_arrays()

    rng = np.random.default_rng(SEED0)
    tasks = []

    for i, theta in enumerate(THETAS):
        for j, T_ns in enumerate(DURATIONS_NS):
            arrays["n_time_steps_used"][i, j] = N_TIME_STEPS
            arrays["segment_duration_ns"][i, j] = float(T_ns) / N_TIME_STEPS

            seed_list = rng.integers(0, 2**31 - 1, size=N_INITS, dtype=np.int64)

            for k in range(N_INITS):
                tasks.append(
                    (
                        i,
                        j,
                        k,
                        float(theta),
                        float(T_ns),
                        int(seed_list[k]),
                        float(J_INIT_MEANS[k % len(J_INIT_MEANS)]),
                        base_config,
                    )
                )

    print("=" * 72)
    print("Generating SQE-gate raw pulse database")
    print(f"Output = {RAW_OUT}")
    print(f"Pulse family label = {PULSE_FAMILY_LABEL}")
    print(f"J_min = {BASE_CONFIG_KWARGS['J_min']}")
    print(f"J_max = {BASE_CONFIG_KWARGS['J_max']}")
    print(f"N theta = {len(THETAS)}")
    print(f"N durations = {len(DURATIONS_NS)}")
    print(f"N inits per point = {N_INITS}")
    print(f"N time steps = {N_TIME_STEPS}")
    print(f"Total seed jobs = {len(tasks)}")
    print("=" * 72)

    chunksize = max(1, len(tasks) // (N_WORKERS * 20))

    with mp.Pool(processes=N_WORKERS) as pool:
        iterator = pool.imap_unordered(run_one_seed, tasks, chunksize=chunksize)

        for n_done, out in enumerate(
            tqdm(iterator, total=len(tasks), desc="SQE-gate pulse jobs", unit="job", dynamic_ncols=True),
            start=1,
        ):
            update_arrays(arrays, out)

            if n_done % CHECKPOINT_EVERY == 0:
                save_raw(arrays)

    save_raw(arrays)

    print("\nDone.")
    print(f"Saved raw all-init pulse database to:\n  {RAW_OUT}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()