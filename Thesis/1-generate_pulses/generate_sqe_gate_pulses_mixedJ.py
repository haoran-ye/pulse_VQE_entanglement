#!/usr/bin/env python3
"""
generate_sqe_gate_pulses_mixedJ.py

Build the raw SQE-gate pulse database using mixed (positive and negative) exchange
coupling (J ∈ [J_min, J_max] = [−5, 5] MHz).

Identical in structure to the positiveJ variant; only J_min, J_max, and the
PULSE_FAMILY_LABEL differ.

Output array shape convention
------------------------------
    baseline_x_final, ent_x_final : (n_theta, n_duration, n_init, N_TIME_STEPS)
    All scalar diagnostic arrays   : (n_theta, n_duration, n_init)

Output
------
    PULSES_DATABASE/sqe_gate_mixedJ_pulses.npz
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# pulse_vqe_entanglement is not pip-installed; add the repo root to sys.path.
import sys as _sys
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
del _sys, _ROOT

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.sqe_gate import sqe_gate_optimisation

# Shared helpers: config manipulation, result extraction.
from thesis_utils import (
    filter_kwargs_for_config,
    safe_replace_config,
    extract_gate_result_arrays,
    fixed_width_x,
    nanmean_safe,
    nanmax_safe,
)


# ============================================================
# Settings
# ============================================================

# B1: resolved relative to this file so the script works on any machine.
PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"

SYSTEM_NAME = "sqe_gate"
PULSE_FAMILY_LABEL = "mixedJ"

OUTDIR = PULSES_DATABASE
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUTDIR / f"{SYSTEM_NAME}_{PULSE_FAMILY_LABEL}_pulses.npz"

# Theta grid: fine near zero (small-angle regime) + coarser large-angle sweep.
small_thetas = np.linspace(0.0 * np.pi, 0.009 * np.pi, 10)
large_thetas = np.linspace(0.01 * np.pi, 0.50 * np.pi, 40)
THETAS = np.concatenate([small_thetas, large_thetas])

DURATIONS_NS = np.arange(1.0, 1200.0, 20.0)   # pulse durations to scan (ns)

N_INITS = 2           # random initialisations per grid point
N_WORKERS = 9         # parallel worker processes
SEED0 = 1234          # master RNG seed for reproducibility
CHECKPOINT_EVERY = 500  # save checkpoint every N completed jobs

INFIDELITY_THRESHOLD = 1e-5   # success criterion for a single run
INFIDELITY_FLOOR = 1e-14      # clip near-zero infidelities to avoid log issues

N_TIME_STEPS = 50             # piecewise-constant pulse segments (fixed for SQE gate)

# Gate entanglement measure used during constrained optimisation.
GATE_ENT_MEASURE = "operator_entanglement"
# S_max = ENTANGLEMENT_MARGIN for gate runs (no theta-dependent target).
ENTANGLEMENT_MARGIN = 0.01

# Augmented Lagrangian / TEM optimiser hyperparameters.
N_OUTER = 3
INNER_MAXITER = 200
EPS_INFIDELITY = 1e-12
DUAL_LR = 1e10
RHO = 0.0
FD_EPS = 1e-6

# B4: was [0.0]*(N//2) + [0.0]*(N-N//2) which always gives zeros.
J_INIT_MEANS = np.zeros(N_INITS, dtype=float)

ZEEMAN = np.array([1000.0, 1008.0])                                    # MHz
DUMMY_INITIAL_STATE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)  # |00⟩

# Default config; per-job overrides applied in run_one_seed via safe_replace_config.
BASE_CONFIG_KWARGS: dict[str, Any] = dict(
    zeeman_splittings=ZEEMAN,
    initial_state=DUMMY_INITIAL_STATE,
    theta_target=0.0,
    n_time_steps=N_TIME_STEPS,
    n_discretisation=1,
    T=200.0,
    max_drive_strength=0.0,
    J_max=5.0,
    J_min=-5.0,                # mixed-J range
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


def make_base_config() -> SimulationConfig:
    """Construct the base SimulationConfig, dropping any unknown kwargs."""
    return SimulationConfig(**filter_kwargs_for_config(BASE_CONFIG_KWARGS))


# ============================================================
# Worker
# ============================================================

def run_one_seed(task: tuple[Any, ...]) -> dict[str, Any]:
    """
    Run one (theta, duration, init) grid point and return results as a flat dict.

    E2 fix: sqe_gate_optimisation with entanglement_optimisation=True runs the
    infidelity-only baseline internally; we read from ent_result.baseline.
    """
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

    # E2: one call only — baseline is stored in ent_result.baseline.
    ent_config = safe_replace_config(base_config, **common_kwargs,
                                     entanglement_optimisation=True)
    ent_result = sqe_gate_optimisation(ent_config)

    baseline_arrays = extract_gate_result_arrays(ent_result.baseline)
    ent_arrays = extract_gate_result_arrays(ent_result)

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

        baseline_success=(np.isfinite(baseline_final_infid)
                          and baseline_final_infid <= INFIDELITY_THRESHOLD),
        baseline_optimization_success=baseline_arrays["optimization_success"],
        baseline_final_infid=baseline_final_infid,
        baseline_ent_avg=nanmean_safe(baseline_arrays["entanglements"]),
        baseline_ent_max=nanmax_safe(baseline_arrays["entanglements"]),
        baseline_x0=fixed_width_x(baseline_arrays["x0"], N_TIME_STEPS),
        baseline_x_final=fixed_width_x(baseline_arrays["x_final"], N_TIME_STEPS),
        baseline_n_iterations=baseline_arrays["n_iterations"],
        baseline_gradient_norm=baseline_arrays["gradient_norm"],

        ent_success=(np.isfinite(ent_final_infid)
                     and ent_final_infid <= INFIDELITY_THRESHOLD),
        ent_optimization_success=ent_arrays["optimization_success"],
        ent_final_infid=ent_final_infid,
        ent_ent_avg=nanmean_safe(ent_arrays["entanglements"]),
        ent_ent_max=nanmax_safe(ent_arrays["entanglements"]),
        ent_x0=fixed_width_x(ent_arrays["x0"], N_TIME_STEPS),
        ent_x_final=fixed_width_x(ent_arrays["x_final"], N_TIME_STEPS),
        ent_n_iterations=ent_arrays["n_iterations"],
        ent_gradient_norm=ent_arrays["gradient_norm"],
    )


# ============================================================
# Array management
# ============================================================

def initialise_arrays() -> dict[str, Any]:
    """Allocate NaN/False-filled output arrays for the full grid."""
    n_theta = len(THETAS)
    n_T = len(DURATIONS_NS)
    shape = (n_theta, n_T, N_INITS)

    return dict(
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


def update_arrays(arrays: dict[str, Any], out: dict[str, Any]) -> None:
    """Write a single worker result into the global arrays at (i, j, k)."""
    i, j, k = out["i_theta"], out["i_T"], out["k_init"]

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


def save_raw(arrays: dict[str, Any]) -> None:
    """Save the current state of all arrays to the output .npz file."""
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Build the full pulse database in parallel using a multiprocessing pool."""
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
                tasks.append((
                    i, j, k,
                    float(theta), float(T_ns),
                    int(seed_list[k]),
                    float(J_INIT_MEANS[k % len(J_INIT_MEANS)]),
                    base_config,
                ))

    print("=" * 72)
    print("Generating SQE-gate pulse database (mixedJ)")
    print(f"Output     = {RAW_OUT}")
    print(f"J_min/max  = {BASE_CONFIG_KWARGS['J_min']} / {BASE_CONFIG_KWARGS['J_max']} MHz")
    print(f"N theta    = {len(THETAS)}")
    print(f"N duration = {len(DURATIONS_NS)}")
    print(f"N inits    = {N_INITS}")
    print(f"N steps    = {N_TIME_STEPS}")
    print(f"N jobs     = {len(tasks)}")
    print("=" * 72)

    chunksize = max(1, len(tasks) // (N_WORKERS * 20))

    with mp.Pool(processes=N_WORKERS) as pool:
        iterator = pool.imap_unordered(run_one_seed, tasks, chunksize=chunksize)

        for n_done, out in enumerate(
            tqdm(iterator, total=len(tasks), desc="SQE-gate mixedJ jobs",
                 unit="job", dynamic_ncols=True),
            start=1,
        ):
            update_arrays(arrays, out)

            if n_done % CHECKPOINT_EVERY == 0:
                save_raw(arrays)

    save_raw(arrays)
    print(f"\nDone.  Saved to:\n  {RAW_OUT}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
