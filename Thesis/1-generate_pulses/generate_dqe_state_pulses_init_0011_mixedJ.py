#!/usr/bin/env python3
"""
generate_dqe_state_pulses_init_0011_mixedJ.py

Build the raw DQE-state pulse database starting from |0011⟩ using mixed exchange
coupling (J ∈ [−5, 5] MHz) on all three channels (J₁₂, J₂₃, J₃₄).

Unlike the positiveJ variant, the number of time steps is adaptive: short
durations use fewer segments (down to 1 ns per segment) so we don't waste
parameters on very short pulses.  The maximum is MAX_N_TIME_STEPS = 50.

P3 fix — state_ent_measure
--------------------------
"linear" has no effect in the DQE optimiser; "von_neumann" is correct.

B6 fix — update_arrays
-----------------------
The original mixedJ script did NOT update n_time_steps_used or segment_duration_ns
in update_arrays (the positiveJ script did).  Both are now written from the worker
result so the saved .npz accurately records the adaptive step count used.

Output array shape convention
------------------------------
    baseline_x_final, ent_x_final : (n_theta, n_duration, n_init, MAX_N_PULSE_PARAMS)
    where MAX_N_PULSE_PARAMS = MAX_N_TIME_STEPS * N_J_CHANNELS = 150
    Entries beyond n_time_steps_used[i,j] * N_J_CHANNELS are NaN (unused).
    All scalar diagnostic arrays   : (n_theta, n_duration, n_init)

Output
------
    PULSES_DATABASE/dqe_state_initstate_0011_mixedJ_pulses.npz
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
from pulse_vqe_entanglement.dqe_state import dqe_optimisation, basis_state

# Shared helpers: config manipulation, result extraction.
from thesis_utils import (
    filter_kwargs_for_config,
    safe_replace_config,
    extract_state_result_arrays,
    fixed_width_x,
    nanmean_safe,
    nanmax_safe,
)


# ============================================================
# Settings
# ============================================================

# B1: resolved relative to this file so the script works on any machine.
PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"

SYSTEM_NAME = "dqe_state"
INITIAL_STATE_LABEL = "initstate_0011"
J_SETTINGS = "mixedJ"

OUTDIR = PULSES_DATABASE
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUTDIR / f"{SYSTEM_NAME}_{INITIAL_STATE_LABEL}_{J_SETTINGS}_pulses.npz"

THETAS = np.linspace(0.0 * np.pi, 0.50 * np.pi, 15)
DURATIONS_NS = np.arange(1.0, 1200.0, 80.0)   # coarser grid than SQE

N_INITS = 1           # random initialisations per grid point
N_WORKERS = 9         # parallel worker processes
SEED0 = 1234          # master RNG seed for reproducibility
CHECKPOINT_EVERY = 100  # save checkpoint every N completed jobs

INFIDELITY_THRESHOLD = 1e-5   # success criterion for a single run
INFIDELITY_FLOOR = 1e-14      # clip near-zero infidelities

MAX_N_TIME_STEPS = 50             # cap on piecewise-constant segments
MIN_SEGMENT_DURATION_NS = 1.0    # floor on segment length (ns)
N_J_CHANNELS = 3                  # J₁₂, J₂₃, J₃₄
MAX_N_PULSE_PARAMS = MAX_N_TIME_STEPS * N_J_CHANNELS  # = 150

# Augmented Lagrangian / TEM optimiser hyperparameters.
N_OUTER = 3
INNER_MAXITER = 200
EPS_INFIDELITY = 1e-12
DUAL_LR = 1e10
RHO = 0.0
FD_EPS = 1e-6

# Fixed S_max for the entanglement constraint.
ENTANGLEMENT_CAP = 0.1

# B4: was [0.0]*(N//2) + [0.0]*(N-N//2) which always gives zeros.
J_INIT_MEANS = np.zeros(N_INITS, dtype=float)

ZEEMAN = np.array([1000.0, 1008.0, 1016.0, 1024.0])  # MHz, 4-spin chain
INITIAL_STATE = basis_state(4, "0011")                # |0011⟩ ∈ ℂ^16

# P3 fix: state_ent_measure="von_neumann" (not "linear" which has no effect in DQE).
BASE_CONFIG_KWARGS: dict[str, Any] = dict(
    zeeman_splittings=ZEEMAN,
    initial_state=INITIAL_STATE,
    theta_target=0.0,
    n_time_steps=MAX_N_TIME_STEPS,   # overridden per-job by n_time_steps_for_duration
    n_discretisation=1,
    T=200.0,
    max_drive_strength=0.0,
    J_max=5.0,
    J_min=-5.0,
    J_init_scale=1e-1,
    J_init_mean=0.0,
    feromagnetic=False,
    gtol=1e-12,
    maxiter=200,
    entanglement_optimisation=False,
    optimisation_target="state",
    state_ent_measure="von_neumann",   # P3 fix: was "linear" which is not implemented
    J_init_seed=None,
    S_max=ENTANGLEMENT_CAP,
    eps_infidelity=EPS_INFIDELITY,
    rho=RHO,
    dual_lr=DUAL_LR,
    n_outer=N_OUTER,
    inner_maxiter=INNER_MAXITER,
    fd_eps=FD_EPS,
    n_J_channels=N_J_CHANNELS,
)


def make_base_config() -> SimulationConfig:
    """Construct the base SimulationConfig, dropping any unknown kwargs."""
    return SimulationConfig(**filter_kwargs_for_config(BASE_CONFIG_KWARGS))


def n_time_steps_for_duration(T_ns: float) -> int:
    """Adaptive step count: at most MAX_N_TIME_STEPS, at least 1 ns per segment."""
    return max(
        1,
        min(MAX_N_TIME_STEPS, int(np.floor(T_ns / MIN_SEGMENT_DURATION_NS))),
    )


# ============================================================
# Worker
# ============================================================

def run_one_seed(task: tuple[Any, ...]) -> dict[str, Any]:
    """
    Run one (theta, duration, init) grid point and return results as a flat dict.

    E2 fix: dqe_optimisation with entanglement_optimisation=True runs the
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

    # Adaptive step count: short pulses get fewer segments.
    n_steps = n_time_steps_for_duration(T_ns)

    common_kwargs = dict(
        theta_target=float(theta),
        T=float(T_ns),
        n_time_steps=int(n_steps),
        J_init_seed=int(seed),
        J_init_mean=float(j_init_mean),
        S_max=float(ENTANGLEMENT_CAP),
        eps_infidelity=float(EPS_INFIDELITY),
        rho=float(RHO),
        dual_lr=float(DUAL_LR),
        n_outer=int(N_OUTER),
        inner_maxiter=int(INNER_MAXITER),
        fd_eps=float(FD_EPS),
        n_J_channels=N_J_CHANNELS,
    )

    # E2: one call only — baseline is stored in ent_result.baseline.
    ent_config = safe_replace_config(base_config, **common_kwargs,
                                     entanglement_optimisation=True)
    ent_result = dqe_optimisation(ent_config)

    baseline_arrays = extract_state_result_arrays(ent_result.baseline)
    ent_arrays = extract_state_result_arrays(ent_result)

    baseline_final_infid = max(float(baseline_arrays["final_infid"]), INFIDELITY_FLOOR)
    ent_final_infid = max(float(ent_arrays["final_infid"]), INFIDELITY_FLOOR)

    return dict(
        i_theta=i_theta,
        i_T=i_T,
        k_init=k_init,
        seed=seed,
        J_init_mean=j_init_mean,
        n_time_steps=n_steps,
        segment_duration_ns=T_ns / n_steps,

        baseline_success=(np.isfinite(baseline_final_infid)
                          and baseline_final_infid <= INFIDELITY_THRESHOLD),
        baseline_optimization_success=baseline_arrays["optimization_success"],
        baseline_final_infid=baseline_final_infid,
        baseline_s_avg=nanmean_safe(baseline_arrays["entropies"]),
        baseline_s_max=nanmax_safe(baseline_arrays["entropies"]),
        # NaN-pad to MAX width so all runs have the same pulse-array shape.
        baseline_x0=fixed_width_x(baseline_arrays["x0"], MAX_N_PULSE_PARAMS),
        baseline_x_final=fixed_width_x(baseline_arrays["x_final"], MAX_N_PULSE_PARAMS),
        baseline_n_iterations=baseline_arrays["n_iterations"],
        baseline_gradient_norm=baseline_arrays["gradient_norm"],

        ent_success=(np.isfinite(ent_final_infid)
                     and ent_final_infid <= INFIDELITY_THRESHOLD),
        ent_optimization_success=ent_arrays["optimization_success"],
        ent_final_infid=ent_final_infid,
        ent_s_avg=nanmean_safe(ent_arrays["entropies"]),
        ent_s_max=nanmax_safe(ent_arrays["entropies"]),
        ent_x0=fixed_width_x(ent_arrays["x0"], MAX_N_PULSE_PARAMS),
        ent_x_final=fixed_width_x(ent_arrays["x_final"], MAX_N_PULSE_PARAMS),
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

        baseline_s_avg=np.full(shape, np.nan),
        ent_s_avg=np.full(shape, np.nan),
        baseline_s_max=np.full(shape, np.nan),
        ent_s_max=np.full(shape, np.nan),

        baseline_x0=np.full(shape + (MAX_N_PULSE_PARAMS,), np.nan),
        ent_x0=np.full(shape + (MAX_N_PULSE_PARAMS,), np.nan),
        baseline_x_final=np.full(shape + (MAX_N_PULSE_PARAMS,), np.nan),
        ent_x_final=np.full(shape + (MAX_N_PULSE_PARAMS,), np.nan),

        baseline_n_iterations=np.full(shape, np.nan),
        ent_n_iterations=np.full(shape, np.nan),
        baseline_gradient_norms=np.full(shape, np.nan),
        ent_gradient_norms=np.full(shape, np.nan),

        seeds=np.full(shape, -1, dtype=np.int64),
        J_init_means=np.full(shape, np.nan),

        # -1 sentinel = not yet computed (adaptive, so varies by grid point).
        n_time_steps_used=np.full((n_theta, n_T), -1, dtype=int),
        segment_duration_ns=np.full((n_theta, n_T), np.nan),
    )


def update_arrays(arrays: dict[str, Any], out: dict[str, Any]) -> None:
    """Write a single worker result into the global arrays at (i, j, k).

    B6 fix: n_time_steps_used and segment_duration_ns are now updated here
    (they were missing in the original mixedJ version, unlike positiveJ).
    """
    i, j, k = out["i_theta"], out["i_T"], out["k_init"]

    arrays["baseline_successes"][i, j, k] = out["baseline_success"]
    arrays["ent_successes"][i, j, k] = out["ent_success"]
    arrays["baseline_optimization_successes"][i, j, k] = out["baseline_optimization_success"]
    arrays["ent_optimization_successes"][i, j, k] = out["ent_optimization_success"]

    arrays["baseline_final_infids"][i, j, k] = out["baseline_final_infid"]
    arrays["ent_final_infids"][i, j, k] = out["ent_final_infid"]

    arrays["baseline_s_avg"][i, j, k] = out["baseline_s_avg"]
    arrays["ent_s_avg"][i, j, k] = out["ent_s_avg"]
    arrays["baseline_s_max"][i, j, k] = out["baseline_s_max"]
    arrays["ent_s_max"][i, j, k] = out["ent_s_max"]

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

    # B6 fix: record adaptive step count per (i, j) — was missing in original.
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
        initial_state_label=INITIAL_STATE_LABEL,
        initial_state=INITIAL_STATE,
        zeeman=ZEEMAN,

        J_min=BASE_CONFIG_KWARGS["J_min"],
        J_max=BASE_CONFIG_KWARGS["J_max"],
        J_init_scale=BASE_CONFIG_KWARGS["J_init_scale"],
        J_init_means_config=J_INIT_MEANS,
        n_inits=N_INITS,
        seed0=SEED0,
        n_J_channels=N_J_CHANNELS,

        infidelity_threshold=INFIDELITY_THRESHOLD,
        infidelity_floor=INFIDELITY_FLOOR,
        entanglement_cap=ENTANGLEMENT_CAP,
        state_ent_measure=BASE_CONFIG_KWARGS["state_ent_measure"],
        eps_infidelity=EPS_INFIDELITY,
        rho=RHO,
        dual_lr=DUAL_LR,
        n_outer=N_OUTER,
        inner_maxiter=INNER_MAXITER,
        fd_eps=FD_EPS,
        max_n_time_steps=MAX_N_TIME_STEPS,
        max_n_pulse_params=MAX_N_PULSE_PARAMS,
        min_segment_duration_ns=MIN_SEGMENT_DURATION_NS,
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

    # Pre-compute adaptive step counts and fill grid metadata before dispatch.
    for i, theta in enumerate(THETAS):
        for j, T_ns in enumerate(DURATIONS_NS):
            n_steps = n_time_steps_for_duration(float(T_ns))
            arrays["n_time_steps_used"][i, j] = n_steps
            arrays["segment_duration_ns"][i, j] = float(T_ns) / n_steps

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
    print("Generating DQE-state pulse database (|0011⟩ initial state, mixedJ)")
    print(f"Output     = {RAW_OUT}")
    print(f"N theta    = {len(THETAS)}")
    print(f"N duration = {len(DURATIONS_NS)}")
    print(f"N inits    = {N_INITS}")
    print(f"N channels = {N_J_CHANNELS}")
    print(f"N jobs     = {len(tasks)}")
    print("=" * 72)

    chunksize = max(1, len(tasks) // (N_WORKERS * 20))

    with mp.Pool(processes=N_WORKERS) as pool:
        iterator = pool.imap_unordered(run_one_seed, tasks, chunksize=chunksize)

        for n_done, out in enumerate(
            tqdm(iterator, total=len(tasks), desc="DQE-state |0011⟩ mixedJ jobs",
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
