#!/usr/bin/env python3
"""
generate_sqe_gate_pulses_positiveJ_rowprop.py

SQE-gate pulse database with theta-continuation (row-propagation).

Grid convention
---------------
    All saved arrays have shape (n_theta, n_T, n_init_slots).

Worker strategy
---------------
One worker process handles one fixed duration T and scans theta from low to high.

For each theta:
  Baseline (infidelity-only):
    - First N_BOOTSTRAP_THETAS thetas: random initialisation only.
    - Later thetas: try propagating the best accepted baseline pulse from the
      previous theta.  If propagation passes the acceptance test and
      SKIP_RANDOM_IF_BASELINE_PROP_ACCEPTED is True, skip the random starts.
    - If propagation fails or is skipped, run N_RANDOM_INITS random starts.

  TEM (entropy-penalised):
    - Same bootstrap/propagation logic as baseline.
    - Propagation: first try infidelity-only repair from the previous TEM pulse;
      if that fails the entanglement acceptance test, follow with a primal-dual
      constrained run starting from the repaired pulse.
    - Acceptance: infidelity below threshold AND (≥20% entropy reduction OR
      S_tem ≤ TEM_ABSOLUTE_ENT_THRESHOLD).

Output
------
    PULSES_DATABASE/sqe_gate_positiveJ_rowprop_pulses.npz

Extra metadata arrays (propagation bookkeeping) are saved alongside the standard
arrays (baseline_final_infids, ent_final_infids, etc.); old plotting scripts can
ignore the extras.
"""

from __future__ import annotations

import os
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# Limit BLAS/OpenMP threads per worker to avoid oversubscription.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# pulse_vqe_entanglement is not pip-installed; add the repo root to sys.path.
import sys as _sys
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
del _sys, _ROOT

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.sqe_gate import sqe_gate_optimisation

# Shared helpers: config manipulation, result extraction, array utilities.
from thesis_utils import (
    CONFIG_FIELDS,           # set of valid SimulationConfig field names
    filter_kwargs_for_config,
    safe_replace_config,
    extract_gate_result_arrays,
    fixed_width_x,
    nanmean_safe,            # replaces average_entanglement
    nanmax_safe,             # replaces max_entanglement
)


# ============================================================
# Settings
# ============================================================

# B1: resolved relative to this file so the script works on any machine.
PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"

SYSTEM_NAME = "sqe_gate"
PULSE_FAMILY_LABEL = "positiveJ_rowprop"

OUTDIR = PULSES_DATABASE
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUTDIR / f"{SYSTEM_NAME}_{PULSE_FAMILY_LABEL}_pulses.npz"

# Theta grid: fine near zero + coarser large-angle sweep.
small_thetas = np.linspace(0.0 * np.pi, 0.009 * np.pi, 10)
large_thetas = np.linspace(0.01 * np.pi, 0.50 * np.pi, 40)
THETAS = np.concatenate([small_thetas, large_thetas])

DURATIONS_NS = np.arange(1.0, 1200.0, 15.0)   # one row-worker per duration

# Each grid point has two init slots:
#   PROP_SLOT (0): used by the propagated-init run (when available)
#   RANDOM_SLOT0 (1): random-init run
N_RANDOM_INITS = 1
N_INIT_SLOTS = N_RANDOM_INITS + 1
PROP_SLOT = 0
RANDOM_SLOT0 = 1

N_WORKERS = 9         # parallel row-worker processes
SEED0 = 1234          # master RNG seed for reproducibility
CHECKPOINT_EVERY_ROWS = 1  # save after every completed duration row

INFIDELITY_THRESHOLD = 1e-5
INFIDELITY_FLOOR = 1e-14

N_TIME_STEPS = 50             # fixed for SQE gate

# Augmented Lagrangian / TEM optimiser hyperparameters.
N_OUTER = 3
INNER_MAXITER = 200
EPS_INFIDELITY = 1e-12
DUAL_LR = 1e10
RHO = 0.0
FD_EPS = 1e-6

GATE_ENT_MEASURE = "operator_entanglement"
ENTANGLEMENT_MARGIN = 0.01  # S_max for gate entanglement constraint

# B4: was [0.0]*(N//2) + [0.0]*(N-N//2) which always gives zeros.
J_INIT_MEANS = np.zeros(N_RANDOM_INITS, dtype=float)

ZEEMAN = np.array([1000.0, 1008.0])                                    # MHz
DUMMY_INITIAL_STATE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

# ------------------------------------------------------------
# Propagation controls
# ------------------------------------------------------------

# Thetas with index < N_BOOTSTRAP_THETAS always use random initialisations
# (no propagation attempt) so the chain has a clean starting point.
N_BOOTSTRAP_THETAS = 2

# Baseline propagation acceptance thresholds.
BASELINE_PROPAGATE_INF_THRESHOLD = 1e-4
BASELINE_MAX_LOG10_WORSENING = 5.0   # allow up to 10^5 worse infidelity

# TEM propagation acceptance thresholds.
TEM_PROPAGATE_INF_THRESHOLD = 1e-4
TEM_MAX_LOG10_WORSENING = 5.0
TEM_MIN_RELATIVE_ENT_REDUCTION = 0.20   # ≥20% entropy reduction required
TEM_ABSOLUTE_ENT_THRESHOLD = 0.5        # OR absolute S_tem below this
ENT_REDUCTION_DENOM_FLOOR = 1e-8       # floor in relative reduction denominator

# Skip random initialisations if propagation is accepted.
SKIP_RANDOM_IF_BASELINE_PROP_ACCEPTED = True
SKIP_RANDOM_IF_TEM_PROP_ACCEPTED = True

# Name of the SimulationConfig field used to inject an explicit initial pulse.
# If the field doesn't exist in the config, propagation will raise at runtime.
CONFIG_X0_FIELD_CANDIDATES = ["x0_override"]

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
    S_max=ENTANGLEMENT_MARGIN,
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
# Propagation helpers (unique to rowprop — not in thesis_utils)
# ============================================================

def infer_x0_field() -> str | None:
    """Return the first CONFIG_X0_FIELD_CANDIDATES name that exists in SimulationConfig."""
    for name in CONFIG_X0_FIELD_CANDIDATES:
        if name in CONFIG_FIELDS:
            return name
    return None


CONFIG_X0_FIELD = infer_x0_field()


def add_initial_x(kwargs: dict[str, Any], x0: np.ndarray | None) -> dict[str, Any]:
    """Inject x0 into the kwargs dict via the detected CONFIG_X0_FIELD."""
    kwargs = dict(kwargs)
    if x0 is None:
        return kwargs
    if CONFIG_X0_FIELD is None:
        raise RuntimeError(
            "No explicit initial-pulse field found in SimulationConfig. "
            "Add the correct name to CONFIG_X0_FIELD_CANDIDATES."
        )
    kwargs[CONFIG_X0_FIELD] = np.asarray(x0, dtype=float).reshape(-1)
    return kwargs


def clean_infid(x: float) -> float:
    """Clip infidelity to INFIDELITY_FLOOR; return NaN for non-finite values."""
    if not np.isfinite(x):
        return np.nan
    return max(float(x), INFIDELITY_FLOOR)


def log10_infid(x: float) -> float:
    """log10 of infidelity (clamped to floor); returns inf for non-finite input."""
    if not np.isfinite(x):
        return np.inf
    return float(np.log10(max(float(x), INFIDELITY_FLOOR)))


def relative_ent_reduction(S_base: float, S_tem: float) -> float:
    """(S_base - S_tem) / max(S_base, floor); NaN if either is non-finite."""
    if not np.isfinite(S_base) or not np.isfinite(S_tem):
        return np.nan
    denom = max(float(S_base), ENT_REDUCTION_DENOM_FLOOR)
    return float((S_base - S_tem) / denom)


# ============================================================
# Single optimisation calls
# ============================================================

def common_config_kwargs(
    theta: float, T_ns: float, seed: int | None, j_init_mean: float
) -> dict[str, Any]:
    """Return per-job config overrides shared by baseline and TEM runs."""
    return dict(
        theta_target=float(theta),
        T=float(T_ns),
        n_time_steps=int(N_TIME_STEPS),
        J_init_seed=None if seed is None else int(seed),
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


def run_baseline_once(
    *,
    base_config: SimulationConfig,
    theta: float,
    T_ns: float,
    seed: int | None,
    j_init_mean: float,
    x0: np.ndarray | None,
) -> dict[str, Any]:
    """Run one infidelity-only optimisation and return a flat result dict."""
    kwargs = common_config_kwargs(theta, T_ns, seed, j_init_mean)
    kwargs = add_initial_x(kwargs, x0)

    config = safe_replace_config(base_config, **kwargs, entanglement_optimisation=False)
    result = sqe_gate_optimisation(config)
    arr = extract_gate_result_arrays(result)

    I = clean_infid(arr["final_infid"])
    return dict(
        success=np.isfinite(I) and I <= INFIDELITY_THRESHOLD,
        optimization_success=arr["optimization_success"],
        final_infid=I,
        ent_avg=nanmean_safe(arr["entanglements"]),
        ent_max=nanmax_safe(arr["entanglements"]),
        # Fall back to x0 if x_initial not recorded by the optimiser.
        x0=fixed_width_x(arr["x0"] if arr["x0"] is not None else x0, N_TIME_STEPS),
        x_final=fixed_width_x(arr["x_final"], N_TIME_STEPS),
        n_iterations=arr["n_iterations"],
        gradient_norm=arr["gradient_norm"],
    )


def run_tem_once(
    *,
    base_config: SimulationConfig,
    theta: float,
    T_ns: float,
    seed: int | None,
    j_init_mean: float,
    x0: np.ndarray | None,
    entanglement_optimisation: bool,
) -> dict[str, Any]:
    """Run one TEM (or baseline-repair) optimisation and return a flat result dict."""
    kwargs = common_config_kwargs(theta, T_ns, seed, j_init_mean)
    kwargs = add_initial_x(kwargs, x0)

    config = safe_replace_config(base_config, **kwargs,
                                 entanglement_optimisation=bool(entanglement_optimisation))
    result = sqe_gate_optimisation(config)
    arr = extract_gate_result_arrays(result)

    I = clean_infid(arr["final_infid"])
    return dict(
        success=np.isfinite(I) and I <= INFIDELITY_THRESHOLD,
        optimization_success=arr["optimization_success"],
        final_infid=I,
        ent_avg=nanmean_safe(arr["entanglements"]),
        ent_max=nanmax_safe(arr["entanglements"]),
        x0=fixed_width_x(arr["x0"] if arr["x0"] is not None else x0, N_TIME_STEPS),
        x_final=fixed_width_x(arr["x_final"], N_TIME_STEPS),
        n_iterations=arr["n_iterations"],
        gradient_norm=arr["gradient_norm"],
        used_primal_dual=bool(entanglement_optimisation),
    )


# ============================================================
# Acceptance tests
# ============================================================

def baseline_propagation_ok(*, I_new: float, I_prev: float) -> bool:
    """True if the propagated pulse passes infidelity and worsening thresholds."""
    if not np.isfinite(I_new):
        return False
    if I_new > BASELINE_PROPAGATE_INF_THRESHOLD:
        return False
    if np.isfinite(I_prev):
        if log10_infid(I_new) > log10_infid(I_prev) + BASELINE_MAX_LOG10_WORSENING:
            return False
    return True


def tem_entanglement_ok(*, S_base: float, S_tem: float) -> bool:
    """True if TEM achieves sufficient entropy reduction OR hits the absolute floor."""
    red = relative_ent_reduction(S_base, S_tem)
    if np.isfinite(red) and red >= TEM_MIN_RELATIVE_ENT_REDUCTION:
        return True
    if np.isfinite(S_tem) and S_tem <= TEM_ABSOLUTE_ENT_THRESHOLD:
        return True
    return False


def tem_propagation_ok(*, I_new: float, I_prev: float, S_base: float, S_tem: float) -> bool:
    """True if the propagated TEM pulse passes both infidelity and entanglement tests."""
    if not np.isfinite(I_new):
        return False
    if I_new > TEM_PROPAGATE_INF_THRESHOLD:
        return False
    if np.isfinite(I_prev):
        if log10_infid(I_new) > log10_infid(I_prev) + TEM_MAX_LOG10_WORSENING:
            return False
    return tem_entanglement_ok(S_base=S_base, S_tem=S_tem)


# ============================================================
# Selection helpers within one grid point
# ============================================================

def select_best_baseline(
    local_results: list[tuple[int, dict[str, Any]]]
) -> tuple[int, dict[str, Any]] | None:
    """Return the (slot, result) with the lowest finite infidelity, or None."""
    valid = [(k, r) for k, r in local_results if np.isfinite(r["final_infid"])]
    if not valid:
        return None
    return min(valid, key=lambda kr: kr[1]["final_infid"])


def select_best_tem(
    local_results: list[tuple[int, dict[str, Any]]]
) -> tuple[int, dict[str, Any]] | None:
    """Return the best TEM result: prefer low entropy among successful runs."""
    successful = [
        (k, r) for k, r in local_results
        if np.isfinite(r["final_infid"])
        and r["final_infid"] <= INFIDELITY_THRESHOLD
        and np.isfinite(r["ent_avg"])
    ]
    if successful:
        return min(successful, key=lambda kr: kr[1]["ent_avg"])
    # Fall back to lowest infidelity if none succeeded.
    valid = [(k, r) for k, r in local_results if np.isfinite(r["final_infid"])]
    if not valid:
        return None
    return min(valid, key=lambda kr: kr[1]["final_infid"])


# ============================================================
# Array storage
# ============================================================

def initialise_arrays() -> dict[str, Any]:
    """Allocate NaN/False-filled output arrays for the full (theta, T, slot) grid."""
    n_theta = len(THETAS)
    n_T = len(DURATIONS_NS)
    shape = (n_theta, n_T, N_INIT_SLOTS)

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

        # Propagation metadata (ignored by plotting scripts that don't know about it).
        baseline_used_propagated_init=np.zeros(shape, dtype=bool),
        ent_used_propagated_init=np.zeros(shape, dtype=bool),
        ent_used_primal_dual=np.zeros(shape, dtype=bool),

        baseline_propagation_accepted=np.zeros((n_theta, n_T), dtype=bool),
        ent_propagation_accepted=np.zeros((n_theta, n_T), dtype=bool),

        baseline_selected_slot=np.full((n_theta, n_T), -1, dtype=int),
        ent_selected_slot=np.full((n_theta, n_T), -1, dtype=int),

        baseline_parent_theta_idx=np.full((n_theta, n_T), -1, dtype=int),
        ent_parent_theta_idx=np.full((n_theta, n_T), -1, dtype=int),

        ent_relative_reduction=np.full(shape, np.nan),
    )

    # Pre-fill static grid metadata (fixed step count for SQE).
    for i in range(n_theta):
        for j, T_ns in enumerate(DURATIONS_NS):
            arrays["n_time_steps_used"][i, j] = N_TIME_STEPS
            arrays["segment_duration_ns"][i, j] = float(T_ns) / N_TIME_STEPS

    return arrays


def store_baseline_result(
    arrays: dict[str, Any], *, i: int, j: int, k: int,
    seed: int, j_init_mean: float, result: dict[str, Any],
    used_propagated: bool, parent_theta_idx: int = -1,
) -> None:
    """Write a baseline run into arrays at grid point (i, j, k)."""
    arrays["baseline_successes"][i, j, k] = result["success"]
    arrays["baseline_optimization_successes"][i, j, k] = result["optimization_success"]
    arrays["baseline_final_infids"][i, j, k] = result["final_infid"]
    arrays["baseline_ent_avg"][i, j, k] = result["ent_avg"]
    arrays["baseline_ent_max"][i, j, k] = result["ent_max"]
    arrays["baseline_x0"][i, j, k] = result["x0"]
    arrays["baseline_x_final"][i, j, k] = result["x_final"]
    arrays["baseline_n_iterations"][i, j, k] = result["n_iterations"]
    arrays["baseline_gradient_norms"][i, j, k] = result["gradient_norm"]
    arrays["seeds"][i, j, k] = seed
    arrays["J_init_means"][i, j, k] = j_init_mean
    arrays["baseline_used_propagated_init"][i, j, k] = used_propagated
    arrays["baseline_parent_theta_idx"][i, j] = parent_theta_idx


def store_tem_result(
    arrays: dict[str, Any], *, i: int, j: int, k: int,
    seed: int, j_init_mean: float, result: dict[str, Any],
    used_propagated: bool, parent_theta_idx: int = -1,
    S_base: float = np.nan,
) -> None:
    """Write a TEM run into arrays at grid point (i, j, k)."""
    arrays["ent_successes"][i, j, k] = result["success"]
    arrays["ent_optimization_successes"][i, j, k] = result["optimization_success"]
    arrays["ent_final_infids"][i, j, k] = result["final_infid"]
    arrays["ent_ent_avg"][i, j, k] = result["ent_avg"]
    arrays["ent_ent_max"][i, j, k] = result["ent_max"]
    arrays["ent_x0"][i, j, k] = result["x0"]
    arrays["ent_x_final"][i, j, k] = result["x_final"]
    arrays["ent_n_iterations"][i, j, k] = result["n_iterations"]
    arrays["ent_gradient_norms"][i, j, k] = result["gradient_norm"]
    arrays["seeds"][i, j, k] = seed
    arrays["J_init_means"][i, j, k] = j_init_mean
    arrays["ent_used_propagated_init"][i, j, k] = used_propagated
    arrays["ent_parent_theta_idx"][i, j] = parent_theta_idx
    arrays["ent_used_primal_dual"][i, j, k] = bool(result.get("used_primal_dual", False))
    arrays["ent_relative_reduction"][i, j, k] = relative_ent_reduction(
        S_base, result["ent_avg"]
    )


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

        n_inits=N_INIT_SLOTS,
        n_random_inits=N_RANDOM_INITS,
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

        n_bootstrap_thetas=N_BOOTSTRAP_THETAS,
        baseline_propagate_inf_threshold=BASELINE_PROPAGATE_INF_THRESHOLD,
        baseline_max_log10_worsening=BASELINE_MAX_LOG10_WORSENING,
        tem_propagate_inf_threshold=TEM_PROPAGATE_INF_THRESHOLD,
        tem_max_log10_worsening=TEM_MAX_LOG10_WORSENING,
        tem_min_relative_ent_reduction=TEM_MIN_RELATIVE_ENT_REDUCTION,
        tem_absolute_ent_threshold=TEM_ABSOLUTE_ENT_THRESHOLD,
        ent_reduction_denom_floor=ENT_REDUCTION_DENOM_FLOOR,
        config_x0_field="" if CONFIG_X0_FIELD is None else CONFIG_X0_FIELD,

        **arrays,
    )


# ============================================================
# Row worker: scans theta for one fixed duration
# ============================================================

def run_one_duration_row(task: tuple[Any, ...]) -> dict[str, Any]:
    """
    Process one full theta sweep at a fixed duration T_ns.

    Returns a dict with keys 'j', 'T_ns', 'arrays' (the full initialised-array
    structure, written only for column j and returned to the main process for
    merging via merge_row).
    """
    j, T_ns, seed_row, base_config = task
    rng = np.random.default_rng(int(seed_row))
    arrays = initialise_arrays()

    # Propagation state: track accepted pulses and their infidelities.
    prev_baseline_x: np.ndarray | None = None
    prev_baseline_I: float = np.nan
    prev_baseline_theta_idx: int = -1

    prev_tem_x: np.ndarray | None = None
    prev_tem_I: float = np.nan
    prev_tem_theta_idx: int = -1

    for i, theta in enumerate(THETAS):
        # ------------------------------------------------
        # Baseline (infidelity-only) scan for this theta
        # ------------------------------------------------
        baseline_results: list[tuple[int, dict[str, Any]]] = []
        baseline_prop_accepted = False

        if i >= N_BOOTSTRAP_THETAS and prev_baseline_x is not None:
            # Try propagated warm-start.
            prop_result = run_baseline_once(
                base_config=base_config, theta=float(theta), T_ns=float(T_ns),
                seed=None, j_init_mean=0.0, x0=prev_baseline_x,
            )
            store_baseline_result(
                arrays, i=i, j=j, k=PROP_SLOT, seed=-1, j_init_mean=np.nan,
                result=prop_result, used_propagated=True,
                parent_theta_idx=prev_baseline_theta_idx,
            )
            baseline_results.append((PROP_SLOT, prop_result))

            baseline_prop_accepted = baseline_propagation_ok(
                I_new=prop_result["final_infid"], I_prev=prev_baseline_I,
            )
            arrays["baseline_propagation_accepted"][i, j] = baseline_prop_accepted

        # Run random initialisations if needed.
        run_randoms = (
            i < N_BOOTSTRAP_THETAS
            or not baseline_prop_accepted
            or not SKIP_RANDOM_IF_BASELINE_PROP_ACCEPTED
        )

        if run_randoms:
            for r in range(N_RANDOM_INITS):
                k = r if i < N_BOOTSTRAP_THETAS else RANDOM_SLOT0 + r
                if k >= N_INIT_SLOTS:
                    continue
                seed = int(rng.integers(0, 2**31 - 1))
                j_mean = float(J_INIT_MEANS[r % len(J_INIT_MEANS)])
                res = run_baseline_once(
                    base_config=base_config, theta=float(theta), T_ns=float(T_ns),
                    seed=seed, j_init_mean=j_mean, x0=None,
                )
                store_baseline_result(
                    arrays, i=i, j=j, k=k, seed=seed, j_init_mean=j_mean,
                    result=res, used_propagated=False,
                )
                baseline_results.append((k, res))

        selected_baseline = select_best_baseline(baseline_results)
        selected_baseline_S = np.nan

        if selected_baseline is not None:
            k_sel, r_sel = selected_baseline
            arrays["baseline_selected_slot"][i, j] = k_sel
            prev_baseline_x = np.asarray(r_sel["x_final"], dtype=float)
            prev_baseline_I = float(r_sel["final_infid"])
            prev_baseline_theta_idx = i
            selected_baseline_S = float(r_sel["ent_avg"])

        # ------------------------------------------------
        # TEM (entropy-penalised) scan for this theta
        # ------------------------------------------------
        tem_results: list[tuple[int, dict[str, Any]]] = []
        tem_prop_accepted = False

        if i >= N_BOOTSTRAP_THETAS and prev_tem_x is not None:
            # Step 1: infidelity-only repair using the propagated TEM pulse.
            repair_result = run_tem_once(
                base_config=base_config, theta=float(theta), T_ns=float(T_ns),
                seed=None, j_init_mean=0.0, x0=prev_tem_x,
                entanglement_optimisation=False,
            )

            repair_ok = tem_propagation_ok(
                I_new=repair_result["final_infid"], I_prev=prev_tem_I,
                S_base=selected_baseline_S, S_tem=repair_result["ent_avg"],
            )

            if repair_ok:
                prop_result = repair_result
            else:
                # Step 2: constrained (primal-dual) run from repaired pulse.
                prop_result = run_tem_once(
                    base_config=base_config, theta=float(theta), T_ns=float(T_ns),
                    seed=None, j_init_mean=0.0,
                    x0=np.asarray(repair_result["x_final"], dtype=float),
                    entanglement_optimisation=True,
                )

            store_tem_result(
                arrays, i=i, j=j, k=PROP_SLOT, seed=-1, j_init_mean=np.nan,
                result=prop_result, used_propagated=True,
                parent_theta_idx=prev_tem_theta_idx, S_base=selected_baseline_S,
            )
            tem_results.append((PROP_SLOT, prop_result))

            tem_prop_accepted = tem_propagation_ok(
                I_new=prop_result["final_infid"], I_prev=prev_tem_I,
                S_base=selected_baseline_S, S_tem=prop_result["ent_avg"],
            )
            arrays["ent_propagation_accepted"][i, j] = tem_prop_accepted

        run_tem_randoms = (
            i < N_BOOTSTRAP_THETAS
            or not tem_prop_accepted
            or not SKIP_RANDOM_IF_TEM_PROP_ACCEPTED
        )

        if run_tem_randoms:
            for r in range(N_RANDOM_INITS):
                k = r if i < N_BOOTSTRAP_THETAS else RANDOM_SLOT0 + r
                if k >= N_INIT_SLOTS:
                    continue
                seed = int(rng.integers(0, 2**31 - 1))
                j_mean = float(J_INIT_MEANS[r % len(J_INIT_MEANS)])
                res = run_tem_once(
                    base_config=base_config, theta=float(theta), T_ns=float(T_ns),
                    seed=seed, j_init_mean=j_mean, x0=None,
                    entanglement_optimisation=True,
                )
                store_tem_result(
                    arrays, i=i, j=j, k=k, seed=seed, j_init_mean=j_mean,
                    result=res, used_propagated=False, S_base=selected_baseline_S,
                )
                tem_results.append((k, res))

        selected_tem = select_best_tem(tem_results)

        if selected_tem is not None:
            k_sel, r_sel = selected_tem
            arrays["ent_selected_slot"][i, j] = k_sel

            # Only propagate if the pulse is both low-infidelity and low-entropy.
            if (
                np.isfinite(r_sel["final_infid"])
                and r_sel["final_infid"] <= TEM_PROPAGATE_INF_THRESHOLD
                and tem_entanglement_ok(S_base=selected_baseline_S,
                                        S_tem=r_sel["ent_avg"])
            ):
                prev_tem_x = np.asarray(r_sel["x_final"], dtype=float)
                prev_tem_I = float(r_sel["final_infid"])
                prev_tem_theta_idx = i
            else:
                # Reset propagation chain on failure.
                prev_tem_x = None
                prev_tem_I = np.nan
                prev_tem_theta_idx = -1

    return dict(j=j, T_ns=T_ns, arrays=arrays)


def merge_row(global_arrays: dict[str, Any], row_out: dict[str, Any]) -> None:
    """Merge column j of the row-worker's arrays into the global arrays."""
    j = int(row_out["j"])
    row_arrays = row_out["arrays"]

    for key, arr in global_arrays.items():
        if arr.ndim >= 2 and arr.shape[1] == len(DURATIONS_NS):
            arr[:, j, ...] = row_arrays[key][:, j, ...]
        else:
            global_arrays[key] = row_arrays[key]


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Build the full row-propagated pulse database using a multiprocessing pool."""
    base_config = make_base_config()
    arrays = initialise_arrays()

    if CONFIG_X0_FIELD is None:
        print("=" * 72)
        print("WARNING: No initial-pulse field found in SimulationConfig.")
        print("Propagation will fail.  Add the field name to CONFIG_X0_FIELD_CANDIDATES.")
        print("=" * 72)
    else:
        print(f"Propagated initial pulse uses SimulationConfig field: {CONFIG_X0_FIELD!r}")

    rng = np.random.default_rng(SEED0)

    # One task per duration column; each worker scans all thetas for that T.
    tasks = [
        (j, float(T_ns), int(rng.integers(0, 2**31 - 1)), base_config)
        for j, T_ns in enumerate(DURATIONS_NS)
    ]

    print("=" * 72)
    print("Generating SQE-gate row-propagated pulse database (positiveJ)")
    print(f"Output     = {RAW_OUT}")
    print(f"J_min/max  = {BASE_CONFIG_KWARGS['J_min']} / {BASE_CONFIG_KWARGS['J_max']} MHz")
    print(f"N theta    = {len(THETAS)}")
    print(f"N duration = {len(DURATIONS_NS)}")
    print(f"N rnd init = {N_RANDOM_INITS}")
    print(f"N slots    = {N_INIT_SLOTS}")
    print(f"N steps    = {N_TIME_STEPS}")
    print(f"Bootstrap  = {N_BOOTSTRAP_THETAS} thetas")
    print(f"N rows     = {len(tasks)}")
    print("=" * 72)

    completed_rows = 0

    with mp.Pool(processes=N_WORKERS) as pool:
        iterator = pool.imap_unordered(run_one_duration_row, tasks, chunksize=1)

        for row_out in tqdm(iterator, total=len(tasks), desc="duration rows",
                             unit="row", dynamic_ncols=True):
            merge_row(arrays, row_out)
            completed_rows += 1

            if CHECKPOINT_EVERY_ROWS and completed_rows % CHECKPOINT_EVERY_ROWS == 0:
                save_raw(arrays)

    save_raw(arrays)
    print(f"\nDone.  Saved to:\n  {RAW_OUT}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
