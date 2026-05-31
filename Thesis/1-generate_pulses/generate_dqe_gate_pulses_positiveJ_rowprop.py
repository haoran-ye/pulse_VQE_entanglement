#!/usr/bin/env python3
"""
generate_dqe_gate_pulses_positiveJ_rowprop.py

DQE-gate pulse database with theta-continuation (row-propagation),
positive exchange coupling (J ∈ [0, 10] MHz).

For each fixed duration T, one worker scans theta from low to high,
warm-starting the next theta from the accepted pulse at the previous theta.
This dramatically reduces failed optimisations at larger angles.

Grid convention
---------------
    All result arrays: (n_theta, n_T, n_init_slots)

Initialisation strategy
-----------------------
    Bootstrap region (i < N_BOOTSTRAP_THETAS):
        Normal random initialisations only.

    Propagation region (i >= N_BOOTSTRAP_THETAS):
        Baseline:
            Try propagated pulse from previous accepted theta first.
            Fall back to random inits if propagation is not accepted.
        TEM:
            Step 1 — infidelity-only repair from propagated TEM pulse.
            Step 2 — if repair does not pass acceptance, run full primal-dual
                     optimisation initialised from the repaired pulse.
            Fall back to random inits if still not accepted.

Adaptive step count
-------------------
    n_time_steps = min(MAX_N_TIME_STEPS, floor(T / MIN_SEGMENT_DURATION_NS))
    Pulse parameter vectors are padded to MAX_N_PULSE_PARAMS with NaN before
    being stored so all arrays have a uniform shape.

TEM acceptance criterion (DQE-specific)
----------------------------------------
    Uses the four-qubit summed operator-entanglement measure
    ("operator_entanglement_sum_4q"), whose absolute scale is larger than the
    single-pair SQE measure.  A fixed absolute threshold is too strict; instead
    accept propagated TEM pulses if either:
        (a) relative entropy reduction >= TEM_MIN_RELATIVE_ENT_REDUCTION (10%), or
        (b) S_tem <= (1 + TEM_ALLOW_RELATIVE_ENT_WORSENING) * S_base (5% slack).

Output
------
    PULSES_DATABASE/dqe_gate_positiveJ_rowprop_pulses.npz
"""

from __future__ import annotations

import os
import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# ============================================================
# Thread control — prevent BLAS/MKL from spawning extra threads
# inside each worker process, which would cause over-subscription
# with N_WORKERS parallel processes.
# ============================================================
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
from pulse_vqe_entanglement.dqe_gate import dqe_gate_optimisation

# B1 fix: resolve output path relative to this file, not hardcoded absolute path.
# E1 fix: import shared utilities from thesis_utils instead of re-defining them.
from thesis_utils import (
    CONFIG_FIELDS,
    filter_kwargs_for_config,
    safe_replace_config,
    fixed_width_x,
    nanmean_safe,
    nanmax_safe,
    extract_gate_result_arrays,
)


# ============================================================
# Settings
# ============================================================

# B1 fix: path resolved relative to this script, not hardcoded.
PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"

SYSTEM_NAME = "dqe_gate"
PULSE_FAMILY_LABEL = "positiveJ_rowprop"

OUTDIR = PULSES_DATABASE
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUTDIR / f"{SYSTEM_NAME}_{PULSE_FAMILY_LABEL}_pulses.npz"

# Theta grid: fine near zero (small-angle regime) + coarser large-angle sweep.
small_thetas = np.linspace(0.0 * np.pi, 0.009 * np.pi, 5)
large_thetas = np.linspace(0.01 * np.pi, 0.50 * np.pi, 20)
THETAS = np.concatenate([small_thetas, large_thetas])

DURATIONS_NS = np.arange(1.0, 1200.0, 80.0)

N_RANDOM_INITS = 1
N_INIT_SLOTS = N_RANDOM_INITS + 1  # slot 0 reserved for propagated init when available
PROP_SLOT = 0                       # index of the propagation slot
RANDOM_SLOT0 = 1                    # index of first random-init slot

N_WORKERS = 9
SEED0 = 1234
CHECKPOINT_EVERY_ROWS = 1   # save checkpoint after every completed duration row

INFIDELITY_THRESHOLD = 1e-5
INFIDELITY_FLOOR = 1e-14

# Adaptive step count: n_steps = min(MAX_N_TIME_STEPS, floor(T / MIN_SEGMENT_NS)).
MAX_N_TIME_STEPS = 50
MIN_SEGMENT_DURATION_NS = 1.0
N_J_CHANNELS = 3                                        # three exchange coupling channels
MAX_N_PULSE_PARAMS = MAX_N_TIME_STEPS * N_J_CHANNELS    # = 150 (storage width)

# Augmented Lagrangian / TEM optimiser hyperparameters.
N_OUTER = 3
INNER_MAXITER = 200
EPS_INFIDELITY = 1e-12
DUAL_LR = 1e10
RHO = 0.0
FD_EPS = 1e-6

# Gate entanglement measure: sum of operator entanglement over all 7 bipartitions.
GATE_ENT_MEASURE = "operator_entanglement_sum_4q"
ENTANGLEMENT_MARGIN = 0.01  # S_max used by the constrained TEM optimiser

# B4 fix: all-zero means; the old code always produced zeros anyway.
J_INIT_MEANS = np.zeros(N_RANDOM_INITS, dtype=float)

ZEEMAN = np.array([1000.0, 1008.0, 1016.0, 1024.0])    # MHz
DUMMY_INITIAL_STATE = np.zeros(16, dtype=np.complex128)
DUMMY_INITIAL_STATE[0] = 1.0                            # |0000⟩ placeholder

# ------------------------------------------------------------
# Propagation controls
# ------------------------------------------------------------

# Warm-starting is skipped for the first N_BOOTSTRAP_THETAS points.
N_BOOTSTRAP_THETAS = 2

# Baseline acceptance: propagated pulse must achieve infidelity <= this threshold
# and must not be more than BASELINE_MAX_LOG10_WORSENING orders of magnitude worse
# than the previous accepted baseline.
BASELINE_PROPAGATE_INF_THRESHOLD = 1e-4
BASELINE_MAX_LOG10_WORSENING = 5.0

# TEM acceptance: same infidelity gate, plus entanglement improvement check.
TEM_PROPAGATE_INF_THRESHOLD = 1e-4
TEM_MAX_LOG10_WORSENING = 5.0

# DQE-specific: accept propagated TEM if it reduces entanglement by >= 10%,
# or if it is no worse than baseline by more than 5%.
TEM_MIN_RELATIVE_ENT_REDUCTION = 0.10
TEM_ALLOW_RELATIVE_ENT_WORSENING = 0.05
ENT_REDUCTION_DENOM_FLOOR = 1e-8   # prevents division by zero in relative reduction

# If a propagated init is accepted, skip random starts for that grid point.
SKIP_RANDOM_IF_BASELINE_PROP_ACCEPTED = True
SKIP_RANDOM_IF_TEM_PROP_ACCEPTED = True

# Field name in SimulationConfig that holds an explicit initial pulse vector.
CONFIG_X0_FIELD_CANDIDATES = ["x0_override"]


BASE_CONFIG_KWARGS: dict[str, Any] = dict(
    zeeman_splittings=ZEEMAN,
    initial_state=DUMMY_INITIAL_STATE,
    theta_target=0.0,
    n_time_steps=MAX_N_TIME_STEPS,
    n_discretisation=1,
    T=200.0,
    max_drive_strength=0.0,
    J_max=10.0,
    J_min=0.0,
    J_init_scale=1e-4,
    J_init_mean=0.0,
    feromagnetic=False,
    gtol=1e-12,
    maxiter=200,
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
    n_J_channels=N_J_CHANNELS,
)


# ============================================================
# Config helpers (propagation-specific)
# ============================================================

def infer_x0_field() -> str | None:
    """
    Find the SimulationConfig field used to pass an explicit initial pulse.

    Checks each name in CONFIG_X0_FIELD_CANDIDATES against the live set of
    dataclass fields.  Returns None if no candidate is found; the caller should
    warn and skip warm-starting in that case.
    """
    for name in CONFIG_X0_FIELD_CANDIDATES:
        if name in CONFIG_FIELDS:
            return name
    return None


CONFIG_X0_FIELD = infer_x0_field()


def make_base_config() -> SimulationConfig:
    """Construct the base SimulationConfig, dropping any unknown kwargs."""
    return SimulationConfig(**filter_kwargs_for_config(BASE_CONFIG_KWARGS))


def add_initial_x(kwargs: dict[str, Any], x0: np.ndarray | None) -> dict[str, Any]:
    """
    Inject an explicit initial pulse vector into a kwargs dict before config creation.

    If *x0* is None the dict is returned unchanged (random init will be used).
    Raises RuntimeError if SimulationConfig has no known x0 field.
    """
    kwargs = dict(kwargs)
    if x0 is None:
        return kwargs
    if CONFIG_X0_FIELD is None:
        raise RuntimeError(
            "No explicit initial-pulse field found in SimulationConfig. "
            "Add the correct field name to CONFIG_X0_FIELD_CANDIDATES."
        )
    kwargs[CONFIG_X0_FIELD] = np.asarray(x0, dtype=float).reshape(-1)
    return kwargs


# ============================================================
# Adaptive step count
# ============================================================

def n_time_steps_for_duration(T_ns: float) -> int:
    """
    Number of piecewise-constant segments for a pulse of duration T_ns.

    Clamps to [1, MAX_N_TIME_STEPS] so that each segment is at least
    MIN_SEGMENT_DURATION_NS nanoseconds long.
    """
    return max(
        1,
        min(MAX_N_TIME_STEPS, int(np.floor(float(T_ns) / MIN_SEGMENT_DURATION_NS))),
    )


def n_pulse_params_for_duration(T_ns: float) -> int:
    """Total number of pulse parameters = n_time_steps * N_J_CHANNELS."""
    return n_time_steps_for_duration(T_ns) * N_J_CHANNELS


# ============================================================
# Infidelity / entropy helpers (propagation acceptance)
# ============================================================

def clean_infid(x: float) -> float:
    """Clip infidelity to INFIDELITY_FLOOR; return NaN if not finite."""
    if not np.isfinite(x):
        return np.nan
    return max(float(x), INFIDELITY_FLOOR)


def log10_infid(x: float) -> float:
    """log10 of infidelity, clipped to INFIDELITY_FLOOR.  +inf if not finite."""
    if not np.isfinite(x):
        return np.inf
    return float(np.log10(max(float(x), INFIDELITY_FLOOR)))


def relative_ent_reduction(S_base: float, S_tem: float) -> float:
    """
    Fractional entropy reduction achieved by TEM relative to baseline.

        reduction = (S_base - S_tem) / max(S_base, ENT_REDUCTION_DENOM_FLOOR)

    Positive = TEM reduced entanglement.  NaN if either input is non-finite.
    """
    if not np.isfinite(S_base) or not np.isfinite(S_tem):
        return np.nan
    denom = max(float(S_base), ENT_REDUCTION_DENOM_FLOOR)
    return float((S_base - S_tem) / denom)


# ============================================================
# Single optimisation calls
# ============================================================

def common_config_kwargs(
    theta: float,
    T_ns: float,
    seed: int | None,
    j_init_mean: float,
    n_steps: int,
) -> dict[str, Any]:
    """Build the keyword arguments shared by baseline and TEM optimisation calls."""
    return dict(
        theta_target=float(theta),
        T=float(T_ns),
        n_time_steps=int(n_steps),
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
        n_J_channels=N_J_CHANNELS,
    )


def run_baseline_once(
    *,
    base_config: SimulationConfig,
    theta: float,
    T_ns: float,
    seed: int | None,
    j_init_mean: float,
    n_steps: int,
    x0: np.ndarray | None,
) -> dict[str, Any]:
    """
    Run one infidelity-only (no TEM) DQE-gate optimisation and return a flat result dict.

    Parameters
    ----------
    base_config : SimulationConfig
        Template config; per-run fields are overridden via safe_replace_config.
    theta : float
        Target rotation angle in radians.
    T_ns : float
        Total pulse duration in nanoseconds.
    seed : int | None
        RNG seed for random init; None means use the explicit *x0*.
    j_init_mean : float
        Mean of the Gaussian used to initialise J amplitudes.
    n_steps : int
        Number of piecewise-constant segments.
    x0 : np.ndarray | None
        Explicit initial pulse vector (warm start); None for random init.

    Returns
    -------
    dict with keys: success, optimization_success, final_infid, ent_avg, ent_max,
        x0, x_final, n_iterations, gradient_norm.
    """
    kwargs = common_config_kwargs(theta, T_ns, seed, j_init_mean, n_steps)
    kwargs = add_initial_x(kwargs, x0)
    config = safe_replace_config(base_config, **kwargs, entanglement_optimisation=False)

    result = dqe_gate_optimisation(config)
    arr = extract_gate_result_arrays(result)

    I = clean_infid(arr["final_infid"])
    return dict(
        success=np.isfinite(I) and I <= INFIDELITY_THRESHOLD,
        optimization_success=arr["optimization_success"],
        final_infid=I,
        ent_avg=nanmean_safe(arr["entanglements"]),
        ent_max=nanmax_safe(arr["entanglements"]),
        # Store x0: use the explicit warm-start if the result has no x0 field.
        x0=fixed_width_x(arr["x0"] if arr["x0"] is not None else x0, MAX_N_PULSE_PARAMS),
        x_final=fixed_width_x(arr["x_final"], MAX_N_PULSE_PARAMS),
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
    n_steps: int,
    x0: np.ndarray | None,
    entanglement_optimisation: bool,
) -> dict[str, Any]:
    """
    Run one TEM (or infidelity-only repair) DQE-gate optimisation.

    When *entanglement_optimisation* is False this acts as an infidelity-only repair
    from a propagated initial pulse, used before deciding whether to escalate to
    full primal-dual optimisation.

    Returns
    -------
    dict with same keys as run_baseline_once plus 'used_primal_dual'.
    """
    kwargs = common_config_kwargs(theta, T_ns, seed, j_init_mean, n_steps)
    kwargs = add_initial_x(kwargs, x0)
    config = safe_replace_config(
        base_config, **kwargs,
        entanglement_optimisation=bool(entanglement_optimisation),
    )

    result = dqe_gate_optimisation(config)
    arr = extract_gate_result_arrays(result)

    I = clean_infid(arr["final_infid"])
    return dict(
        success=np.isfinite(I) and I <= INFIDELITY_THRESHOLD,
        optimization_success=arr["optimization_success"],
        final_infid=I,
        ent_avg=nanmean_safe(arr["entanglements"]),
        ent_max=nanmax_safe(arr["entanglements"]),
        x0=fixed_width_x(arr["x0"] if arr["x0"] is not None else x0, MAX_N_PULSE_PARAMS),
        x_final=fixed_width_x(arr["x_final"], MAX_N_PULSE_PARAMS),
        n_iterations=arr["n_iterations"],
        gradient_norm=arr["gradient_norm"],
        used_primal_dual=bool(entanglement_optimisation),
    )


# ============================================================
# Propagation acceptance tests
# ============================================================

def baseline_propagation_ok(*, I_new: float, I_prev: float) -> bool:
    """
    Accept a propagated baseline pulse if:
      1. I_new is finite.
      2. I_new <= BASELINE_PROPAGATE_INF_THRESHOLD.
      3. I_new is not more than BASELINE_MAX_LOG10_WORSENING orders of magnitude
         worse than the previous accepted baseline.
    """
    if not np.isfinite(I_new):
        return False
    if I_new > BASELINE_PROPAGATE_INF_THRESHOLD:
        return False
    if np.isfinite(I_prev):
        if log10_infid(I_new) > log10_infid(I_prev) + BASELINE_MAX_LOG10_WORSENING:
            return False
    return True


def tem_entanglement_ok(*, S_base: float, S_tem: float) -> bool:
    """
    DQE TEM entanglement acceptance criterion.

    Uses a relative criterion because the 4-qubit summed operator-entanglement
    measure has a larger absolute scale than the 2-qubit SQE measure.

    Accept if either:
      (a) TEM reduces entanglement by >= TEM_MIN_RELATIVE_ENT_REDUCTION (10%), or
      (b) S_tem <= (1 + TEM_ALLOW_RELATIVE_ENT_WORSENING) * S_base  (5% slack).
    """
    if not np.isfinite(S_base) or not np.isfinite(S_tem):
        return False
    red = relative_ent_reduction(S_base, S_tem)
    if np.isfinite(red) and red >= TEM_MIN_RELATIVE_ENT_REDUCTION:
        return True
    return bool(S_tem <= (1.0 + TEM_ALLOW_RELATIVE_ENT_WORSENING) * S_base)


def tem_propagation_ok(
    *, I_new: float, I_prev: float, S_base: float, S_tem: float
) -> bool:
    """
    Accept a propagated TEM pulse if infidelity AND entanglement checks pass.

    Infidelity check mirrors baseline_propagation_ok (with TEM-specific thresholds).
    Entanglement check delegates to tem_entanglement_ok.
    """
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
    """
    Choose the baseline run with the lowest finite infidelity.

    Returns None if no run produced a finite infidelity.
    """
    valid = [(k, r) for k, r in local_results if np.isfinite(r["final_infid"])]
    if not valid:
        return None
    return min(valid, key=lambda kr: kr[1]["final_infid"])


def select_best_tem(
    local_results: list[tuple[int, dict[str, Any]]]
) -> tuple[int, dict[str, Any]] | None:
    """
    Choose the best TEM run, prioritising:
      1. Successful runs (infidelity <= threshold) with smallest ent_avg.
      2. If none succeed, the run with the smallest finite infidelity.

    Returns None if no run produced a finite infidelity.
    """
    successful = [
        (k, r) for k, r in local_results
        if np.isfinite(r["final_infid"])
        and r["final_infid"] <= INFIDELITY_THRESHOLD
        and np.isfinite(r["ent_avg"])
    ]
    if successful:
        return min(successful, key=lambda kr: kr[1]["ent_avg"])

    valid = [(k, r) for k, r in local_results if np.isfinite(r["final_infid"])]
    if not valid:
        return None
    return min(valid, key=lambda kr: kr[1]["final_infid"])


# ============================================================
# Array management
# ============================================================

def initialise_arrays() -> dict[str, Any]:
    """Allocate NaN/False-filled output arrays for the full (theta, T, init) grid."""
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

        # Pulse vectors padded to MAX_N_PULSE_PARAMS with NaN.
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

        # Adaptive step count per (theta, T) grid point.
        n_time_steps_used=np.full((n_theta, n_T), -1, dtype=int),
        segment_duration_ns=np.full((n_theta, n_T), np.nan),

        # Propagation metadata (ignored by old plotting scripts).
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

    # Pre-fill adaptive step-count arrays.
    for i in range(n_theta):
        for j, T_ns in enumerate(DURATIONS_NS):
            n_steps = n_time_steps_for_duration(float(T_ns))
            arrays["n_time_steps_used"][i, j] = n_steps
            arrays["segment_duration_ns"][i, j] = float(T_ns) / n_steps

    return arrays


def store_baseline_result(
    arrays: dict[str, Any],
    *,
    i: int,
    j: int,
    k: int,
    seed: int,
    j_init_mean: float,
    result: dict[str, Any],
    used_propagated: bool,
    parent_theta_idx: int = -1,
) -> None:
    """Write one baseline run result into the global arrays at indices (i, j, k)."""
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
    arrays: dict[str, Any],
    *,
    i: int,
    j: int,
    k: int,
    seed: int,
    j_init_mean: float,
    result: dict[str, Any],
    used_propagated: bool,
    parent_theta_idx: int = -1,
    S_base: float = np.nan,
) -> None:
    """
    Write one TEM run result into the global arrays at indices (i, j, k).

    S_base is the baseline entanglement at this grid point, used to compute
    the relative entropy reduction stored in ent_relative_reduction.
    """
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
    """Save the current state of all result arrays to the output .npz file."""
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

        max_n_time_steps=MAX_N_TIME_STEPS,
        max_n_pulse_params=MAX_N_PULSE_PARAMS,
        min_segment_duration_ns=MIN_SEGMENT_DURATION_NS,
        n_J_channels=N_J_CHANNELS,
        n_workers=N_WORKERS,

        n_bootstrap_thetas=N_BOOTSTRAP_THETAS,
        baseline_propagate_inf_threshold=BASELINE_PROPAGATE_INF_THRESHOLD,
        baseline_max_log10_worsening=BASELINE_MAX_LOG10_WORSENING,
        tem_propagate_inf_threshold=TEM_PROPAGATE_INF_THRESHOLD,
        tem_max_log10_worsening=TEM_MAX_LOG10_WORSENING,
        tem_min_relative_ent_reduction=TEM_MIN_RELATIVE_ENT_REDUCTION,
        tem_allow_relative_ent_worsening=TEM_ALLOW_RELATIVE_ENT_WORSENING,
        ent_reduction_denom_floor=ENT_REDUCTION_DENOM_FLOOR,
        config_x0_field="" if CONFIG_X0_FIELD is None else CONFIG_X0_FIELD,

        **arrays,
    )


# ============================================================
# Row worker (one process per duration T)
# ============================================================

def run_one_duration_row(task: tuple[Any, ...]) -> dict[str, Any]:
    """
    Scan all theta values for a fixed duration T_ns, warm-starting each theta
    from the previous accepted pulse.

    This function is the unit of parallelism: one call per duration column.
    It returns the full arrays dict (only column j is populated).

    Propagation state:
        prev_baseline_x / _I / _theta_idx -- accepted baseline pulse and metadata
        prev_tem_x / _I / _theta_idx      -- accepted TEM pulse and metadata

    When a propagated pulse does not pass the acceptance test and random inits
    are run, the selected best result from random inits updates the propagation
    state for the next theta.
    """
    j, T_ns, seed_row, base_config = task

    rng = np.random.default_rng(int(seed_row))
    arrays = initialise_arrays()

    # State carried from one theta to the next.
    prev_baseline_x: np.ndarray | None = None
    prev_baseline_I: float = np.nan
    prev_baseline_theta_idx: int = -1

    prev_tem_x: np.ndarray | None = None
    prev_tem_I: float = np.nan
    prev_tem_theta_idx: int = -1

    for i, theta in enumerate(THETAS):
        # Adaptive step count: coarser for short pulses, capped at MAX_N_TIME_STEPS.
        n_steps = n_time_steps_for_duration(float(T_ns))
        n_params = int(n_steps) * N_J_CHANNELS  # active pulse params for this T
        arrays["n_time_steps_used"][i, j] = n_steps
        arrays["segment_duration_ns"][i, j] = float(T_ns) / n_steps

        # ---------------------------------------------------------------
        # Baseline
        # ---------------------------------------------------------------
        baseline_results: list[tuple[int, dict[str, Any]]] = []
        baseline_prop_accepted = False

        if i >= N_BOOTSTRAP_THETAS and prev_baseline_x is not None:
            # Attempt propagation: use the previous accepted baseline pulse as x0.
            # Slice to active params so the optimiser works in the correct space.
            prop_result = run_baseline_once(
                base_config=base_config,
                theta=float(theta),
                T_ns=float(T_ns),
                seed=None,
                j_init_mean=0.0,
                n_steps=n_steps,
                x0=prev_baseline_x[:n_params],
            )

            store_baseline_result(
                arrays, i=i, j=j, k=PROP_SLOT,
                seed=-1, j_init_mean=np.nan,
                result=prop_result, used_propagated=True,
                parent_theta_idx=prev_baseline_theta_idx,
            )
            baseline_results.append((PROP_SLOT, prop_result))

            baseline_prop_accepted = baseline_propagation_ok(
                I_new=prop_result["final_infid"],
                I_prev=prev_baseline_I,
            )
            arrays["baseline_propagation_accepted"][i, j] = baseline_prop_accepted

        # Run random inits during bootstrap or if propagation was not accepted.
        run_baseline_randoms = (
            i < N_BOOTSTRAP_THETAS
            or not baseline_prop_accepted
            or not SKIP_RANDOM_IF_BASELINE_PROP_ACCEPTED
        )

        if run_baseline_randoms:
            for r in range(N_RANDOM_INITS):
                # Bootstrap: use slot r; propagation region: use RANDOM_SLOT0+r.
                k = r if i < N_BOOTSTRAP_THETAS else RANDOM_SLOT0 + r
                if k >= N_INIT_SLOTS:
                    continue

                seed = int(rng.integers(0, 2**31 - 1))
                j_mean = float(J_INIT_MEANS[r % len(J_INIT_MEANS)])

                res = run_baseline_once(
                    base_config=base_config,
                    theta=float(theta),
                    T_ns=float(T_ns),
                    seed=seed,
                    j_init_mean=j_mean,
                    n_steps=n_steps,
                    x0=None,
                )

                store_baseline_result(
                    arrays, i=i, j=j, k=k,
                    seed=seed, j_init_mean=j_mean,
                    result=res, used_propagated=False,
                    parent_theta_idx=-1,
                )
                baseline_results.append((k, res))

        # Update propagation state from the best baseline run.
        selected_baseline = select_best_baseline(baseline_results)
        selected_baseline_S = np.nan

        if selected_baseline is not None:
            k_sel, r_sel = selected_baseline
            arrays["baseline_selected_slot"][i, j] = k_sel
            # Slice to active params for warm-starting the next theta.
            prev_baseline_x = np.asarray(r_sel["x_final"][:n_params], dtype=float)
            prev_baseline_I = float(r_sel["final_infid"])
            prev_baseline_theta_idx = i
            selected_baseline_S = float(r_sel["ent_avg"])

        # ---------------------------------------------------------------
        # TEM
        # ---------------------------------------------------------------
        tem_results: list[tuple[int, dict[str, Any]]] = []
        tem_prop_accepted = False

        if i >= N_BOOTSTRAP_THETAS and prev_tem_x is not None:
            # Step 1: infidelity-only repair from propagated TEM pulse.
            repair_result = run_tem_once(
                base_config=base_config,
                theta=float(theta),
                T_ns=float(T_ns),
                seed=None,
                j_init_mean=0.0,
                n_steps=n_steps,
                x0=prev_tem_x[:n_params],
                entanglement_optimisation=False,
            )

            repair_ok = tem_propagation_ok(
                I_new=repair_result["final_infid"],
                I_prev=prev_tem_I,
                S_base=selected_baseline_S,
                S_tem=repair_result["ent_avg"],
            )

            if repair_ok:
                # Repair alone was sufficient.
                prop_result = repair_result
            else:
                # Step 2: primal-dual TEM from the repaired pulse.
                prop_result = run_tem_once(
                    base_config=base_config,
                    theta=float(theta),
                    T_ns=float(T_ns),
                    seed=None,
                    j_init_mean=0.0,
                    n_steps=n_steps,
                    x0=np.asarray(repair_result["x_final"][:n_params], dtype=float),
                    entanglement_optimisation=True,
                )

            store_tem_result(
                arrays, i=i, j=j, k=PROP_SLOT,
                seed=-1, j_init_mean=np.nan,
                result=prop_result, used_propagated=True,
                parent_theta_idx=prev_tem_theta_idx,
                S_base=selected_baseline_S,
            )
            tem_results.append((PROP_SLOT, prop_result))

            tem_prop_accepted = tem_propagation_ok(
                I_new=prop_result["final_infid"],
                I_prev=prev_tem_I,
                S_base=selected_baseline_S,
                S_tem=prop_result["ent_avg"],
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
                    base_config=base_config,
                    theta=float(theta),
                    T_ns=float(T_ns),
                    seed=seed,
                    j_init_mean=j_mean,
                    n_steps=n_steps,
                    x0=None,
                    entanglement_optimisation=True,
                )

                store_tem_result(
                    arrays, i=i, j=j, k=k,
                    seed=seed, j_init_mean=j_mean,
                    result=res, used_propagated=False,
                    parent_theta_idx=-1,
                    S_base=selected_baseline_S,
                )
                tem_results.append((k, res))

        # Update TEM propagation state.
        selected_tem = select_best_tem(tem_results)

        if selected_tem is not None:
            k_sel, r_sel = selected_tem
            arrays["ent_selected_slot"][i, j] = k_sel

            # Only propagate TEM pulses that pass both infidelity and entanglement tests.
            if (
                np.isfinite(r_sel["final_infid"])
                and r_sel["final_infid"] <= TEM_PROPAGATE_INF_THRESHOLD
                and tem_entanglement_ok(S_base=selected_baseline_S,
                                        S_tem=r_sel["ent_avg"])
            ):
                prev_tem_x = np.asarray(r_sel["x_final"][:n_params], dtype=float)
                prev_tem_I = float(r_sel["final_infid"])
                prev_tem_theta_idx = i
            else:
                # Discard propagation state; next theta starts fresh.
                prev_tem_x = None
                prev_tem_I = np.nan
                prev_tem_theta_idx = -1

    return dict(j=j, T_ns=T_ns, arrays=arrays)


def merge_row(global_arrays: dict[str, Any], row_out: dict[str, Any]) -> None:
    """
    Merge a completed duration-row result into the global arrays in-place.

    Only column j (the duration index) is updated; other columns are unchanged.
    For 1D arrays (no duration axis) the entire array is replaced.
    """
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
    """Build the full DQE-gate row-propagated pulse database in parallel."""
    base_config = make_base_config()
    arrays = initialise_arrays()

    if CONFIG_X0_FIELD is None:
        print("=" * 72)
        print("WARNING: No explicit initial-pulse field detected in SimulationConfig.")
        print("Propagation will fail unless you add the correct name to:")
        print("    CONFIG_X0_FIELD_CANDIDATES")
        print("=" * 72)
    else:
        print(f"Using SimulationConfig field for propagated initial pulse: {CONFIG_X0_FIELD}")

    rng = np.random.default_rng(SEED0)
    tasks = []

    for j, T_ns in enumerate(DURATIONS_NS):
        seed_row = int(rng.integers(0, 2**31 - 1))
        tasks.append((j, float(T_ns), seed_row, base_config))

    print("=" * 72)
    print("Generating DQE-gate row-propagated pulse database (positiveJ)")
    print(f"Output             = {RAW_OUT}")
    print(f"J_min / J_max      = {BASE_CONFIG_KWARGS['J_min']} / {BASE_CONFIG_KWARGS['J_max']} MHz")
    print(f"N theta            = {len(THETAS)}")
    print(f"N durations        = {len(DURATIONS_NS)}")
    print(f"N random inits     = {N_RANDOM_INITS}")
    print(f"N init slots saved = {N_INIT_SLOTS}")
    print(f"Max N time steps   = {MAX_N_TIME_STEPS}")
    print(f"N J channels       = {N_J_CHANNELS}")
    print(f"N bootstrap thetas = {N_BOOTSTRAP_THETAS}")
    print(f"Total row jobs     = {len(tasks)}")
    print("=" * 72)

    completed_rows = 0

    with mp.Pool(processes=N_WORKERS) as pool:
        iterator = pool.imap_unordered(run_one_duration_row, tasks, chunksize=1)

        for row_out in tqdm(
            iterator,
            total=len(tasks),
            desc="DQE-gate positiveJ row-prop rows",
            unit="row",
            dynamic_ncols=True,
        ):
            merge_row(arrays, row_out)
            completed_rows += 1

            if CHECKPOINT_EVERY_ROWS and completed_rows % CHECKPOINT_EVERY_ROWS == 0:
                save_raw(arrays)

    save_raw(arrays)
    print(f"\nDone.  Saved to:\n  {RAW_OUT}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
