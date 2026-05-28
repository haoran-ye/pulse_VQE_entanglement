#!/usr/bin/env python3
"""
generate_dqe_gate_pulses_positiveJ_rowprop.py

DQE-gate pulse database with theta-continuation.

Grid convention:
    arrays have shape (n_theta, n_T, n_init_slots)

For each fixed duration T, one worker scans theta from low to high.

Initialisation strategy:
    - first N_BOOTSTRAP_THETAS theta values: normal random initialisations
    - later theta values:
        baseline:
            try propagated baseline x from previous accepted theta first
        TEM:
            try propagated TEM x from previous accepted theta first
            first with infidelity-only repair
            then, if needed, primal-dual entanglement optimisation
        fallback:
            if propagation is not accepted, run normal random initialisations

Compatibility:
    The usual arrays are still saved:
        baseline_final_infids
        ent_final_infids
        baseline_ent_avg
        ent_ent_avg
        baseline_x_final
        ent_x_final
        etc.

Extra propagation metadata arrays are also saved, but old plotting scripts should ignore them.
"""

from __future__ import annotations

import os
import sys
import multiprocessing as mp
from pathlib import Path
from dataclasses import replace, fields, is_dataclass
from typing import Any

import numpy as np
from tqdm import tqdm


# ============================================================
# Thread control
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.dqe_gate import dqe_gate_optimisation


# ============================================================
# Settings
# ============================================================

PULSES_DATABASE = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/PULSES_DATABASE"
)

SYSTEM_NAME = "dqe_gate"
PULSE_FAMILY_LABEL = "positiveJ_rowprop"

OUTDIR = PULSES_DATABASE
OUTDIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = OUTDIR / f"{SYSTEM_NAME}_{PULSE_FAMILY_LABEL}_pulses.npz"

small_thetas = np.linspace(0.0 * np.pi, 0.009 * np.pi, 5)
large_thetas = np.linspace(0.01 * np.pi, 0.50 * np.pi, 20)
THETAS = np.concatenate([small_thetas, large_thetas])

DURATIONS_NS = np.arange(1.0, 1200.0, 80.0)

N_RANDOM_INITS = 1
N_INIT_SLOTS = N_RANDOM_INITS + 1      # slot 0 reserved for propagation when available
PROP_SLOT = 0
RANDOM_SLOT0 = 1

N_WORKERS = 9
SEED0 = 1234
CHECKPOINT_EVERY_ROWS = 1

INFIDELITY_THRESHOLD = 1e-5
INFIDELITY_FLOOR = 1e-14

MAX_N_TIME_STEPS = 50
MIN_SEGMENT_DURATION_NS = 1.0
N_J_CHANNELS = 3
MAX_N_PULSE_PARAMS = MAX_N_TIME_STEPS * N_J_CHANNELS

N_OUTER = 3
INNER_MAXITER = 200
EPS_INFIDELITY = 1e-12
DUAL_LR = 1e10
RHO = 0.0
FD_EPS = 1e-6

GATE_ENT_MEASURE = "operator_entanglement_sum_4q"
ENTANGLEMENT_MARGIN = 0.01

J_INIT_MEANS = np.array(
    [0.0] * (N_RANDOM_INITS // 2)
    + [0.0] * (N_RANDOM_INITS - N_RANDOM_INITS // 2),
    dtype=float,
)

ZEEMAN = np.array([1000.0, 1008.0, 1016.0, 1024.0])
DUMMY_INITIAL_STATE = np.zeros(16, dtype=np.complex128)
DUMMY_INITIAL_STATE[0] = 1.0

# ------------------------------------------------------------
# Propagation controls
# ------------------------------------------------------------

N_BOOTSTRAP_THETAS = 2

BASELINE_PROPAGATE_INF_THRESHOLD = 1e-4
BASELINE_MAX_LOG10_WORSENING = 5.0

TEM_PROPAGATE_INF_THRESHOLD = 1e-4
TEM_MAX_LOG10_WORSENING = 5.0

TEM_MIN_RELATIVE_ENT_REDUCTION = 0.10
TEM_ALLOW_RELATIVE_ENT_WORSENING = 0.05
ENT_REDUCTION_DENOM_FLOOR = 1e-8

# If a propagated pulse succeeds and passes these tests, random starts are skipped.
SKIP_RANDOM_IF_BASELINE_PROP_ACCEPTED = True
SKIP_RANDOM_IF_TEM_PROP_ACCEPTED = True

# Candidate field names for explicit initial pulse in SimulationConfig.
# If your SimulationConfig uses a different name, add it here.
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
# Dataclass helpers
# ============================================================

def dataclass_field_names(cls: type) -> set[str]:
    if not is_dataclass(cls):
        return set()
    return {f.name for f in fields(cls)}


CONFIG_FIELDS = dataclass_field_names(SimulationConfig)


def infer_x0_field() -> str | None:
    for name in CONFIG_X0_FIELD_CANDIDATES:
        if name in CONFIG_FIELDS:
            return name
    return None


CONFIG_X0_FIELD = infer_x0_field()


def filter_kwargs_for_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in CONFIG_FIELDS}


def make_base_config() -> SimulationConfig:
    return SimulationConfig(**filter_kwargs_for_config(BASE_CONFIG_KWARGS))


def safe_replace_config(config: SimulationConfig, **kwargs: Any) -> SimulationConfig:
    return replace(config, **filter_kwargs_for_config(kwargs))


def add_initial_x(kwargs: dict[str, Any], x0: np.ndarray | None) -> dict[str, Any]:
    kwargs = dict(kwargs)
    if x0 is None:
        return kwargs

    if CONFIG_X0_FIELD is None:
        raise RuntimeError(
            "No explicit initial-pulse field was found in SimulationConfig. "
            "Add the correct field name to CONFIG_X0_FIELD_CANDIDATES."
        )

    kwargs[CONFIG_X0_FIELD] = np.asarray(x0, dtype=float).reshape(-1)
    return kwargs


# ============================================================
# Result helpers
# ============================================================

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


def n_time_steps_for_duration(T_ns: float) -> int:
    return max(
        1,
        min(MAX_N_TIME_STEPS, int(np.floor(float(T_ns) / MIN_SEGMENT_DURATION_NS))),
    )


def n_pulse_params_for_duration(T_ns: float) -> int:
    return n_time_steps_for_duration(T_ns) * N_J_CHANNELS


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


def clean_infid(x: float) -> float:
    if not np.isfinite(x):
        return np.nan
    return max(float(x), INFIDELITY_FLOOR)


def log10_infid(x: float) -> float:
    if not np.isfinite(x):
        return np.inf
    return float(np.log10(max(float(x), INFIDELITY_FLOOR)))


def relative_ent_reduction(S_base: float, S_tem: float) -> float:
    if not np.isfinite(S_base) or not np.isfinite(S_tem):
        return np.nan
    denom = max(float(S_base), ENT_REDUCTION_DENOM_FLOOR)
    return float((S_base - S_tem) / denom)


# ============================================================
# Single optimisation calls
# ============================================================

def common_config_kwargs(theta: float, T_ns: float, seed: int | None, j_init_mean: float, n_steps: int) -> dict[str, Any]:
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
    kwargs = common_config_kwargs(theta, T_ns, seed, j_init_mean, n_steps)
    kwargs = add_initial_x(kwargs, x0)

    config = safe_replace_config(
        base_config,
        **kwargs,
        entanglement_optimisation=False,
    )

    result = dqe_gate_optimisation(config)
    arr = extract_result_arrays(result)

    I = clean_infid(arr["final_infid"])
    return dict(
        success=np.isfinite(I) and I <= INFIDELITY_THRESHOLD,
        optimization_success=arr["optimization_success"],
        final_infid=I,
        ent_avg=average_entanglement(arr["entanglements"]),
        ent_max=max_entanglement(arr["entanglements"]),
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
    kwargs = common_config_kwargs(theta, T_ns, seed, j_init_mean, n_steps)
    kwargs = add_initial_x(kwargs, x0)

    config = safe_replace_config(
        base_config,
        **kwargs,
        entanglement_optimisation=bool(entanglement_optimisation),
    )

    result = dqe_gate_optimisation(config)
    arr = extract_result_arrays(result)

    I = clean_infid(arr["final_infid"])
    return dict(
        success=np.isfinite(I) and I <= INFIDELITY_THRESHOLD,
        optimization_success=arr["optimization_success"],
        final_infid=I,
        ent_avg=average_entanglement(arr["entanglements"]),
        ent_max=max_entanglement(arr["entanglements"]),
        x0=fixed_width_x(arr["x0"] if arr["x0"] is not None else x0, MAX_N_PULSE_PARAMS),
        x_final=fixed_width_x(arr["x_final"], MAX_N_PULSE_PARAMS),
        n_iterations=arr["n_iterations"],
        gradient_norm=arr["gradient_norm"],
        used_primal_dual=bool(entanglement_optimisation),
    )


# ============================================================
# Acceptance tests
# ============================================================

def baseline_propagation_ok(
    *,
    I_new: float,
    I_prev: float,
) -> bool:
    if not np.isfinite(I_new):
        return False

    if I_new > BASELINE_PROPAGATE_INF_THRESHOLD:
        return False

    if np.isfinite(I_prev):
        if log10_infid(I_new) > log10_infid(I_prev) + BASELINE_MAX_LOG10_WORSENING:
            return False

    return True


def tem_entanglement_ok(
    *,
    S_base: float,
    S_tem: float,
) -> bool:
    """
    DQE TEM propagation condition.

    DQE uses a four-qubit summed operator-entanglement measure, so the absolute
    scale is larger and a fixed SQE-style entropy threshold is too harsh.

    Accept propagation if either:
      1. TEM reduces entanglement by at least TEM_MIN_RELATIVE_ENT_REDUCTION; or
      2. TEM is not worse than baseline by more than TEM_ALLOW_RELATIVE_ENT_WORSENING.
    """
    if not np.isfinite(S_base) or not np.isfinite(S_tem):
        return False

    red = relative_ent_reduction(S_base, S_tem)

    if np.isfinite(red) and red >= TEM_MIN_RELATIVE_ENT_REDUCTION:
        return True

    return bool(S_tem <= (1.0 + TEM_ALLOW_RELATIVE_ENT_WORSENING) * S_base)


def tem_propagation_ok(
    *,
    I_new: float,
    I_prev: float,
    S_base: float,
    S_tem: float,
) -> bool:
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

def select_best_baseline(local_results: list[tuple[int, dict[str, Any]]]) -> tuple[int, dict[str, Any]] | None:
    valid = [(k, r) for k, r in local_results if np.isfinite(r["final_infid"])]
    if not valid:
        return None
    return min(valid, key=lambda kr: kr[1]["final_infid"])


def select_best_tem(local_results: list[tuple[int, dict[str, Any]]]) -> tuple[int, dict[str, Any]] | None:
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
# Array storage
# ============================================================

def initialise_arrays() -> dict[str, Any]:
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

        n_time_steps_used=np.full((n_theta, n_T), -1, dtype=int),
        segment_duration_ns=np.full((n_theta, n_T), np.nan),

        # Extra metadata. Old plotting scripts should ignore these.
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
    arrays["ent_relative_reduction"][i, j, k] = relative_ent_reduction(S_base, result["ent_avg"])


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
# Row worker
# ============================================================

def run_one_duration_row(task: tuple[Any, ...]) -> dict[str, Any]:
    (
        j,
        T_ns,
        seed_row,
        base_config,
    ) = task

    rng = np.random.default_rng(int(seed_row))
    arrays = initialise_arrays()

    prev_baseline_x: np.ndarray | None = None
    prev_baseline_I: float = np.nan
    prev_baseline_theta_idx: int = -1

    prev_tem_x: np.ndarray | None = None
    prev_tem_I: float = np.nan
    prev_tem_theta_idx: int = -1

    for i, theta in enumerate(THETAS):
        n_steps = n_time_steps_for_duration(float(T_ns))
        n_params = int(n_steps) * N_J_CHANNELS
        arrays["n_time_steps_used"][i, j] = n_steps
        arrays["segment_duration_ns"][i, j] = float(T_ns) / n_steps

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------
        baseline_results: list[tuple[int, dict[str, Any]]] = []
        baseline_prop_accepted = False

        if i >= N_BOOTSTRAP_THETAS and prev_baseline_x is not None:
            prop_result = run_baseline_once(
                base_config=base_config,
                theta=float(theta),
                T_ns=float(T_ns),
                seed=None,
                j_init_mean=0.0,
                n_steps=n_steps,
                x0=prev_baseline_x,
            )

            store_baseline_result(
                arrays,
                i=i,
                j=j,
                k=PROP_SLOT,
                seed=-1,
                j_init_mean=np.nan,
                result=prop_result,
                used_propagated=True,
                parent_theta_idx=prev_baseline_theta_idx,
            )
            baseline_results.append((PROP_SLOT, prop_result))

            baseline_prop_accepted = baseline_propagation_ok(
                I_new=prop_result["final_infid"],
                I_prev=prev_baseline_I,
            )
            arrays["baseline_propagation_accepted"][i, j] = baseline_prop_accepted

        run_baseline_randoms = (
            i < N_BOOTSTRAP_THETAS
            or not baseline_prop_accepted
            or not SKIP_RANDOM_IF_BASELINE_PROP_ACCEPTED
        )

        if run_baseline_randoms:
            for r in range(N_RANDOM_INITS):
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
                    arrays,
                    i=i,
                    j=j,
                    k=k,
                    seed=seed,
                    j_init_mean=j_mean,
                    result=res,
                    used_propagated=False,
                    parent_theta_idx=-1,
                )
                baseline_results.append((k, res))

        selected_baseline = select_best_baseline(baseline_results)
        selected_baseline_S = np.nan

        if selected_baseline is not None:
            k_sel, r_sel = selected_baseline
            arrays["baseline_selected_slot"][i, j] = k_sel
            prev_baseline_x = np.asarray(r_sel["x_final"][:n_params], dtype=float)
            prev_baseline_I = float(r_sel["final_infid"])
            prev_baseline_theta_idx = i
            selected_baseline_S = float(r_sel["ent_avg"])

        # ----------------------------------------------------
        # TEM
        # ----------------------------------------------------
        tem_results: list[tuple[int, dict[str, Any]]] = []
        tem_prop_accepted = False

        if i >= N_BOOTSTRAP_THETAS and prev_tem_x is not None:
            # First: infidelity-only repair from propagated TEM pulse.
            repair_result = run_tem_once(
                base_config=base_config,
                theta=float(theta),
                T_ns=float(T_ns),
                seed=None,
                j_init_mean=0.0,
                n_steps=n_steps,
                x0=prev_tem_x,
                entanglement_optimisation=False,
            )

            repair_ok = tem_propagation_ok(
                I_new=repair_result["final_infid"],
                I_prev=prev_tem_I,
                S_base=selected_baseline_S,
                S_tem=repair_result["ent_avg"],
            )

            if repair_ok:
                prop_result = repair_result
            else:
                # Second: primal-dual from repaired pulse.
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
                arrays,
                i=i,
                j=j,
                k=PROP_SLOT,
                seed=-1,
                j_init_mean=np.nan,
                result=prop_result,
                used_propagated=True,
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
                    arrays,
                    i=i,
                    j=j,
                    k=k,
                    seed=seed,
                    j_init_mean=j_mean,
                    result=res,
                    used_propagated=False,
                    parent_theta_idx=-1,
                    S_base=selected_baseline_S,
                )
                tem_results.append((k, res))

        selected_tem = select_best_tem(tem_results)

        if selected_tem is not None:
            k_sel, r_sel = selected_tem
            arrays["ent_selected_slot"][i, j] = k_sel

            # Only propagate accepted TEM pulses:
            # successful infidelity plus entanglement condition.
            if (
                np.isfinite(r_sel["final_infid"])
                and r_sel["final_infid"] <= TEM_PROPAGATE_INF_THRESHOLD
                and tem_entanglement_ok(S_base=selected_baseline_S, S_tem=r_sel["ent_avg"])
            ):
                prev_tem_x = np.asarray(r_sel["x_final"][:n_params], dtype=float)
                prev_tem_I = float(r_sel["final_infid"])
                prev_tem_theta_idx = i
            else:
                prev_tem_x = None
                prev_tem_I = np.nan
                prev_tem_theta_idx = -1

    return dict(
        j=j,
        T_ns=T_ns,
        arrays=arrays,
    )


def merge_row(global_arrays: dict[str, Any], row_out: dict[str, Any]) -> None:
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
    base_config = make_base_config()
    arrays = initialise_arrays()

    if CONFIG_X0_FIELD is None:
        print("=" * 72)
        print("WARNING")
        print("No explicit initial-pulse field was detected in SimulationConfig.")
        print("Propagation will fail unless you add the correct field name to:")
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
    print("Generating DQE-gate row-propagated pulse database")
    print(f"Output = {RAW_OUT}")
    print(f"Pulse family label = {PULSE_FAMILY_LABEL}")
    print(f"J_min = {BASE_CONFIG_KWARGS['J_min']}")
    print(f"J_max = {BASE_CONFIG_KWARGS['J_max']}")
    print(f"N theta = {len(THETAS)}")
    print(f"N durations = {len(DURATIONS_NS)}")
    print(f"N random inits = {N_RANDOM_INITS}")
    print(f"N init slots saved = {N_INIT_SLOTS}")
    print(f"Max N time steps = {MAX_N_TIME_STEPS}")
    print(f"N J channels = {N_J_CHANNELS}")
    print(f"N bootstrap theta values = {N_BOOTSTRAP_THETAS}")
    print(f"Total duration-row jobs = {len(tasks)}")
    print("=" * 72)

    completed_rows = 0

    with mp.Pool(processes=N_WORKERS) as pool:
        iterator = pool.imap_unordered(run_one_duration_row, tasks, chunksize=1)

        for row_out in tqdm(
            iterator,
            total=len(tasks),
            desc="duration rows",
            unit="row",
            dynamic_ncols=True,
        ):
            merge_row(arrays, row_out)
            completed_rows += 1

            if CHECKPOINT_EVERY_ROWS and completed_rows % CHECKPOINT_EVERY_ROWS == 0:
                save_raw(arrays)

    save_raw(arrays)

    print("\nDone.")
    print(f"Saved row-propagated pulse database to:\n  {RAW_OUT}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()