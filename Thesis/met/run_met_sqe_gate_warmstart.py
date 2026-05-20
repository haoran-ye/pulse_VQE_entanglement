#!/usr/bin/env python3
# run_met_sqe_gate_warmstart.py

from __future__ import annotations

import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import get_context
from dataclasses import fields, is_dataclass

import numpy as np


# =============================================================================
# Import project modules from upper directory
# =============================================================================

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.sqe_gate import sqe_gate_optimisation


# =============================================================================
# User settings
# =============================================================================

OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met/met_sqe_gate_warmstart"
)

N_WORKERS = 9
N_INITS = 4

ZEEMAN_SPLITTINGS = np.array([1000.0, 1008.0])
INITIAL_STATE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

J_MIN = 0.0
J_MAX = 10.0

J_INIT_SCALE_SMALL = 1e-4
J_INIT_SCALE_RANDOM = 1e-1
J_INIT_MEAN_NEGATIVE = -1.0 + 1e-4
J_INIT_MEAN_ZERO = 0.0

N_TIME_STEPS = 50
N_DISCRETISATION = 1

MAXITER = 200
GTOL = 1e-6
XRTOL = 1e-4

MAIN_THETAS = np.arange(0.0125, 0.2501, 0.0125) * 2.0 * np.pi
SMALL_THETAS = np.array(
    [0.0, 0.0125 / 4.0, 0.0125 / 2.0, 0.0125 * 3.0 / 4.0]
) * 2.0 * np.pi
EXCITATION_STRENGTHS = np.concatenate([SMALL_THETAS, MAIN_THETAS])

PULSE_DURATIONS = np.arange(5.0, 1200.0, 30.0)
PULSE_DURATIONS_DESC = PULSE_DURATIONS[::-1]

INFIDELITY_FLOOR = 1e-13

PROPAGATION_THRESHOLD_LOG10 = -4.0
MAX_WORSE_THAN_PREVIOUS_LOG10 = 3.0
INITIAL_PREVIOUS_LOG10 = -5.0

RESULTS_FILENAME = "met_sqe_gate_warmstart_results.npz"


# =============================================================================
# Dataclass compatibility
# =============================================================================

def dataclass_field_names(cls: type) -> set[str]:
    if not is_dataclass(cls):
        return set()
    return {f.name for f in fields(cls)}


CONFIG_FIELDS = dataclass_field_names(SimulationConfig)


def filter_config_kwargs(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if k in CONFIG_FIELDS}


def make_config(**kwargs) -> SimulationConfig:
    return SimulationConfig(**filter_config_kwargs(kwargs))


def supports_field(name: str) -> bool:
    return name in CONFIG_FIELDS


def add_warm_start(kwargs: dict, x0: np.ndarray | None) -> dict:
    """
    Requires SimulationConfig.x0_override and sqe_gate_optimisation to use it.

    In sqe_gate.py, initialise x0 with something like:

        if getattr(config, "x0_override", None) is not None:
            x0 = np.asarray(config.x0_override, dtype=float).copy()
        else:
            x0 = _initialise_x0(config)
    """
    if x0 is None:
        return kwargs

    if not supports_field("x0_override"):
        raise RuntimeError(
            "Warm start requested, but SimulationConfig has no field 'x0_override'.\n"
            "Add this field to SimulationConfig:\n\n"
            "    x0_override: np.ndarray | None = None\n\n"
            "and modify sqe_gate.py so _initialise_x0 is bypassed when "
            "config.x0_override is not None."
        )

    kwargs["x0_override"] = np.asarray(x0, dtype=float).copy()
    return kwargs


# =============================================================================
# Initialisation strategy
# =============================================================================

def init_settings_for_index(init_index: int) -> tuple[float, float]:
    if init_index == 0:
        return J_INIT_MEAN_NEGATIVE, J_INIT_SCALE_SMALL

    if init_index <= N_INITS // 2:
        return J_INIT_MEAN_ZERO, J_INIT_SCALE_RANDOM

    return J_INIT_MEAN_NEGATIVE, J_INIT_SCALE_RANDOM


def deterministic_seed(theta_index: int, init_index: int, duration_index: int = 0) -> int:
    return 100000 * theta_index + 1000 * init_index + duration_index


# =============================================================================
# Worker: one warm-start column
# =============================================================================

def run_theta_init_column(job):
    theta_index, init_index, theta = job

    J_init_mean, J_init_scale = init_settings_for_index(init_index)

    propagated_x = None
    secondary_seed_offset = 0
    loginfidelity_from_previous = INITIAL_PREVIOUS_LOG10

    column_results = []

    for T in PULSE_DURATIONS_DESC:
        duration_index = int(np.where(PULSE_DURATIONS == T)[0][0])
        seed = deterministic_seed(
            theta_index,
            init_index,
            duration_index + secondary_seed_offset,
        )

        kwargs = dict(
            zeeman_splittings=ZEEMAN_SPLITTINGS,
            initial_state=INITIAL_STATE,
            theta_target=float(theta),
            T=float(T),
            n_time_steps=N_TIME_STEPS,
            n_discretisation=N_DISCRETISATION,
            J_min=J_MIN,
            J_max=J_MAX,
            J_init_scale=J_init_scale,
            J_init_mean=J_init_mean,
            J_init_seed=seed,
            maxiter=MAXITER,
            gtol=GTOL,
            xrtol=XRTOL,
            entanglement_optimisation=False,
            optimisation_target="gate",
            n_J_channels=1,
        )

        kwargs = add_warm_start(kwargs, propagated_x)
        config = make_config(**kwargs)

        result = sqe_gate_optimisation(config)

        final_infid = float(np.real(result.infid_final[-1]))
        final_infid = max(final_infid, 0.0)
        logI = np.log10(max(final_infid, INFIDELITY_FLOOR))

        success = bool(result.optimization_success)

        x_final = np.asarray(result.x_final, dtype=float)
        x0 = np.asarray(result.x0, dtype=float)

        if propagated_x is not None:
            warm_start_diff = float(np.linalg.norm(x0 - propagated_x))
        else:
            warm_start_diff = np.nan

        column_results.append(
            {
                "theta_index": theta_index,
                "duration_index": duration_index,
                "init_index": init_index,
                "theta": float(theta),
                "T": float(T),
                "n_time_steps": int(N_TIME_STEPS),
                "segment_duration_ns": float(T) / int(N_TIME_STEPS),
                "seed": seed,
                "J_init_mean": J_init_mean,
                "J_init_scale": J_init_scale,
                "infidelity": final_infid,
                "log10_infidelity": logI,
                "x0": x0,
                "x_final": x_final,
                "success": success,
                "nit": int(result.n_iterations),
                "grad_norm": float(result.gradient_norm),
                "used_warm_start": propagated_x is not None,
                "warm_start_diff": warm_start_diff,
            }
        )

        should_propagate = False

        if success:
            not_too_much_worse = (
                logI < loginfidelity_from_previous + MAX_WORSE_THAN_PREVIOUS_LOG10
            )
            below_threshold = logI < PROPAGATION_THRESHOLD_LOG10

            if not_too_much_worse and below_threshold:
                should_propagate = True

        if should_propagate:
            propagated_x = x_final
            loginfidelity_from_previous = logI
        else:
            propagated_x = None
            secondary_seed_offset += 10000000
            loginfidelity_from_previous = INITIAL_PREVIOUS_LOG10

    return column_results


# =============================================================================
# Build jobs
# =============================================================================

def build_jobs():
    jobs = []

    for i_theta, theta in enumerate(EXCITATION_STRENGTHS):
        for i_init in range(N_INITS):
            jobs.append((i_theta, i_init, float(theta)))

    rng = np.random.default_rng(44)
    rng.shuffle(jobs)

    return jobs


# =============================================================================
# Processing
# =============================================================================

def apply_monotonic_duration_correction(best_infid):
    """
    Saved duration axis is short -> long.

    If a shorter duration has lower infidelity, overwrite the longer duration
    with the shorter-duration result, then continue comparing to the next longer.
    """
    corrected = best_infid.copy()

    for i_theta in range(corrected.shape[0]):
        running_best = np.inf

        for i_T in range(corrected.shape[1]):
            if not np.isfinite(corrected[i_theta, i_T]):
                continue

            if corrected[i_theta, i_T] < running_best:
                running_best = corrected[i_theta, i_T]
            else:
                corrected[i_theta, i_T] = running_best

    return corrected


def save_results(raw_results):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    n_theta = len(EXCITATION_STRENGTHS)
    n_T = len(PULSE_DURATIONS)

    infidelities = np.full((n_theta, n_T, N_INITS), np.nan)
    log10_infidelities = np.full((n_theta, n_T, N_INITS), np.nan)
    successes = np.full((n_theta, n_T, N_INITS), False)
    n_iterations = np.full((n_theta, n_T, N_INITS), -1)
    grad_norms = np.full((n_theta, n_T, N_INITS), np.nan)
    seeds = np.full((n_theta, n_T, N_INITS), -1)
    J_init_means = np.full((n_theta, n_T, N_INITS), np.nan)
    J_init_scales = np.full((n_theta, n_T, N_INITS), np.nan)
    n_time_steps_used = np.full((n_theta, n_T, N_INITS), -1)
    segment_duration_ns = np.full((n_theta, n_T, N_INITS), np.nan)
    used_warm_start = np.full((n_theta, n_T, N_INITS), False)
    warm_start_diff = np.full((n_theta, n_T, N_INITS), np.nan)

    x0_all = np.empty((n_theta, n_T, N_INITS), dtype=object)
    x_final_all = np.empty((n_theta, n_T, N_INITS), dtype=object)

    for r in raw_results:
        i = r["theta_index"]
        j = r["duration_index"]
        k = r["init_index"]

        infidelities[i, j, k] = r["infidelity"]
        log10_infidelities[i, j, k] = r["log10_infidelity"]
        successes[i, j, k] = r["success"]
        n_iterations[i, j, k] = r["nit"]
        grad_norms[i, j, k] = r["grad_norm"]
        seeds[i, j, k] = r["seed"]
        J_init_means[i, j, k] = r["J_init_mean"]
        J_init_scales[i, j, k] = r["J_init_scale"]
        n_time_steps_used[i, j, k] = r["n_time_steps"]
        segment_duration_ns[i, j, k] = r["segment_duration_ns"]
        used_warm_start[i, j, k] = r["used_warm_start"]
        warm_start_diff[i, j, k] = r["warm_start_diff"]
        x0_all[i, j, k] = r["x0"]
        x_final_all[i, j, k] = r["x_final"]

    valid_points = np.any(np.isfinite(infidelities), axis=2)

    best_infid = np.full((n_theta, n_T), np.nan)
    best_seed_index = np.full((n_theta, n_T), -1)

    best_infid[valid_points] = np.nanmin(infidelities[valid_points], axis=1)
    best_seed_index[valid_points] = np.nanargmin(infidelities[valid_points], axis=1)

    best_infid[(best_infid <= 0.0) & valid_points] = INFIDELITY_FLOOR

    corrected_best_infid = apply_monotonic_duration_correction(best_infid)

    np.savez(
        OUTDIR / RESULTS_FILENAME,
        excitation_strengths=EXCITATION_STRENGTHS,
        pulse_durations=PULSE_DURATIONS,
        pulse_durations_desc=PULSE_DURATIONS_DESC,
        infidelities=infidelities,
        log10_infidelities=log10_infidelities,
        best_infid=best_infid,
        corrected_best_infid=corrected_best_infid,
        best_seed_index=best_seed_index,
        successes=successes,
        n_iterations=n_iterations,
        grad_norms=grad_norms,
        seeds=seeds,
        J_init_means=J_init_means,
        J_init_scales=J_init_scales,
        n_time_steps_used=n_time_steps_used,
        segment_duration_ns=segment_duration_ns,
        used_warm_start=used_warm_start,
        warm_start_diff=warm_start_diff,
        x0_all=x0_all,
        x_final_all=x_final_all,
        n_time_steps=N_TIME_STEPS,
        n_discretisation=N_DISCRETISATION,
        zeeman_splittings=ZEEMAN_SPLITTINGS,
        initial_state=INITIAL_STATE,
        J_min=J_MIN,
        J_max=J_MAX,
        J_init_scale_small=J_INIT_SCALE_SMALL,
        J_init_scale_random=J_INIT_SCALE_RANDOM,
        J_init_mean_zero=J_INIT_MEAN_ZERO,
        J_init_mean_negative=J_INIT_MEAN_NEGATIVE,
        maxiter=MAXITER,
        gtol=GTOL,
        xrtol=XRTOL,
        infidelity_floor=INFIDELITY_FLOOR,
        propagation_threshold_log10=PROPAGATION_THRESHOLD_LOG10,
        max_worse_than_previous_log10=MAX_WORSE_THAN_PREVIOUS_LOG10,
    )

    return best_infid, corrected_best_infid


# =============================================================================
# Main
# =============================================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if not supports_field("x0_override"):
        raise RuntimeError(
            "SimulationConfig does not contain x0_override, so warm-starting "
            "cannot work. Add:\n\n"
            "    x0_override: np.ndarray | None = None\n\n"
            "to SimulationConfig, and modify sqe_gate.py to use it before "
            "calling _initialise_x0(config)."
        )

    print(f"Number of SQE gate warm-start columns: {len(EXCITATION_STRENGTHS) * N_INITS}")
    print(f"Each column sweeps {len(PULSE_DURATIONS)} durations from long to short.")
    print(f"Total optimisation runs: {len(EXCITATION_STRENGTHS) * N_INITS * len(PULSE_DURATIONS)}")
    print(f"Saving results to: {OUTDIR.resolve()}")
    print(f"Number of workers: {N_WORKERS}")
    print(f"Pulse segments fixed to: {N_TIME_STEPS}")
    print("Saved duration order: short to long")
    print("Optimisation order inside each column: long to short")
    print("Post-processing correction: short to long")
    print("Warm start field: x0_override")

    jobs = build_jobs()

    ctx = get_context("spawn")
    raw_results = []

    with ctx.Pool(processes=N_WORKERS) as pool:
        iterator = pool.imap_unordered(run_theta_init_column, jobs)

        iterator = tqdm(
            iterator,
            total=len(jobs),
            desc="SQE gate warm-start columns",
            dynamic_ncols=True,
        )

        for column in iterator:
            raw_results.extend(column)

    save_results(raw_results)

    print("Done.")
    print(f"Raw and processed data saved to: {OUTDIR / RESULTS_FILENAME}")


if __name__ == "__main__":
    main()