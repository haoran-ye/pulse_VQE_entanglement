#!/usr/bin/env python3
# run_met_sqe_gate.py

from __future__ import annotations

import sys
from pathlib import Path
from tqdm import tqdm
from multiprocessing import get_context

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

OUTDIR = Path("/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_sqe_gate")

N_WORKERS = 9
N_INITS = 4  # better to keep this even, so half mean 0 and half mean -1

ZEEMAN_SPLITTINGS = np.array([1000.0, 1008.0])
INITIAL_STATE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)  # dummy state for gate optimiser

J_MIN = 0.0
J_MAX = 10.0
J_INIT_SCALE = 1e-4
J_INIT_MEAN_ZERO = 0.0
J_INIT_MEAN_NEGATIVE = -1.0 + 1e-3

MAX_N_TIME_STEPS = 50
N_DISCRETISATION = 1
MIN_SEGMENT_DURATION_NS = 1.0

MAXITER = 200
GTOL = 1e-12

EXCITATION_STRENGTHS = np.linspace(0.0, 0.5 * np.pi, 50)
PULSE_DURATIONS = np.arange(1, 500.0 + 1e-9, 10.0)

INFIDELITY_FLOOR = 1e-13

RESULTS_FILENAME = "met_sqe_gate_results.npz"


# =============================================================================
# Segment logic
# =============================================================================

def n_time_steps_for_duration(T: float) -> int:
    """
    Use as many time segments as possible up to MAX_N_TIME_STEPS,
    while keeping each segment at least MIN_SEGMENT_DURATION_NS long.
    """
    n_allowed_by_min_segment = int(np.floor(float(T) / MIN_SEGMENT_DURATION_NS))
    return max(1, min(MAX_N_TIME_STEPS, n_allowed_by_min_segment))


# =============================================================================
# Worker
# =============================================================================

def init_mean_for_index(init_index: int) -> float:
    """
    Use half the random initialisations around mean 0,
    and half around mean -1.

    If N_INITS is odd, the extra initialisation goes to mean 0.
    """
    n_zero = (N_INITS + 1) // 2
    if init_index < n_zero:
        return J_INIT_MEAN_ZERO
    return J_INIT_MEAN_NEGATIVE


def run_single_optimisation(job):
    theta_index, duration_index, init_index, theta, T, n_time_steps = job

    seed = 100000 * theta_index + 1000 * duration_index + init_index
    J_init_mean = init_mean_for_index(init_index)

    config = SimulationConfig(
        zeeman_splittings=ZEEMAN_SPLITTINGS,
        initial_state=INITIAL_STATE,
        theta_target=float(theta),
        T=float(T),
        n_time_steps=int(n_time_steps),
        n_discretisation=N_DISCRETISATION,
        J_min=J_MIN,
        J_max=J_MAX,
        J_init_scale=J_INIT_SCALE,
        J_init_mean=J_init_mean,
        J_init_seed=seed,
        maxiter=MAXITER,
        gtol=GTOL,
        entanglement_optimisation=False,
        optimisation_target="gate",
        n_J_channels=1,
    )

    result = sqe_gate_optimisation(config)

    final_infid = float(np.real(result.infid_final[-1]))
    final_infid = max(final_infid, 0.0)

    return {
        "theta_index": theta_index,
        "duration_index": duration_index,
        "init_index": init_index,
        "theta": float(theta),
        "T": float(T),
        "n_time_steps": int(n_time_steps),
        "segment_duration_ns": float(T) / int(n_time_steps),
        "seed": seed,
        "J_init_mean": J_init_mean,
        "infidelity": final_infid,
        "x0": result.x0,
        "x_final": result.x_final,
        "success": bool(result.optimization_success),
        "nit": int(result.n_iterations),
        "grad_norm": float(result.gradient_norm),
    }


# =============================================================================
# Helpers
# =============================================================================

def build_jobs():
    jobs = []

    for i_theta, theta in enumerate(EXCITATION_STRENGTHS):
        for i_T, T in enumerate(PULSE_DURATIONS):
            n_time_steps = n_time_steps_for_duration(float(T))

            for i_init in range(N_INITS):
                jobs.append((i_theta, i_T, i_init, float(theta), float(T), n_time_steps))

    return jobs


def apply_monotonic_duration_correction(best_infid):
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
    successes = np.full((n_theta, n_T, N_INITS), False)
    n_iterations = np.full((n_theta, n_T, N_INITS), -1)
    grad_norms = np.full((n_theta, n_T, N_INITS), np.nan)
    seeds = np.full((n_theta, n_T, N_INITS), -1)
    J_init_means = np.full((n_theta, n_T, N_INITS), np.nan)
    n_time_steps_used = np.full((n_theta, n_T, N_INITS), -1)
    segment_duration_ns = np.full((n_theta, n_T, N_INITS), np.nan)

    x0_all = np.empty((n_theta, n_T, N_INITS), dtype=object)
    x_final_all = np.empty((n_theta, n_T, N_INITS), dtype=object)

    for r in raw_results:
        i = r["theta_index"]
        j = r["duration_index"]
        k = r["init_index"]

        infidelities[i, j, k] = r["infidelity"]
        successes[i, j, k] = r["success"]
        n_iterations[i, j, k] = r["nit"]
        grad_norms[i, j, k] = r["grad_norm"]
        seeds[i, j, k] = r["seed"]
        J_init_means[i, j, k] = r["J_init_mean"]
        n_time_steps_used[i, j, k] = r["n_time_steps"]
        segment_duration_ns[i, j, k] = r["segment_duration_ns"]
        x0_all[i, j, k] = r["x0"]
        x_final_all[i, j, k] = r["x_final"]

    valid_points = np.any(np.isfinite(infidelities), axis=2)

    best_infid = np.full((n_theta, n_T), np.nan)
    best_seed_index = np.full((n_theta, n_T), -1)

    best_infid[valid_points] = np.nanmin(infidelities[valid_points], axis=1)
    best_seed_index[valid_points] = np.nanargmin(infidelities[valid_points], axis=1)

    # Numerical guard before monotonic correction:
    # exact zeros or tiny negatives from cancellation become floor values.
    best_infid[(best_infid <= 0.0) & valid_points] = INFIDELITY_FLOOR

    corrected_best_infid = apply_monotonic_duration_correction(best_infid)

    np.savez(
        OUTDIR / RESULTS_FILENAME,
        excitation_strengths=EXCITATION_STRENGTHS,
        pulse_durations=PULSE_DURATIONS,
        infidelities=infidelities,
        best_infid=best_infid,
        corrected_best_infid=corrected_best_infid,
        best_seed_index=best_seed_index,
        successes=successes,
        n_iterations=n_iterations,
        grad_norms=grad_norms,
        seeds=seeds,
        J_init_means=J_init_means,
        n_time_steps_used=n_time_steps_used,
        segment_duration_ns=segment_duration_ns,
        x0_all=x0_all,
        x_final_all=x_final_all,
        max_n_time_steps=MAX_N_TIME_STEPS,
        n_discretisation=N_DISCRETISATION,
        min_segment_duration_ns=MIN_SEGMENT_DURATION_NS,
        zeeman_splittings=ZEEMAN_SPLITTINGS,
        initial_state=INITIAL_STATE,
        J_min=J_MIN,
        J_max=J_MAX,
        J_init_scale=J_INIT_SCALE,
        J_init_mean_zero=J_INIT_MEAN_ZERO,
        J_init_mean_negative=J_INIT_MEAN_NEGATIVE,
        maxiter=MAXITER,
        gtol=GTOL,
        infidelity_floor=INFIDELITY_FLOOR,
    )

    return best_infid, corrected_best_infid


# =============================================================================
# Main
# =============================================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs()

    print(f"Number of gate-based SQE optimisation runs: {len(jobs)}")
    print(f"Saving results to: {OUTDIR.resolve()}")
    print(f"Maximum number of pulse segments: {MAX_N_TIME_STEPS}")
    print(f"Minimum segment duration: {MIN_SEGMENT_DURATION_NS} ns")
    print(f"Initialisation strategy: first {(N_INITS + 1) // 2} inits use mean {J_INIT_MEAN_ZERO}, "
          f"remaining {N_INITS // 2} use mean {J_INIT_MEAN_NEGATIVE}")

    ctx = get_context("spawn")

    raw_results = []

    with ctx.Pool(processes=N_WORKERS) as pool:
        iterator = pool.imap_unordered(run_single_optimisation, jobs)

        if tqdm is not None:
            iterator = tqdm(iterator, total=len(jobs), desc="SQE gate MET runs")

        for result in iterator:
            raw_results.append(result)

    save_results(raw_results)

    print("Done.")
    print(f"Raw and processed data saved to: {OUTDIR / RESULTS_FILENAME}")


if __name__ == "__main__":
    main()
