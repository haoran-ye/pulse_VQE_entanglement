#!/usr/bin/env python3
# plot_met_sqe_state.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

RESULTS_FILE = Path("/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_sqe_state/met_sqe_state_results.npz")
OUTDIR = RESULTS_FILE.parent

PLOT_FILENAME = "met_sqe_state_map.png"

INFIDELITY_FLOOR = 1e-13

MET_THRESHOLD = 1e-5
MET_LINE_COLOR = "white"


# =============================================================================
# Helpers
# =============================================================================

def compute_met_line(corrected_best_infid, pulse_durations):
    met_T = np.full(corrected_best_infid.shape[0], np.nan)

    for i_theta in range(corrected_best_infid.shape[0]):
        good = np.where(corrected_best_infid[i_theta, :] < MET_THRESHOLD)[0]
        if len(good) > 0:
            met_T[i_theta] = pulse_durations[good[0]]

    return met_T


def plot_met_map(results):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    excitation_strengths = results["excitation_strengths"]
    pulse_durations = results["pulse_durations"]
    corrected_best_infid = results["corrected_best_infid"]

    if "infidelity_floor" in results.files:
        infidelity_floor = float(results["infidelity_floor"])
    else:
        infidelity_floor = INFIDELITY_FLOOR

    theta_grid, T_grid = np.meshgrid(excitation_strengths / np.pi, pulse_durations)

    # Numerical guard before log10:
    # zeros/negatives become the lowest plotted value.
    safe_infid = corrected_best_infid.T.copy()
    safe_infid[safe_infid <= 0.0] = infidelity_floor

    z = np.log10(safe_infid)

    plt.figure(figsize=(7.2, 5.2))
    mesh = plt.pcolormesh(
        theta_grid,
        T_grid,
        z,
        shading="auto",
        cmap="inferno",
        vmin=np.log10(infidelity_floor),
        # vmax=0.0,
    )

    met_T = compute_met_line(corrected_best_infid, pulse_durations)

    plt.plot(
        excitation_strengths / np.pi,
        met_T,
        linestyle="--",
        linewidth=2.0,
        color=MET_LINE_COLOR,
        label="MET line",
    )

    cbar = plt.colorbar(mesh)
    cbar.set_label(r"$\log_{10}(infidelity)$")

    plt.xlabel(r"Excitation strength $\theta/\pi$")
    plt.ylabel(r"Pulse duration $T$ / ns")
    plt.title("SQE MET scan")
    plt.legend(frameon=False, labelcolor="white")

    plt.tight_layout()
    plt.savefig(OUTDIR / PLOT_FILENAME, dpi=300)
    plt.close()

    print(f"Plot saved to: {OUTDIR / PLOT_FILENAME}")


# =============================================================================
# Main
# =============================================================================

def main():
    results = np.load(RESULTS_FILE, allow_pickle=True)
    plot_met_map(results)


if __name__ == "__main__":
    main()
