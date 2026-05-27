#!/usr/bin/env python3
"""
plot_pulse_database_met_single.py

Single MET-style plotter for raw pulse databases.

This script:
  1. loads a raw pulse database,
  2. selects the best init by lowest infidelity at each (theta, T),
  3. applies the same duration-envelope logic used in MET plotting,
  4. plots one infidelity map with one MET line,
  5. saves only one PNG.

No entanglement plots.
No selected .npz output.

Works for both state and gate pulse databases as long as the file contains:
  - thetas
  - durations_ns
  - baseline_final_infids
  - ent_final_infids
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

INPUT_FILE = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/PULSES_DATABASE/"
    "sqe_gate_mixedJ_pulses.npz"
)

# Output goes to the same directory as this plotting script.
OUTDIR = Path(__file__).resolve().parent

# Choose which pulse family to plot:
#   "baseline" or "ent"
PULSE_TYPE = "baseline"
# PULSE_TYPE = "ent"

PLOT_FILENAME = None
#PLOT_FILENAME = "met_map_dqe_state_positiveJ.png"
# If None, filename is generated automatically from INPUT_FILE and PULSE_TYPE.

INFIDELITY_FLOOR = 1e-13
MET_THRESHOLD = 1e-5
MET_LINE_COLOR = "white"


# =============================================================================
# Helpers
# =============================================================================

def get_array(data, *names):
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of these arrays found in database: {names}")


def select_best_infid_over_inits(final_infids):
    """
    Select lowest infidelity over all saved inits at each (theta, T).

    final_infids shape:
        (n_theta, n_T, n_init)

    returns:
        best_infid shape:
            (n_theta, n_T)
    """
    n_theta, n_T, _ = final_infids.shape
    best_infid = np.full((n_theta, n_T), np.nan)

    for i_theta in range(n_theta):
        for i_T in range(n_T):
            vals = final_infids[i_theta, i_T]
            finite = np.isfinite(vals)

            if not np.any(finite):
                continue

            best_infid[i_theta, i_T] = np.nanmin(vals[finite])

    return np.maximum(best_infid, INFIDELITY_FLOOR)


def apply_duration_envelope(best_infid):
    """
    Same idea as corrected_best_infid in the original MET scripts.

    If a target is achievable at a shorter duration, longer durations should
    not look worse in the MET map. Therefore for each theta we keep the
    cumulative minimum along increasing T.
    """
    corrected = best_infid.copy()

    for i_theta in range(corrected.shape[0]):
        running = np.inf

        for i_T in range(corrected.shape[1]):
            val = corrected[i_theta, i_T]

            if not np.isfinite(val):
                continue

            if val < running:
                running = val
            else:
                corrected[i_theta, i_T] = running

    return corrected


def compute_met_line(corrected_best_infid, pulse_durations):
    met_T = np.full(corrected_best_infid.shape[0], np.nan)

    for i_theta in range(corrected_best_infid.shape[0]):
        good = np.where(corrected_best_infid[i_theta, :] < MET_THRESHOLD)[0]
        if len(good) > 0:
            met_T[i_theta] = pulse_durations[good[0]]

    return met_T


def infer_cbar_label(data):
    if "system_name" in data.files:
        system_name = str(data["system_name"])
    else:
        system_name = ""

    if "gate" in system_name:
        return r"$\log_{10}(gate\ infidelity)$"

    return r"$\log_{10}(infidelity)$"


def plot_met_map(data):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    thetas = get_array(data, "thetas", "excitation_strengths")
    pulse_durations = get_array(data, "durations_ns", "pulse_durations")

    if PULSE_TYPE == "baseline":
        final_infids = get_array(data, "baseline_final_infids")
    elif PULSE_TYPE == "ent":
        final_infids = get_array(data, "ent_final_infids")
    else:
        raise ValueError("PULSE_TYPE must be 'baseline' or 'ent'.")

    best_infid = select_best_infid_over_inits(final_infids)
    corrected_best_infid = apply_duration_envelope(best_infid)

    if "infidelity_floor" in data.files:
        infidelity_floor = float(data["infidelity_floor"])
    else:
        infidelity_floor = INFIDELITY_FLOOR

    theta_grid, T_grid = np.meshgrid(thetas / np.pi, pulse_durations)

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
    )

    met_T = compute_met_line(corrected_best_infid, pulse_durations)

    plt.plot(
        thetas / np.pi,
        met_T,
        linestyle="--",
        linewidth=2.0,
        color=MET_LINE_COLOR,
        label="MET line",
    )

    cbar = plt.colorbar(mesh)
    cbar.set_label(infer_cbar_label(data))

    plt.xlabel(r"Excitation strength $\theta/\pi$")
    plt.ylabel(r"Pulse duration $T$ / ns")
    plt.legend(frameon=False, labelcolor="white")

    plt.tight_layout()

    if PLOT_FILENAME is None:
        plot_filename = f"{INPUT_FILE.stem}_{PULSE_TYPE}_met_map.png"
    else:
        plot_filename = PLOT_FILENAME

    output_path = OUTDIR / plot_filename
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved to: {output_path}")

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Input file: {INPUT_FILE}")
    print(f"Pulse type: {PULSE_TYPE}")
    print(f"Output dir: {OUTDIR}")
    print(f"Best infidelity before envelope = {np.nanmin(best_infid):.6e}")
    print(f"Best infidelity after envelope  = {np.nanmin(corrected_best_infid):.6e}")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================

def main():
    data = np.load(INPUT_FILE, allow_pickle=True)
    plot_met_map(data)


if __name__ == "__main__":
    main()
