#!/usr/bin/env python3
"""
2-plot_pulse_database_met.py

Single MET-style plotter for raw pulse databases.

Steps
-----
  1. Load a raw pulse database (.npz).
  2. Select the best init by lowest infidelity at each (theta, T).
  3. Apply the duration-envelope correction (MET logic: if achievable at
     shorter T, longer T should not look worse).
  4. Plot a log₁₀(infidelity) heatmap with the MET line overlaid.
  5. Save one PNG to OUTDIR.

Compatible databases
--------------------
  Works for both state and gate pulse databases, provided the file contains:
    thetas, durations_ns, baseline_final_infids, ent_final_infids.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

# B1 fix: INPUT_FILE resolved relative to this script.
PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"
INPUT_FILE = PULSES_DATABASE / "dqe_gate_positiveJ_pulses.npz"

# Output goes to the same directory as this plotting script.
OUTDIR = Path(__file__).resolve().parent

# Choose which pulse family to plot: "baseline" or "ent".
PULSE_TYPE = "baseline"
# PULSE_TYPE = "ent"

# Override the auto-generated filename, or leave as None.
PLOT_FILENAME = None

INFIDELITY_FLOOR = 1e-13
MET_THRESHOLD = 1e-5
MET_LINE_COLOR = "white"


# =============================================================================
# Helpers
# =============================================================================

def get_array(data, *names):
    """Return the first of *names* that exists in *data.files*."""
    for name in names:
        if name in data.files:
            return data[name]
    raise KeyError(f"None of these arrays found in database: {names}")


def select_best_infid_over_inits(final_infids: np.ndarray) -> np.ndarray:
    """
    Minimum infidelity over all saved inits at each (theta, T) grid point.

    Parameters
    ----------
    final_infids : (n_theta, n_T, n_init)

    Returns
    -------
    best_infid : (n_theta, n_T), clipped to INFIDELITY_FLOOR.
    """
    # E3 fix: vectorised nanmin replaces the nested Python loop.
    # np.nanmin on an all-NaN slice returns NaN with a RuntimeWarning; suppress it.
    with np.errstate(all="ignore"):
        best_infid = np.nanmin(final_infids, axis=2)
    return np.maximum(best_infid, INFIDELITY_FLOOR)


def apply_duration_envelope(best_infid: np.ndarray) -> np.ndarray:
    """
    Duration-envelope (MET) correction: running minimum along the T axis.

    If a target is achievable at a shorter duration T', a longer duration T > T'
    should not appear worse in the MET map.  This is identical to the
    cumulative minimum of best_infid along the T axis.

    Parameters
    ----------
    best_infid : (n_theta, n_T)

    Returns
    -------
    corrected : (n_theta, n_T), NaN where no finite value has been seen yet.
    """
    # E3 fix: np.minimum.accumulate replaces the nested Python loop.
    # Replace NaN with +inf so they are transparent to the running minimum.
    filled = np.where(np.isfinite(best_infid), best_infid, np.inf)
    running_min = np.minimum.accumulate(filled, axis=1)
    # Restore NaN where no finite value had appeared at or before this T.
    return np.where(np.isinf(running_min), np.nan, running_min)


def compute_met_line(corrected_best_infid: np.ndarray, pulse_durations: np.ndarray) -> np.ndarray:
    """
    For each theta, find the shortest T at which infidelity < MET_THRESHOLD.

    Returns NaN for theta values where the threshold is never reached.
    """
    met_T = np.full(corrected_best_infid.shape[0], np.nan)
    for i_theta in range(corrected_best_infid.shape[0]):
        good = np.where(corrected_best_infid[i_theta, :] < MET_THRESHOLD)[0]
        if len(good) > 0:
            met_T[i_theta] = pulse_durations[good[0]]
    return met_T


def infer_cbar_label(data) -> str:
    """Choose a colorbar label based on whether the database is for state or gate."""
    if "system_name" in data.files:
        system_name = str(data["system_name"])
    else:
        system_name = ""
    #if "gate" in system_name:
    #    return r"$\log_{10}(\mathrm{gate\ infidelity})$"
    return r"$\log_{10}(\mathrm{infidelity})$"


def plot_met_map(data) -> None:
    """Load arrays from *data*, compute the MET line, and save a heatmap PNG."""
    OUTDIR.mkdir(parents=True, exist_ok=True)

    thetas = get_array(data, "thetas", "excitation_strengths")
    pulse_durations = get_array(data, "durations_ns", "pulse_durations")

    if PULSE_TYPE == "baseline":
        final_infids = get_array(data, "baseline_final_infids")
    elif PULSE_TYPE == "ent":
        final_infids = get_array(data, "ent_final_infids")
    else:
        raise ValueError("PULSE_TYPE must be 'baseline' or 'ent'.")

    # Select best init and apply envelope.
    best_infid = select_best_infid_over_inits(final_infids)
    corrected_best_infid = apply_duration_envelope(best_infid)

    infidelity_floor = (
        float(data["infidelity_floor"])
        if "infidelity_floor" in data.files
        else INFIDELITY_FLOOR
    )

    # pcolormesh expects (T, theta) orientation (T on y-axis).
    theta_grid, T_grid = np.meshgrid(thetas / np.pi, pulse_durations)
    safe_infid = corrected_best_infid.T.copy()
    safe_infid[safe_infid <= 0.0] = infidelity_floor
    z = np.log10(safe_infid)

    plt.figure(figsize=(7.2, 5.2))
    mesh = plt.pcolormesh(
        theta_grid, T_grid, z,
        shading="auto",
        cmap="inferno",
        vmin=np.log10(infidelity_floor),
    )

    met_T = compute_met_line(corrected_best_infid, pulse_durations)
    plt.plot(thetas / np.pi, met_T,
             linestyle="--", linewidth=2.0, color=MET_LINE_COLOR, label="MET line")

    cbar = plt.colorbar(mesh)
    cbar.set_label(infer_cbar_label(data))

    plt.xlabel(r"Excitation strength $\theta/\pi$")
    plt.ylabel(r"Pulse duration $T$ / ns")
    plt.legend(frameon=False, labelcolor="white")
    plt.tight_layout()

    plot_filename = PLOT_FILENAME or f"{INPUT_FILE.stem}_{PULSE_TYPE}_met_map.png"
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

def main() -> None:
    data = np.load(INPUT_FILE, allow_pickle=True)
    plot_met_map(data)


if __name__ == "__main__":
    main()
