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

PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"
INPUT_FILE = PULSES_DATABASE / "dqe_gate_positiveJ_pulses.npz"

OUTDIR = Path(__file__).resolve().parent

# Choose which pulse family to plot: "baseline" or "ent".
PULSE_TYPE = "baseline"
# PULSE_TYPE = "ent"

PLOT_FILENAME = None

INFIDELITY_FLOOR = 1e-13
MET_THRESHOLD = 1e-5
MET_LINE_COLOR = "white"

# =============================================================================
# Plot appearance
# =============================================================================

FIGSIZE = (7.2, 5.2)

FONT_SIZE = 14
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 14
CBAR_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 14

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
})


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
    with np.errstate(all="ignore"):
        best_infid = np.nanmin(final_infids, axis=2)

    return np.maximum(best_infid, INFIDELITY_FLOOR)


def apply_duration_envelope(best_infid: np.ndarray) -> np.ndarray:
    """
    Duration-envelope (MET) correction: running minimum along the T axis.

    If a target is achievable at a shorter duration T', a longer duration T > T'
    should not appear worse in the MET map.
    """
    filled = np.where(np.isfinite(best_infid), best_infid, np.inf)
    running_min = np.minimum.accumulate(filled, axis=1)

    return np.where(np.isinf(running_min), np.nan, running_min)


def compute_met_line(
    corrected_best_infid: np.ndarray,
    pulse_durations: np.ndarray,
) -> np.ndarray:
    """
    For each theta, find the shortest T at which infidelity < MET_THRESHOLD.
    """
    met_T = np.full(corrected_best_infid.shape[0], np.nan)

    for i_theta in range(corrected_best_infid.shape[0]):
        good = np.where(
            corrected_best_infid[i_theta, :] < MET_THRESHOLD
        )[0]

        if len(good) > 0:
            met_T[i_theta] = pulse_durations[good[0]]

    return met_T


def infer_cbar_label(data) -> str:
    """Choose a colorbar label."""
    return r"$\log_{10}$(infidelity)"


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

    best_infid = select_best_infid_over_inits(final_infids)
    corrected_best_infid = apply_duration_envelope(best_infid)

    infidelity_floor = (
        float(data["infidelity_floor"])
        if "infidelity_floor" in data.files
        else INFIDELITY_FLOOR
    )

    theta_grid, T_grid = np.meshgrid(
        thetas / np.pi,
        pulse_durations,
    )

    safe_infid = corrected_best_infid.T.copy()
    safe_infid[safe_infid <= 0.0] = infidelity_floor

    z = np.log10(safe_infid)

    plt.figure(figsize=FIGSIZE)

    mesh = plt.pcolormesh(
        theta_grid,
        T_grid,
        z,
        shading="auto",
        cmap="inferno",
        vmin=np.log10(infidelity_floor),
    )

    met_T = compute_met_line(
        corrected_best_infid,
        pulse_durations,
    )

    plt.plot(
        thetas / np.pi,
        met_T,
        linestyle="--",
        linewidth=2.0,
        color=MET_LINE_COLOR,
        label="MET line",
    )

    cbar = plt.colorbar(mesh)
    cbar.set_label(
        infer_cbar_label(data),
        fontsize=CBAR_LABEL_SIZE,
    )

    plt.xlabel(
        r"Excitation strength $\theta/\pi$",
        fontsize=AXIS_LABEL_SIZE,
    )

    plt.ylabel(
        r"Pulse duration $T$ / ns",
        fontsize=AXIS_LABEL_SIZE,
    )

    plt.legend(
        frameon=False,
        labelcolor="white",
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout()

    plot_filename = (
        PLOT_FILENAME
        or f"{INPUT_FILE.stem}_{PULSE_TYPE}_met_map.png"
    )

    output_path = OUTDIR / plot_filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

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