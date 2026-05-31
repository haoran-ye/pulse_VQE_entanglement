#!/usr/bin/env python3
"""
3-plot_pulse_database_met_ent.py

Three-panel MET + entanglement plotter for raw pulse databases.

Produces two PNG files per run:

  1. Infidelity three-panel (baseline | TEM | Δlog₁₀I)
  2. Entanglement three-panel (baseline | TEM | reduction)

Each panel shows a MET line for the baseline and/or TEM pulse family.

Compatible databases
--------------------
  Works for state databases (keys: baseline_s_avg, ent_s_avg, …) and gate
  databases (keys: baseline_ent_avg, ent_ent_avg, …).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

# B1 fix: INPUT_FILE resolved relative to this script.
PULSES_DATABASE = Path(__file__).resolve().parent.parent / "PULSES_DATABASE"
INPUT_FILE = PULSES_DATABASE / "dqe_gate_positiveJ_pulses.npz"

OUTDIR = Path(__file__).resolve().parent
PLOT_PREFIX = INPUT_FILE.stem

INFIDELITY_FLOOR = 1e-14
MET_THRESHOLD = 1e-5

ENTANGLEMENT_STAT = "avg"
# ENTANGLEMENT_STAT = "max"


# =============================================================================
# Array helpers
# =============================================================================

def get_array(raw, *names):
    """Return the first of *names* that exists in *raw.files*."""
    for name in names:
        if name in raw.files:
            return raw[name]
    raise KeyError(f"None of these arrays found: {names}")


def infer_database_kind(raw) -> str:
    """Distinguish state databases (s_avg/s_max) from gate databases (ent_avg/ent_max)."""
    if "ent_s_avg" in raw.files or "ent_s_max" in raw.files:
        return "state"
    if "ent_ent_avg" in raw.files or "ent_ent_max" in raw.files:
        return "gate"
    raise KeyError("Could not infer database kind.")


def get_entanglement_arrays(raw, kind: str):
    """
    Return baseline and TEM entanglement arrays plus colorbar labels.

    Returns
    -------
    before_ent_all, after_ent_all, shared_cbar, reduction_cbar, reduction_title
    """
    if kind == "state":
        if ENTANGLEMENT_STAT == "avg":
            before_ent_all = get_array(raw, "baseline_s_avg", "baseline_s_max")
            after_ent_all = get_array(raw, "ent_s_avg", "ent_s_max")
            reduction_title = "(c) Reduction in state entanglement"
        elif ENTANGLEMENT_STAT == "max":
            before_ent_all = get_array(raw, "baseline_s_max", "baseline_s_avg")
            after_ent_all = get_array(raw, "ent_s_max", "ent_s_avg")
            reduction_title = "(c) Maximum reduction in state entanglement"
        else:
            raise ValueError("ENTANGLEMENT_STAT must be 'avg' or 'max'.")

    elif kind == "gate":
        if ENTANGLEMENT_STAT == "avg":
            before_ent_all = get_array(raw, "baseline_ent_avg", "baseline_ent_max")
            after_ent_all = get_array(raw, "ent_ent_avg", "ent_ent_max")
            reduction_title = "(c) Reduction in unitary entanglement"
        elif ENTANGLEMENT_STAT == "max":
            before_ent_all = get_array(raw, "baseline_ent_max", "baseline_ent_avg")
            after_ent_all = get_array(raw, "ent_ent_max", "ent_ent_avg")
            reduction_title = "(c) Maximum reduction in unitary entanglement"
        else:
            raise ValueError("ENTANGLEMENT_STAT must be 'avg' or 'max'.")

    else:
        raise ValueError(kind)

    shared_cbar = r"$\overline{S}_{\rm transient}$"
    reduction_cbar = (
        r"$\overline{S}_{\rm transient}"
        r"-\overline{S}^{\rm TEM}_{\rm transient}$"
    )
    return before_ent_all, after_ent_all, shared_cbar, reduction_cbar, reduction_title


def select_before_from_raw(raw, before_ent_all: np.ndarray):
    """
    For each (theta, T), select the init with the lowest finite baseline infidelity.

    Returns
    -------
    before_infid : (n_theta, n_T), clipped to INFIDELITY_FLOOR.
    before_ent   : (n_theta, n_T), entanglement at the selected init.
    """
    baseline_final_infids = raw["baseline_final_infids"]

    # E3 fix: vectorised argmin replaces the double Python loop.
    # Replace NaN with inf so argmin selects the smallest finite value.
    filled = np.where(np.isfinite(baseline_final_infids), baseline_final_infids, np.inf)
    k_best = np.argmin(filled, axis=2)  # (n_theta, n_T)

    # Fancy-index the 3D arrays using the best-k indices.
    i_idx = np.arange(baseline_final_infids.shape[0])[:, None]
    j_idx = np.arange(baseline_final_infids.shape[1])[None, :]

    before_infid = baseline_final_infids[i_idx, j_idx, k_best]
    before_ent = before_ent_all[i_idx, j_idx, k_best]

    # Where all inits were non-finite, argmin is arbitrary; restore NaN.
    all_non_finite = ~np.any(np.isfinite(baseline_final_infids), axis=2)
    before_infid[all_non_finite] = np.nan
    before_ent[all_non_finite] = np.nan

    return np.maximum(before_infid, INFIDELITY_FLOOR), before_ent


def local_tem_selection_at_point(ent_infids: np.ndarray, ent_values: np.ndarray, threshold: float):
    """
    Select the best TEM init at a single (theta, T) grid point.

    Priority: (1) lowest entanglement among successful runs, (2) lowest
    infidelity among all finite runs, (3) -1 if nothing finite.
    """
    finite = np.isfinite(ent_infids) & np.isfinite(ent_values)
    successful = finite & (ent_infids <= threshold)

    if np.any(successful):
        idx = np.where(successful)[0]
        k = idx[np.nanargmin(ent_values[successful])]
        return int(k), True

    finite_I = np.isfinite(ent_infids)
    if np.any(finite_I):
        idx = np.where(finite_I)[0]
        k = idx[np.nanargmin(ent_infids[finite_I])]
        return int(k), False

    return -1, False


def select_after_from_raw(raw, after_ent_all: np.ndarray, threshold: float):
    """
    Select TEM results using the MET-override logic: once a T gives a successful
    low-entanglement result, that result carries forward to larger T.

    Returns
    -------
    after_infid : (n_theta, n_T)
    after_ent   : (n_theta, n_T)
    after_uses_successful_override : (n_theta, n_T) bool
    """
    ent_final_infids = raw["ent_final_infids"]
    n_theta, n_T, _ = ent_final_infids.shape

    after_infid = np.full((n_theta, n_T), np.nan)
    after_ent = np.full((n_theta, n_T), np.nan)
    after_uses_successful_override = np.full((n_theta, n_T), False, dtype=bool)

    for i in range(n_theta):
        best_success_ent = np.inf
        best_success_infid = np.nan

        for j in range(n_T):
            k_local, local_success = local_tem_selection_at_point(
                ent_final_infids[i, j], after_ent_all[i, j], threshold,
            )

            if local_success:
                local_ent = after_ent_all[i, j, k_local]
                local_infid = ent_final_infids[i, j, k_local]
                if local_ent < best_success_ent:
                    best_success_ent = local_ent
                    best_success_infid = local_infid

            if np.isfinite(best_success_ent):
                after_infid[i, j] = best_success_infid
                after_ent[i, j] = best_success_ent
                after_uses_successful_override[i, j] = True
            elif k_local >= 0:
                after_infid[i, j] = ent_final_infids[i, j, k_local]
                after_ent[i, j] = after_ent_all[i, j, k_local]

    return (
        np.maximum(after_infid, INFIDELITY_FLOOR),
        after_ent,
        after_uses_successful_override,
    )


def extract_met_line(infid: np.ndarray, durations_ns: np.ndarray, threshold: float) -> np.ndarray:
    """
    For each theta, return the first T at which infidelity < threshold.
    Uses np.argmax on a boolean mask, which returns the first True index.
    """
    met = np.full(infid.shape[0], np.nan)
    for i in range(infid.shape[0]):
        valid = np.isfinite(infid[i]) & (infid[i] <= threshold)
        if np.any(valid):
            met[i] = durations_ns[np.argmax(valid)]
    return met


def plot_met_line(ax, theta_pi: np.ndarray, met: np.ndarray, label: str, color: str,
                  linestyle: str = "--") -> None:
    """Overlay a MET line on an existing axes, skipping NaN theta values."""
    valid = np.isfinite(met)
    if np.any(valid):
        ax.plot(theta_pi[valid], met[valid],
                color=color, linestyle=linestyle, linewidth=2.2, label=label)


def finite_minmax(*arrays) -> tuple[float | None, float | None]:
    """Global min and max over all finite values across multiple arrays."""
    values = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(arr)):
            values.append(np.nanmin(arr))
            values.append(np.nanmax(arr))
    if not values:
        return None, None
    return min(values), max(values)


def symmetric_absmax(arr: np.ndarray) -> float:
    """Max absolute finite value; returns 1.0 if none finite or all zero."""
    arr = np.asarray(arr, dtype=float)
    if not np.any(np.isfinite(arr)):
        return 1.0
    val = np.nanmax(np.abs(arr))
    if not np.isfinite(val) or val == 0:
        return 1.0
    return float(val)


# =============================================================================
# Plotting
# =============================================================================

def plot_three_panel(
    X: np.ndarray,
    Y: np.ndarray,
    theta_pi: np.ndarray,
    panels: list[dict],
    outfile: Path,
    shared_left_middle_cbar: bool = False,
) -> None:
    """
    Create a 1×3 heatmap figure and save it.

    Parameters
    ----------
    X, Y          : meshgrid arrays for pcolormesh (theta/pi, duration).
    theta_pi      : 1D theta values (for MET line).
    panels        : list of dicts with keys z, title, cbar, cmap, vmin, vmax,
                    optionally before_met and after_met.
    outfile       : output PNG path.
    shared_left_middle_cbar : if True, panels[0] and [1] share one colorbar.

    B8 fix: constrained_layout=True replaces all manual colorbar positioning.
    """
    # B8 fix: constrained_layout=True lets matplotlib handle colorbar sizing.
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 4.8),
        constrained_layout=True,
        sharey=True,
    )

    images = []

    for i_ax, (ax, panel) in enumerate(zip(axes, panels)):
        im = ax.pcolormesh(
            X, Y,
            np.ma.masked_invalid(panel["z"].T),
            shading="auto",
            cmap=panel["cmap"],
            vmin=panel.get("vmin"),
            vmax=panel.get("vmax"),
        )
        images.append(im)

        if panel.get("before_met") is not None:
            plot_met_line(ax, theta_pi, panel["before_met"], "MET (baseline)", "white")
        if panel.get("after_met") is not None:
            plot_met_line(ax, theta_pi, panel["after_met"], "MET (TEM)", "cyan")

        ax.set_title(panel["title"])
        ax.set_xlabel(r"Excitation strength $\theta/\pi$")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_box_aspect(1)

        if panel.get("before_met") is not None or panel.get("after_met") is not None:
            ax.legend(frameon=True)

    axes[0].set_ylabel(r"Pulse duration $T$ / ns")

    if shared_left_middle_cbar:
        # Shared colorbar spanning axes[0] and axes[1]; right panel gets its own.
        fig.colorbar(images[1], ax=axes[:2], label=panels[0]["cbar"])
        fig.colorbar(images[2], ax=axes[2], label=panels[2]["cbar"])
    else:
        for ax, im, panel in zip(axes, images, panels):
            fig.colorbar(im, ax=ax, label=panel["cbar"])

    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved plot to\n{outfile}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(INPUT_FILE, allow_pickle=True)
    kind = infer_database_kind(raw)

    thetas = get_array(raw, "thetas", "excitation_strengths")
    theta_pi = thetas / np.pi
    durations_ns = get_array(raw, "durations_ns", "pulse_durations")

    threshold = (
        float(raw["infidelity_threshold"])
        if "infidelity_threshold" in raw.files
        else MET_THRESHOLD
    )

    X, Y = np.meshgrid(theta_pi, durations_ns)

    (
        before_ent_all, after_ent_all,
        shared_ent_cbar, reduction_ent_cbar, reduction_ent_title,
    ) = get_entanglement_arrays(raw, kind)

    before_infid, before_ent = select_before_from_raw(raw, before_ent_all)
    after_infid, after_ent, after_uses_successful_override = select_after_from_raw(
        raw, after_ent_all, threshold,
    )

    before_met = extract_met_line(before_infid, durations_ns, threshold)
    after_met = extract_met_line(after_infid, durations_ns, threshold)

    log_before_infid = np.log10(before_infid)
    log_after_infid = np.log10(after_infid)
    delta_log_infid = log_after_infid - log_before_infid

    infid_vmin, infid_vmax = finite_minmax(log_before_infid, log_after_infid)
    delta_abs = symmetric_absmax(delta_log_infid)

    # --- Infidelity three-panel ---
    plot_three_panel(
        X, Y, theta_pi,
        panels=[
            dict(z=log_before_infid, title="(a) Baseline",
                 cbar=r"$\log_{10}(\mathrm{infidelity})$",
                 cmap="inferno", vmin=infid_vmin, vmax=infid_vmax,
                 before_met=before_met),
            dict(z=log_after_infid, title="(b) TEM",
                 cbar=r"$\log_{10}(\mathrm{infidelity})$",
                 cmap="inferno", vmin=infid_vmin, vmax=infid_vmax,
                 after_met=after_met),
            dict(z=delta_log_infid, title="(c) Infidelity ratio",
                 cbar=r"-$\Delta \log_{10}(\mathrm{infidelity})$",
                 cmap="coolwarm", vmin=-delta_abs, vmax=delta_abs,
                 before_met=before_met, after_met=after_met),
        ],
        outfile=OUTDIR / f"{PLOT_PREFIX}_ent_infidelity_three_panel_met.png",
        shared_left_middle_cbar=False,
    )

    # --- Normalise entanglement to [0, 1] ---
    ent_norm = np.nanmax([np.nanmax(before_ent), np.nanmax(after_ent)])
    if not np.isfinite(ent_norm) or ent_norm == 0.0:
        ent_norm = 1.0
    before_ent = before_ent / ent_norm
    after_ent = after_ent / ent_norm
    reduction_ent = before_ent - after_ent

    ent_vmin, ent_vmax = finite_minmax(before_ent, after_ent)
    red_abs = 1.0

    # --- Entanglement three-panel ---
    plot_three_panel(
        X, Y, theta_pi,
        panels=[
            dict(z=before_ent, title="(a) Baseline",
                 cbar=r"$\overline{S}_{\rm transient}$",
                 cmap="viridis", vmin=ent_vmin, vmax=ent_vmax,
                 before_met=before_met),
            dict(z=after_ent, title="(b) TEM",
                 cbar=r"$\overline{S}_{\rm transient}$",
                 cmap="viridis", vmin=ent_vmin, vmax=ent_vmax,
                 after_met=after_met),
            dict(z=reduction_ent, title=reduction_ent_title,
                 cbar=(
                     r"$\overline{S}_{\rm transient}"
                     r"-\overline{S}^{\rm TEM}_{\rm transient}$"
                 ),
                 cmap="coolwarm", vmin=-red_abs, vmax=red_abs,
                 before_met=before_met, after_met=after_met),
        ],
        outfile=OUTDIR / f"{PLOT_PREFIX}_entanglement_three_panel_met.png",
        shared_left_middle_cbar=True,
    )

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output dir: {OUTDIR}")
    print(f"Database kind: {kind}")
    print(f"ENTANGLEMENT_STAT: {ENTANGLEMENT_STAT}")
    print(f"Best before infidelity: {np.nanmin(before_infid):.6e}")
    print(f"Best after infidelity:  {np.nanmin(after_infid):.6e}")
    print(f"Entanglement normalisation: {ent_norm:.6e}")
    print(f"Max ent reduction:      {np.nanmax(reduction_ent):.6e}")
    print(f"After override points:  {np.sum(after_uses_successful_override)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
