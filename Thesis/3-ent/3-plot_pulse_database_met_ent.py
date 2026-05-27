#!/usr/bin/env python3
"""
plot_pulse_database_met_ent.py

Plot MET-entanglement maps from a raw pulse database.

This version uses the raw PULSES_DATABASE format, but applies the intended
TEM-after selection logic:

Before / baseline:
    At each fixed (theta, T), choose the baseline init with lowest infidelity.

After / TEM:
    For each fixed theta, scan durations in increasing T.

    At each T:
      1. Locally select the lowest-entanglement successful TEM init
         among runs satisfying infidelity <= threshold.
      2. If no successful TEM init exists at this T, locally select the
         lowest-infidelity TEM init only as a fallback for pre-MET plotting.
      3. Maintain the best successful TEM pulse found so far along increasing T,
         ranked by lowest entanglement.
      4. Once a successful TEM pulse has been found, use the lowest-entanglement
         successful pulse so far for all later durations.

Therefore:
    - before has no duration overriding;
    - after/TEM has entanglement-based duration overriding after MET;
    - after_infid and after_ent always correspond to the same selected run;
    - no selected .npz file is saved.

Expected raw database arrays:
    thetas
    durations_ns
    baseline_final_infids
    ent_final_infids

For state database:
    baseline_s_avg / baseline_s_max
    ent_s_avg / ent_s_max

For gate database:
    baseline_ent_avg / baseline_ent_max
    ent_ent_avg / ent_ent_max
"""

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

PLOT_PREFIX = INPUT_FILE.stem

INFIDELITY_FLOOR = 1e-14
MET_THRESHOLD = 1e-5

# Choose "avg" for the current met_ent story.
# Choose "max" only if you explicitly want max-entanglement plots.
ENTANGLEMENT_STAT = "avg"
# ENTANGLEMENT_STAT = "max"


# =============================================================================
# Helpers
# =============================================================================

def add_colorbar(fig, ax, im, label):
    cbar = fig.colorbar(
        im,
        ax=ax,
        pad=0.02,
        fraction=0.046,
        shrink=1.0,
        aspect=25,
    )
    cbar.set_label(label)
    return cbar


def get_array(raw, *names):
    for name in names:
        if name in raw.files:
            return raw[name]
    raise KeyError(f"None of these arrays found: {names}")


def infer_database_kind(raw):
    if "ent_s_avg" in raw.files or "ent_s_max" in raw.files:
        return "state"
    if "ent_ent_avg" in raw.files or "ent_ent_max" in raw.files:
        return "gate"
    raise KeyError("Could not infer database kind: no recognised entanglement arrays found.")


def get_entanglement_arrays(raw, kind):
    if kind == "state":
        if ENTANGLEMENT_STAT == "avg":
            before_ent_all = get_array(raw, "baseline_s_avg", "baseline_s_max")
            after_ent_all = get_array(raw, "ent_s_avg", "ent_s_max")
            before_cbar = r"$\overline{S}_{\rm before}$"
            after_cbar = r"$\overline{S}_{\rm after}$"
            reduction_cbar = r"$\overline{S}_{\rm before}-\overline{S}_{\rm after}$"
            reduction_title = "Average entanglement reduction"
        elif ENTANGLEMENT_STAT == "max":
            before_ent_all = get_array(raw, "baseline_s_max", "baseline_s_avg")
            after_ent_all = get_array(raw, "ent_s_max", "ent_s_avg")
            before_cbar = r"$S_{\max,\rm before}$"
            after_cbar = r"$S_{\max,\rm after}$"
            reduction_cbar = r"$S_{\max,\rm before}-S_{\max,\rm after}$"
            reduction_title = "Maximum entanglement reduction"
        else:
            raise ValueError("ENTANGLEMENT_STAT must be 'avg' or 'max'.")

        return before_ent_all, after_ent_all, before_cbar, after_cbar, reduction_cbar, reduction_title

    if kind == "gate":
        if ENTANGLEMENT_STAT == "avg":
            before_ent_all = get_array(raw, "baseline_ent_avg", "baseline_ent_max")
            after_ent_all = get_array(raw, "ent_ent_avg", "ent_ent_max")
            before_cbar = r"$\overline{E}_{\rm op,before}$"
            after_cbar = r"$\overline{E}_{\rm op,after}$"
            reduction_cbar = r"$\overline{E}_{\rm op,before}-\overline{E}_{\rm op,after}$"
            reduction_title = "Average gate-entanglement reduction"
        elif ENTANGLEMENT_STAT == "max":
            before_ent_all = get_array(raw, "baseline_ent_max", "baseline_ent_avg")
            after_ent_all = get_array(raw, "ent_ent_max", "ent_ent_avg")
            before_cbar = r"$E_{\rm op,max,before}$"
            after_cbar = r"$E_{\rm op,max,after}$"
            reduction_cbar = r"$E_{\rm op,max,before}-E_{\rm op,max,after}$"
            reduction_title = "Maximum gate-entanglement reduction"
        else:
            raise ValueError("ENTANGLEMENT_STAT must be 'avg' or 'max'.")

        return before_ent_all, after_ent_all, before_cbar, after_cbar, reduction_cbar, reduction_title

    raise ValueError(kind)


def select_before_from_raw(raw, before_ent_all):
    """
    Before / baseline:
        At each (theta, T), choose the baseline init with lowest infidelity.

    No duration overriding is applied to before.
    """
    baseline_final_infids = raw["baseline_final_infids"]

    n_theta, n_T, _ = baseline_final_infids.shape

    before_infid = np.full((n_theta, n_T), np.nan)
    before_ent = np.full((n_theta, n_T), np.nan)
    before_seed_index = np.full((n_theta, n_T), -1, dtype=int)

    for i in range(n_theta):
        for j in range(n_T):
            finite = np.isfinite(baseline_final_infids[i, j])
            if not np.any(finite):
                continue

            idx = np.where(finite)[0]
            k = idx[np.nanargmin(baseline_final_infids[i, j, finite])]

            before_seed_index[i, j] = k
            before_infid[i, j] = baseline_final_infids[i, j, k]
            before_ent[i, j] = before_ent_all[i, j, k]

    return np.maximum(before_infid, INFIDELITY_FLOOR), before_ent, before_seed_index


def local_tem_selection_at_point(ent_infids, ent_values, threshold):
    """
    Local TEM selection at one fixed (theta, T).

    Returns:
        k_local
        local_is_successful

    Rule:
        - If any successful TEM init exists, choose lowest entanglement among them.
        - Otherwise choose lowest infidelity among finite runs.
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


def select_after_from_raw(raw, after_ent_all, threshold):
    """
    After / TEM:

    For each theta, scan durations in increasing T.

    Before the first successful TEM run:
        plot the local lowest-infidelity TEM run, so every finite point is visible.

    Once successful TEM runs exist:
        keep the lowest-entanglement successful TEM run found so far and use it
        for this and all later durations, unless a later successful run gives
        even lower entanglement.

    This is the duration-overriding logic requested for TEM.
    """
    ent_final_infids = raw["ent_final_infids"]

    n_theta, n_T, _ = ent_final_infids.shape

    after_infid = np.full((n_theta, n_T), np.nan)
    after_ent = np.full((n_theta, n_T), np.nan)
    after_seed_index = np.full((n_theta, n_T), -1, dtype=int)
    after_source_T_index = np.full((n_theta, n_T), -1, dtype=int)
    after_uses_successful_override = np.full((n_theta, n_T), False, dtype=bool)

    for i in range(n_theta):
        best_success_k = -1
        best_success_T_index = -1
        best_success_ent = np.inf
        best_success_infid = np.nan

        for j in range(n_T):
            k_local, local_success = local_tem_selection_at_point(
                ent_final_infids[i, j],
                after_ent_all[i, j],
                threshold,
            )

            if local_success:
                local_ent = after_ent_all[i, j, k_local]
                local_infid = ent_final_infids[i, j, k_local]

                if local_ent < best_success_ent:
                    best_success_k = k_local
                    best_success_T_index = j
                    best_success_ent = local_ent
                    best_success_infid = local_infid

            if best_success_k >= 0:
                # Duration override: use the lowest-entanglement successful
                # TEM pulse found so far along increasing T.
                after_seed_index[i, j] = best_success_k
                after_source_T_index[i, j] = best_success_T_index
                after_infid[i, j] = best_success_infid
                after_ent[i, j] = best_success_ent
                after_uses_successful_override[i, j] = True

            elif k_local >= 0:
                # Pre-MET fallback: no successful TEM pulse has appeared yet.
                # Plot the local best-infidelity TEM run so the point is visible.
                after_seed_index[i, j] = k_local
                after_source_T_index[i, j] = j
                after_infid[i, j] = ent_final_infids[i, j, k_local]
                after_ent[i, j] = after_ent_all[i, j, k_local]
                after_uses_successful_override[i, j] = False

    return (
        np.maximum(after_infid, INFIDELITY_FLOOR),
        after_ent,
        after_seed_index,
        after_source_T_index,
        after_uses_successful_override,
    )


def extract_met_line(infid, durations_ns, threshold):
    met = np.full(infid.shape[0], np.nan)

    for i in range(infid.shape[0]):
        valid = np.isfinite(infid[i]) & (infid[i] <= threshold)
        if np.any(valid):
            met[i] = durations_ns[np.argmax(valid)]

    return met


def plot_met_line(ax, theta_pi, met, label, color, linestyle="--"):
    valid = np.isfinite(met)
    if np.any(valid):
        ax.plot(
            theta_pi[valid],
            met[valid],
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            label=label,
        )


def finite_minmax(*arrays):
    values = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if np.any(np.isfinite(arr)):
            values.append(np.nanmin(arr))
            values.append(np.nanmax(arr))
    if not values:
        return None, None
    return min(values), max(values)


def symmetric_absmax(arr):
    arr = np.asarray(arr, dtype=float)
    if not np.any(np.isfinite(arr)):
        return 1.0
    val = np.nanmax(np.abs(arr))
    if not np.isfinite(val) or val == 0:
        return 1.0
    return float(val)


def plot_three_panel(X, Y, theta_pi, panels, outfile):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    for ax, panel in zip(axes, panels):
        im = ax.pcolormesh(
            X,
            Y,
            np.ma.masked_invalid(panel["z"].T),
            shading="auto",
            cmap=panel["cmap"],
            vmin=panel.get("vmin"),
            vmax=panel.get("vmax"),
        )

        add_colorbar(fig, ax, im, panel["cbar"])

        if panel.get("before_met") is not None:
            plot_met_line(ax, theta_pi, panel["before_met"], "Before MET", "white")

        if panel.get("after_met") is not None:
            plot_met_line(ax, theta_pi, panel["after_met"], "After MET", "cyan")

        ax.set_title(panel["title"])
        ax.set_xlabel(r"Excitation strength $\theta/\pi$")
        ax.set_ylabel(r"Pulse duration $T$ / ns")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_box_aspect(1)

        if panel.get("before_met") is not None or panel.get("after_met") is not None:
            ax.legend(frameon=True)

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to\n{outfile}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(INPUT_FILE, allow_pickle=True)

    kind = infer_database_kind(raw)

    thetas = get_array(raw, "thetas", "excitation_strengths")
    theta_pi = thetas / np.pi
    durations_ns = get_array(raw, "durations_ns", "pulse_durations")

    if "infidelity_threshold" in raw.files:
        threshold = float(raw["infidelity_threshold"])
    else:
        threshold = MET_THRESHOLD

    X, Y = np.meshgrid(theta_pi, durations_ns)

    (
        before_ent_all,
        after_ent_all,
        before_ent_cbar,
        after_ent_cbar,
        reduction_ent_cbar,
        reduction_ent_title,
    ) = get_entanglement_arrays(raw, kind)

    before_infid, before_ent, before_seed_index = select_before_from_raw(
        raw,
        before_ent_all,
    )

    (
        after_infid,
        after_ent,
        after_seed_index,
        after_source_T_index,
        after_uses_successful_override,
    ) = select_after_from_raw(
        raw,
        after_ent_all,
        threshold,
    )

    before_met = extract_met_line(before_infid, durations_ns, threshold)
    after_met = extract_met_line(after_infid, durations_ns, threshold)

    log_before_infid = np.log10(before_infid)
    log_after_infid = np.log10(after_infid)
    delta_log_infid = log_after_infid - log_before_infid

    infid_vmin, infid_vmax = finite_minmax(log_before_infid, log_after_infid)
    delta_abs = symmetric_absmax(delta_log_infid)

    plot_three_panel(
        X,
        Y,
        theta_pi,
        panels=[
            dict(
                z=log_before_infid,
                title="Before entanglement optimisation",
                cbar=r"$\log_{10}(\mathcal{I}_{\rm before})$",
                cmap="inferno",
                vmin=infid_vmin,
                vmax=infid_vmax,
                before_met=before_met,
            ),
            dict(
                z=log_after_infid,
                title="After entanglement optimisation",
                cbar=r"$\log_{10}(\mathcal{I}_{\rm after})$",
                cmap="inferno",
                vmin=infid_vmin,
                vmax=infid_vmax,
                after_met=after_met,
            ),
            dict(
                z=delta_log_infid,
                title="Change in infidelity",
                cbar=r"$\Delta \log_{10}\mathcal{I}$",
                cmap="coolwarm",
                vmin=-delta_abs,
                vmax=delta_abs,
                before_met=before_met,
                after_met=after_met,
            ),
        ],
        outfile=OUTDIR / f"{PLOT_PREFIX}_ent_infidelity_three_panel_met.png",
    )

    reduction_ent = before_ent - after_ent

    ent_vmin, ent_vmax = finite_minmax(before_ent, after_ent)
    red_abs = symmetric_absmax(reduction_ent)

    plot_three_panel(
        X,
        Y,
        theta_pi,
        panels=[
            dict(
                z=before_ent,
                title="Before entanglement optimisation",
                cbar=before_ent_cbar,
                cmap="viridis",
                vmin=ent_vmin,
                vmax=ent_vmax,
                before_met=before_met,
            ),
            dict(
                z=after_ent,
                title="After entanglement optimisation",
                cbar=after_ent_cbar,
                cmap="viridis",
                vmin=ent_vmin,
                vmax=ent_vmax,
                after_met=after_met,
            ),
            dict(
                z=reduction_ent,
                title=reduction_ent_title,
                cbar=reduction_ent_cbar,
                cmap="coolwarm",
                vmin=-red_abs,
                vmax=red_abs,
                before_met=before_met,
                after_met=after_met,
            ),
        ],
        outfile=OUTDIR / f"{PLOT_PREFIX}_entanglement_three_panel_met.png",
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
    print(f"Max ent reduction:      {np.nanmax(reduction_ent):.6e}")
    print(f"After override points:  {np.sum(after_uses_successful_override)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
