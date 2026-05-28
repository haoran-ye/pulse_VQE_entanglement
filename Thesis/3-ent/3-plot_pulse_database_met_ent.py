#!/usr/bin/env python3
"""
plot_pulse_database_met_ent.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

INPUT_FILE = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/PULSES_DATABASE/"
    "dqe_gate_positiveJ_rowprop_pulses.npz"
)

OUTDIR = Path(__file__).resolve().parent
PLOT_PREFIX = INPUT_FILE.stem

INFIDELITY_FLOOR = 1e-14
MET_THRESHOLD = 1e-5

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


def add_shared_colorbar(fig, axes, im, label):
    cbar = fig.colorbar(
        im,
        ax=axes,
        pad=0.02,
        fraction=0.046,
        shrink=1.0,
        aspect=25,
    )

    # This is the version that keeps the left/middle colourbar correct.
    pos0 = axes[0].get_position()

    cbar.ax.set_position([
        cbar.ax.get_position().x0,
        pos0.y0,
        cbar.ax.get_position().width,
        pos0.y1 - pos0.y0,
    ])

    cbar.set_label(label)
    return cbar


def match_colorbar_height_to_axis(cbar, ax):
    """
    Force a single-panel colourbar to have exactly the same vertical extent
    as its corresponding heatmap axis.

    This only modifies the right/reduction colourbar and leaves the existing
    shared left/middle colourbar behaviour unchanged.
    """
    pos = ax.get_position()
    cbar_pos = cbar.ax.get_position()

    cbar.ax.set_position([
        cbar_pos.x0,
        pos.y0,
        cbar_pos.width,
        pos.y1 - pos.y0,
    ])


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
    raise KeyError("Could not infer database kind.")


def get_entanglement_arrays(raw, kind):
    if kind == "state":
        if ENTANGLEMENT_STAT == "avg":
            before_ent_all = get_array(raw, "baseline_s_avg", "baseline_s_max")
            after_ent_all = get_array(raw, "ent_s_avg", "ent_s_max")
            reduction_title = "Reduction in state entanglement"
        elif ENTANGLEMENT_STAT == "max":
            before_ent_all = get_array(raw, "baseline_s_max", "baseline_s_avg")
            after_ent_all = get_array(raw, "ent_s_max", "ent_s_avg")
            reduction_title = "Maximum reduction in state entanglement"
        else:
            raise ValueError("ENTANGLEMENT_STAT must be 'avg' or 'max'.")

    elif kind == "gate":
        if ENTANGLEMENT_STAT == "avg":
            before_ent_all = get_array(raw, "baseline_ent_avg", "baseline_ent_max")
            after_ent_all = get_array(raw, "ent_ent_avg", "ent_ent_max")
            reduction_title = "Reduction in unitary entanglement"
        elif ENTANGLEMENT_STAT == "max":
            before_ent_all = get_array(raw, "baseline_ent_max", "baseline_ent_avg")
            after_ent_all = get_array(raw, "ent_ent_max", "ent_ent_avg")
            reduction_title = "Maximum reduction in unitary entanglement"
        else:
            raise ValueError("ENTANGLEMENT_STAT must be 'avg' or 'max'.")

    else:
        raise ValueError(kind)

    shared_cbar = r"$\overline{S}_{\rm transient}$"

    reduction_cbar = (
        r"$\overline{S}_{\rm transient}"
        r"-\overline{S}^{\rm TEM}_{\rm transient}$"
    )

    return (
        before_ent_all,
        after_ent_all,
        shared_cbar,
        reduction_cbar,
        reduction_title,
    )


def select_before_from_raw(raw, before_ent_all):
    baseline_final_infids = raw["baseline_final_infids"]

    n_theta, n_T, _ = baseline_final_infids.shape

    before_infid = np.full((n_theta, n_T), np.nan)
    before_ent = np.full((n_theta, n_T), np.nan)

    for i in range(n_theta):
        for j in range(n_T):
            finite = np.isfinite(baseline_final_infids[i, j])

            if not np.any(finite):
                continue

            idx = np.where(finite)[0]
            k = idx[np.nanargmin(baseline_final_infids[i, j, finite])]

            before_infid[i, j] = baseline_final_infids[i, j, k]
            before_ent[i, j] = before_ent_all[i, j, k]

    return np.maximum(before_infid, INFIDELITY_FLOOR), before_ent


def local_tem_selection_at_point(ent_infids, ent_values, threshold):
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
    ent_final_infids = raw["ent_final_infids"]

    n_theta, n_T, _ = ent_final_infids.shape

    after_infid = np.full((n_theta, n_T), np.nan)
    after_ent = np.full((n_theta, n_T), np.nan)

    after_uses_successful_override = np.full(
        (n_theta, n_T),
        False,
        dtype=bool,
    )

    for i in range(n_theta):
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


def plot_three_panel(
    X,
    Y,
    theta_pi,
    panels,
    outfile,
    shared_left_middle_cbar=False,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 4.8),
        constrained_layout=False,
        sharey=True,
    )

    images = []
    right_cbar = None

    for i_ax, (ax, panel) in enumerate(zip(axes, panels)):

        im = ax.pcolormesh(
            X,
            Y,
            np.ma.masked_invalid(panel["z"].T),
            shading="auto",
            cmap=panel["cmap"],
            vmin=panel.get("vmin"),
            vmax=panel.get("vmax"),
        )

        images.append(im)

        if not shared_left_middle_cbar:
            cbar = add_colorbar(fig, ax, im, panel["cbar"])
            match_colorbar_height_to_axis(cbar, ax)

        elif i_ax == 2:
            right_cbar = add_colorbar(fig, ax, im, panel["cbar"])

        if panel.get("before_met") is not None:
            plot_met_line(
                ax,
                theta_pi,
                panel["before_met"],
                "MET (baseline)",
                "white",
            )

        if panel.get("after_met") is not None:
            plot_met_line(
                ax,
                theta_pi,
                panel["after_met"],
                "MET (TEM)",
                "cyan",
            )

        ax.set_title(panel["title"])
        ax.set_xlabel(r"Excitation strength $\theta/\pi$")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_box_aspect(1)

        if (
            panel.get("before_met") is not None
            or panel.get("after_met") is not None
        ):
            ax.legend(frameon=True)

    axes[0].set_ylabel(r"Pulse duration $T$ / ns")

    if shared_left_middle_cbar:
        shared_cbar = add_shared_colorbar(
            fig,
            axes[:2],
            images[1],
            panels[0]["cbar"],
        )

        # Do NOT modify shared_cbar; this preserves the previous correct
        # left/middle colourbar behaviour.

        if right_cbar is not None:
            # Match the right/reduction colourbar height to the already-correct
            # shared left/middle colourbar, not by recomputing layout.
            shared_pos = shared_cbar.ax.get_position()
            right_pos = right_cbar.ax.get_position()

            right_cbar.ax.set_position([
                right_pos.x0,
                shared_pos.y0,
                right_pos.width,
                shared_pos.height,
            ])

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
        shared_ent_cbar,
        reduction_ent_cbar,
        reduction_ent_title,
    ) = get_entanglement_arrays(raw, kind)

    before_infid, before_ent = select_before_from_raw(
        raw,
        before_ent_all,
    )

    (
        after_infid,
        after_ent,
        after_uses_successful_override,
    ) = select_after_from_raw(
        raw,
        after_ent_all,
        threshold,
    )

    before_met = extract_met_line(
        before_infid,
        durations_ns,
        threshold,
    )

    after_met = extract_met_line(
        after_infid,
        durations_ns,
        threshold,
    )

    log_before_infid = np.log10(before_infid)
    log_after_infid = np.log10(after_infid)

    delta_log_infid = log_after_infid - log_before_infid

    infid_vmin, infid_vmax = finite_minmax(
        log_before_infid,
        log_after_infid,
    )

    delta_abs = symmetric_absmax(delta_log_infid)

    plot_three_panel(
        X,
        Y,
        theta_pi,
        panels=[
            dict(
                z=log_before_infid,
                title="Baseline",
                cbar=r"$\log_{10}(\mathcal{I}_{\rm before})$",
                cmap="inferno",
                vmin=infid_vmin,
                vmax=infid_vmax,
                before_met=before_met,
            ),
            dict(
                z=log_after_infid,
                title="TEM",
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
        shared_left_middle_cbar=False,
    )

    # ============================================================
    # Normalise entanglement to max value = 1
    # ============================================================

    ent_norm = np.nanmax([
        np.nanmax(before_ent),
        np.nanmax(after_ent),
    ])

    if not np.isfinite(ent_norm) or ent_norm == 0.0:
        ent_norm = 1.0

    before_ent = before_ent / ent_norm
    after_ent = after_ent / ent_norm

    reduction_ent = before_ent - after_ent

    ent_vmin, ent_vmax = finite_minmax(
        before_ent,
        after_ent,
    )

    red_abs = 1.0

    plot_three_panel(
        X,
        Y,
        theta_pi,
        panels=[
            dict(
                z=before_ent,
                title="Baseline",
                cbar=r"$\overline{S}_{\rm transient}$",
                cmap="viridis",
                vmin=ent_vmin,
                vmax=ent_vmax,
                before_met=before_met,
            ),
            dict(
                z=after_ent,
                title="TEM",
                cbar=r"$\overline{S}_{\rm transient}$",
                cmap="viridis",
                vmin=ent_vmin,
                vmax=ent_vmax,
                after_met=after_met,
            ),
            dict(
                z=reduction_ent,
                title=reduction_ent_title,
                cbar=(
                    r"$\overline{S}_{\rm transient}"
                    r"-\overline{S}^{\rm TEM}_{\rm transient}$"
                ),
                cmap="coolwarm",
                vmin=-red_abs,
                vmax=red_abs,
                before_met=before_met,
                after_met=after_met,
            ),
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