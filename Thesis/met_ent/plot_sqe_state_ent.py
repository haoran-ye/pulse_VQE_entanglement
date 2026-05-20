#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# Choose: "raw" or "processed"
PLOT_MODE = "processed"

OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_ent/sqe_state_ent/"
)

RAW_DATA = OUTDIR / "sqe_state_entanglement_scan_raw.npz"
PROCESSED_DATA = OUTDIR / "sqe_state_entanglement_scan_processed.npz"


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


def select_best_raw(data):
    successes = data["successes"]
    baseline_final_infids = data["baseline_final_infids"]
    ent_final_infids = data["ent_final_infids"]
    baseline_s_max = data["baseline_s_max"]
    ent_s_max = data["ent_s_max"]

    n_theta, n_T, _ = successes.shape

    best_baseline_infid = np.full((n_theta, n_T), np.nan)
    best_ent_infid = np.full((n_theta, n_T), np.nan)
    best_baseline_s = np.full((n_theta, n_T), np.nan)
    best_ent_s = np.full((n_theta, n_T), np.nan)

    for i in range(n_theta):
        for j in range(n_T):
            valid = successes[i, j] & np.isfinite(ent_s_max[i, j])

            if not np.any(valid):
                continue

            valid_indices = np.where(valid)[0]
            k_best = valid_indices[np.nanargmin(ent_s_max[i, j, valid])]

            best_baseline_infid[i, j] = baseline_final_infids[i, j, k_best]
            best_ent_infid[i, j] = ent_final_infids[i, j, k_best]
            best_baseline_s[i, j] = baseline_s_max[i, j, k_best]
            best_ent_s[i, j] = ent_s_max[i, j, k_best]

    return best_baseline_infid, best_ent_infid, best_baseline_s, best_ent_s


def load_data():
    if PLOT_MODE == "raw":
        data = np.load(RAW_DATA, allow_pickle=True)
        thetas = data["thetas"]
        durations_ns = data["durations_ns"]
        baseline_infid, ent_infid, baseline_s, ent_s = select_best_raw(data)

    elif PLOT_MODE == "processed":
        data = np.load(PROCESSED_DATA, allow_pickle=True)
        thetas = data["thetas"]
        durations_ns = data["durations_ns"]
        baseline_infid = data["corrected_best_baseline_infid"]
        ent_infid = data["corrected_best_ent_infid"]
        baseline_s = data["corrected_best_baseline_s_max"]
        ent_s = data["corrected_best_ent_s_max"]

    else:
        raise ValueError("PLOT_MODE must be 'raw' or 'processed'.")

    return thetas, durations_ns, baseline_infid, ent_infid, baseline_s, ent_s


def main():

    thetas, durations_ns, baseline_infid, ent_infid, baseline_s, ent_s = load_data()

    baseline_infid = np.where(baseline_infid > 0, baseline_infid, np.nan)
    ent_infid = np.where(ent_infid > 0, ent_infid, np.nan)

    log10_baseline_infid = np.log10(baseline_infid)

    reduction = baseline_s - ent_s

    ent_vmin = np.nanmin([np.nanmin(baseline_s), np.nanmin(ent_s)])
    ent_vmax = np.nanmax([np.nanmax(baseline_s), np.nanmax(ent_s)])

    red_vmin = 0.0
    red_vmax = np.nanmax(reduction)

    X, Y = np.meshgrid(thetas / np.pi, durations_ns)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(22, 5.8),
        constrained_layout=True,
    )

    panels = [
        (
            log10_baseline_infid.T,
            "Baseline infidelity",
            r"$\log_{10}(\mathcal{I}_{\rm baseline})$",
            "inferno",
            None,
            None,
        ),
        (
            baseline_s.T,
            "Baseline entanglement",
            r"$S_{\max}$",
            "viridis",
            ent_vmin,
            ent_vmax,
        ),
        (
            ent_s.T,
            "After entanglement optimisation",
            r"$S_{\max}$",
            "viridis",
            ent_vmin,
            ent_vmax,
        ),
        (
            reduction.T,
            "Entanglement reduction",
            r"$S_{\max}^{\rm baseline}-S_{\max}^{\rm opt}$",
            "magma",
            red_vmin,
            red_vmax,
        ),
    ]

    for ax, (z, title, cbar_label, cmap, vmin, vmax) in zip(axes, panels):

        im = ax.pcolormesh(
            X,
            Y,
            np.ma.masked_invalid(z),
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

        add_colorbar(fig, ax, im, cbar_label)

        ax.set_title(f"{title} ({PLOT_MODE})")
        ax.set_xlabel(r"Excitation angle $\theta/\pi$")
        ax.set_ylabel(r"Pulse duration $T$ (ns)")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_box_aspect(1)

    outfile = OUTDIR / f"sqe_state_entanglement_four_panel_{PLOT_MODE}.png"

    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to\n{outfile}")


if __name__ == "__main__":
    main()