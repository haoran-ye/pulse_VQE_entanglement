#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_ent/sqe_state_ent/"
)
DATA = OUTDIR / "sqe_state_entanglement_scan_processed.npz"


def main():
    data = np.load(DATA, allow_pickle=True)

    thetas = data["thetas"]           # shape (n_theta,)
    durations_ns = data["durations_ns"]  # shape (n_T,)
    best_delta_s_max = data["best_delta_s_max"]  # shape (n_theta, n_T)
    success_rate = data["success_rate"]           # shape (n_theta, n_T)
    any_success = data["any_success"]             # shape (n_theta, n_T), bool
    best_infid = data["best_infid"]               # shape (n_theta, n_T)

    # Mask delta_s_max where no seed succeeded
    delta_s_masked = np.where(any_success, best_delta_s_max, np.nan)
    masked_delta_s = np.ma.masked_invalid(delta_s_masked)
    masked_success = np.ma.masked_invalid(success_rate.astype(float))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))

    # --- Panel 1: excess entanglement suppression ---
    ax = axes[0]
    im0 = ax.pcolormesh(
        durations_ns,
        thetas / np.pi,
        masked_delta_s,
        shading="auto",
        cmap="viridis",
    )
    cbar0 = fig.colorbar(im0, ax=ax)
    cbar0.set_label(r"$\Delta S_{\max}$ (bits)")
    ax.set_xlabel(r"Pulse duration $T$ (ns)")
    ax.set_ylabel(r"Excitation strength $\theta/\pi$")
    ax.set_title(r"Min excess transient entanglement $\Delta S_{\max}$")
    ax.grid(True, alpha=0.25, linestyle="--")

    # --- Panel 2: success rate ---
    ax = axes[1]
    im1 = ax.pcolormesh(
        durations_ns,
        thetas / np.pi,
        masked_success,
        shading="auto",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
    )
    cbar1 = fig.colorbar(im1, ax=ax)
    cbar1.set_label("Success rate")
    ax.set_xlabel(r"Pulse duration $T$ (ns)")
    ax.set_ylabel(r"Excitation strength $\theta/\pi$")
    ax.set_title("Fraction of seeds below infidelity threshold")
    ax.grid(True, alpha=0.25, linestyle="--")

    fig.suptitle("SQE state entanglement suppression scan", y=1.01)
    fig.tight_layout()

    outfile = OUTDIR / "sqe_state_delta_s_max_heatmap.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {outfile}")

    # Summary statistics
    n_success_pts = int(np.sum(any_success))
    n_total = any_success.size
    print(f"Grid points with at least one successful seed: {n_success_pts}/{n_total}")
    if n_success_pts > 0:
        print(f"  best delta_s_max range: [{np.nanmin(delta_s_masked):.4f}, {np.nanmax(delta_s_masked):.4f}] bits")
        print(f"  best infidelity range:  [{np.nanmin(best_infid):.2e}, {np.nanmax(best_infid):.2e}]")


if __name__ == "__main__":
    main()
