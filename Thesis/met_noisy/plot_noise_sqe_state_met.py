#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


OUTDIR = Path(
    "/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/Thesis/results/met_noisy/sqe_state_noisy_met/"
)

DATA = OUTDIR / "sqe_state_noisy_met_scan.npz"

MET_THRESHOLD = 1e-5
INFID_FLOOR = 1e-16


def savefig(name):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def add_met_contour(ax, theta_over_pi, durations_ns, logI_T_theta):
    level = np.log10(MET_THRESHOLD)

    if not np.any(np.isfinite(logI_T_theta)):
        return None

    zmin = np.nanmin(logI_T_theta)
    zmax = np.nanmax(logI_T_theta)

    if not (zmin <= level <= zmax):
        return None

    cs = ax.contour(
        theta_over_pi,
        durations_ns,
        logI_T_theta,
        levels=[level],
        colors="white",
        linestyles="--",
        linewidths=1.8,
    )

    return cs


def main():
    data = np.load(DATA)

    thetas = data["thetas"]
    durations_ns = data["durations_ns"]
    theta_over_pi = thetas / np.pi

    base = np.maximum(data["base_noisy_infids"], INFID_FLOOR)
    ent = np.maximum(data["ent_noisy_infids"], INFID_FLOOR)

    # Stored shape:
    #   base[theta_index, T_index]
    #
    # Desired plotting axes:
    #   x-axis = theta/pi
    #   y-axis = T/ns
    #
    # Therefore transpose for plotting:
    #   Z[T_index, theta_index]
    log_base = np.log10(base).T
    log_ent = np.log10(ent).T

    assert log_base.shape == (len(durations_ns), len(theta_over_pi))
    assert log_ent.shape == (len(durations_ns), len(theta_over_pi))

    vmin = np.nanmin([np.nanmin(log_base), np.nanmin(log_ent)])
    vmax = np.nanmax([np.nanmax(log_base), np.nanmax(log_ent)])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    plots = [
        (axes[0], log_base, "Baseline pulse"),
        (axes[1], log_ent, "Entanglement-suppressed pulse"),
    ]

    contour_handle = None

    for ax, Z, title in plots:
        im = ax.pcolormesh(
            theta_over_pi,
            durations_ns,
            Z,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )

        cs = add_met_contour(ax, theta_over_pi, durations_ns, Z)

        if cs is not None and contour_handle is None:
            contour_handle = cs.legend_elements()[0][0]

        ax.set_title(title)
        ax.set_xlabel(r"$\theta/\pi$")

    axes[0].set_ylabel("Duration / ns")

    cbar = fig.colorbar(im, ax=axes, pad=0.02, fraction=0.046)
    cbar.set_label(r"$\log_{10}(\mathrm{noisy\ final\ infidelity})$")

    if contour_handle is not None:
        axes[1].legend(
            [contour_handle],
            [r"$I=10^{-5}$"],
            loc="upper right",
            frameon=True,
        )

    fig.suptitle(
        rf"SQE state noisy MET, $T_1={float(data['T1_us']):.2e}\,\mu s$, "
        rf"$T_2={float(data['T2_us']):.2e}\,\mu s$",
    )

    fig.savefig(
        OUTDIR / "sqe_state_noisy_met_side_by_side.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Saved {OUTDIR / 'sqe_state_noisy_met_side_by_side.png'}")

    ratio = base / ent
    log_ratio = np.log10(ratio).T

    assert log_ratio.shape == (len(durations_ns), len(theta_over_pi))

    plt.figure(figsize=(6.8, 5.2))
    im = plt.pcolormesh(
        theta_over_pi,
        durations_ns,
        log_ratio,
        shading="auto",
        cmap="inferno",
    )
    plt.xlabel(r"$\theta/\pi$")
    plt.ylabel("Duration / ns")
    plt.title(r"SQE state: $\log_{10}(I_{\rm base}/I_{\rm ent})$")
    cbar = plt.colorbar(im)
    cbar.set_label(r"$\log_{10}(I_{\rm base}/I_{\rm ent})$")
    savefig("sqe_state_noisy_met_improvement_ratio.png")

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"T1 = {float(data['T1_us']):.6e} us")
    print(f"T2 = {float(data['T2_us']):.6e} us")
    print(f"Best baseline noisy infidelity = {np.nanmin(base):.6e}")
    print(f"Best ent noisy infidelity      = {np.nanmin(ent):.6e}")
    print(f"Max improvement ratio          = {np.nanmax(ratio):.6e}")
    print("=" * 72)


if __name__ == "__main__":
    main()