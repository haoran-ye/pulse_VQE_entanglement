from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from .device import get_sqe_target_state, single_qubit_entanglement_entropy, MHZ_TO_RAD_PER_S

# Plot styling defaults
plt.rcParams.update({
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'lines.linewidth': 1.8,
})


def _get_snapshot_style(label: str) -> Tuple[float, float, str]:
    """Return (alpha, linewidth, color) for snapshot label."""
    if label == "init":
        return 0.4, 1.2, '#1f77b4'
    elif label == "final":
        return 1.0, 2.5, '#d62728'
    else:
        return 0.25, 1.0, '#7f7f7f'


def plot_optimisation_results(results):
    config = results.config
    t_segments = np.linspace(0, config.T_ns, config.N_SEGMENTS + 1)

    # Create figure with 4 rows: 3 full-width trajectories + 1 centered convergence plot
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.2], hspace=0.4, wspace=0.3)

    # Row 1: Pulse evolution (full width)
    ax1 = fig.add_subplot(gs[0, :])
    for label, x_mhz in results.pulse_snapshots:
        alpha, lw, color = _get_snapshot_style(label)
        pulse = x_mhz * MHZ_TO_RAD_PER_S / 1e6
        y = np.r_[pulse, pulse[-1]]
        ax1.step(t_segments, y, where="post", alpha=alpha, linewidth=lw,
                color=color, label=label)

    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel(r"$J(t)$ ($10^6$ rad/s)")
    ax1.set_title(rf"Pulse Evolution ($\theta={config.theta:.3f}$ rad)")
    ax1.legend(loc='best')

    # Row 2: Infidelity trajectory (full width)
    ax2 = fig.add_subplot(gs[1, :])
    for label, t_arr, c_arr in results.snapshot_data:
        alpha, lw, color = _get_snapshot_style(label)
        ax2.plot(t_arr, c_arr, alpha=alpha, linewidth=lw, color=color, label=label)

    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Infidelity")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Infidelity Trajectory")
    ax2.legend(loc='best')

    # Row 3: Entanglement evolution (full width)
    ax3 = fig.add_subplot(gs[2, :])
    if results.entropy_snapshots is not None:
        psi_target = get_sqe_target_state(config.theta)
        S_target = single_qubit_entanglement_entropy(psi_target)

        for label, t_arr, S_t in results.entropy_snapshots:
            alpha, lw, color = _get_snapshot_style(label)
            ax3.plot(t_arr, S_t, alpha=alpha, linewidth=lw, color=color, label=label)

        ax3.axhline(S_target, ls='--', color='black', lw=2, alpha=0.7,
                   label=f'Target S={S_target:.3f}')
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlabel("Time (ns)")
        ax3.set_ylabel("Entanglement Entropy")
        ax3.set_title(rf"Entanglement Evolution ($\theta={config.theta:.3f}$ rad)")
        ax3.legend(loc='best', ncol=2)
    else:
        ax3.text(0.5, 0.5, 'No entanglement data\n(set entanglement_on=True)',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_xlabel("Time (ns)")
        ax3.set_ylabel("Entanglement Entropy")
        ax3.set_title("Entanglement Evolution")

    # Row 4: Optimisation convergence (centered, normal aspect ratio)
    ax4 = fig.add_subplot(gs[3, 1])
    iter_data = []

    for label, _, c_arr in results.snapshot_data:
        if label == "init":
            iter_data.append((0, float(c_arr[-1])))
        elif label.startswith("iter"):
            it_num = int(label.split()[1])
            iter_data.append((it_num, float(c_arr[-1])))
        elif label == "final":
            iter_data.append((results.total_iterations, float(c_arr[-1])))

    iter_nums, iter_costs = zip(*sorted(iter_data)) if iter_data else ([], [])

    ax4.plot(iter_nums, iter_costs, marker='o', markersize=6, color='#2ca02c',
            markerfacecolor='white', markeredgewidth=2, linewidth=2)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Final Infidelity")
    ax4.set_title("Optimisation Convergence")
    ax4.set_yscale('log')
    if config.infidelity_threshold:
        ax4.axhline(config.infidelity_threshold, ls='--', color='red',
                   alpha=0.7, label=f'Threshold={config.infidelity_threshold:.1e}')
        ax4.legend()

    plt.show()


def plot_entanglement(results):
    if results.entropy_snapshots is None:
        raise ValueError("No entanglement data. Set config.entanglement_on=True before optimise().")

    psi_target = get_sqe_target_state(results.config.theta)
    S_target = single_qubit_entanglement_entropy(psi_target)

    fig, ax = plt.subplots(figsize=(12, 3.5))

    for label, t_arr, S_t in results.entropy_snapshots:
        alpha, lw, color = _get_snapshot_style(label)
        ax.plot(t_arr, S_t, alpha=alpha, linewidth=lw, color=color, label=label)

    ax.axhline(S_target, ls='--', color='black', lw=2, alpha=0.7, label=f'Target S={S_target:.3f}')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Entanglement Entropy")
    ax.set_title(rf"Entanglement Evolution ($\theta={results.config.theta:.3f}$ rad)")
    ax.legend(loc='best', ncol=2)

    plt.tight_layout()
    plt.show()


def plot_MET_vs_theta(theta_values, MET_results, INFIDELITY_THRESHOLDS):
    thresholds = sorted(INFIDELITY_THRESHOLDS, reverse=True)
    cmap = cm.viridis
    norm = Normalize(vmin=np.log10(min(thresholds)), vmax=np.log10(max(thresholds)))

    fig, ax = plt.subplots(figsize=(9, 6))

    for I_thresh in thresholds:
        color = cmap(norm(np.log10(I_thresh)))
        ax.plot(theta_values / np.pi, MET_results[I_thresh],
               marker='o', markersize=5, linewidth=2.5, color=color,
               label=rf"$10^{{{int(np.log10(I_thresh))}}}$")

    ax.set_xlabel(r"$\theta / \pi$", fontsize=13)
    ax.set_ylabel(r"$T_{\mathrm{MET}}$ (ns)", fontsize=13)
    ax.set_title("Minimal Evolution Time vs Entangling Angle", fontsize=14, pad=15)
    ax.legend(title="Infidelity Threshold", loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()


def _list_threshold_dirs(results_dir: Union[str, Path]) -> List[Path]:
    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")
    thr_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("I_"))
    if not thr_dirs:
        raise ValueError(f"No threshold directories (I_*) found in {root}")
    return thr_dirs


def _parse_threshold_from_dirname(name: str) -> float:
    if not name.startswith("I_"):
        raise ValueError(f"Invalid threshold directory name: {name}")
    return float(name[2:])


def load_MET_tables_from_dir(results_dir: Union[str, Path]) -> Tuple[np.ndarray, Dict[float, Dict[str, np.ndarray]]]:
    """Load MET scan results from directory structure with I_* subdirectories."""
    thr_dirs = _list_threshold_dirs(results_dir)
    theta_values_ref = None
    tables = {}

    for d in thr_dirs:
        I = _parse_threshold_from_dirname(d.name)
        met_file = d / "met_table.npz"
        if not met_file.exists():
            raise FileNotFoundError(f"Missing {met_file}")

        data = np.load(met_file)
        theta_values = np.asarray(data["theta_values"], dtype=float)
        MET_runs = np.asarray(data["MET_runs"], dtype=float)

        if theta_values_ref is None:
            theta_values_ref = theta_values
        elif not np.allclose(theta_values_ref, theta_values):
            raise ValueError(f"Theta values mismatch in {d}")

        tables[float(I)] = {
            "MET_runs": MET_runs,
            "MET_mean": np.nanmean(MET_runs, axis=1),
            "MET_std": np.nanstd(MET_runs, axis=1),
        }

    return theta_values_ref, tables


def plot_MET_vs_theta_from_dir(
    results_dir: Union[str, Path],
    *,
    use: str = "mean",
    show_errorbars: bool = False,
    errorbar_kind: str = "std",
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """Load and plot MET vs theta from directory with I_* subdirectories.

    Args:
        use: "mean" or "median" aggregation across runs
        show_errorbars: Whether to show error bars (only for mean)
        errorbar_kind: "std" or "sem" (standard error of mean)

    Returns:
        (theta_values, MET_results) where MET_results[I_thresh] = plotted array
    """
    theta_values, tables = load_MET_tables_from_dir(results_dir)
    thresholds = sorted(tables.keys(), reverse=True)
    cmap = cm.viridis
    norm = Normalize(vmin=np.log10(min(thresholds)), vmax=np.log10(max(thresholds)))

    fig, ax = plt.subplots(figsize=(9, 6))
    MET_results = {}

    for I in thresholds:
        MET_runs = tables[I]["MET_runs"]
        color = cmap(norm(np.log10(I)))

        y = tables[I]["MET_mean"] if use == "mean" else np.nanmedian(MET_runs, axis=1)
        MET_results[I] = y

        if show_errorbars and use == "mean":
            std = tables[I]["MET_std"]
            if errorbar_kind == "sem":
                n_eff = np.sum(np.isfinite(MET_runs), axis=1)
                yerr = std / np.maximum(np.sqrt(n_eff), 1.0)
            else:
                yerr = std

            ax.errorbar(theta_values / np.pi, y, yerr=yerr, fmt='o-',
                       markersize=4, linewidth=2.5, color=color, capsize=3,
                       label=rf"$10^{{{int(np.log10(I))}}}$")
        else:
            ax.plot(theta_values / np.pi, y, marker='o', markersize=5,
                   linewidth=2.5, color=color,
                   label=rf"$10^{{{int(np.log10(I))}}}$")

    ax.set_xlabel(r"$\theta / \pi$", fontsize=13)
    ax.set_ylabel(r"$T_{\mathrm{MET}}$ (ns)", fontsize=13)
    ax.set_title("Minimal Evolution Time vs Entangling Angle", fontsize=14, pad=15)
    ax.legend(title="Infidelity Threshold", loc="upper left", frameon=True)

    plt.tight_layout()
    plt.show()

    return theta_values, MET_results
