from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from .dataclasses import SQEResults
from .sqe_optimisation import single_qubit_excitation, get_Sent_single, get_Sent_traj
from .dqe_optimisation import entropy_bipartite_traj, dqe_unitary, entropy_bipartite_single


plt.rcParams.update({
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'lines.linewidth': 1.5,
    'legend.fontsize': 9,
    'legend.framealpha': 0.9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

COLOR_INITIAL   = '#1f77b4'   # blue (unchanged)
COLOR_FIDELITY_ONLY  = '#2ca02c'   # purple (fidelity-only)
COLOR_ENTANGLEMENT  = '#d62728'   # red (entanglement-aware)
COLOR_TARGET = 'black'

def _rescale_J(x, J_min, J_max):
    return 0.5 * (J_max - J_min) * x + 0.5 * (J_max + J_min)

def plot_control_pulses(results: SQEResults, ax=None):
    if ax is None:
        ax = plt.gca()

    config = results.config
    n_time_steps = config.n_time_steps
    T = config.T
    t = np.arange(n_time_steps + 1) * T / n_time_steps

    J_min = float(config.J_min)
    J_max = float(config.J_max)

    # ---- Initial pulse ----
    x0_plot = np.concatenate([results.x0, [results.x0[-1]]])
    J0_plot = _rescale_J(x0_plot, J_min, J_max)

    ax.step(
        t, J0_plot,
        where='post',
        label='Initial',
        color=COLOR_INITIAL,
        linestyle='-',
        zorder=2,
    )

    # ---- Fidelity-only baseline ----
    if results.baseline is not None:
        xb = results.baseline.x_final
        xb_plot = np.concatenate([xb, [xb[-1]]])
        Jb_plot = _rescale_J(xb_plot, J_min, J_max)

        ax.step(
            t, Jb_plot,
            where='post',
            label='I-optimised',
            color=COLOR_FIDELITY_ONLY,
            linestyle='-',
            zorder=3,
        )

    # ---- Final optimised trajectory ----
    xf_plot = np.concatenate([results.x_final, [results.x_final[-1]]])
    Jf_plot = _rescale_J(xf_plot, J_min, J_max)

    if results.baseline is None:
        label = 'I-optimised'
        color = COLOR_FIDELITY_ONLY
    else:
        label = 'S-optimised'
        color = COLOR_ENTANGLEMENT

    ax.step(
        t, Jf_plot,
        where='post',
        label=label,
        color=color,
        linestyle='-',
        zorder=4,
    )

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$J$ (MHz)')
    ax.set_title('Control Pulse (Exchange Coupling)')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')

    return ax


def plot_infidelity_evolution(results: SQEResults, ax=None):
    if ax is None:
        ax = plt.gca()
    config = results.config
    n_time_steps = config.n_time_steps
    n_disc = config.n_discretisation
    T = config.T
    # Time array for fine discretization
    nT_fine = n_time_steps * n_disc
    t = np.arange(nT_fine + 1) * T / nT_fine

    ax.plot(t, results.val0, label='Initial', color=COLOR_INITIAL, linestyle='-', zorder=2)

    if results.baseline is not None:
        ax.plot(
            t, results.baseline.val_final,
            label='I-optimised',
            color=COLOR_FIDELITY_ONLY,
            linestyle='-',
            zorder=3,
        )

    if results.baseline is None:
        label = 'I-optimised'
        color = COLOR_FIDELITY_ONLY
    else:
        label = 'S-optimised'
        color = COLOR_ENTANGLEMENT

    ax.plot(
        t, results.val_final,
        label=label,
        color=color,
        linestyle='-',
        zorder=4,
    )

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Infidelity')
    ax.set_yscale('log')
    ax.set_ylim(top=5.0)
    ax.set_title('Infidelity Evolution')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    return ax


def plot_entanglement_evolution(results: SQEResults, ax=None):
    if ax is None:
        ax = plt.gca()
    config = results.config
    n_time_steps = config.n_time_steps
    n_disc = config.n_discretisation
    T = config.T
    theta_target = config.theta_target
    initial_state = config.initial_state
    # Time array for fine discretization
    nT_fine = n_time_steps * n_disc
    t = np.arange(nT_fine + 1) * T / nT_fine

    Sent_initial = get_Sent_traj(results.initial_traj)
    ax.plot(t, Sent_initial, label='Initial', color=COLOR_INITIAL, linestyle='-', zorder=2)

    if results.baseline is not None:
        Sent_baseline = get_Sent_traj(results.baseline.final_traj)
        ax.plot(
            t, Sent_baseline,
            label='I-optimised',
            color=COLOR_FIDELITY_ONLY,
            linestyle='-',
            zorder=3,
        )

    Sent_final = get_Sent_traj(results.final_traj)

    if results.baseline is None:
        label = 'I-optimised'
        color = COLOR_FIDELITY_ONLY
    else:
        label = 'S-optimised'
        color = COLOR_ENTANGLEMENT

    ax.plot(
        t, Sent_final,
        label=label,
        color=color,
        linestyle='-',
        zorder=4,
    )

    target_state = single_qubit_excitation(theta_target) @ initial_state
    target_entropy = get_Sent_single(target_state)
    ax.hlines(
        target_entropy, 0, T,
        colors=COLOR_TARGET,
        linestyles='dashed',
        label='Target S',
        zorder=1,
    )

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('S (bits)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Entanglement Entropy Evolution')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    return ax


def plot_sqe_optimisation(results: SQEResults, figsize=(12, 8)):
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    plot_control_pulses(results, ax=axes[0])
    plot_infidelity_evolution(results, ax=axes[1])
    plot_entanglement_evolution(results, ax=axes[2])
    plt.tight_layout()
    plt.show()


def plot_control_pulses_multiJ(results: SQEResults, ax=None):
    if ax is None:
        ax = plt.gca()
    cfg = results.config
    nT = cfg.n_time_steps
    T = cfg.T
    t = np.arange(nT + 1) * T / nT

    nJ = int(getattr(cfg, "n_J_channels", 1))
    x0 = np.asarray(results.x0, float).reshape(nT, nJ)
    xf = np.asarray(results.x_final, float).reshape(nT, nJ)

    # plot initial
    for j in range(nJ):
        y0 = np.concatenate([x0[:, j], [x0[-1, j]]])
        ax.step(t, y0, where="post", color=COLOR_INITIAL, alpha=0.35 if nJ > 1 else 1.0,
                label="Initial Pulse" if j == 0 else None)

    # plot baseline if exists
    if results.baseline is not None:
        xb = np.asarray(results.baseline.x_final, float).reshape(nT, nJ)
        for j in range(nJ):
            yb = np.concatenate([xb[:, j], [xb[-1, j]]])
            ax.step(t, yb, where="post", color=COLOR_FIDELITY_ONLY, linestyle="--",
                    alpha=0.8 if nJ > 1 else 1.0,
                    label="Optimised (fidelity only)" if j == 0 else None)

    # plot entanglement-aware
    for j in range(nJ):
        yf = np.concatenate([xf[:, j], [xf[-1, j]]])
        ax.step(t, yf, where="post", color= COLOR_ENTANGLEMENT,
                alpha=0.8 if nJ > 1 else 1.0,
                label="Optimised (entanglement-aware)" if j == 0 else None)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("J (arb.)")
    ax.set_title("Control Pulse")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    return ax


# ---------- DQE: entanglement evolution uses bipartition entropy ----------
def plot_entanglement_evolution_dqe(results: SQEResults, ax=None):
    if ax is None:
        ax = plt.gca()
    cfg = results.config
    nT = cfg.n_time_steps
    n_disc = cfg.n_discretisation
    T = cfg.T
    # Time array for fine discretization
    nT_fine = nT * n_disc
    t = np.arange(nT_fine + 1) * T / nT_fine

    subsystem = tuple(cfg.entropy_subsystem) if cfg.entropy_subsystem is not None else (0, 1)

    S0 = entropy_bipartite_traj(results.initial_traj, subsystem=subsystem, n_qubits=4)
    Sf = entropy_bipartite_traj(results.final_traj, subsystem=subsystem, n_qubits=4)

    ax.plot(t, S0, label="Initial S", color=COLOR_INITIAL)

    if results.baseline is not None:
        Sb = entropy_bipartite_traj(results.baseline.final_traj, subsystem=subsystem, n_qubits=4)
        ax.plot(t, Sb, label="Final S (fidelity only)", color=COLOR_FIDELITY_ONLY, linestyle="--")

    ax.plot(t, Sf, label="Final S (entanglement-aware)", color=COLOR_ENTANGLEMENT)

    # target entropy
    target_state = dqe_unitary(cfg.theta_target) @ cfg.initial_state
    target_entropy = entropy_bipartite_single(target_state, subsystem=subsystem, n_qubits=4)
    ax.hlines(target_entropy, 0, T, colors=COLOR_TARGET, linestyles="dashed",
              label="Target Entanglement Entropy")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("S (bits)")
    ax.set_ylim(-0.05, 2.05)  # 4 qubits can have up to 2 bits across 2|2 cut
    ax.set_title(f"Entanglement Entropy Evolution (A={subsystem})")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    return ax


def plot_dqe_optimisation(results: SQEResults, figsize=(12, 8)):
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # control pulses (3 channels)
    plot_control_pulses_multiJ(results, ax=axes[0])

    # infidelity evolution (reuse your existing function if you want,
    # or keep a copy with the baseline curve — simplest is to reuse yours if it checks baseline)
    # Here I assume your plot_infidelity_evolution already supports baseline.
    from .plotting import plot_infidelity_evolution  # if you keep it in same file remove this import
    plot_infidelity_evolution(results, ax=axes[1])

    # DQE entropy plot
    plot_entanglement_evolution_dqe(results, ax=axes[2])

    plt.tight_layout()
    plt.show()

# ============================================================================
# MET Plotting Functions
# ============================================================================

def plot_MET_vs_theta(
    scan_results: dict,
    figsize=(10, 6),
    show_errorbars: bool = False,
):
    """
    Plot Minimum Excitation Time (MET) vs theta for multiple infidelity thresholds.

    This function plots results directly from scan_MET_over_theta().

    Parameters
    ----------
    scan_results : dict
        Dictionary returned by scan_MET_over_theta().
        Format: {threshold: {'theta_values': array, 'MET_mean': array, 'MET_std': array}}
    figsize : tuple
        Figure size in inches (width, height).
    show_errorbars : bool
        Whether to show error bars using standard deviation.

    Examples
    --------
    >>> from met import scan_MET_over_theta
    >>> from plotting import plot_MET_vs_theta
    >>>
    >>> results = scan_MET_over_theta(...)
    >>> plot_MET_vs_theta(results, show_errorbars=True)
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    thresholds = sorted(scan_results.keys(), reverse=True)
    cmap = cm.viridis
    norm = Normalize(
        vmin=np.log10(min(thresholds)),
        vmax=np.log10(max(thresholds))
    )

    fig, ax = plt.subplots(figsize=figsize)

    for I_thresh in thresholds:
        color = cmap(norm(np.log10(I_thresh)))
        data = scan_results[I_thresh]

        theta_values = data['theta_values']
        y = data['MET_mean']
        yerr = data.get('MET_std', None)

        # Plot with or without error bars
        if show_errorbars and yerr is not None:
            ax.errorbar(
                theta_values / np.pi, y, yerr=yerr,
                fmt='o-', markersize=5, linewidth=2.5,
                color=color, capsize=3,
                label=rf"$10^{{{int(np.log10(I_thresh))}}}$"
            )
        else:
            ax.plot(
                theta_values / np.pi, y,
                marker='o', markersize=5, linewidth=2.5,
                color=color,
                label=rf"$10^{{{int(np.log10(I_thresh))}}}$"
            )

    ax.set_xlabel(r"$\theta / \pi$", fontsize=13)
    ax.set_ylabel(r"$T_{\mathrm{MET}}$ (ns)", fontsize=13)
    ax.set_title("Minimal Evolution Time vs Entangling Angle", fontsize=14, pad=15)
    ax.legend(title="Infidelity Threshold", loc="upper left", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()


def load_MET_tables_from_dir(results_dir):
    """
    Load MET scan results from directory structure with I_* subdirectories.

    Parameters
    ----------
    results_dir : str or Path
        Root directory containing I_* subdirectories with met_table.npz files.

    Returns
    -------
    theta_values : np.ndarray
        Array of theta values.
    tables : dict
        Dictionary mapping infidelity threshold to data dict with keys:
        'MET_runs', 'MET_mean', 'MET_std'.
    """
    from pathlib import Path

    root = Path(results_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Results directory not found: {root}")

    # Find all threshold directories
    thr_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("I_"))
    if not thr_dirs:
        raise ValueError(f"No threshold directories (I_*) found in {root}")

    theta_values_ref = None
    n_theta_ref = None
    tables = {}
    skipped = []

    for d in thr_dirs:
        # Parse threshold from directory name (e.g., "I_1e-06" -> 1e-06)
        threshold_str = d.name[2:]  # Remove "I_"
        I = float(threshold_str)

        # Load MET table
        met_file = d / "met_table.npz"
        if not met_file.exists():
            print(f"Warning: Missing {met_file}, skipping...")
            continue

        data = np.load(met_file)
        theta_values = np.asarray(data["theta_values"], dtype=float)
        MET_runs = np.asarray(data["MET_runs"], dtype=float)

        # Check if theta values match the reference
        if theta_values_ref is None:
            theta_values_ref = theta_values
            n_theta_ref = len(theta_values)
        else:
            # Check if shapes match first
            if len(theta_values) != n_theta_ref:
                skipped.append((d.name, len(theta_values), n_theta_ref))
                print(f"Warning: Skipping {d.name} - has {len(theta_values)} theta values, expected {n_theta_ref}")
                continue

            # Check if values match
            if not np.allclose(theta_values_ref, theta_values):
                skipped.append((d.name, "values mismatch", n_theta_ref))
                print(f"Warning: Skipping {d.name} - theta values don't match reference")
                continue

        tables[float(I)] = {
            "MET_runs": MET_runs,
            "MET_mean": np.nanmean(MET_runs, axis=1),
            "MET_std": np.nanstd(MET_runs, axis=1),
        }

    if not tables:
        raise ValueError(f"No valid MET tables found in {root}. All directories were skipped.")

    if skipped:
        print(f"\nNote: Skipped {len(skipped)} directories due to theta value mismatches.")
        print("This usually means you have results from different scan runs.")
        print(f"Loaded {len(tables)} thresholds successfully.\n")

    return theta_values_ref, tables


def plot_MET_vs_theta_from_dir(
    results_dir,
    use: str = "mean",
    show_errorbars: bool = False,
    errorbar_kind: str = "std",
    figsize=(6, 4),
):
    """
    Load and plot MET vs theta from directory with I_* subdirectories.

    Parameters
    ----------
    results_dir : str or Path
        Root directory containing MET scan results.
    use : str
        Aggregation method: "mean" or "median".
    show_errorbars : bool
        Whether to show error bars (only for mean).
    errorbar_kind : str
        Type of error bars: "std" (standard deviation) or "sem" (standard error of mean).
    figsize : tuple
        Figure size in inches (width, height).

    Returns
    -------
    theta_values : np.ndarray
        Array of theta values.
    MET_results : dict
        Dictionary mapping infidelity threshold to plotted MET values.
    """
    from matplotlib import cm
    from matplotlib.colors import Normalize

    theta_values, tables = load_MET_tables_from_dir(results_dir)
    thresholds = sorted(tables.keys(), reverse=True)
    cmap = cm.viridis
    norm = Normalize(
        vmin=np.log10(min(thresholds)),
        vmax=np.log10(max(thresholds))
    )

    fig, ax = plt.subplots(figsize=figsize)
    MET_results = {}

    for I in thresholds:
        MET_runs = tables[I]["MET_runs"]
        color = cmap(norm(np.log10(I)))

        # Compute aggregated values
        if use == "mean":
            y = tables[I]["MET_mean"]
        else:  # median
            y = np.nanmedian(MET_runs, axis=1)

        MET_results[I] = y

        # Plot with or without error bars
        if show_errorbars and use == "mean":
            std = tables[I]["MET_std"]
            if errorbar_kind == "sem":
                n_eff = np.sum(np.isfinite(MET_runs), axis=1)
                yerr = std / np.maximum(np.sqrt(n_eff), 1.0)
            else:
                yerr = std

            ax.errorbar(
                theta_values / np.pi, y, yerr=yerr,
                fmt='o-', markersize=5, linewidth=2.5,
                color=color, capsize=3,
                label=rf"$10^{{{int(np.log10(I))}}}$"
            )
        else:
            ax.plot(
                theta_values / np.pi, y,
                marker='o', markersize=5, linewidth=2.5,
                color=color,
                label=rf"$10^{{{int(np.log10(I))}}}$"
            )

    ax.set_xlabel(r"$\theta / \pi$", fontsize=13)
    ax.set_ylabel(r"$T_{\mathrm{MET}}$ (ns)", fontsize=13)
    ax.set_ylim(0.0, 20)
    ax.set_title("Minimal Evolution Time vs Entangling Angle", fontsize=14, pad=15)
    ax.legend(title="Infidelity Threshold", loc="upper left", frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()

    return theta_values, MET_results


def plot_MET_colormap_from_dir(
    results_dir,
    *,
    use: str = "mean",                 # "mean" uses MET_mean; "median" uses nanmedian over runs
    T_max: float | None = None,        # if None, auto from data
    n_T: int = 250,                    # vertical resolution
    levels: int = 40,                  # contour levels
    met_line_I: float | None = None,   # overlay MET line for this infidelity threshold (e.g. 1e-4)
    cmap: str = "inferno",
    figsize=(6, 4),
    vmin: float | None = None,         # in log10-space
    vmax: float | None = None,         # in log10-space
):
    """
    Make a coloured contour plot like your example:
    x = theta/pi, y = T (ns), colour = log10(best achievable infidelity).

    Requires results laid out as I_* subfolders containing met_table.npz,
    loadable via load_MET_tables_from_dir(results_dir).
    """
    theta_values, tables = load_MET_tables_from_dir(results_dir)

    # Thresholds available (infidelity values). Sort from smallest->largest for "best achievable".
    thresholds = np.array(sorted(tables.keys()), dtype=float)  # ascending: best first

    # Build MET(θ; I) array: shape (n_I, n_theta)
    n_theta = len(theta_values)
    n_I = len(thresholds)

    MET_I_theta = np.full((n_I, n_theta), np.nan, dtype=float)

    for j, I in enumerate(thresholds):
        if use == "mean":
            MET_I_theta[j, :] = np.asarray(tables[I]["MET_mean"], dtype=float)
        elif use == "median":
            MET_runs = np.asarray(tables[I]["MET_runs"], dtype=float)
            MET_I_theta[j, :] = np.nanmedian(MET_runs, axis=1)
        else:
            raise ValueError("use must be 'mean' or 'median'")

    # Choose T grid
    finite_MET = MET_I_theta[np.isfinite(MET_I_theta)]
    if T_max is None:
        if finite_MET.size == 0:
            raise ValueError("No finite MET values found in the loaded tables.")
        T_max = float(np.nanmax(finite_MET)) * 1.05  # small headroom

    T_grid = np.linspace(0.0, T_max, int(n_T), dtype=float)  # y-axis
    # Z will store best achievable infidelity at each (θ, T)
    Z = np.full((len(T_grid), n_theta), np.nan, dtype=float)

    # Invert MET curves:
    # For each theta index i and time index t, pick smallest I with MET(I, theta) <= T.
    for i in range(n_theta):
        met_vs_I = MET_I_theta[:, i]  # shape (n_I,)
        for k, T in enumerate(T_grid):
            ok = np.isfinite(met_vs_I) & (met_vs_I <= T)
            if np.any(ok):
                Z[k, i] = np.min(thresholds[ok])
            else:
                # If nothing achieved by time T, set to 1.0 (log10=0) or leave NaN.
                # Using NaN makes it show as blank; using 1.0 makes it dark/light depending on cmap.
                Z[k, i] = np.nan

    # Plot in log10 space
    logZ = np.log10(Z)

    # Sensible defaults for color scaling
    if vmin is None:
        vmin = float(np.nanmin(logZ)) if np.isfinite(logZ).any() else -12.0
    if vmax is None:
        vmax = float(np.nanmax(logZ)) if np.isfinite(logZ).any() else 0.0

    fig, ax = plt.subplots(figsize=figsize)

    # Filled contours
    cf = ax.contourf(
        theta_values / np.pi,
        T_grid,
        logZ,
        levels=levels,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        antialiased=True,
    )

    # Optional: add thin contour lines to mimic the “layered” look
    ax.contour(
        theta_values / np.pi,
        T_grid,
        logZ,
        levels=levels,
        linewidths=0.35,
        colors="w",
        alpha=0.25,
    )

    # Overlay a chosen MET line (for one threshold)
    if met_line_I is not None:
        met_line_I = float(met_line_I)
        if met_line_I not in tables:
            # allow approximate match (e.g. 1e-4 vs 0.0001 saved as float)
            idx = int(np.argmin(np.abs(thresholds - met_line_I)))
            I_match = float(thresholds[idx])
        else:
            I_match = met_line_I

        if use == "mean":
            y_line = np.asarray(tables[I_match]["MET_mean"], dtype=float)
        else:
            y_line = np.nanmedian(np.asarray(tables[I_match]["MET_runs"], dtype=float), axis=1)

        ax.plot(theta_values / np.pi, y_line, color="k", linewidth=2.0)
        ax.text(
            0.55, 0.25, "MET line",
            transform=ax.transAxes,
            color="k",
            fontsize=12,
            alpha=0.85,
        )

    ax.set_xlabel(r"excitation strength $\theta$  (in units of $\pi$)", fontsize=12)
    ax.set_ylabel(r"pulse duration $T$ in ns", fontsize=12)
    ax.set_ylim(0.0, 20)
    ax.set_title("Single-qubit excitation", fontsize=14, pad=10)

    ax.grid(alpha=0.25, linestyle="--")

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r"$\log_{10}(\mathrm{Infidelity})$", fontsize=12)

    plt.tight_layout()
    plt.show()

    return theta_values, T_grid, 