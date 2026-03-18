"""
Plotting functions for DQE (Double Qubit Excitation) optimization results.

Standard layout (plot_dqe_optimisation)
----------------------------------------
Panels 0-2 : Control pulses for J₁₂, J₂₃, J₃₄ — each shows
             Initial / I-optimised / S-optimised curves.
Panel  3   : State infidelity evolution (log scale).
Panel  4   : Total entanglement cost — sum of von Neumann entropies over all
             7 bipartite partitions of 4 qubits.
Panels 5+  : One panel per entry in config.dqe_plot_partitions — the
             bipartite entropy S(A|B) for the specified qubit subset A.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .dataclasses import DQEResults
from .dqe_state import dqe_unitary
from .entanglement_measures import (
    entropy_bipartite_single,
    entropy_bipartite_traj,
    entropy_sum_single,
    entropy_sum_traj,
    ALL_BIPARTITIONS_4Q,
)


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

COLOR_INITIAL       = '#1f77b4'   # blue
COLOR_FIDELITY_ONLY = '#2ca02c'   # green
COLOR_ENTANGLEMENT  = '#d62728'   # red
COLOR_TARGET        = 'black'

_CHANNEL_NAMES = [r'$J_{12}$', r'$J_{23}$', r'$J_{34}$']


# =============================================================================
# Private helpers
# =============================================================================

def _rescale_J(x, J_min, J_max):
    return 0.5 * (J_max - J_min) * x + 0.5 * (J_max + J_min)


def _run_label_color(run_mode):
    if run_mode == "infidelity":
        return 'I-optimised', COLOR_FIDELITY_ONLY
    return 'S-optimised', COLOR_ENTANGLEMENT


def _fine_time(config):
    n_fine = config.n_time_steps * config.n_discretisation
    return np.arange(n_fine + 1) * config.T / n_fine


# =============================================================================
# Individual panel functions
# =============================================================================

def plot_control_channel_dqe(results: DQEResults, channel_idx: int, ax=None):
    """
    Plot one J-channel control pulse (Initial / I-optimised / S-optimised).

    Parameters
    ----------
    channel_idx : int
        0 → J₁₂, 1 → J₂₃, 2 → J₃₄.
    """
    if ax is None:
        ax = plt.gca()
    config = results.config
    n_T   = config.n_time_steps
    t     = np.arange(n_T + 1) * config.T / n_T
    J_min, J_max = float(config.J_min), float(config.J_max)

    def _channel(x):
        return np.asarray(x, float).reshape(n_T, 3)[:, channel_idx]

    def _step(vals):
        return _rescale_J(np.concatenate([vals, [vals[-1]]]), J_min, J_max)

    ax.step(t, _step(_channel(results.x0)), where='post',
            label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.step(t, _step(_channel(results.baseline.x_final)), where='post',
                label='I-optimised', color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.run_mode)
    ax.step(t, _step(_channel(results.x_final)), where='post',
            label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$J$ (MHz)')
    ax.set_title(f'Control Pulse ({_CHANNEL_NAMES[channel_idx]})')
    ax.legend()
    return ax


def plot_infidelity_evolution_dqe(results: DQEResults, ax=None):
    """Plot state infidelity evolution (log scale)."""
    if ax is None:
        ax = plt.gca()
    config = results.config
    t = _fine_time(config)

    ax.plot(t, results.val0, label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, results.baseline.val_final, label='I-optimised',
                color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.run_mode)
    ax.plot(t, results.val_final, label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Infidelity')
    ax.set_yscale('log')
    ax.set_ylim(top=5.0)
    ax.set_title('Infidelity Evolution')
    ax.legend()
    return ax


def plot_entanglement_cost_dqe(results: DQEResults, ax=None):
    """
    Plot the total entanglement cost: sum of S over all 7 bipartitions.

    This is the quantity directly minimized in run_mode="entropy".
    Shows a target-entropy dashed line for the DQE target state.
    """
    if ax is None:
        ax = plt.gca()
    config = results.config
    t = _fine_time(config)

    ax.plot(t, entropy_sum_traj(results.initial_traj),
            label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, entropy_sum_traj(results.baseline.final_traj),
                label='I-optimised', color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.run_mode)
    ax.plot(t, entropy_sum_traj(results.final_traj),
            label=label, color=color, zorder=4)

    # Target line
    if config.initial_state is not None:
        target_state  = dqe_unitary(config.theta_target) @ config.initial_state
        target_S_sum  = entropy_sum_single(target_state)
        ax.hlines(target_S_sum, 0, config.T, colors=COLOR_TARGET,
                  linestyles='dashed', label='Target', zorder=1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$\Sigma_P\, S_P$ (bits)')
    n_parts = len(ALL_BIPARTITIONS_4Q)
    ax.set_title(f'Total Entanglement Cost (sum over {n_parts} bipartitions)')
    ax.legend()
    return ax


def plot_partition_entropy_dqe(results: DQEResults, ax=None, *,
                                partition=(0, 1)):
    """
    Plot the bipartite entropy S(A|B) for a specific partition.

    Parameters
    ----------
    partition : tuple[int, ...]
        Qubit indices forming subsystem A (big-endian, 0-indexed).
        E.g. (0, 1) → entropy of qubits {0,1} vs {2,3}.
    """
    if ax is None:
        ax = plt.gca()
    config    = results.config
    t         = _fine_time(config)
    partition = tuple(partition)
    complement = tuple(q for q in range(4) if q not in partition)

    def _S(traj):
        return entropy_bipartite_traj(traj, subsystem=partition, n_qubits=4)

    ax.plot(t, _S(results.initial_traj),
            label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, _S(results.baseline.final_traj),
                label='I-optimised', color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.run_mode)
    ax.plot(t, _S(results.final_traj), label=label, color=color, zorder=4)

    # Target line
    if config.initial_state is not None:
        target_state = dqe_unitary(config.theta_target) @ config.initial_state
        target_S = entropy_bipartite_single(
            target_state, subsystem=partition, n_qubits=4)
        ax.hlines(target_S, 0, config.T, colors=COLOR_TARGET,
                  linestyles='dashed', label='Target', zorder=1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('S (bits)')
    ax.set_ylim(-0.05, 2.05)
    ax.set_title(f'Entropy S({list(partition)} | {list(complement)})')
    ax.legend()
    return ax


# =============================================================================
# Combined multi-panel figure
# =============================================================================

def plot_dqe_optimisation(results: DQEResults, figsize=None):
    """
    Multi-panel figure for DQE state optimization results.

    Panels
    ------
    0   : J₁₂ control pulse
    1   : J₂₃ control pulse
    2   : J₃₄ control pulse
    3   : Infidelity evolution (log scale)
    4   : Total entanglement cost (Σ S over all 7 bipartitions)
    5+  : One panel per entry in config.dqe_plot_partitions
    """
    extra    = [tuple(p) for p in results.config.dqe_plot_partitions]
    n_panels = 5 + len(extra)
    figsize  = figsize or (12, 4 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    plot_control_channel_dqe(results, 0, axes[0])
    plot_control_channel_dqe(results, 1, axes[1])
    plot_control_channel_dqe(results, 2, axes[2])
    plot_infidelity_evolution_dqe(results, axes[3])
    plot_entanglement_cost_dqe(results, axes[4])

    for i, part in enumerate(extra):
        plot_partition_entropy_dqe(results, axes[5 + i], partition=part)

    plt.tight_layout()
    plt.show()
