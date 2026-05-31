"""
Plotting functions for SQE and DQE state optimization results.

SQE state layout (plot_sqe_optimisation)
-----------------------------------------
Panel 0 : Control pulse J(t) in physical units (MHz)
Panel 1 : State infidelity evolution (log scale)
Panel 2 : Main state entanglement measure (config.state_ent_measure)
Panel 3+: One panel per extra measure in config.extra_ent_measures

DQE state layout (plot_dqe_optimisation)
-----------------------------------------
Panels 0-2 : Control pulses J₁₂, J₂₃, J₃₄ (one per coupling channel)
Panel  3   : State infidelity evolution (log scale)
Panel  4   : Total entanglement cost Σ_P S_P over all 7 bipartitions
Panel  5+  : One panel per entry in config.dqe_plot_partitions

Color convention
----------------
Blue  (#1f77b4) : initial (unoptimized) trajectory
Green (#2ca02c) : infidelity-only optimized (I-optimised)
Red   (#d62728) : entropy-constrained optimized (S-optimised)
Black (dashed)  : target value

Note on target lines
--------------------
Target entropy values are computed from the ideal gate matrix applied to
config.initial_state, *without* the rotating frame correction.  This is
intentional: the rotating frame unitary adds a global phase structure that
does not affect entropy, so the simpler (no-frame) calculation gives the
same target entropy at T and is used for a dashed reference line only.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from .dataclasses import SQEResults, DQEResults
from .sqe_state import single_qubit_excitation
from .dqe_state import dqe_unitary
from .entanglement_measures import (
    STATE_MEASURES, STATE_MEASURE_LABELS, STATE_MEASURE_TITLES,
    entropy_bipartite_single, entropy_bipartite_traj,
    entropy_sum_single, entropy_sum_traj,
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

COLOR_INITIAL        = '#1f77b4'  # blue
COLOR_FIDELITY_ONLY  = '#2ca02c'  # green
COLOR_ENTANGLEMENT   = '#d62728'  # red
COLOR_TARGET         = 'black'


def _rescale_J(x, J_min, J_max):
    # Affine map from optimiser range [-1, 1] to physical J range [J_min, J_max] MHz
    return 0.5 * (J_max - J_min) * x + 0.5 * (J_max + J_min)


def _run_label_color(entanglement_optimisation):
    if not entanglement_optimisation:
        return 'I-optimised', COLOR_FIDELITY_ONLY
    return 'S-optimised', COLOR_ENTANGLEMENT


# =============================================================================
# Individual panel functions
# =============================================================================

def plot_control_pulses(results: SQEResults, ax=None):
    if ax is None:
        ax = plt.gca()
    config = results.config
    t = np.arange(config.n_time_steps + 1) * config.T / config.n_time_steps
    J_min, J_max = float(config.J_min), float(config.J_max)

    x0_plot = np.concatenate([results.x0, [results.x0[-1]]])
    ax.step(t, _rescale_J(x0_plot, J_min, J_max), where='post',
            label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        xb = results.baseline.x_final
        ax.step(t, _rescale_J(np.concatenate([xb, [xb[-1]]]), J_min, J_max),
                where='post', label='I-optimised', color=COLOR_FIDELITY_ONLY, zorder=3)

    xf = results.x_final
    label, color = _run_label_color(config.entanglement_optimisation)
    ax.step(t, _rescale_J(np.concatenate([xf, [xf[-1]]]), J_min, J_max),
            where='post', label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$J$ (MHz)')
    ax.set_title(r'(a) Control pulse $J(t)$')
    ax.legend()
    return ax


def plot_infidelity_evolution(results: SQEResults, ax=None):
    if ax is None:
        ax = plt.gca()
    config = results.config
    n_fine = config.n_time_steps * config.n_discretisation
    t = np.arange(n_fine + 1) * config.T / n_fine

    ax.plot(t, results.val0, label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, results.baseline.val_final, label='I-optimised',
                color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.plot(t, results.val_final, label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Infidelity')
    ax.set_yscale('log')
    ax.set_ylim(top=5.0)
    ax.set_title(r'(b) Infidelity evolution')
    ax.legend()
    return ax


def plot_entanglement_evolution(results: SQEResults, ax=None, *,
                                measure_name: str | None = None):
    """
    Plot entanglement evolution for a given state measure.

    Parameters
    ----------
    measure_name : str, optional
        Key in STATE_MEASURES. Defaults to config.state_ent_measure.
    """
    if ax is None:
        ax = plt.gca()
    config = results.config
    measure_name = measure_name or config.state_ent_measure

    single_fn, traj_fn = STATE_MEASURES[measure_name]
    n_fine = config.n_time_steps * config.n_discretisation
    t = np.arange(n_fine + 1) * config.T / n_fine

    ax.plot(t, traj_fn(results.initial_traj), label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, traj_fn(results.baseline.final_traj), label='I-optimised',
                color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.plot(t, traj_fn(results.final_traj), label=label, color=color, zorder=4)

    # Target entropy line: apply gate to initial state (no rotating frame correction
    # needed here — the frame unitary is global-phase-like and does not change entropy)
    if config.initial_state is not None:
        target_state = single_qubit_excitation(config.theta_target) @ config.initial_state
        target_value = single_fn(target_state)
        ax.hlines(target_value, 0, config.T, colors=COLOR_TARGET, linestyles='dashed',
                  label='Target', zorder=1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(STATE_MEASURE_LABELS.get(measure_name, measure_name))
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"(c) {STATE_MEASURE_TITLES.get(measure_name, f'{measure_name} evolution')}")
    ax.legend()
    return ax


def plot_sqe_optimisation(results: SQEResults, figsize=None):
    """
    Multi-panel figure for SQE state optimization results.

    Panels
    ------
    0 : Control pulse
    1 : Infidelity evolution (log scale)
    2 : Main entanglement measure (config.state_ent_measure)
    3+ : One panel per measure in config.extra_ent_measures
    """
    extra = list(results.config.extra_ent_measures)
    n_panels = 3 + len(extra)
    figsize = figsize or (12, 8 + 3 * len(extra))
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    plot_control_pulses(results, ax=axes[0])
    plot_infidelity_evolution(results, ax=axes[1])
    plot_entanglement_evolution(results, ax=axes[2])

    for i, name in enumerate(extra):
        plot_entanglement_evolution(results, ax=axes[3 + i], measure_name=name)

    plt.tight_layout()
    plt.show()


_CHANNEL_NAMES_DQE = [r'$J_{12}$', r'$J_{23}$', r'$J_{34}$']


def _fine_time(config):
    n_fine = config.n_time_steps * config.n_discretisation
    return np.arange(n_fine + 1) * config.T / n_fine


# =============================================================================
# DQE state plotting
# =============================================================================

def plot_control_channel_dqe(results: DQEResults, channel_idx: int, ax=None):
    """
    Plot one J-channel control pulse for DQE state results.

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

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.step(t, _step(_channel(results.x_final)), where='post',
            label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$J$ (MHz)')
    ax.set_title(f'Control Pulse ({_CHANNEL_NAMES_DQE[channel_idx]})')
    ax.legend()
    return ax


def plot_infidelity_evolution_dqe(results: DQEResults, ax=None):
    """Plot state infidelity evolution for DQE state results (log scale)."""
    if ax is None:
        ax = plt.gca()
    config = results.config
    t = _fine_time(config)

    ax.plot(t, results.val0, label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, results.baseline.val_final, label='I-optimised',
                color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.entanglement_optimisation)
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

    Shows a dashed target line for the DQE target state.
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

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.plot(t, entropy_sum_traj(results.final_traj),
            label=label, color=color, zorder=4)

    if config.initial_state is not None:
        target_state = dqe_unitary(config.theta_target) @ config.initial_state
        target_S_sum = entropy_sum_single(target_state)
        ax.hlines(target_S_sum, 0, config.T, colors=COLOR_TARGET,
                  linestyles='dashed', label='Target', zorder=1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$\Sigma_P\, S_P$ (bits)')
    ax.set_title(f'Total Entanglement Cost (sum over {len(ALL_BIPARTITIONS_4Q)} bipartitions)')
    ax.legend()
    return ax


def plot_partition_entropy_dqe(results: DQEResults, ax=None, *, partition=(0, 1)):
    """
    Plot the bipartite von Neumann entropy S(A|B) for a specific partition.

    Parameters
    ----------
    partition : tuple[int, ...]
        Qubit indices forming subsystem A (big-endian, 0-indexed).
        E.g. (0, 1) → entropy of qubits {0,1} vs {2,3}.

    Notes
    -----
    The complement B = {0,1,2,3} \\ A is shown in the panel title.
    S(A) = S(B) for pure states, so either side gives the same number.
    """
    if ax is None:
        ax = plt.gca()
    config     = results.config
    t          = _fine_time(config)
    partition  = tuple(partition)
    # Complement qubits: the other side of the bipartition
    complement = tuple(q for q in range(4) if q not in partition)

    def _S(traj):
        return entropy_bipartite_traj(traj, subsystem=partition, n_qubits=4)

    ax.plot(t, _S(results.initial_traj),
            label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, _S(results.baseline.final_traj),
                label='I-optimised', color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.plot(t, _S(results.final_traj), label=label, color=color, zorder=4)

    if config.initial_state is not None:
        target_state = dqe_unitary(config.theta_target) @ config.initial_state
        target_S = entropy_bipartite_single(target_state, subsystem=partition, n_qubits=4)
        ax.hlines(target_S, 0, config.T, colors=COLOR_TARGET,
                  linestyles='dashed', label='Target', zorder=1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('S (bits)')
    ax.set_ylim(-0.05, 2.05)
    ax.set_title(f'Entropy S({list(partition)} | {list(complement)})')
    ax.legend()
    return ax


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
    figsize  = figsize or (12, 3 * n_panels)
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
