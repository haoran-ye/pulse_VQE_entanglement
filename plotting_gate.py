"""
Plotting functions for gate-based optimization results.

SQE gate layout (plot_sqe_gate_optimisation)
--------------------------------------------
Panel 0 : Control pulse J(t) in physical units (MHz)
Panel 1 : Gate infidelity evolution (log scale)
Panel 2 : Main gate entanglement measure (config.gate_ent_measure)
Panel 3+: One panel per measure in config.extra_ent_measures

DQE gate layout (plot_dqe_gate_optimisation)
--------------------------------------------
Panels 0-2 : Control pulses J₁₂, J₂₃, J₃₄ (one per subplot)
Panel  3   : Gate infidelity evolution (log scale)
Panel  4   : Main gate entanglement measure (config.gate_ent_measure)
Panel  5+  : One panel per measure in config.extra_ent_measures

Color convention
----------------
Blue  (#1f77b4) : initial (unoptimized) trajectory
Green (#2ca02c) : infidelity-only optimized (I-optimised)
Red   (#d62728) : entropy-constrained optimized (EP-optimised)
Black (dashed)  : target value

Target lines
------------
Target lines are drawn only for deterministic gate measures
(GATE_MEASURE_DETERMINISTIC = True).  Monte Carlo-based measures such as
"average_entangling_power" and "max_output_entanglement" are skipped at
plot time because re-evaluation is too slow.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from .dataclasses import SQEGateResults, DQEGateResults
from .sqe_state import single_qubit_excitation
from .dqe_state import dqe_unitary
from .entanglement_measures import (
    GATE_MEASURES,
    GATE_MEASURE_LABELS,
    GATE_MEASURE_TITLES,
    GATE_MEASURE_DETERMINISTIC,
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
    return 'EP-optimised', COLOR_ENTANGLEMENT


def _fine_time(config):
    n_fine = config.n_time_steps * config.n_discretisation
    return np.arange(n_fine + 1) * config.T / n_fine


# =============================================================================
# Individual panel functions
# =============================================================================

def plot_control_pulses_gate(results: SQEGateResults, ax=None):
    """Plot exchange coupling control pulses."""
    if ax is None:
        ax = plt.gca()
    config = results.config
    t = np.arange(config.n_time_steps + 1) * config.T / config.n_time_steps
    J_min, J_max = float(config.J_min), float(config.J_max)

    x0 = results.x0
    ax.step(t, _rescale_J(np.concatenate([x0, [x0[-1]]]), J_min, J_max),
            where='post', label='Initial', color=COLOR_INITIAL, zorder=2)

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
    ax.set_title('Control Pulse')
    ax.legend()
    return ax


def plot_infidelity_evolution_gate(results: SQEGateResults, ax=None):
    """Plot gate infidelity evolution (log scale)."""
    if ax is None:
        ax = plt.gca()
    config = results.config
    t = _fine_time(config)

    ax.plot(t, results.infid_initial, label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, results.baseline.infid_final, label='I-optimised',
                color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.plot(t, results.infid_final, label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Gate Infidelity')
    ax.set_yscale('log')
    ax.set_ylim(top=5.0)
    ax.set_title('Gate Infidelity Evolution')
    ax.legend()
    return ax


def plot_entanglement_evolution_gate(results: SQEGateResults, ax=None, *,
                                     measure_name: str | None = None):
    """
    Plot gate entanglement measure evolution.

    Parameters
    ----------
    measure_name : str, optional
        Key in GATE_MEASURES. Defaults to config.gate_ent_measure (the main
        optimized measure). Pass a name from config.extra_ent_measures to plot
        a pre-computed extra measure.
    """
    if ax is None:
        ax = plt.gca()
    config = results.config
    measure_name = measure_name or config.gate_ent_measure
    t = _fine_time(config)
    is_main = (measure_name == config.gate_ent_measure)

    # The main measure is stored directly on the result; extra measures in a dict
    def _get_traj(res, final: bool):
        if is_main:
            return res.ent_final if final else res.ent_initial
        store = res.extra_ent_final if final else res.extra_ent_initial
        return store[measure_name]

    ax.plot(t, _get_traj(results, False), label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, _get_traj(results.baseline, True), label='I-optimised',
                color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.plot(t, _get_traj(results, True), label=label, color=color, zorder=4)

    # Deterministic measures can be evaluated quickly at plot time for a target line.
    # MC-based measures (average_entangling_power etc.) are skipped — too slow.
    if GATE_MEASURE_DETERMINISTIC.get(measure_name, False):
        target_gate = single_qubit_excitation(config.theta_target)
        target_value = GATE_MEASURES[measure_name](target_gate)
        ax.hlines(target_value, 0, config.T, colors=COLOR_TARGET, linestyles='dashed',
                  label='Target', zorder=1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(GATE_MEASURE_LABELS.get(measure_name, measure_name))
    ax.set_ylim(-0.05, 2.05)
    ax.set_title(GATE_MEASURE_TITLES.get(measure_name, f'{measure_name} Evolution'))
    ax.legend()
    return ax


# =============================================================================
# Combined multi-panel figure
# =============================================================================

def plot_sqe_gate_optimisation(results: SQEGateResults, figsize=None):
    """
    Multi-panel figure for gate optimization results.

    Panels
    ------
    0 : Control pulse
    1 : Gate infidelity evolution (log scale)
    2 : Main gate entanglement measure (config.gate_ent_measure)
    3+ : One panel per measure in config.extra_ent_measures
    """
    extra = list(results.config.extra_ent_measures)
    n_panels = 3 + len(extra)
    figsize = figsize or (12, 8 + 3 * len(extra))
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    plot_control_pulses_gate(results, ax=axes[0])
    plot_infidelity_evolution_gate(results, ax=axes[1])
    plot_entanglement_evolution_gate(results, ax=axes[2])

    for i, name in enumerate(extra):
        plot_entanglement_evolution_gate(results, ax=axes[3 + i], measure_name=name)

    plt.tight_layout()
    plt.show()


# =============================================================================
# DQE gate plotting
# =============================================================================

_CHANNEL_NAMES = [r'$J_{12}$', r'$J_{23}$', r'$J_{34}$']


def plot_control_channel_dqe_gate(results: DQEGateResults, channel_idx: int, ax=None):
    """
    Plot one J-channel control pulse for DQE gate results.

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
    ax.set_title(f'Control Pulse ({_CHANNEL_NAMES[channel_idx]})')
    ax.legend()
    return ax


def plot_infidelity_evolution_dqe_gate(results: DQEGateResults, ax=None):
    """Plot gate infidelity evolution for DQE gate results (log scale)."""
    if ax is None:
        ax = plt.gca()
    config = results.config
    t = _fine_time(config)

    ax.plot(t, results.infid_initial, label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, results.baseline.infid_final, label='I-optimised',
                color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.plot(t, results.infid_final, label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Gate Infidelity')
    ax.set_yscale('log')
    ax.set_ylim(top=5.0)
    ax.set_title('Gate Infidelity Evolution')
    ax.legend()
    return ax


def plot_entanglement_evolution_dqe_gate(results: DQEGateResults, ax=None, *,
                                          measure_name: str | None = None):
    """
    Plot gate entanglement measure evolution for DQE gate results.

    Parameters
    ----------
    measure_name : str, optional
        Key in GATE_MEASURES. Defaults to config.gate_ent_measure.
    """
    if ax is None:
        ax = plt.gca()
    config = results.config
    measure_name = measure_name or config.gate_ent_measure
    t = _fine_time(config)
    is_main = (measure_name == config.gate_ent_measure)

    # The main measure is stored directly on the result; extra measures in a dict
    def _get_traj(res, final: bool):
        if is_main:
            return res.ent_final if final else res.ent_initial
        store = res.extra_ent_final if final else res.extra_ent_initial
        return store[measure_name]

    ax.plot(t, _get_traj(results, False), label='Initial', color=COLOR_INITIAL, zorder=2)

    if results.baseline is not None:
        ax.plot(t, _get_traj(results.baseline, True), label='I-optimised',
                color=COLOR_FIDELITY_ONLY, zorder=3)

    label, color = _run_label_color(config.entanglement_optimisation)
    ax.plot(t, _get_traj(results, True), label=label, color=color, zorder=4)

    # Deterministic measures can be evaluated quickly at plot time for a target line.
    # MC-based measures (average_entangling_power etc.) are skipped — too slow.
    if GATE_MEASURE_DETERMINISTIC.get(measure_name, False):
        target_gate = dqe_unitary(config.theta_target)
        target_value = GATE_MEASURES[measure_name](target_gate)
        ax.hlines(target_value, 0, config.T, colors=COLOR_TARGET, linestyles='dashed',
                  label='Target', zorder=1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(GATE_MEASURE_LABELS.get(measure_name, measure_name))
    ax.set_ylim(bottom=-0.05)
    ax.set_title(GATE_MEASURE_TITLES.get(measure_name, f'{measure_name} Evolution'))
    ax.legend()
    return ax


def plot_dqe_gate_optimisation(results: DQEGateResults, figsize=None):
    """
    Multi-panel figure for DQE gate optimization results.

    Panels
    ------
    0   : J₁₂ control pulse
    1   : J₂₃ control pulse
    2   : J₃₄ control pulse
    3   : Gate infidelity evolution (log scale)
    4   : Main gate entanglement measure
    5+  : One panel per measure in config.extra_ent_measures
    """
    extra    = list(results.config.extra_ent_measures)
    n_panels = 5 + len(extra)
    figsize  = figsize or (12, 3 * n_panels)
    fig, axes = plt.subplots(n_panels, 1, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    plot_control_channel_dqe_gate(results, 0, axes[0])
    plot_control_channel_dqe_gate(results, 1, axes[1])
    plot_control_channel_dqe_gate(results, 2, axes[2])
    plot_infidelity_evolution_dqe_gate(results, axes[3])
    plot_entanglement_evolution_dqe_gate(results, axes[4])

    for i, name in enumerate(extra):
        plot_entanglement_evolution_dqe_gate(results, axes[5 + i], measure_name=name)

    plt.tight_layout()
    plt.show()
