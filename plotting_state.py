from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from .dataclasses import SQEResults
from .sqe_state import single_qubit_excitation
from .entanglement_measures import STATE_MEASURES, STATE_MEASURE_LABELS, STATE_MEASURE_TITLES


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
    return 0.5 * (J_max - J_min) * x + 0.5 * (J_max + J_min)


def _run_label_color(run_mode):
    if run_mode == "infidelity":
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
    label, color = _run_label_color(config.run_mode)
    ax.step(t, _rescale_J(np.concatenate([xf, [xf[-1]]]), J_min, J_max),
            where='post', label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$J$ (MHz)')
    ax.set_title('Control Pulse (Exchange Coupling)')
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

    label, color = _run_label_color(config.run_mode)
    ax.plot(t, results.val_final, label=label, color=color, zorder=4)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Infidelity')
    ax.set_yscale('log')
    ax.set_ylim(top=5.0)
    ax.set_title('Infidelity Evolution')
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

    label, color = _run_label_color(config.run_mode)
    ax.plot(t, traj_fn(results.final_traj), label=label, color=color, zorder=4)

    # Target entropy line
    if config.initial_state is not None:
        target_state = single_qubit_excitation(config.theta_target) @ config.initial_state
        target_value = single_fn(target_state)
        ax.hlines(target_value, 0, config.T, colors=COLOR_TARGET, linestyles='dashed',
                  label='Target', zorder=1)

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(STATE_MEASURE_LABELS.get(measure_name, measure_name))
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(STATE_MEASURE_TITLES.get(measure_name, f'{measure_name} Evolution'))
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

