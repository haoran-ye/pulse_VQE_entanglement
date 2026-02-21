"""
Plotting functions for gate-based SQE optimization results.

Produces the same 3-panel layout as plotting.py:
1. Control pulse (exchange coupling)
2. Infidelity evolution (gate or state)
3. Entanglement measure evolution (operator entanglement or entropy)
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from .dataclasses import SQEGateResults
from .sqe_state import single_qubit_excitation, get_Sent_single
from .sqe_gate import operator_entanglement


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

COLOR_INITIAL = '#1f77b4'        # blue
COLOR_FIDELITY_ONLY = '#2ca02c'  # green (infidelity-only)
COLOR_ENTANGLEMENT = '#d62728'   # red (entanglement-constrained)
COLOR_TARGET = 'black'


def _rescale_J(x, J_min, J_max):
    return 0.5 * (J_max - J_min) * x + 0.5 * (J_max + J_min)


# ========== Individual Plot Functions ==========

def plot_control_pulses_gate(results: SQEGateResults, ax=None):
    """Plot exchange coupling control pulses (identical format to state version)."""
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
        where='post', label='Initial', color=COLOR_INITIAL,
        linestyle='-', zorder=2,
    )

    # ---- Baseline (infidelity-only) ----
    if results.baseline is not None:
        xb = results.baseline.x_final
        xb_plot = np.concatenate([xb, [xb[-1]]])
        Jb_plot = _rescale_J(xb_plot, J_min, J_max)
        ax.step(
            t, Jb_plot,
            where='post', label='I-optimised', color=COLOR_FIDELITY_ONLY,
            linestyle='-', zorder=3,
        )

    # ---- Final optimised ----
    xf_plot = np.concatenate([results.x_final, [results.x_final[-1]]])
    Jf_plot = _rescale_J(xf_plot, J_min, J_max)

    run_mode = config.run_mode
    if run_mode == "infidelity":
        label = 'I-optimised'
        color = COLOR_FIDELITY_ONLY
    else:
        label = 'EP-optimised'
        color = COLOR_ENTANGLEMENT

    ax.step(
        t, Jf_plot,
        where='post', label=label, color=color,
        linestyle='-', zorder=4,
    )

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel(r'$J$ (MHz)')
    ax.set_title('Control Pulse (Exchange Coupling)')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    return ax


def plot_infidelity_evolution_gate(results: SQEGateResults, ax=None):
    """
    Plot infidelity evolution (gate or state depending on optimisation_target).

    Uses log scale on the y-axis.
    """
    if ax is None:
        ax = plt.gca()

    config = results.config
    n_fine = config.n_time_steps * config.n_discretisation
    t = np.arange(n_fine + 1) * config.T / n_fine

    is_gate = config.optimisation_target == "gate"

    # ---- Initial ----
    ax.plot(
        t, results.infid_initial,
        label='Initial', color=COLOR_INITIAL, linestyle='-', zorder=2,
    )

    # ---- Baseline ----
    if results.baseline is not None:
        ax.plot(
            t, results.baseline.infid_final,
            label='I-optimised', color=COLOR_FIDELITY_ONLY,
            linestyle='-', zorder=3,
        )

    # ---- Final optimised ----
    run_mode = config.run_mode
    if run_mode == "infidelity":
        label = 'I-optimised'
        color = COLOR_FIDELITY_ONLY
    else:
        label = 'EP-optimised'
        color = COLOR_ENTANGLEMENT

    ax.plot(
        t, results.infid_final,
        label=label, color=color, linestyle='-', zorder=4,
    )

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Gate Infidelity' if is_gate else 'Infidelity')
    ax.set_yscale('log')
    ax.set_ylim(top=5.0)
    ax.set_title('Gate Infidelity Evolution' if is_gate else 'Infidelity Evolution')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    return ax


def plot_entanglement_evolution_gate(results: SQEGateResults, ax=None):
    """
    Plot entanglement measure evolution.

    For gate optimization: plots operator entanglement (entangling power).
    For state optimization: plots entanglement entropy.
    """
    if ax is None:
        ax = plt.gca()

    config = results.config
    n_fine = config.n_time_steps * config.n_discretisation
    t = np.arange(n_fine + 1) * config.T / n_fine
    T = config.T
    theta_target = config.theta_target

    is_gate = config.optimisation_target == "gate"

    # ---- Initial ----
    ax.plot(
        t, results.ent_initial,
        label='Initial', color=COLOR_INITIAL, linestyle='-', zorder=2,
    )

    # ---- Baseline ----
    if results.baseline is not None:
        ax.plot(
            t, results.baseline.ent_final,
            label='I-optimised', color=COLOR_FIDELITY_ONLY,
            linestyle='-', zorder=3,
        )

    # ---- Final optimised ----
    run_mode = config.run_mode
    if run_mode == "infidelity":
        label = 'I-optimised'
        color = COLOR_FIDELITY_ONLY
    else:
        label = 'EP-optimised'
        color = COLOR_ENTANGLEMENT

    ax.plot(
        t, results.ent_final,
        label=label, color=color, linestyle='-', zorder=4,
    )

    # ---- Target line ----
    target_gate = single_qubit_excitation(theta_target)
    if is_gate:
        target_value = operator_entanglement(target_gate)
        target_label = 'Target EP'
    else:
        if config.initial_state is not None:
            target_state = target_gate @ config.initial_state
            target_value = get_Sent_single(target_state)
        else:
            target_value = None
        target_label = 'Target S'

    if target_value is not None:
        ax.hlines(
            target_value, 0, T,
            colors=COLOR_TARGET, linestyles='dashed',
            label=target_label, zorder=1,
        )

    ax.set_xlabel('Time (ns)')
    if is_gate:
        ax.set_ylabel('Operator Entanglement (bits)')
        ax.set_ylim(-0.05, 2.05)
        ax.set_title('Entangling Power Evolution')
    else:
        ax.set_ylabel('S (bits)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Entanglement Entropy Evolution')

    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    return ax


# ========== Combined 3-Panel Plot ==========

def plot_sqe_gate_optimisation(results: SQEGateResults, figsize=(12, 8)):
    """
    Produce 3-panel figure for gate optimization results.

    Panel 1: Control pulses (exchange coupling)
    Panel 2: Infidelity evolution (gate or state, log scale)
    Panel 3: Entanglement measure evolution (operator entanglement or entropy)
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    plot_control_pulses_gate(results, ax=axes[0])
    plot_infidelity_evolution_gate(results, ax=axes[1])
    plot_entanglement_evolution_gate(results, ax=axes[2])
    plt.tight_layout()
    plt.show()
