"""Adjoint gradient of the seven-partition DQE state entropy cost.

The four-spin propagator matches SpinChain and PySTE's first-order Trotter
ordering: drift, then exchange channels 0, 1, and 2. Physical spin 0 is the
rightmost tensor factor in the simulator, and its Pauli Z is diag(-1, 1).
These conventions matter for four spins even though they cancel in the SQE
two-spin case.
"""
from __future__ import annotations

from functools import reduce

import numpy as np

from .dataclasses import NS_TO_US, SimulationConfig
from .entanglement_measures import ALL_BIPARTITIONS_4Q

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.diag([1, -1]).astype(np.complex128)
_IDENTITY = np.eye(16, dtype=np.complex128)


def _tensor(operators):
    return reduce(np.kron, operators)


def _site(operator: np.ndarray, physical_spin: int) -> np.ndarray:
    """Place an operator on a physical spin (spin 0 is rightmost)."""
    return _tensor([operator if axis == 3 - physical_spin else _I for axis in range(4)])


_EXCHANGE = tuple(
    sum(
        _tensor([pauli if axis in (3 - spin, 2 - spin) else _I
                 for axis in range(4)])
        for pauli in (_X, _Y, _Z)
    ) / 4.0
    for spin in range(3)
)
_EXCHANGE_EIG = tuple(np.linalg.eigh(h) for h in _EXCHANGE)
_PARTITIONS = tuple(
    (partition + tuple(q for q in range(4) if q not in partition),
     2 ** len(partition), 2 ** (4 - len(partition)))
    for partition in ALL_BIPARTITIONS_4Q
)


def _entropy_sum_and_state_gradient(state: np.ndarray) -> tuple[float, np.ndarray]:
    """Return sum of seven entropies and q with dS = 2 Re(q^H dpsi)."""
    tensor = state.reshape((2,) * 4)
    total_entropy = 0.0
    gradient = np.zeros(16, dtype=np.complex128)
    for axes, dim_a, dim_b in _PARTITIONS:
        amplitudes = np.transpose(tensor, axes).reshape(dim_a, dim_b)
        rho = amplitudes @ amplitudes.conj().T
        eigenvalues, vectors = np.linalg.eigh(rho)
        clipped = np.clip(eigenvalues, 1e-15, 1.0)
        total_entropy -= float(np.sum(clipped * np.log2(clipped)))
        entropy_derivative = (
            vectors * (-(np.log(clipped) + 1.0) / np.log(2.0))
        ) @ vectors.conj().T
        q_partition = (entropy_derivative @ amplitudes).reshape((2,) * 4)
        gradient += np.transpose(q_partition, np.argsort(axes)).reshape(16)
    return total_entropy, gradient


def _step_matrices(amplitudes: np.ndarray, drift_step: np.ndarray,
                   dt: float, exchange_derivatives: tuple[np.ndarray, ...]
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return the Trotter step and its three exact control derivatives."""
    factors = [drift_step]
    for channel, amplitude in enumerate(amplitudes):
        eigenvalues, vectors = _EXCHANGE_EIG[channel]
        phases = np.exp(-1j * dt * amplitude * eigenvalues)
        factors.append((vectors * phases) @ vectors.conj().T)

    prefixes = [_IDENTITY]
    for factor in factors:
        prefixes.append(factor @ prefixes[-1])
    suffix = _IDENTITY
    derivatives = np.empty((3, 16, 16), dtype=np.complex128)
    for channel in range(2, -1, -1):
        factor_index = channel + 1
        derivatives[channel] = (
            suffix @ (-1j * dt * exchange_derivatives[channel])
            @ factors[factor_index] @ prefixes[factor_index]
        )
        suffix = suffix @ factors[factor_index]
    return prefixes[-1], derivatives


def entropy_cost_and_grad(
    config: SimulationConfig, x: np.ndarray, *, return_trajectory: bool = False,
):
    """Return the DQE entropy cost and analytic gradient.

    ``x`` is flat in time-major order with three exchange controls per time
    step. If requested, the trajectory has shape (16, n_fine + 1).
    """
    n_steps = int(config.n_time_steps)
    n_disc = int(config.n_discretisation)
    if n_steps < 1 or n_disc < 1:
        raise ValueError("n_time_steps and n_discretisation must be positive")
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (3 * n_steps,):
        raise ValueError(f"expected x.shape == ({3 * n_steps},), got {x.shape}")
    initial = np.asarray(config.initial_state, dtype=np.complex128)
    if initial.shape != (16,):
        raise ValueError("DQE requires an initial state of shape (16,)")
    zeeman = np.asarray(config.zeeman_splittings, dtype=np.float64)
    if zeeman.shape != (4,):
        raise ValueError("DQE requires four Zeeman splittings")

    n_fine = n_steps * n_disc
    # QuGrad passes the Python timestep through float32 before PySTE uses it.
    dt = float(np.float32(2.0 * np.pi * config.T * NS_TO_US / n_fine))
    detuning = zeeman - zeeman.mean()
    # The simulator's Pauli Z is diag(-1, 1), the negative of _Z.
    drift = -sum(0.5 * detuning[i] * _site(_Z, i) for i in range(4))
    drift_step = np.diag(np.exp(-1j * dt * np.diag(drift)))
    sign = -1.0 if config.feromagnetic else 1.0
    exchange_slope = sign * 0.5 * (config.J_max - config.J_min)
    exchange_derivatives = tuple(exchange_slope * h for h in _EXCHANGE)

    steps = np.empty((n_steps, 16, 16), dtype=np.complex128)
    derivatives = np.empty((n_steps, 3, 16, 16), dtype=np.complex128)
    for j, row in enumerate(x.reshape(n_steps, 3)):
        physical_j = 0.5 * (config.J_min + config.J_max
                            + (config.J_max - config.J_min) * row)
        steps[j], derivatives[j] = _step_matrices(
            sign * physical_j, drift_step, dt, exchange_derivatives
        )

    states = np.empty((n_fine + 1, 16), dtype=np.complex128)
    states[0] = initial
    for k in range(n_fine):
        states[k + 1] = steps[k // n_disc] @ states[k]

    entropies = np.empty(n_fine + 1, dtype=np.float64)
    q = np.empty_like(states)
    for k, state in enumerate(states):
        entropies[k], q[k] = _entropy_sum_and_state_gradient(state)
    violation = np.maximum(entropies - float(config.S_max), 0.0)
    cost = float(np.mean(violation**2))
    q *= (2.0 * violation / (n_fine + 1))[:, None]

    gradient = np.zeros((n_steps, 3), dtype=np.float64)
    costate = q[-1].copy()
    for k in range(n_fine - 1, -1, -1):
        j = k // n_disc
        for channel in range(3):
            gradient[j, channel] += 2.0 * np.vdot(
                costate, derivatives[j, channel] @ states[k]
            ).real
        costate = q[k] + steps[j].conj().T @ costate

    if return_trajectory:
        return cost, gradient.ravel(), states.T
    return cost, gradient.ravel()
