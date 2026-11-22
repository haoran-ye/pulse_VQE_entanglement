"""Adjoint gradient of the transient SQE state-entanglement cost.

For psi[k+1] = U[k] psi[k], define C = mean_k max(S(psi[k])-S_max, 0)^2.
Writing dC = 2 Re(sum_k q[k]^H dpsi[k]), the backward costate is

    p[N] = q[N],  p[k] = q[k] + U[k]^H p[k+1],
    dC/dx_j = 2 Re(sum_{k in segment j} p[k+1]^H (dU[k]/dx_j) psi[k]).

The entropy derivative q is analytic. PySTE uses a first-order Suzuki-Trotter
step, U = exp(-i dt J H_exchange) exp(-i dt H_drift). Since the exchange
generator commutes with its own exponential, dU/dx = -i dt (dJ/dx)
H_exchange U. This requires O(n_time_steps) small matrix operations instead
of two full simulator propagations for every pulse parameter.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .dataclasses import NS_TO_US, SimulationConfig

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.diag([1, -1]).astype(np.complex128)
_EXCHANGE = sum(np.kron(p, p) for p in (_X, _Y, _Z)) / 4


def _entropy_and_state_gradient(state: np.ndarray, measure: str):
    """Return S and q with dS = 2 Re(q^H dpsi), for a two-qubit state."""
    psi = state.reshape(2, 2)
    rho = psi @ psi.conj().T
    if measure == "von_neumann":
        eig, vectors = np.linalg.eigh(rho)
        clipped = np.clip(eig, 1e-15, 1.0)
        entropy = -np.sum(clipped * np.log2(clipped))
        matrix = (vectors * (-(np.log(clipped) + 1.0) / np.log(2.0))) @ vectors.conj().T
        q = matrix @ psi
    elif measure == "linear":
        entropy = 1.0 - np.trace(rho @ rho).real
        q = -2.0 * rho @ psi
    else:
        raise ValueError("SQE adjoint gradient supports 'von_neumann' and 'linear'.")
    return float(entropy), q.reshape(4)


def entropy_cost_and_grad(
    config: SimulationConfig, x: np.ndarray, *,
    return_trajectory: bool = False,
):
    """Return (cost, gradient), optionally with the trajectory (4, n_fine+1).

    ``x`` uses the same [-1,1] exchange controls, big-endian basis, rotating
    frame, and ns-to-us conversion as ``sqe_state_adjoint``.
    """
    n_steps = int(config.n_time_steps)
    n_disc = int(config.n_discretisation)
    if n_steps < 1 or n_disc < 1:
        raise ValueError("n_time_steps and n_discretisation must be positive")
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (n_steps,):
        raise ValueError(f"expected x.shape == ({n_steps},), got {x.shape}")
    initial = np.asarray(config.initial_state, dtype=np.complex128)
    if initial.shape != (4,):
        raise ValueError("SQE requires an initial state of shape (4,)")
    if config.state_ent_measure not in ("von_neumann", "linear"):
        raise ValueError("SQE adjoint gradient supports 'von_neumann' and 'linear'.")

    delta_b = np.asarray(config.zeeman_splittings, dtype=float)
    if delta_b.shape != (2,):
        raise ValueError("SQE requires two Zeeman splittings")
    delta_b = delta_b - delta_b.mean()
    drift = 0.5 * (delta_b[0] * np.kron(_Z, _I) + delta_b[1] * np.kron(_I, _Z))
    sign = -1.0 if config.feromagnetic else 1.0
    d_h = sign * 0.5 * (config.J_max - config.J_min) * _EXCHANGE
    n_fine = n_steps * n_disc
    # QuGrad's pulse-form path passes the Python timestep through tf.float32
    # before handing it to PySTE, even when controls and states are complex128.
    dt = float(np.float32(2.0 * np.pi * config.T * NS_TO_US / n_fine))

    drift_propagator = expm(-1j * dt * drift)
    exchange_eig, exchange_vectors = np.linalg.eigh(_EXCHANGE)
    propagators = []
    derivatives = []
    for amplitude in x:
        physical_j = 0.5 * (config.J_min + config.J_max + (config.J_max - config.J_min) * amplitude)
        phases = np.exp(-1j * dt * sign * physical_j * exchange_eig)
        exchange_propagator = (exchange_vectors * phases) @ exchange_vectors.conj().T
        u = exchange_propagator @ drift_propagator
        du = (-1j * dt * d_h) @ u
        propagators.append(u)
        derivatives.append(du)

    states = np.empty((n_fine + 1, 4), dtype=np.complex128)
    states[0] = initial
    for k in range(n_fine):
        states[k + 1] = propagators[k // n_disc] @ states[k]

    entropies = np.empty(n_fine + 1)
    q = np.empty_like(states)
    for k, state in enumerate(states):
        entropies[k], q[k] = _entropy_and_state_gradient(state, config.state_ent_measure)
    violation = np.maximum(entropies - config.S_max, 0.0)
    cost = float(np.mean(violation**2))
    q *= (2.0 * violation / (n_fine + 1))[:, None]

    costate = q[-1].copy()
    gradient = np.zeros(n_steps, dtype=np.float64)
    for k in range(n_fine - 1, -1, -1):
        j = k // n_disc
        gradient[j] += 2.0 * np.vdot(costate, derivatives[j] @ states[k]).real
        costate = q[k] + propagators[j].conj().T @ costate

    if return_trajectory:
        return cost, gradient, states.T
    return cost, gradient
