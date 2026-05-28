"""
Shared utilities for pulse_vqe_entanglement optimiser modules.

Consolidates functions that were previously duplicated across sqe_state,
sqe_gate, sqe_state_noise, dqe_state, and dqe_gate:
  - TensorFlow dtype conversion
  - Random initial-parameter generation (thread-safe, no global state)
  - Gate matrices: SQE and DQE rotations
  - Gate infidelity / trajectory
  - Rotating-frame construction for an arbitrary number of spins
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.linalg import expm

from .dataclasses import SimulationConfig, NS_TO_US


# ---------------------------------------------------------------------------
# TensorFlow helpers
# ---------------------------------------------------------------------------

def convert_dtype(np_array) -> tf.Tensor:
    """Convert a numpy array to a TensorFlow complex128 tensor."""
    return tf.cast(tf.convert_to_tensor(np_array), dtype=tf.complex128)


# ---------------------------------------------------------------------------
# Initial parameters
# ---------------------------------------------------------------------------

def _initialise_x0(config: SimulationConfig) -> np.ndarray:
    """
    Draw initial control parameters, clipped to [-1, 1].

    Uses config.J_init_seed for reproducibility without touching the global
    random state.  If config.x0_override is set it is returned directly.
    """
    if getattr(config, "x0_override", None) is not None:
        x0 = np.asarray(config.x0_override, dtype=float).copy()
    else:
        rng = np.random.RandomState(
            int(config.J_init_seed) if config.J_init_seed is not None else None
        )
        n   = int(config.n_time_steps) * int(config.n_J_channels)
        x0  = config.J_init_mean + float(config.J_init_scale) * rng.uniform(-1.0, 1.0, size=n)
    return np.clip(x0, -1.0, 1.0).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Gate matrices
# ---------------------------------------------------------------------------

def single_qubit_excitation(theta: float) -> np.ndarray:
    """
    SQE rotation in the |01⟩ ↔ |10⟩ subspace.

    Identity on |00⟩ and |11⟩.  Basis order: [|00⟩, |01⟩, |10⟩, |11⟩].
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [[1, 0,  0, 0],
         [0, c, -s, 0],
         [0, s,  c, 0],
         [0, 0,  0, 1]],
        dtype=np.complex128,
    )


def dqe_unitary(theta: float) -> np.ndarray:
    """
    DQE gate in the 4-spin (16-dim) Hilbert space.

    Rotates the |0011⟩ ↔ |1100⟩ subspace by theta; identity everywhere else.
    Basis order: big-endian, index = int(bitstring, 2).
    """
    U = np.eye(16, dtype=np.complex128)
    c, s     = np.cos(theta), np.sin(theta)
    U[3, 3]  =  c
    U[3, 12] = -s
    U[12, 3] =  s
    U[12, 12] = c
    return U


# ---------------------------------------------------------------------------
# Gate infidelity
# ---------------------------------------------------------------------------

def gate_infidelity(U: np.ndarray, target: np.ndarray) -> float:
    """Average gate infidelity: 1 − (|Tr(target† U)|² + d) / (d(d+1))."""
    d = U.shape[0]
    return float(1.0 - (np.abs(np.trace(target.conj().T @ U)) ** 2 + d) / (d * (d + 1)))


def get_gate_infidelity_traj(U_traj: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gate infidelity at each time step of U_traj (shape (n_t, d, d))."""
    return np.array([gate_infidelity(U_traj[t], target) for t in range(U_traj.shape[0])])


# ---------------------------------------------------------------------------
# Rotating-frame construction
# ---------------------------------------------------------------------------

def make_rotating_frame(
    zeeman_splittings: np.ndarray | list,
    T_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the rotating-frame unitary and shifted Zeeman splittings for n spins.

    Parameters
    ----------
    zeeman_splittings : array-like, shape (n,)
        Zeeman splittings in MHz.
    T_ns : float
        Total gate time in nanoseconds.

    Returns
    -------
    deltaB : ndarray, shape (n,)
        zeeman_splittings − mean(zeeman_splittings).
    U_rot : ndarray, shape (2^n, 2^n)
        expm(−i π Σ_k deltaB_k Z_k · T_ns · NS_TO_US).
    """
    B      = np.asarray(zeeman_splittings, dtype=np.float64)
    deltaB = B - np.mean(B)
    n      = len(deltaB)

    Z       = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    dim     = 2 ** n
    H_frame = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(n):
        Zk = np.kron(
            np.kron(np.eye(2 ** k,           dtype=np.complex128), Z),
                     np.eye(2 ** (n - k - 1), dtype=np.complex128),
        )
        H_frame += deltaB[k] * Zk

    U_rot = expm(-1j * np.pi * H_frame * T_ns * NS_TO_US)
    return deltaB, U_rot
