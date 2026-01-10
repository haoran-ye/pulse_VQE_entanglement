from __future__ import annotations

import numpy as np
import tensorflow as tf

from qugrad import QuantumSystem
from qugradlab.hilbert_spaces import QubitSpace

from .dataclasses import *


# Units: inputs (ns, MHz), internals (rad/s, s), cost (infidelity = 1 - fidelity)
MHZ_TO_RAD_PER_S = 2.0 * np.pi * 1e6
JMAX_MHZ = 5.0

# Pauli operators
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def build_hilbert_space():
    qubit_space = QubitSpace(qubits=2)
    hilbert_space = qubit_space.computational_subspace()
    return hilbert_space, hilbert_space.dim


def build_device(config: SimulationConfig, hilbert_space):
    X1, X2 = np.kron(X, I2), np.kron(I2, X)
    Y1, Y2 = np.kron(Y, I2), np.kron(I2, Y)
    Z1, Z2 = np.kron(Z, I2), np.kron(I2, Z)

    Hc = 0.25 * (X1 @ X2 + Y1 @ Y2 + Z1 @ Z2)
    H0 = -0.5 * (config.B1 * MHZ_TO_RAD_PER_S * Z1 + config.B2 * MHZ_TO_RAD_PER_S * Z2)

    return QuantumSystem(H0, [Hc], hilbert_space)


def build_pulse_form_and_initial(config: SimulationConfig, DIM: int):
    N_STEPS = config.N_SEGMENTS * config.PTS_PER_SEG
    dt = (config.T_ns * 1e-9) / float(N_STEPS)

    rng = np.random.default_rng(config.seed)
    x0 = config.pulse_init_scale * rng.standard_normal(config.N_SEGMENTS).astype(np.float64)

    def pulse_form(amps_flat: tf.Tensor):
        amps = tf.reshape(tf.cast(amps_flat, tf.complex128), (config.N_SEGMENTS, 1))
        ctrl_amp_mhz = tf.repeat(amps, repeats=config.PTS_PER_SEG, axis=0)
        ctrl_amp = ctrl_amp_mhz * tf.cast(MHZ_TO_RAD_PER_S, tf.complex128)

        psi0 = np.zeros(DIM, dtype=np.complex128)
        psi0[1] = 1.0

        return ctrl_amp, psi0, dt, tf.constant([0.0], dtype=tf.complex128), [1]

    return x0, pulse_form


def get_sqe_target_state(theta: float) -> np.ndarray:
    psi = np.zeros(4, dtype=np.complex128)
    psi[1] = np.cos(theta)
    psi[2] = np.sin(theta)
    return psi


def get_observable(config: SimulationConfig, DIM: int):
    psi_target = get_sqe_target_state(config.theta)
    return np.eye(DIM, dtype=np.complex128) - np.outer(psi_target, psi_target.conj())


def single_qubit_entanglement_entropy(psi: np.ndarray, base: float = 2.0) -> float:
    psi_mat = np.asarray(psi).reshape(2, 2)
    rho_A = psi_mat @ psi_mat.conj().T
    eigvals = np.clip(np.linalg.eigvalsh(rho_A).real, 0.0, 1.0)

    nonzero = eigvals > 0
    if not np.any(nonzero):
        return 0.0

    logs = np.log2(eigvals[nonzero]) if base == 2.0 else np.log(eigvals[nonzero]) / np.log(base)
    return float(-np.sum(eigvals[nonzero] * logs))


def trajectory_metrics_once(driven_device, x_mhz: np.ndarray, observable: np.ndarray, compute_entanglement: bool):
    ctrl_amp, psi0, dt = driven_device.get_driving_pulses(x_mhz)
    states = np.asarray(driven_device.evolver.propagate_all(ctrl_amp, psi0, dt))

    n_steps_plus_1 = states.shape[1]
    times_ns = np.arange(n_steps_plus_1) * dt * 1e9

    infidelity_t = np.array([np.real(np.vdot(states[:, i], observable @ states[:, i]))
                             for i in range(n_steps_plus_1)])

    entropy_t = None
    if compute_entanglement:
        entropy_t = np.array([single_qubit_entanglement_entropy(states[:, i])
                              for i in range(n_steps_plus_1)])

    return times_ns, infidelity_t, entropy_t
