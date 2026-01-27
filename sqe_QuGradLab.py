"""
Minimal Single-Qubit Excitation (SQE) using QuGradLab/QuGrad built-in tools.

Implements optimization of a 2-qubit system with antiferromagnetic J coupling
to perform single-qubit excitation from |01⟩ to cos(θ)|01⟩ + sin(θ)|10⟩.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qugrad import QuantumSystem
from qugradlab.hilbert_spaces import QubitSpace


# ============================================================================
# Configuration
# ============================================================================

class SQEConfig:
    """Configuration for single-qubit excitation optimization."""
    def __init__(self):
        # Target state parameter
        self.theta = 0.4 * np.pi  # Target angle

        # Time and discretization
        self.T_ns = 1000.0  # Total time in nanoseconds
        self.N_SEGMENTS = 100  # Number of piecewise constant segments

        # Optimization
        self.max_iter = 200
        self.infidelity_threshold = 1e-5
        self.seed = 42

        # Coupling strength (antiferromagnetic, no magnetic field)
        self.J_coupling = 1.0  # Will be optimized in units of rad/s

        # Pulse bounds (MHz)
        self.J_max_MHz = 5.0


# ============================================================================
# Quantum System Setup
# ============================================================================

def setup_system(config: SQEConfig):
    """Build 2-qubit Hilbert space and Hamiltonian."""
    # Pauli operators
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)

    # 2-qubit Hilbert space
    qubit_space = QubitSpace(qubits=2)
    hilbert_space = qubit_space.computational_subspace()
    DIM = hilbert_space.dim

    # Two-qubit Pauli operators
    X1, X2 = np.kron(X, I2), np.kron(I2, X)
    Y1, Y2 = np.kron(Y, I2), np.kron(I2, Y)
    Z1, Z2 = np.kron(Z, I2), np.kron(I2, Z)

    # Antiferromagnetic Heisenberg exchange (control Hamiltonian)
    # H_control = J(t) * (X1⊗X2 + Y1⊗Y2 + Z1⊗Z2) / 4
    Hc = 0.25 * (X1 @ X2 + Y1 @ Y2 + Z1 @ Z2)

    # No static magnetic field (drift Hamiltonian)
    H0 = np.zeros((DIM, DIM), dtype=np.complex128)

    return hilbert_space, DIM, H0, Hc


def setup_pulse_parameterization(config: SQEConfig, DIM: int):
    """Setup piecewise constant pulse parameterization."""
    N_STEPS = config.N_SEGMENTS
    dt = (config.T_ns * 1e-9) / float(N_STEPS)  # Time step in seconds
    MHZ_TO_RAD_PER_S = 2.0 * np.pi * 1e6

    # Initial pulse parameters (random small values in MHz)
    rng = np.random.default_rng(config.seed)
    x0 = 0.2 * rng.standard_normal(config.N_SEGMENTS).astype(np.float64)

    # Initial state: |01⟩
    psi0 = np.zeros(DIM, dtype=np.complex128)
    psi0[1] = 1.0

    def pulse_form(amps_MHz: tf.Tensor):
        """
        Convert optimization parameters to control amplitudes.

        Args:
            amps_MHz: Piecewise constant pulse amplitudes in MHz (N_SEGMENTS,)

        Returns:
            ctrl_amp: Control amplitudes in rad/s (N_STEPS, 1)
            psi0: Initial state
            dt: Time step in seconds
            freqs: Carrier frequencies (not used, set to 0)
            number_channels: Number of control channels [1]
        """
        # Convert to complex128 and reshape
        amps = tf.reshape(tf.cast(amps_MHz, tf.complex128), (config.N_SEGMENTS, 1))

        # Convert MHz to rad/s
        ctrl_amp = amps * tf.cast(MHZ_TO_RAD_PER_S, tf.complex128)

        # No carrier frequency
        freqs = tf.constant([0.0], dtype=tf.complex128)

        return ctrl_amp, psi0, dt, freqs, [1]

    return x0, pulse_form


def setup_target_observable(config: SQEConfig, DIM: int):
    """Setup target state and infidelity observable."""
    # Target state: |ψ_target⟩ = cos(θ)|01⟩ + sin(θ)|10⟩
    psi_target = np.zeros(DIM, dtype=np.complex128)
    psi_target[1] = np.cos(config.theta)  # |01⟩
    psi_target[2] = np.sin(config.theta)  # |10⟩

    # Infidelity observable: I - |ψ_target⟩⟨ψ_target|
    # Expectation value gives 1 - fidelity
    observable = np.eye(DIM, dtype=np.complex128) - np.outer(psi_target, psi_target.conj())

    return psi_target, observable


# ============================================================================
# Optimization
# ============================================================================

def optimize_pulse(config: SQEConfig):
    """Run pulse optimization using QuGrad."""
    print("=" * 60)
    print("Single-Qubit Excitation Optimization")
    print("=" * 60)
    print(f"Target angle: θ = {config.theta:.4f} rad ({config.theta/np.pi:.4f}π)")
    print(f"Total time: {config.T_ns} ns")
    print(f"Segments: {config.N_SEGMENTS}")
    print(f"Max iterations: {config.max_iter}")
    print(f"Infidelity threshold: {config.infidelity_threshold:.1e}")
    print("=" * 60 + "\n")

    # Setup system
    hilbert_space, DIM, H0, Hc = setup_system(config)
    x0, pulse_form = setup_pulse_parameterization(config, DIM)
    psi_target, observable = setup_target_observable(config, DIM)

    # Create quantum system with control
    device = QuantumSystem(H0, [Hc], hilbert_space)
    driven_device = device.pulse_form(pulse_form)

    # Optimization bounds (MHz)
    Jmax = config.J_max_MHz
    bounds = [(-Jmax, +Jmax)] * config.N_SEGMENTS

    # Cost function: infidelity at final time
    def cost_function(x):
        val, grad = driven_device.gradient(x, observable)
        return float(np.real(val)), np.real(grad).astype(np.float64)

    # Initial infidelity
    I_init = float(np.real(driven_device.evolved_expectation_value(x0, observable)))
    print(f"Initial infidelity: {I_init:.6e}\n")

    # Optimize
    print("Starting optimization...")
    result = minimize(
        cost_function,
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": config.max_iter, "disp": True, "gtol": 1e-8}
    )

    # Final infidelity
    I_final = float(np.real(driven_device.evolved_expectation_value(result.x, observable)))

    print("\n" + "=" * 60)
    print(f"Optimization complete!")
    print(f"Final infidelity: {I_final:.6e}")
    print(f"Iterations: {result.nit}")
    print(f"Success: {I_final <= config.infidelity_threshold}")
    print("=" * 60 + "\n")

    return {
        'config': config,
        'device': driven_device,
        'observable': observable,
        'psi_target': psi_target,
        'x0': x0,
        'x_opt': result.x,
        'I_init': I_init,
        'I_final': I_final,
        'result': result,
        'DIM': DIM
    }


# ============================================================================
# Analysis and Plotting
# ============================================================================

def analyze_trajectory(results):
    """Compute infidelity and entanglement along the trajectory."""
    config = results['config']
    driven_device = results['device']
    observable = results['observable']
    x_opt = results['x_opt']

    # Get optimal pulse and propagate
    ctrl_amp, psi0, dt = driven_device.get_driving_pulses(x_opt)
    states = np.asarray(driven_device.evolver.propagate_all(ctrl_amp, psi0, dt))

    n_steps = states.shape[1]
    times_ns = np.arange(n_steps) * dt * 1e9

    # Compute infidelity at each time
    infidelity_t = np.array([np.real(np.vdot(states[:, i], observable @ states[:, i]))
                             for i in range(n_steps)])

    # Compute entanglement entropy at each time
    def entanglement_entropy(psi):
        psi_mat = psi.reshape(2, 2)
        rho_A = psi_mat @ psi_mat.conj().T
        eigvals = np.clip(np.linalg.eigvalsh(rho_A).real, 0.0, 1.0)
        nonzero = eigvals > 0
        if not np.any(nonzero):
            return 0.0
        logs = np.log2(eigvals[nonzero])
        return float(-np.sum(eigvals[nonzero] * logs))

    entropy_t = np.array([entanglement_entropy(states[:, i]) for i in range(n_steps)])

    # Target entanglement
    S_target = entanglement_entropy(results['psi_target'])

    return times_ns, infidelity_t, entropy_t, S_target, x_opt


def plot_results(results):
    """Plot optimization results."""
    config = results['config']
    times_ns, infidelity_t, entropy_t, S_target, x_opt = analyze_trajectory(results)

    MHZ_TO_RAD_PER_S = 2.0 * np.pi * 1e6
    t_segments = np.linspace(0, config.T_ns, config.N_SEGMENTS + 1)
    pulse_MHz = x_opt
    pulse_plot = np.r_[pulse_MHz, pulse_MHz[-1]]

    fig = plt.figure(figsize=(12, 10))

    # Pulse shape
    ax1 = plt.subplot(3, 1, 1)
    ax1.step(t_segments, pulse_plot, where='post', linewidth=2, color='#d62728')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel(r'$J(t)$ (MHz)')
    ax1.set_title(rf'Optimized Pulse ($\theta={config.theta:.3f}$ rad)', fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Infidelity trajectory
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(times_ns, infidelity_t, linewidth=2, color='#1f77b4')
    ax2.axhline(config.infidelity_threshold, ls='--', color='red', alpha=0.7,
               label=f'Threshold={config.infidelity_threshold:.1e}')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Infidelity')
    ax2.set_yscale('log')
    ax2.set_title('Infidelity Evolution', fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend()

    # Entanglement entropy
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(times_ns, entropy_t, linewidth=2, color='#2ca02c')
    ax3.axhline(S_target, ls='--', color='black', lw=2, alpha=0.7,
               label=f'Target S={S_target:.3f}')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Entanglement Entropy')
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title('Entanglement Evolution', fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.legend()

    plt.tight_layout()
    plt.show()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Create configuration
    config = SQEConfig()

    # Run optimization
    results = optimize_pulse(config)

    # Plot results
    plot_results(results)
