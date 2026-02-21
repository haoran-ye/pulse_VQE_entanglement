# pulse_vqe_entanglement/dqe_optimisation.py
"""
Double Qubit Excitation (DQE) optimization for 4-spin systems.

Optimizes piecewise-constant exchange coupling pulses (3 channels for nearest-
neighbor chain) to implement two-qubit gates while optionally constraining
bipartite entanglement entropy via primal-dual methods.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, SQEResults, NS_TO_US


def convert_dtype(np_array) -> tf.Tensor:
    """Convert numpy array to TensorFlow complex128 tensor."""
    return tf.cast(tf.convert_to_tensor(np_array), dtype=tf.complex128)


def basis_state(n_qubits: int, bitstring: str) -> np.ndarray:
    """
    Create computational basis state from bitstring.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    bitstring : str
        Binary string (e.g., "0011").

    Returns
    -------
    np.ndarray
        Basis state vector of length 2^n_qubits.
    """
    if len(bitstring) != n_qubits:
        raise ValueError("bitstring length must equal n_qubits")
    idx = int(bitstring, 2)
    v = np.zeros((2**n_qubits,), dtype=np.complex128)
    v[idx] = 1.0
    return v


def dqe_unitary(theta: float) -> np.ndarray:
    """
    Double qubit excitation gate in 4-spin subspace.

    Acts as identity except on span{|0011⟩, |1100⟩} where it applies rotation:
        |0011⟩ → cos(θ)|0011⟩ - sin(θ)|1100⟩
        |1100⟩ → sin(θ)|0011⟩ + cos(θ)|1100⟩

    This is analogous to the SQE gate but for two-qubit excitation.

    Parameters
    ----------
    theta : float
        Rotation angle (radians).

    Returns
    -------
    np.ndarray
        16×16 unitary gate matrix.
    """
    U = np.eye(16, dtype=np.complex128)
    i = int("0011", 2)
    j = int("1100", 2)
    c = np.cos(theta)
    s = np.sin(theta)

    U[i, i] = c
    U[i, j] = -s
    U[j, i] = s
    U[j, j] = c
    return U


# ========== Bipartite Entanglement Entropy ==========

def _reshape_state_for_partition(state: np.ndarray, subsystem: tuple[int, ...], n_qubits: int):
    """
    Reshape state for bipartite entropy calculation.

    Reorders qubits so subsystem A comes first, then B, and reshapes into
    (dim_A, dim_B) matrix for Schmidt decomposition.

    Convention: Computational basis index follows big-endian bitstring
    q0 q1 ... q(n-1).

    Parameters
    ----------
    state : np.ndarray
        Pure state vector of length 2^n_qubits.
    subsystem : tuple[int, ...]
        Qubit indices for subsystem A.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray
        Reshaped state matrix (dim_A, dim_B).
    """
    state = np.asarray(state, dtype=np.complex128).reshape((2,) * n_qubits)

    A = tuple(subsystem)
    B = tuple(q for q in range(n_qubits) if q not in A)
    perm = A + B
    psi = np.transpose(state, perm)
    dim_A = 2 ** len(A)
    dim_B = 2 ** (n_qubits - len(A))
    return psi.reshape(dim_A, dim_B)


def entropy_bipartite_single(state: np.ndarray, *, subsystem=(0, 1), n_qubits=4) -> float:
    """
    Compute bipartite entanglement entropy for pure state.

    Parameters
    ----------
    state : np.ndarray
        Pure state vector.
    subsystem : tuple[int, ...]
        Qubit indices for subsystem A.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    float
        Von Neumann entropy S(A) = S(B) (in bits).
    """
    psi_AB = _reshape_state_for_partition(state, tuple(subsystem), n_qubits)
    rho_A = psi_AB @ psi_AB.conj().T
    eig = np.linalg.eigvalsh(rho_A)
    eig = np.clip(eig, 1e-15, 1.0)
    return float(-np.sum(eig * np.log2(eig)))


def entropy_bipartite_traj(traj: np.ndarray, *, subsystem=(0, 1), n_qubits=4) -> np.ndarray:
    """
    Compute bipartite entropy along state trajectory.

    Parameters
    ----------
    traj : np.ndarray
        State trajectory, shape (2^n_qubits, n_time_points).
    subsystem : tuple[int, ...]
        Qubit indices for subsystem A.
    n_qubits : int
        Total number of qubits.

    Returns
    -------
    np.ndarray
        Entropy at each time point, shape (n_time_points,).
    """
    traj = np.asarray(traj)
    n_times = traj.shape[1]
    entropy = np.zeros((n_times,), dtype=float)
    for t in range(n_times):
        entropy[t] = entropy_bipartite_single(traj[:, t], subsystem=subsystem, n_qubits=n_qubits)
    return entropy


# ========== Entropy-Constrained Optimization (Primal-Dual) ==========

def _cost_entropy(driven_sys, x, *, S_max: float, subsystem: tuple[int, ...]) -> float:
    """
    Entropy constraint violation cost (squared, time-averaged).

    Uses bipartite entropy for specified subsystem.

    Returns
    -------
    float
        Mean squared violation: mean((S(t) - S_max)_+^2).
    """
    traj = np.asarray(driven_sys.propagate_all(x))
    S = np.asarray(entropy_bipartite_traj(traj, subsystem=subsystem, n_qubits=4), float)
    viol = np.maximum(0.0, S - S_max)
    return float(np.mean(viol**2))


def _entropy_grad_fd(driven_sys, x, *, S_max: float, subsystem: tuple[int, ...], eps: float) -> np.ndarray:
    """Finite-difference gradient of entropy cost."""
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    f0 = _cost_entropy(driven_sys, x, S_max=S_max, subsystem=subsystem)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_cost_entropy(driven_sys, xp, S_max=S_max, subsystem=subsystem) - f0) / eps
    return g


def _infidelity_and_grad(driven_sys, x, observable):
    """Compute infidelity and gradient via automatic differentiation."""
    infidelity, grad = driven_sys.gradient(x, observable)
    return float(np.real(infidelity)), np.asarray(grad, float)


def _lagrangian_and_grad(driven_sys, x, *, observable, lam, S_max, subsystem, fd_eps, rho, eps_infidelity):
    """
    Augmented Lagrangian and gradient for entropy-constrained DQE optimization.

    Lagrangian: L(x, λ) = S_cost(x) + λ(I(x) - ε) + (ρ/2)(I(x) - ε)²

    Parameters
    ----------
    lam : float
        Lagrange multiplier for infidelity constraint.
    subsystem : tuple[int, ...]
        Bipartite entropy subsystem.
    rho : float
        Augmented Lagrangian penalty parameter.
    eps_infidelity : float
        Target infidelity threshold ε.

    Returns
    -------
    L : float
        Lagrangian value.
    grad_L : np.ndarray
        Lagrangian gradient.
    """
    S = _cost_entropy(driven_sys, x, S_max=S_max, subsystem=subsystem)
    gS = _entropy_grad_fd(driven_sys, x, S_max=S_max, subsystem=subsystem, eps=fd_eps)

    infidelity, grad_infidelity = _infidelity_and_grad(driven_sys, x, observable)
    constraint = infidelity - eps_infidelity

    L = S + lam * constraint
    grad_L = gS + lam * grad_infidelity

    if rho != 0.0:
        L += 0.5 * rho * (constraint ** 2)
        grad_L += rho * constraint * grad_infidelity

    return float(L), np.asarray(grad_L, float)


def _primal_dual_optimise(driven_sys, x0, *, observable, bounds, config: SimulationConfig, subsystem: tuple[int, ...], verbose=True):
    """
    Primal-dual optimization for entropy-constrained DQE gate synthesis.

    Minimizes bipartite entropy while constraining infidelity via augmented
    Lagrangian.

    Parameters
    ----------
    driven_sys : DrivenSystem
        System with pulse function.
    x0 : np.ndarray
        Initial parameters.
    observable : np.ndarray
        Infidelity observable.
    bounds : list
        Parameter bounds for L-BFGS-B.
    config : SimulationConfig
        Configuration with primal-dual settings.
    subsystem : tuple[int, ...]
        Bipartite entropy subsystem indices.
    verbose : bool
        Print progress.

    Returns
    -------
    x_final : np.ndarray
        Optimized parameters.
    lam_final : float
        Final Lagrange multiplier.
    history : list[dict]
        Per-iteration diagnostics.
    """
    x = np.asarray(x0, float).copy()
    lam = 1e-1
    hist = []

    for k in range(int(config.n_outer)):
        def fun_and_jac(z):
            return _lagrangian_and_grad(
                driven_sys, z,
                observable=observable,
                lam=lam,
                S_max=float(config.S_max),
                subsystem=subsystem,
                fd_eps=float(config.fd_eps),
                rho=float(config.rho),
                eps_infidelity=float(config.eps_infidelity),
            )

        res = minimize(
            fun_and_jac,
            x0=x,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(config.inner_maxiter), "gtol": float(config.gtol)},
        )
        x = np.asarray(res.x, float)

        infidelity, _ = _infidelity_and_grad(driven_sys, x, observable)
        constraint = infidelity - float(config.eps_infidelity)
        lam = max(0.0, lam + float(config.dual_lr) * constraint)

        hist.append({
            "outer_iter": k,
            "lam": lam,
            "I_final": infidelity,
            "constraint_residual": constraint,
            "inner_success": bool(res.success),
            "inner_nit": int(res.nit),
            "inner_fun": float(res.fun),
            "inner_grad_norm": float(np.linalg.norm(res.jac)) if res.jac is not None else np.nan,
            "message": str(res.message),
        })

        if verbose:
            print(f"[outer {k:02d}] I={infidelity:.3e}, c=I-eps={constraint:.3e}, lam={lam:.3e}, inner_nit={res.nit}, inner_success={res.success}")

    return x, lam, hist


# ========== Public API ==========

def dqe_optimisation(config: SimulationConfig) -> SQEResults:
    """
    Optimize double-qubit gate via exchange coupling pulses (4-spin chain).

    Analogous to sqe_optimisation but for 4 qubits with 3 nearest-neighbor
    exchange couplings. Supports three modes (config.run_mode):
    - "infidelity": Minimize infidelity only (baseline).
    - "entropy": Minimize bipartite entropy subject to infidelity constraint.
    - "both": Run both and return entropy-constrained with baseline attached.

    Parameters
    ----------
    config : SimulationConfig
        Full configuration. Key DQE-specific settings:
        - entropy_subsystem: Bipartition for entropy (default: (0,1) vs (2,3))
        - n_J_channels: Automatically set to 3 for DQE

    Returns
    -------
    SQEResults
        Optimization results with trajectories at fine discretization.
    """
    # Initialize random seed
    if config.init_seed is not None:
        np.random.seed(int(config.init_seed))
    elif config.drive_seed is not None or config.J_init_seed is not None:
        seed = config.drive_seed if config.drive_seed is not None else config.J_init_seed
        np.random.seed(int(seed))

    # DQE configuration
    config.max_drive_strength = 0.0
    config.n_J_channels = 3  # 3 nearest-neighbor exchange couplings

    # Bipartite entropy subsystem (default: qubits 0,1 vs 2,3)
    subsystem = tuple(config.entropy_subsystem) if config.entropy_subsystem is not None else (0, 1)

    # Build target and observable
    U = dqe_unitary(config.theta_target)
    target_state = U @ config.initial_state
    observable = np.eye(16, dtype=np.complex128) - np.outer(target_state, target_state.conj())

    # Create 4-spin chain system
    sys = SpinChain(
        spins=4,
        zeeman_splittings=config.zeeman_splittings,
        max_drive_strength=0.0,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )

    # Pulse discretization
    n_steps = config.n_time_steps
    n_disc = config.n_discretisation
    T = config.T

    def pulse_function(x):
        """
        Convert coarse control parameters to fine pulse amplitudes.

        Parameters
        ----------
        x : tf.Tensor or np.ndarray
            Coarse parameters, shape (n_steps × 3,) for 3 couplings.

        Returns
        -------
        tuple
            (drive_amp, freqs, J_amp, init_state, dt) for QuGrad.
        """
        # Reshape to coarse grid and expand to fine resolution
        x_tf = convert_dtype(x)
        x_coarse = tf.reshape(x_tf, (n_steps, 3))
        x_fine = tf.repeat(x_coarse, repeats=n_disc, axis=0)

        n_fine = n_steps * n_disc
        # Convert dt from ns to μs for qugradlab (frequencies in MHz, time in μs)
        dt_fine_us = (T / n_fine) * NS_TO_US

        drive_ctrl_amp = convert_dtype(np.zeros((n_fine, 4)))
        J_ctrl_amp = x_fine
        init = convert_dtype(config.initial_state)
        freqs = convert_dtype(config.zeeman_splittings)

        return drive_ctrl_amp, freqs, J_ctrl_amp, init, dt_fine_us

    driven_sys = sys.pulse_form(pulse_function)

    # Initialize controls and bounds
    n_params = n_steps * 3
    x0 = config.J_init_scale * np.random.uniform(-1.0, 1.0, size=n_params)
    x0 = np.clip(x0, -1.0, 1.0)
    bounds = [(-1.0, 1.0)] * len(x0)

    def package_results(x_final, *, success, nit, grad_norm, lam_final=None, history=None, baseline=None):
        """Package DQE optimization results."""
        val0 = np.real(driven_sys.evolved_expectation_value_all(x0, observable))
        val_final = np.real(driven_sys.evolved_expectation_value_all(x_final, observable))
        traj0 = driven_sys.propagate_all(x0)
        traj_final = driven_sys.propagate_all(x_final)

        return SQEResults(
            config=config,
            x0=np.asarray(x0, float),
            x_final=np.asarray(x_final, float),
            val0=np.asarray(val0, float),
            val_final=np.asarray(val_final, float),
            initial_traj=np.asarray(traj0),
            final_traj=np.asarray(traj_final),
            gradient_norm=float(grad_norm),
            n_iterations=int(nit),
            optimization_success=bool(success),
            lam_final=lam_final,
            history=history,
            baseline=baseline,
        )

    def run_baseline():
        """Run infidelity-only optimization (baseline)."""
        res = minimize(
            driven_sys.gradient,
            x0=x0,
            args=(observable,),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": int(config.maxiter), "gtol": float(config.gtol)},
        )
        grad_norm = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        return package_results(
            np.asarray(res.x, float),
            success=bool(res.success),
            nit=int(res.nit),
            grad_norm=grad_norm
        )

    def run_entropy():
        """Run entropy-constrained optimization via primal-dual."""
        x_final, lam_final, hist = _primal_dual_optimise(
            driven_sys, x0,
            observable=observable,
            bounds=bounds,
            config=config,
            subsystem=subsystem,
            verbose=True,
        )
        _, grad_final = driven_sys.gradient(x_final, observable)
        grad_norm = float(np.linalg.norm(np.asarray(grad_final, float)))
        success = bool(hist) and (float(hist[-1]["constraint_residual"]) <= 0.0)
        return package_results(
            x_final,
            success=success,
            nit=len(hist),
            grad_norm=grad_norm,
            lam_final=float(lam_final),
            history=hist
        )

    # Execute based on run mode
    if config.run_mode == "infidelity":
        return run_baseline()

    if config.run_mode == "entropy":
        return run_entropy()

    if config.run_mode == "both":
        baseline = run_baseline()
        constrained = run_entropy()
        constrained.baseline = baseline
        return constrained

    raise ValueError(f"Unknown run_mode: {config.run_mode!r}")
