"""
Single Qubit Excitation (SQE) optimization for 2-spin systems.

Optimizes piecewise-constant exchange coupling pulses to implement single-qubit
gates while optionally constraining entanglement entropy via primal-dual methods.
"""
from __future__ import annotations
from tabnanny import verbose

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, SQEResults, NS_TO_US


def single_qubit_excitation(theta: float) -> np.ndarray:
    """
    Single-qubit rotation gate in two-spin subspace.

    Rotates the |01⟩ ↔ |10⟩ subspace by angle theta while leaving |00⟩ and |11⟩
    unchanged. Basis order: [|00⟩, |01⟩, |10⟩, |11⟩].

    Parameters
    ----------
    theta : float
        Rotation angle (radians).

    Returns
    -------
    np.ndarray
        4×4 unitary gate matrix.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta), 0.0],
            [0.0, np.sin(theta),  np.cos(theta), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def convert_dtype(np_array) -> tf.Tensor:
    """Convert numpy array to TensorFlow complex128 tensor."""
    return tf.cast(tf.convert_to_tensor(np_array), dtype=tf.complex128)


def get_Sent_single(state: np.ndarray) -> float:
    """
    Compute entanglement entropy for a single 2-qubit pure state.

    Parameters
    ----------
    state : np.ndarray
        Pure state vector of length 4.

    Returns
    -------
    float
        Von Neumann entropy of qubit A (in bits).
    """
    psi = state.reshape(2, 2)
    rho_A = psi @ psi.conj().T
    eig = np.linalg.eigvalsh(rho_A)
    eig = np.clip(eig, 1e-15, 1.0)
    return float(-np.sum(eig * np.log2(eig)))


def get_Sent_traj(traj: np.ndarray) -> np.ndarray:
    """
    Compute entanglement entropy along a 2-qubit state trajectory.

    Parameters
    ----------
    traj : np.ndarray
        State trajectory, shape (4, n_time_points).

    Returns
    -------
    np.ndarray
        Entropy at each time point, shape (n_time_points,).
    """
    traj = np.asarray(traj)
    psi = traj.T.reshape(-1, 2, 2)
    rho_A = psi @ np.conj(psi).transpose(0, 2, 1)
    eig = np.linalg.eigvalsh(rho_A)
    eig = np.clip(eig, 1e-15, 1.0)
    return -np.sum(eig * np.log2(eig), axis=1)


# ========== Internal Helpers ==========

def _initialise_seed(config: SimulationConfig) -> None:
    """Initialize random seed from config."""
    if config.init_seed is not None:
        np.random.seed(int(config.init_seed))
        return
    if config.drive_seed is not None or config.J_init_seed is not None:
        seed = config.drive_seed if config.drive_seed is not None else config.J_init_seed
        np.random.seed(int(seed))


def _initialise_x0(config: SimulationConfig) -> np.ndarray:
    """
    Initialize control parameters.

    Returns
    -------
    np.ndarray
        Initial parameters in [-1, 1], shape (n_time_steps × n_J_channels,).
    """
    n_params = config.n_time_steps * config.n_J_channels
    if config.max_drive_strength == 0.0:
        #x0 = -1.0 + config.J_init_scale * np.random.uniform(-1.0, 1.0, size=n_params)
        #x0 = np.clip(x0, -1.0, 1.0)
        x0 = config.J_init_scale * np.random.uniform(-1.0, 1.0, size=n_params)
        return np.clip(x0, -1.0, 1.0)

    raise NotImplementedError("Drive controls not enabled in this configuration.")


def _make_system_and_observable(config: SimulationConfig):
    """
    Create spin system and infidelity observable for SQE.

    Returns
    -------
    sys : SpinChain
        2-spin chain system.
    observable : np.ndarray
        Infidelity observable: I - |ψ_target⟩⟨ψ_target|.
    """
    target_state = single_qubit_excitation(config.theta_target) @ config.initial_state
    observable = np.eye(4, dtype=np.complex128) - np.outer(target_state, target_state.conj())

    sys = SpinChain(
        spins=2,
        zeeman_splittings=config.zeeman_splittings,
        max_drive_strength=config.max_drive_strength,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )
    return sys, observable


def _make_driven_sys(config: SimulationConfig, sys: SpinChain):
    """
    Create driven system with fine discretization.

    Optimization parameters are at coarse resolution (n_time_steps), but state
    evolution uses fine resolution (n_time_steps × n_discretisation) via piecewise
    constant interpolation.

    Parameters
    ----------
    config : SimulationConfig
        Configuration with discretization parameters.
    sys : SpinChain
        Spin system to drive.

    Returns
    -------
    DrivenSystem
        System with pulse function for gradient computation.
    """
    n_steps = config.n_time_steps
    n_disc = config.n_discretisation
    T = config.T

    def pulse_function(x):
        """
        Convert coarse control parameters to fine pulse amplitudes.

        Parameters
        ----------
        x : tf.Tensor or np.ndarray
            Coarse parameters, shape (n_steps × n_J_channels,).

        Returns
        -------
        tuple
            (drive_amp, freqs, J_amp, init_state, dt) for QuGrad.
        """
        # Reshape to coarse grid and expand to fine resolution
        x_tf = convert_dtype(x)
        x_coarse = tf.reshape(x_tf, (n_steps, config.n_J_channels))
        x_fine = tf.repeat(x_coarse, repeats=n_disc, axis=0)

        n_fine = n_steps * n_disc
        # Convert dt from ns to μs for qugradlab (frequencies in MHz, time in μs)
        dt_fine_us = (T / n_fine) * NS_TO_US

        drive_ctrl_amp = convert_dtype(np.zeros((n_fine, 2)))
        J_ctrl_amp = x_fine
        init = convert_dtype(config.initial_state)
        freqs = convert_dtype(config.zeeman_splittings)

        return drive_ctrl_amp, freqs, J_ctrl_amp, init, dt_fine_us

    return sys.pulse_form(pulse_function)


def _package_results(*, config, driven_sys, observable, x0, xf, success, nit, grad_norm, lam_final=None, history=None, baseline=None):
    """Package optimization results into SQEResults dataclass."""
    val0 = np.real(driven_sys.evolved_expectation_value_all(x0, observable))
    valf = np.real(driven_sys.evolved_expectation_value_all(xf, observable))
    traj0 = driven_sys.propagate_all(x0)
    trajf = driven_sys.propagate_all(xf)

    return SQEResults(
        config=config,
        x0=np.asarray(x0, float),
        x_final=np.asarray(xf, float),
        val0=np.asarray(val0, float),
        val_final=np.asarray(valf, float),
        initial_traj=np.asarray(traj0),
        final_traj=np.asarray(trajf),
        gradient_norm=float(grad_norm),
        n_iterations=int(nit),
        optimization_success=bool(success),
        lam_final=lam_final,
        history=history,
        baseline=baseline,
    )


# ========== Entropy-Constrained Optimization ==========

def _cost_entropy(driven_sys, x, *, S_max: float) -> float:
    """
    Entropy constraint violation cost (squared, time-averaged).

    Returns
    -------
    float
        Mean squared violation: mean((S(t) - S_max)_+^2).
    """
    traj = np.asarray(driven_sys.propagate_all(x))
    S = np.asarray(get_Sent_traj(traj), float)
    viol = np.maximum(0.0, S - S_max)
    return float(np.mean(viol**2))


def _entropy_grad_fd(driven_sys, x, *, S_max: float, eps: float) -> np.ndarray:
    """Finite-difference gradient of entropy cost."""
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    f0 = _cost_entropy(driven_sys, x, S_max=S_max)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_cost_entropy(driven_sys, xp, S_max=S_max) - f0) / eps
    return g


def _infidelity_and_grad(driven_sys, x, observable):
    """Compute infidelity and gradient via automatic differentiation."""
    infidelity, grad = driven_sys.gradient(x, observable)
    return float(np.real(infidelity)), np.asarray(grad, float)


def _lagrangian_and_grad(driven_sys, x, *, observable, lam, S_max, fd_eps, rho, eps_infidelity):
    """
    Augmented Lagrangian and gradient for entropy-constrained optimization.

    Lagrangian: L(x, λ) = S_cost(x) + λ(I(x) - ε) + (ρ/2)(I(x) - ε)²

    Parameters
    ----------
    lam : float
        Lagrange multiplier for infidelity constraint.
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
    S = _cost_entropy(driven_sys, x, S_max=S_max)
    gS = _entropy_grad_fd(driven_sys, x, S_max=S_max, eps=fd_eps)

    infidelity, grad_infidelity = _infidelity_and_grad(driven_sys, x, observable)
    constraint = infidelity - eps_infidelity

    L = S + lam * constraint
    grad_L = gS + lam * grad_infidelity

    if rho != 0.0:
        L += 0.5 * rho * (constraint ** 2)
        grad_L += rho * constraint * grad_infidelity

    return float(L), np.asarray(grad_L, float)


def _primal_dual_optimise(driven_sys, x0, *, observable, bounds, config: SimulationConfig, verbose=True):
    """
    Primal-dual optimization for entropy-constrained gate synthesis.

    Minimizes entropy while constraining infidelity via augmented Lagrangian.

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
    lam = 1e0
    hist = []

    for k in range(int(config.n_outer)):
        def fun_and_jac(z):
            return _lagrangian_and_grad(
                driven_sys, z,
                observable=observable,
                lam=lam,
                S_max=float(config.S_max),
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
            print(f"[outer {k:02d}] I={infidelity:.3e}, lam={lam:.6e}, inner_nit={res.nit}, inner_success={res.success}")

    return x, lam, hist


# ========== Public API ==========

def sqe_optimisation(config: SimulationConfig) -> SQEResults:
    """
    Optimize single-qubit gate via exchange coupling pulses.

    Supports three modes (config.run_mode):
    - "infidelity": Minimize infidelity only (baseline).
    - "entropy": Minimize entropy subject to infidelity constraint.
    - "both": Run both and return entropy-constrained with baseline attached.

    Parameters
    ----------
    config : SimulationConfig
        Full configuration including system, optimization, and discretization
        parameters.

    Returns
    -------
    SQEResults
        Optimization results with trajectories at fine discretization.
    """
    _initialise_seed(config)

    # SQE uses 1 exchange coupling channel
    config.n_J_channels = 1

    # Build system and target observable
    sys, observable = _make_system_and_observable(config)
    driven_sys = _make_driven_sys(config, sys)

    # Initialize controls and bounds
    x0 = _initialise_x0(config)
    bounds = [(-1.0, 1.0)] * len(x0)

    def run_baseline(x_init):
        """Run infidelity-only optimization (baseline)."""
        res = minimize(
            driven_sys.gradient,
            x0=x_init,
            args=(observable,),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": int(config.maxiter), "gtol": float(config.gtol)},
        )
        grad_norm = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0

        #final_state = driven_sys.propagate_all(res.x)[:, -1]
        #print(final_state)
        #print(np.abs(final_state[1]/np.abs(final_state[2]))) 
        #print(np.angle(final_state[1]/np.angle(final_state[2])))

        print("Infidelity optimisation results:")
        print(f"I={res.fun:.3e}, nit={res.nit}, success={res.success}, grad_norm={grad_norm}")

        return _package_results(
            config=config, driven_sys=driven_sys, observable=observable,
            x0=x_init, xf=np.asarray(res.x, float),
            success=bool(res.success), nit=int(res.nit), grad_norm=grad_norm,
        )

    def run_entropy(x_init):
        """Run entropy-constrained optimization via primal-dual."""
        x_final, lam_final, hist = _primal_dual_optimise(
            driven_sys, x_init,
            observable=observable,
            bounds=bounds,
            config=config,
            verbose=True,
        )
        _, grad_final = driven_sys.gradient(x_final, observable)
        grad_norm = float(np.linalg.norm(np.asarray(grad_final, float)))
        success = bool(hist) and (float(hist[-1]["constraint_residual"]) <= 0.0)
        return _package_results(
            config=config, driven_sys=driven_sys, observable=observable,
            x0=x_init, xf=x_final,
            success=success, nit=len(hist), grad_norm=grad_norm,
            lam_final=float(lam_final), history=hist,
        )

    # Execute based on run mode
    if config.run_mode == "infidelity":
        return run_baseline(x0)

    if config.run_mode == "entropy":
        return run_entropy(x0)

    if config.run_mode == "both":
        baseline = run_baseline(x0.copy())
        constrained = run_entropy(x0.copy())
        constrained.baseline = baseline
        return constrained

    raise ValueError(f"Unknown run_mode: {config.run_mode!r}")
