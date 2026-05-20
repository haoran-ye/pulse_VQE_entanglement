"""
Single Qubit Excitation (SQE) optimization for 2-spin systems.

Optimizes piecewise-constant exchange coupling pulses to implement single-qubit
gates while optionally constraining entanglement entropy via primal-dual methods.

entanglement_optimisation flag
-------------------------------
False : Minimize state infidelity only (uses QuGrad autodiff gradient).
True  : First minimize infidelity, then run the entropy-constrained primal-dual
        loop from the same x0; baseline result stored in results.baseline.

Entanglement measure
--------------------
Set config.state_ent_measure to any key in entanglement_measures.STATE_MEASURES.
Currently supported: "von_neumann" (default), "linear".
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from scipy.linalg import expm

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, SQEResults, NS_TO_US
from .entanglement_measures import (
    STATE_MEASURES,
    von_neumann_entropy_single, von_neumann_entropy_traj,
    linear_entropy_single, linear_entropy_traj,
)

# =============================================================================
# System construction
# =============================================================================

def convert_dtype(np_array) -> tf.Tensor:
    """Convert numpy array to TensorFlow complex128 tensor."""
    return tf.cast(tf.convert_to_tensor(np_array), dtype=tf.complex128)


def single_qubit_excitation(theta: float) -> np.ndarray:
    """
    Single-qubit rotation gate in the two-spin subspace.

    Rotates the |01⟩ ↔ |10⟩ subspace by angle theta while leaving |00⟩ and
    |11⟩ unchanged. Basis order: [|00⟩, |01⟩, |10⟩, |11⟩].
    """
    return np.array(
        [[1, 0, 0, 0],
         [0, np.cos(theta), -np.sin(theta), 0],
         [0, np.sin(theta),  np.cos(theta), 0],
         [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def _initialise_seed(config: SimulationConfig) -> None:
    if config.J_init_seed is not None:
        np.random.seed(int(config.J_init_seed))


def _initialise_x0(config: SimulationConfig) -> np.ndarray:
    """Initial control parameters drawn uniformly in [-1, 1]."""
    if getattr(config, "x0_override", None) is not None:
        x0 = np.asarray(config.x0_override, dtype=float).copy()
    else:
        n = int(config.n_time_steps) * int(config.n_J_channels)
        x0 = config.J_init_mean + float(config.J_init_scale) * np.random.uniform(-1.0, 1.0, size=n)
    return np.clip(x0, -1.0, 1.0).astype(np.float64, copy=False)


def _make_system_and_observable(config: SimulationConfig):
    """
    Create the 2-spin system and the infidelity observable I - |ψ_target⟩⟨ψ_target|.

    The system is constructed in the rotating frame (symmetric Zeeman splittings
    ±ΔB/2) and the target state is dressed by the corresponding frame unitary.
    """
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    Z1 = np.kron(Z, I2)
    Z2 = np.kron(I2, Z)

    B = np.array(config.zeeman_splittings, dtype=np.float64)
    B_avg = np.mean(B)
    deltaB = B - B_avg
    U_rot = expm(-1j * np.pi 
                 * (deltaB[0] * Z1 + deltaB[1] * Z2) 
                 * config.T * NS_TO_US)

    target_state = U_rot @ single_qubit_excitation(config.theta_target) @ config.initial_state
    observable = np.eye(4, dtype=np.complex128) - np.outer(target_state, target_state.conj())

    sys = SpinChain(
        spins=2,
        zeeman_splittings=deltaB,
        max_drive_strength=config.max_drive_strength,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )
    return sys, observable


def _make_driven_sys(config: SimulationConfig, sys: SpinChain):
    """
    Create a driven system with piecewise-constant pulse interpolation.

    Optimization parameters live on a coarse grid (n_time_steps); state
    evolution uses a finer grid (n_time_steps × n_discretisation).
    """
    n_steps = config.n_time_steps
    n_disc = config.n_discretisation
    n_fine = n_steps * n_disc
    dt_fine_us = (config.T / n_fine) * NS_TO_US

    def pulse_function(x):
        x_coarse = tf.reshape(convert_dtype(x), (n_steps, config.n_J_channels))
        x_fine = tf.repeat(x_coarse, repeats=n_disc, axis=0)
        return (
            convert_dtype(np.zeros((n_fine, 2))),       # drive amplitudes (disabled)
            convert_dtype(config.zeeman_splittings),    # drive frequencies
            x_fine,                                     # J coupling amplitudes
            convert_dtype(config.initial_state),        # initial state
            dt_fine_us * 2 * np.pi,                     # timestep (rad/μs)
        )

    return sys.pulse_form(pulse_function)


def _package_results(*, config, driven_sys, observable, x0, xf, success, nit,
                     grad_norm, lam_final=None, history=None, baseline=None) -> SQEResults:
    """Compute expectation value and trajectory for x0 and xf, pack into SQEResults."""
    return SQEResults(
        config=config,
        x0=np.asarray(x0, float),
        x_final=np.asarray(xf, float),
        val0=np.asarray(np.real(driven_sys.evolved_expectation_value_all(x0, observable)), float),
        val_final=np.asarray(np.real(driven_sys.evolved_expectation_value_all(xf, observable)), float),
        initial_traj=np.asarray(driven_sys.propagate_all(x0)),
        final_traj=np.asarray(driven_sys.propagate_all(xf)),
        gradient_norm=float(grad_norm),
        n_iterations=int(nit),
        optimization_success=bool(success),
        lam_final=lam_final,
        history=history,
        baseline=baseline,
    )


# =============================================================================
# Entropy-constrained (primal-dual) optimisation
# =============================================================================

def _cost_entropy(driven_sys, x, *, S_max: float, ent_traj_fn) -> float:
    """
    Mean squared trajectory violation of the entanglement cap.

    cost = mean((E(t) - S_max)₊²) where E is computed by ent_traj_fn.

    Parameters
    ----------
    ent_traj_fn : callable
        A traj_fn from STATE_MEASURES: takes (4, n_t) trajectory, returns (n_t,).
    """
    traj = np.asarray(driven_sys.propagate_all(x))
    viol = np.maximum(0.0, ent_traj_fn(traj) - S_max)
    return float(np.mean(viol**2))


def _entropy_grad_fd(driven_sys, x, *, S_max: float, eps: float,
                     ent_traj_fn, f0: float | None = None) -> np.ndarray:
    """
    Forward-difference gradient of the entropy cost.

    Parameters
    ----------
    ent_traj_fn : callable
        Trajectory entanglement function; see _cost_entropy.
    f0 : float, optional
        Cost at the unperturbed x. Providing it avoids a redundant evaluation.
    """
    x = np.asarray(x, float)
    if f0 is None:
        f0 = _cost_entropy(driven_sys, x, S_max=S_max, ent_traj_fn=ent_traj_fn)
    g = np.empty_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_cost_entropy(driven_sys, xp, S_max=S_max, ent_traj_fn=ent_traj_fn) - f0) / eps
    return g


def _lagrangian_and_grad(driven_sys, x, *, observable, lam, S_max, fd_eps,
                         rho, eps_infidelity, ent_traj_fn):
    """
    Augmented Lagrangian value and gradient.

    Objective: L = S_cost + λ(I - ε) + (ρ/2)(I - ε)²

    where S_cost is the mean-squared entanglement violation (using ent_traj_fn)
    and I is the state infidelity (the constraint I ≤ ε is enforced via λ).
    """
    S = _cost_entropy(driven_sys, x, S_max=S_max, ent_traj_fn=ent_traj_fn)
    gS = _entropy_grad_fd(driven_sys, x, S_max=S_max, eps=fd_eps, ent_traj_fn=ent_traj_fn, f0=S)

    infidelity, grad_inf = driven_sys.gradient(x, observable)
    infidelity = max(0.0, float(np.real(infidelity)))   # guard against float64 cancellation
    grad_inf = np.asarray(grad_inf, float)
    c = infidelity - eps_infidelity

    L = S + lam * c + 0.5 * rho * c**2
    gL = gS + (lam + rho * c) * grad_inf
    return float(L), gL


def _primal_dual_optimise(driven_sys, x0, *, observable, bounds, config: SimulationConfig,
                          ent_traj_fn, verbose: bool = True):
    """
    Augmented Lagrangian primal-dual loop.

    Each outer iteration minimises the Lagrangian via L-BFGS-B, then updates
    the dual variable λ ← max(0, λ + α(I - ε)).

    Parameters
    ----------
    ent_traj_fn : callable
        Trajectory entanglement function resolved from config.state_ent_measure.

    Returns
    -------
    x_final : np.ndarray
    lam_final : float
    history : list[dict]  – per-outer-iteration diagnostics
    """
    x = np.asarray(x0, float).copy()
    lam = 1.0
    hist = []

    for k in range(int(config.n_outer)):
        res = minimize(
            lambda z: _lagrangian_and_grad(
                driven_sys, z, observable=observable, lam=lam,
                S_max=float(config.S_max), fd_eps=float(config.fd_eps),
                rho=float(config.rho), eps_infidelity=float(config.eps_infidelity),
                ent_traj_fn=ent_traj_fn,
            ),
            x0=x, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": int(config.inner_maxiter), "gtol": float(config.gtol)},
        )
        x = np.asarray(res.x, float)

        S = _cost_entropy(driven_sys, x, S_max=float(config.S_max), ent_traj_fn=ent_traj_fn)

        infidelity = np.real(driven_sys.gradient(x, observable)[0])
        c = infidelity - float(config.eps_infidelity)
        lam = max(0.0, lam + float(config.dual_lr) * c)

        hist.append({
            "outer_iter": k, "lam": lam, "I_final": infidelity, "constraint_residual": c, "S": S,
            "inner_success": bool(res.success), "inner_nit": int(res.nit),
            "inner_fun": float(res.fun),
            "inner_grad_norm": float(np.linalg.norm(res.jac)) if res.jac is not None else np.nan,
            "message": str(res.message),
        })
        if verbose:
            print(f"[outer {k:02d}] I={infidelity:.3e}, lam={lam:.6e}, "
                  f"inner_nit={res.nit}, inner_success={res.success}, "
                  f"constraint_residual={c:.3e}, S={S:.3e}")

    return x, lam, hist


# =============================================================================
# Public API
# =============================================================================

def sqe_optimisation(config: SimulationConfig) -> SQEResults:
    """
    Optimize a single-qubit gate via exchange coupling pulses (state formulation).

    The entanglement measure used for the entropy constraint is determined by
    config.state_ent_measure (default: "von_neumann"). Any key in
    entanglement_measures.STATE_MEASURES is valid.

    Parameters
    ----------
    config : SimulationConfig
        Full configuration. Key fields: entanglement_optimisation,
        state_ent_measure, n_time_steps, T, theta_target, zeeman_splittings,
        initial_state, S_max, eps_infidelity.

    Returns
    -------
    SQEResults
    """
    _initialise_seed(config)
    config.n_J_channels = 1

    # Resolve entanglement measure once
    if config.state_ent_measure not in STATE_MEASURES:
        raise ValueError(f"Unknown state_ent_measure: {config.state_ent_measure!r}. "
                         f"Valid: {list(STATE_MEASURES)}")
    _, ent_traj_fn = STATE_MEASURES[config.state_ent_measure]

    sys, observable = _make_system_and_observable(config)
    driven_sys = _make_driven_sys(config, sys)

    x0 = _initialise_x0(config)
    bounds = [(-1.0, 1.0)] * len(x0)

    def run_baseline(x_init):
        res = minimize(
            driven_sys.gradient, x0=x_init, args=(observable,),
            jac=True, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": int(config.maxiter), "gtol": float(config.gtol)},
        )
        gn = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        S = _cost_entropy(driven_sys, res.x, S_max=float(config.S_max), ent_traj_fn=ent_traj_fn)
        print(f"Infidelity: I={res.fun:.3e}, nit={res.nit}, success={res.success}, grad_norm={gn:.3e}, S={S:.3e}")
        return _package_results(
            config=config, driven_sys=driven_sys, observable=observable,
            x0=x_init, xf=np.asarray(res.x, float),
            success=bool(res.success), nit=int(res.nit), grad_norm=gn,
        )

    def run_entropy(x_init):
        xf, lam, hist = _primal_dual_optimise(
            driven_sys, x_init, observable=observable, bounds=bounds,
            config=config, ent_traj_fn=ent_traj_fn, verbose=True,
        )
        gn = float(np.linalg.norm(np.asarray(driven_sys.gradient(xf, observable)[1], float)))
        ok = bool(hist) and float(hist[-1]["constraint_residual"]) <= 0.0
        return _package_results(
            config=config, driven_sys=driven_sys, observable=observable,
            x0=x_init, xf=xf, success=ok, nit=len(hist), grad_norm=gn,
            lam_final=float(lam), history=hist,
        )

    if not config.entanglement_optimisation:
        return run_baseline(x0)

    baseline = run_baseline(x0.copy())
    constrained = run_entropy(x0.copy())
    constrained.baseline = baseline
    return constrained
