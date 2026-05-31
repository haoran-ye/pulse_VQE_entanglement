"""
Single Qubit Excitation (SQE) state optimization for 2-spin systems.

Physics
-------
A 2-spin chain (semiconductor spin qubits) is driven by a single piecewise-
constant exchange coupling J(t).  The target operation is the SQE gate: a
rotation by theta_target in the {|01⟩, |10⟩} subspace, acting as identity
on |00⟩ and |11⟩.

The system is built in the rotating frame of the average Zeeman splitting.
Free Zeeman precession at mean(B) is removed; only the relative detunings
deltaB = B − mean(B) appear in the drift Hamiltonian.  The frame unitary
U_rot dresses the target state so the infidelity is computed in that frame.

Algorithm
---------
entanglement_optimisation = False
    Minimize state infidelity

        I = ⟨ψ(T) | I − |ψ_target⟩⟨ψ_target| | ψ(T)⟩

    via L-BFGS-B using QuGrad autodiff gradients.

entanglement_optimisation = True
    1. Run the baseline (infidelity-only) optimisation to convergence.
    2. Run the entropy-constrained primal-dual loop from the same x0.
    The baseline result is stored in results.baseline for comparison.

Entropy cost (primal-dual)
--------------------------
    cost(x) = mean_t( (E(t) − S_max)₊² )

where E(t) is the state entanglement measure at time t, (·)₊ = max(0, ·)
is the hinge function, and S_max is the entanglement cap.

Augmented Lagrangian
--------------------
    L(x, λ) = cost(x) + λ(I(x) − ε) + (ρ/2)(I(x) − ε)²

Dual update after each inner solve:
    λ ← max(0, λ + α(I − ε))

The max(0, ·) enforces non-negativity of λ (KKT condition).

Entanglement measure
--------------------
Set config.state_ent_measure to any key in entanglement_measures.STATE_MEASURES.
Currently supported: "von_neumann" (default), "linear".
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, SQEResults, NS_TO_US
from .entanglement_measures import STATE_MEASURES
from ._utils import (
    convert_dtype,
    _initialise_x0,
    single_qubit_excitation,
    make_rotating_frame,
)

# =============================================================================
# System construction
# =============================================================================

def _make_system_and_observable(config: SimulationConfig):
    """
    Build the 2-spin SpinChain in the rotating frame and the infidelity observable.

    Returns
    -------
    sys : SpinChain
        2-spin system with drift Hamiltonian using deltaB (relative detunings).
    observable : ndarray, shape (4, 4)
        I − |ψ_target⟩⟨ψ_target|.  Minimising ⟨ψ(T)|obs|ψ(T)⟩ is equivalent
        to maximising the overlap with the target state.
    """
    # Build rotating frame: deltaB = B − mean(B); U_rot = expm(−iπ H_frame T·1e-3)
    deltaB, U_rot = make_rotating_frame(config.zeeman_splittings, config.T)

    # Target state in the rotating frame: apply gate, then dress with U_rot
    target_state = U_rot @ single_qubit_excitation(config.theta_target) @ config.initial_state
    # Projector onto the complement of |ψ_target⟩; expectation value = infidelity
    observable   = np.eye(4, dtype=np.complex128) - np.outer(target_state, target_state.conj())

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

    The optimiser works on a coarse grid of n_time_steps parameters per J channel.
    Each parameter is held constant for n_discretisation fine steps, giving a
    total of n_fine = n_time_steps × n_discretisation integration points.
    """
    n_steps    = config.n_time_steps
    n_disc     = config.n_discretisation
    n_fine     = n_steps * n_disc
    # Convert gate time from ns to µs (qugradlab time unit); × 2π for rad/µs
    dt_fine_us = (config.T / n_fine) * NS_TO_US

    def pulse_function(x):
        # x is a flat vector of length n_steps × n_J_channels in [-1, 1]
        x_coarse = tf.reshape(convert_dtype(x), (n_steps, config.n_J_channels))
        # Repeat each coarse step n_disc times → piecewise-constant fine grid
        x_fine   = tf.repeat(x_coarse, repeats=n_disc, axis=0)
        return (
            convert_dtype(np.zeros((n_fine, 2))),       # drive amplitudes (disabled)
            convert_dtype(config.zeeman_splittings),    # drive frequencies (needed by SpinChain API)
            x_fine,                                     # J coupling amplitudes on fine grid
            convert_dtype(config.initial_state),        # |ψ₀⟩
            dt_fine_us * 2 * np.pi,                     # timestep in rad/µs
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

    cost = mean((E(t) − S_max)₊²) where E is computed by ent_traj_fn.
    """
    traj = np.asarray(driven_sys.propagate_all(x))
    viol = np.maximum(0.0, ent_traj_fn(traj) - S_max)
    return float(np.mean(viol**2))


def _entropy_grad_fd(driven_sys, x, *, S_max: float, eps: float,
                     ent_traj_fn) -> np.ndarray:
    """
    Central-difference gradient of the entropy cost (O(eps²) error).
    """
    x = np.asarray(x, float)
    g = np.empty_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (
            _cost_entropy(driven_sys, xp, S_max=S_max, ent_traj_fn=ent_traj_fn)
            - _cost_entropy(driven_sys, xm, S_max=S_max, ent_traj_fn=ent_traj_fn)
        ) / (2.0 * eps)
    return g


def _lagrangian_and_grad(driven_sys, x, *, observable, lam, S_max, fd_eps,
                         rho, eps_infidelity, ent_traj_fn):
    """
    Augmented Lagrangian value and gradient.

    L(x, λ) = S_cost(x) + λ·c + (ρ/2)·c²,  c = I(x) − ε

    where S_cost is the mean-squared entanglement hinge violation (primary
    objective), I is the state infidelity (the constraint I ≤ ε enforced via λ),
    and c is the constraint residual.

    Gradient: ∇L = ∇S + (λ + ρ·c)·∇I
    """
    # Primary objective: entanglement cost and its finite-difference gradient
    S  = _cost_entropy(driven_sys, x, S_max=S_max, ent_traj_fn=ent_traj_fn)
    gS = _entropy_grad_fd(driven_sys, x, S_max=S_max, eps=fd_eps, ent_traj_fn=ent_traj_fn)

    # Constraint function: infidelity and its autodiff gradient (QuGrad)
    infidelity, grad_inf = driven_sys.gradient(x, observable)
    infidelity = max(0.0, float(np.real(infidelity)))   # guard numerical noise from complex arithmetic
    grad_inf   = np.asarray(grad_inf, float)
    c = infidelity - eps_infidelity                      # constraint residual I − ε

    # Augmented Lagrangian and its gradient
    L  = S + lam * c + 0.5 * rho * c**2
    gL = gS + (lam + rho * c) * grad_inf
    return float(L), gL


def _primal_dual_optimise(driven_sys, x0, *, observable, bounds, config: SimulationConfig,
                          ent_traj_fn, verbose: bool = True):
    """
    Augmented Lagrangian primal-dual loop.

    Each outer iteration minimises the Lagrangian via L-BFGS-B, then updates
    the dual variable λ ← max(0, λ + α(I − ε)).

    Returns
    -------
    x_final : np.ndarray
    lam_final : float
    history : list[dict]  – per-outer-iteration diagnostics
    """
    x   = np.asarray(x0, float).copy()
    lam = 1.0
    hist: list[dict] = []

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

        S          = _cost_entropy(driven_sys, x, S_max=float(config.S_max), ent_traj_fn=ent_traj_fn)
        infidelity = np.real(driven_sys.gradient(x, observable)[0])
        c          = infidelity - float(config.eps_infidelity)
        # Dual ascent: λ ← max(0, λ + α(I−ε)); max(0,·) is the KKT projection
        lam        = max(0.0, lam + float(config.dual_lr) * c)

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
    config = SimulationConfig(**vars(config))   # local copy — do not mutate caller's config
    config.n_J_channels = 1

    if config.state_ent_measure not in STATE_MEASURES:
        raise ValueError(f"Unknown state_ent_measure: {config.state_ent_measure!r}. "
                         f"Valid: {list(STATE_MEASURES)}")
    _, ent_traj_fn = STATE_MEASURES[config.state_ent_measure]

    sys, observable = _make_system_and_observable(config)
    driven_sys      = _make_driven_sys(config, sys)

    x0     = _initialise_x0(config)
    bounds = [(-1.0, 1.0)] * len(x0)

    def run_baseline(x_init):
        res = minimize(
            driven_sys.gradient, x0=x_init, args=(observable,),
            jac=True, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": int(config.maxiter), "gtol": float(config.gtol)},
        )
        gn = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        S  = _cost_entropy(driven_sys, res.x, S_max=float(config.S_max), ent_traj_fn=ent_traj_fn)
        print(f"Infidelity: I={res.fun:.3e}, nit={res.nit}, success={res.success}, "
              f"grad_norm={gn:.3e}, S={S:.3e}")
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

    baseline   = run_baseline(x0.copy())
    constrained = run_entropy(x0.copy())
    constrained.baseline = baseline
    return constrained
