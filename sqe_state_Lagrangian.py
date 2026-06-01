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
from scipy.optimize import minimize, Bounds, NonlinearConstraint

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
# Entropy-constrained optimisation
# =============================================================================

def _cost_entropy(driven_sys, x, *, S_max: float, ent_traj_fn) -> float:
    """
    Mean squared trajectory violation of the entanglement cap.

    cost = mean((E(t) − S_max)_+^2)
    """
    traj = np.asarray(driven_sys.propagate_all(x))
    viol = np.maximum(0.0, ent_traj_fn(traj) - S_max)
    return float(np.mean(viol**2))


def _entropy_grad_fd(driven_sys, x, *, S_max: float, eps: float,
                     ent_traj_fn) -> np.ndarray:
    """
    Central-difference gradient of the entropy cost.
    """
    x = np.asarray(x, float)
    g = np.empty_like(x)

    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps

        g[i] = (
            _cost_entropy(driven_sys, xp, S_max=S_max, ent_traj_fn=ent_traj_fn)
            - _cost_entropy(driven_sys, xm, S_max=S_max, ent_traj_fn=ent_traj_fn)
        ) / (2.0 * eps)

    return g


def _entropy_objective_and_grad(driven_sys, x, *, S_max, fd_eps, ent_traj_fn):
    """
    Objective for trust-constr:
        minimise S(x)
    """
    S = _cost_entropy(
        driven_sys,
        x,
        S_max=S_max,
        ent_traj_fn=ent_traj_fn,
    )

    gS = _entropy_grad_fd(
        driven_sys,
        x,
        S_max=S_max,
        eps=fd_eps,
        ent_traj_fn=ent_traj_fn,
    )

    return float(S), np.asarray(gS, float)


def _infidelity_constraint_value(driven_sys, x, *, observable):
    """
    Constraint value:
        I(x)
    """
    infidelity = driven_sys.gradient(x, observable)[0]
    return float(max(0.0, np.real(infidelity)))


def _infidelity_constraint_jac(driven_sys, x, *, observable):
    """
    Constraint Jacobian:
        dI/dx
    """
    _, grad_inf = driven_sys.gradient(x, observable)
    return np.asarray(grad_inf, float)


def _constrained_entropy_optimise(
    driven_sys,
    x0,
    *,
    observable,
    bounds,
    config: SimulationConfig,
    ent_traj_fn,
    verbose: bool = True,
):
    """
    Strict constrained optimisation:

        minimise_x S(x)
        subject to I(x) <= eps_infidelity

    Uses scipy trust-constr with a NonlinearConstraint.
    """

    x0 = np.asarray(x0, float).copy()

    eps_constraint = float(getattr(config, "eps_internal", config.eps_infidelity))
    fd_eps = float(getattr(config, "fd_eps", 1e-6))
    S_max = float(config.S_max)

    scipy_bounds = Bounds(
        lb=np.array([b[0] for b in bounds], dtype=float),
        ub=np.array([b[1] for b in bounds], dtype=float),
    )

    fidelity_constraint = NonlinearConstraint(
        fun=lambda x: _infidelity_constraint_value(
            driven_sys,
            x,
            observable=observable,
        ),
        lb=-np.inf,
        ub=eps_constraint,
        jac=lambda x: _infidelity_constraint_jac(
            driven_sys,
            x,
            observable=observable,
        ),
    )

    I0 = _infidelity_constraint_value(driven_sys, x0, observable=observable)
    S0 = _cost_entropy(driven_sys, x0, S_max=S_max, ent_traj_fn=ent_traj_fn)

    if verbose:
        print(
            f"[trust-constr start] "
            f"I0={I0:.3e}, S0={S0:.3e}, "
            f"constraint={eps_constraint:.3e}"
        )

    res = minimize(
        lambda x: _entropy_objective_and_grad(
            driven_sys,
            x,
            S_max=S_max,
            fd_eps=fd_eps,
            ent_traj_fn=ent_traj_fn,
        ),
        x0=x0,
        jac=True,
        method="trust-constr",
        bounds=scipy_bounds,
        constraints=[fidelity_constraint],
        options={
            "maxiter": int(config.maxiter),
            "gtol": float(config.gtol),
            "xtol": float(getattr(config, "xtol", 1e-10)),
            "barrier_tol": float(getattr(config, "barrier_tol", 1e-10)),
            "verbose": 3 if verbose else 0,
        },
    )

    xf = np.asarray(res.x, float)

    If = _infidelity_constraint_value(driven_sys, xf, observable=observable)
    Sf = _cost_entropy(driven_sys, xf, S_max=S_max, ent_traj_fn=ent_traj_fn)
    gS = _entropy_grad_fd(
        driven_sys,
        xf,
        S_max=S_max,
        eps=fd_eps,
        ent_traj_fn=ent_traj_fn,
    )

    hist = [{
        "success": bool(res.success),
        "message": str(res.message),
        "status": int(res.status),
        "nit": int(res.niter),
        "fun": float(res.fun),
        "S": float(Sf),
        "I_final": float(If),
        "constraint_residual": float(If - eps_constraint),
        "entropy_grad_norm": float(np.linalg.norm(gS)),
        "constr_violation": float(getattr(res, "constr_violation", np.nan)),
        "optimality": float(getattr(res, "optimality", np.nan)),
    }]

    if verbose:
        print(
            f"[trust-constr final] "
            f"I={If:.3e}, S={Sf:.3e}, "
            f"I-eps={If - eps_constraint:.3e}, "
            f"success={res.success}, "
            f"nit={res.niter}"
        )

    return xf, hist


# =============================================================================
# Public API
# =============================================================================

def sqe_optimisation(config: SimulationConfig) -> SQEResults:
    """
    Optimise a single-qubit excitation pulse.

    entanglement_optimisation = False:
        minimise final infidelity using L-BFGS-B.

    entanglement_optimisation = True:
        1. First obtain a high-fidelity baseline pulse.
        2. Starting from that feasible pulse, minimise transient entanglement
           subject to I(x) <= eps_infidelity using trust-constr.
    """
    config = SimulationConfig(**vars(config))
    config.n_J_channels = 1

    if config.state_ent_measure not in STATE_MEASURES:
        raise ValueError(
            f"Unknown state_ent_measure: {config.state_ent_measure!r}. "
            f"Valid: {list(STATE_MEASURES)}"
        )

    _, ent_traj_fn = STATE_MEASURES[config.state_ent_measure]

    sys, observable = _make_system_and_observable(config)
    driven_sys = _make_driven_sys(config, sys)

    x0 = _initialise_x0(config)
    bounds = [(-1.0, 1.0)] * len(x0)

    def run_baseline(x_init):
        res = minimize(
            driven_sys.gradient,
            x0=np.asarray(x_init, float),
            args=(observable,),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={
                "maxiter": int(config.maxiter),
                "gtol": float(config.gtol),
            },
        )

        gn = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        S = _cost_entropy(
            driven_sys,
            res.x,
            S_max=float(config.S_max),
            ent_traj_fn=ent_traj_fn,
        )

        print(
            f"Baseline: I={res.fun:.3e}, nit={res.nit}, "
            f"success={res.success}, grad_norm={gn:.3e}, S={S:.3e}"
        )

        return _package_results(
            config=config,
            driven_sys=driven_sys,
            observable=observable,
            x0=x_init,
            xf=np.asarray(res.x, float),
            success=bool(res.success),
            nit=int(res.nit),
            grad_norm=gn,
        )

    def run_entropy(x_init):
        xf, hist = _constrained_entropy_optimise(
            driven_sys,
            x_init,
            observable=observable,
            bounds=bounds,
            config=config,
            ent_traj_fn=ent_traj_fn,
            verbose=True,
        )

        infidelity, grad_inf = driven_sys.gradient(xf, observable)
        infidelity = max(0.0, float(np.real(infidelity)))
        gn = float(np.linalg.norm(np.asarray(grad_inf, float)))

        ok = (
            bool(hist)
            and float(hist[-1]["constraint_residual"]) <= 0.0
        )

        return _package_results(
            config=config,
            driven_sys=driven_sys,
            observable=observable,
            x0=x_init,
            xf=xf,
            success=ok,
            nit=int(hist[-1]["nit"]) if hist else 0,
            grad_norm=gn,
            lam_final=None,
            history=hist,
        )

    if not config.entanglement_optimisation:
        return run_baseline(x0)

    baseline = run_baseline(x0.copy())

    baseline_infidelity = float(baseline.val_final[-1])
    if baseline_infidelity > float(config.eps_infidelity):
        print(
            f"Warning: baseline is not feasible: "
            f"I={baseline_infidelity:.3e} > eps={float(config.eps_infidelity):.3e}. "
            f"Constrained optimisation may fail."
        )

    constrained = run_entropy(baseline.x_final.copy())
    constrained.baseline = baseline

    return constrained