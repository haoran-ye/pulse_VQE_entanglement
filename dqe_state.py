"""
Double Qubit Excitation (DQE) optimization for 4-spin systems.

Optimizes piecewise-constant exchange coupling pulses (3 channels for a
nearest-neighbor chain) to implement a target DQE gate — a rotation in the
|0011⟩ ↔ |1100⟩ subspace — while optionally minimizing multipartite
entanglement via primal-dual methods.

entanglement_optimisation flag
-------------------------------
False : Minimize state infidelity only (uses QuGrad autodiff gradient).
True  : First minimize infidelity, then run the entropy-constrained primal-dual
        loop from the same x0; baseline result stored in results.baseline.

Frame transformation
--------------------
The system is constructed in the rotating frame of the average Zeeman splitting,
while the infidelity observable is constructed in the frame of each qubit.

Entanglement cost
-----------------
The cost function is the mean-squared violation of a cap on the *sum* of von
Neumann entropies over all seven distinct bipartite partitions of 4 qubits
(four single-qubit cuts + three two-qubit cuts).  Controlled by config.S_max.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, DQEResults, NS_TO_US
from .entanglement_measures import (
    entropy_bipartite_single,
    entropy_bipartite_traj,
    entropy_sum_single,
    entropy_sum_traj,
    ALL_BIPARTITIONS_4Q,
)
from ._utils import (
    convert_dtype,
    _initialise_x0,
    dqe_unitary,
    make_rotating_frame,
)


# =============================================================================
# Utilities
# =============================================================================

def basis_state(n_qubits: int, bitstring: str) -> np.ndarray:
    """Create a computational basis state from a binary string."""
    if len(bitstring) != n_qubits:
        raise ValueError("bitstring length must equal n_qubits")
    v = np.zeros(2 ** n_qubits, dtype=np.complex128)
    v[int(bitstring, 2)] = 1.0
    return v


# =============================================================================
# System construction
# =============================================================================

def _make_system_and_observable(config: SimulationConfig):
    """
    Build the rotating-frame SpinChain and the infidelity observable.

    The system is constructed in the rotating frame of the average Zeeman
    splitting; the observable is constructed in the frame of each qubit.
    """
    deltaB, U_rot = make_rotating_frame(config.zeeman_splittings, config.T)

    target_state = U_rot @ dqe_unitary(config.theta_target) @ config.initial_state
    observable   = np.eye(16, dtype=np.complex128) - np.outer(target_state, target_state.conj())

    sys = SpinChain(
        spins=4,
        zeeman_splittings=deltaB,
        max_drive_strength=0.0,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )
    return sys, observable


def _make_driven_sys(config: SimulationConfig, sys: SpinChain):
    """
    Create a driven system with piecewise-constant interpolation.

    The J channels are packed as x.reshape(n_time_steps, n_J_channels) on the
    coarse grid and repeated n_discretisation times to obtain the fine grid.
    """
    n_steps    = config.n_time_steps
    n_disc     = config.n_discretisation
    n_fine     = n_steps * n_disc
    dt_fine_us = (config.T / n_fine) * NS_TO_US

    def pulse_function(x):
        x_coarse = tf.reshape(convert_dtype(x), (n_steps, config.n_J_channels))
        x_fine   = tf.repeat(x_coarse, repeats=n_disc, axis=0)
        return (
            convert_dtype(np.zeros((n_fine, 4))),    # drive amplitudes (disabled)
            tf.zeros(4, dtype=tf.complex128),        # drive frequencies
            x_fine,                                  # J-channel amplitudes
            convert_dtype(config.initial_state),     # initial state
            dt_fine_us * 2 * np.pi,                  # timestep (rad/µs)
        )

    return sys.pulse_form(pulse_function)


def _package_results(*, config, driven_sys, observable, x0, xf, success, nit,
                     grad_norm, lam_final=None, history=None, baseline=None) -> DQEResults:
    """Evaluate trajectories for x0 and xf, pack into DQEResults."""
    return DQEResults(
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

def _cost_entropy(driven_sys, x, *, S_max: float) -> float:
    """
    Mean-squared trajectory violation of the multipartite entanglement cap.

    cost = mean( (Σ_P S_P(t) − S_max)₊² )

    where Σ_P sums over all 7 bipartitions in ALL_BIPARTITIONS_4Q.
    """
    traj    = np.asarray(driven_sys.propagate_all(x))   # (16, n_t)
    total_S = entropy_sum_traj(traj)
    viol    = np.maximum(0.0, total_S - S_max)
    return float(np.mean(viol ** 2))


def _entropy_grad_fd(driven_sys, x, *, S_max: float, eps: float) -> np.ndarray:
    """
    Central-difference gradient of the entropy cost (O(eps²) error).
    """
    x = np.asarray(x, float)
    g = np.empty_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        g[i] = (
            _cost_entropy(driven_sys, xp, S_max=S_max)
            - _cost_entropy(driven_sys, xm, S_max=S_max)
        ) / (2.0 * eps)
    return g


def _lagrangian_and_grad(driven_sys, x, *, observable, lam, S_max, fd_eps,
                         rho, eps_infidelity):
    """
    Augmented Lagrangian value and gradient.

    Objective: L = S_cost + λ(I − ε) + (ρ/2)(I − ε)²

    where S_cost is the mean-squared multipartite entanglement violation and
    I is the state infidelity (the constraint I ≤ ε is enforced via λ).
    """
    S  = _cost_entropy(driven_sys, x, S_max=S_max)
    gS = _entropy_grad_fd(driven_sys, x, S_max=S_max, eps=fd_eps)

    infidelity, grad_inf = driven_sys.gradient(x, observable)
    infidelity = max(0.0, float(np.real(infidelity)))   # guard against float64 cancellation
    grad_inf   = np.asarray(grad_inf, float)
    c = infidelity - eps_infidelity

    L  = S + lam * c + 0.5 * rho * c ** 2
    gL = gS + (lam + rho * c) * grad_inf
    return float(L), gL


def _primal_dual_optimise(driven_sys, x0, *, observable, bounds,
                          config: SimulationConfig, verbose: bool = True):
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
                driven_sys, z,
                observable=observable, lam=lam,
                S_max=float(config.S_max), fd_eps=float(config.fd_eps),
                rho=float(config.rho), eps_infidelity=float(config.eps_infidelity),
            ),
            x0=x, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": int(config.inner_maxiter), "gtol": float(config.gtol)},
        )
        x = np.asarray(res.x, float)

        S          = _cost_entropy(driven_sys, x, S_max=float(config.S_max))
        infidelity = np.real(driven_sys.gradient(x, observable)[0])
        c          = infidelity - float(config.eps_infidelity)
        lam        = max(0.0, lam + float(config.dual_lr) * c)

        hist.append({
            "outer_iter": k, "lam": lam, "I_final": infidelity, "S": S,
            "constraint_residual": c,
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

def dqe_optimisation(config: SimulationConfig) -> DQEResults:
    """
    Optimize a double-qubit excitation gate via piecewise-constant exchange
    coupling pulses on a 4-spin nearest-neighbor chain (3 J channels).

    The rotating frame transformation removes free Zeeman precession of the
    |0011⟩ ↔ |1100⟩ transition (see module docstring).

    The entropy cost is the mean-squared violation of a cap on the sum of von
    Neumann entropies over all 7 bipartite partitions of 4 qubits.

    Parameters
    ----------
    config : SimulationConfig
        Key fields: entanglement_optimisation, theta_target,
        zeeman_splittings (length 4), initial_state (length 16),
        T, n_time_steps, S_max, eps_infidelity.

    Returns
    -------
    DQEResults
    """
    config = SimulationConfig(**vars(config))   # local copy — do not mutate caller's config
    config.n_J_channels = 3

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
        S  = _cost_entropy(driven_sys, res.x, S_max=float(config.S_max))
        print(f"Infidelity: I={res.fun:.3e}, nit={res.nit}, "
              f"success={res.success}, grad_norm={gn:.3e}, S={S:.3e}")
        return _package_results(
            config=config, driven_sys=driven_sys, observable=observable,
            x0=x_init, xf=np.asarray(res.x, float),
            success=bool(res.success), nit=int(res.nit), grad_norm=gn,
        )

    def run_entropy(x_init):
        xf, lam, hist = _primal_dual_optimise(
            driven_sys, x_init, observable=observable, bounds=bounds,
            config=config, verbose=True,
        )
        gn = float(np.linalg.norm(
            np.asarray(driven_sys.gradient(xf, observable)[1], float)
        ))
        ok = bool(hist) and float(hist[-1]["constraint_residual"]) <= 0.0
        return _package_results(
            config=config, driven_sys=driven_sys, observable=observable,
            x0=x_init, xf=xf, success=ok, nit=len(hist), grad_norm=gn,
            lam_final=float(lam), history=hist,
        )

    if not config.entanglement_optimisation:
        return run_baseline(x0)

    baseline    = run_baseline(x0.copy())
    constrained = run_entropy(x0.copy())
    constrained.baseline = baseline
    return constrained
