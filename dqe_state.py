"""
Double Qubit Excitation (DQE) state optimization for 4-spin systems.

Physics
-------
A 4-spin nearest-neighbor chain (semiconductor spin qubits) is driven by three
piecewise-constant exchange couplings J₁₂(t), J₂₃(t), J₃₄(t).  The target
operation is the DQE gate: a rotation by theta_target in the {|0011⟩, |1100⟩}
subspace, acting as identity everywhere else.

The system is built in the rotating frame of the average Zeeman splitting, which
removes free precession and leaves only the relative detunings deltaB = B − mean(B)
in the drift Hamiltonian.  The frame unitary U_rot dresses the target state.

Algorithm
---------
entanglement_optimisation = False
    Minimize state infidelity I = ⟨ψ(T)|I − |ψ_target⟩⟨ψ_target||ψ(T)⟩ via
    L-BFGS-B using QuGrad autodiff gradients.

entanglement_optimisation = True
    1. Run the baseline (infidelity-only) optimisation to convergence.
    2. Run the entropy-constrained primal-dual loop from the same x0.
    Baseline stored in results.baseline for comparison.

Entanglement cost (primal-dual)
-------------------------------
    cost(x) = mean_t( (Σ_P S_P(t) − S_max)₊² )

where the sum runs over all 7 distinct bipartite partitions of 4 qubits
(ALL_BIPARTITIONS_4Q: four single-qubit cuts + three two-qubit cuts), S_P(t)
is the von Neumann entropy for partition P, and (·)₊ = max(0, ·).

Augmented Lagrangian
--------------------
    L(x, λ) = cost(x) + λ(I(x) − ε) + (ρ/2)(I(x) − ε)²

Dual update after each inner solve:
    λ ← max(0, λ + α(I − ε))
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
    Build the 4-spin SpinChain in the rotating frame and the infidelity observable.

    Returns
    -------
    sys : SpinChain
        4-spin system; drift uses deltaB (relative detunings); drive disabled.
    observable : ndarray, shape (16, 16)
        I − |ψ_target⟩⟨ψ_target|.  Its expectation value equals the state infidelity.
    """
    # Build rotating frame: deltaB = B − mean(B); U_rot = expm(−iπ H_frame T·1e-3)
    deltaB, U_rot = make_rotating_frame(config.zeeman_splittings, config.T)

    # Target state in the rotating frame: apply DQE gate, then dress with U_rot
    target_state = U_rot @ dqe_unitary(config.theta_target) @ config.initial_state
    # Projector onto the complement of |ψ_target⟩ in the 16-dimensional Hilbert space
    observable   = np.eye(16, dtype=np.complex128) - np.outer(target_state, target_state.conj())

    sys = SpinChain(
        spins=4,
        zeeman_splittings=deltaB,
        max_drive_strength=0.0,           # no microwave drive in this experiment
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )
    return sys, observable


def _make_driven_sys(config: SimulationConfig, sys: SpinChain):
    """
    Create a driven system with piecewise-constant interpolation.

    x is a flat vector of length n_time_steps × n_J_channels (= 3 for DQE).
    It is reshaped to (n_time_steps, 3) on the coarse grid, then each row is
    repeated n_discretisation times to obtain the fine-grid pulse tensor.
    """
    n_steps    = config.n_time_steps
    n_disc     = config.n_discretisation
    n_fine     = n_steps * n_disc
    # T is in ns; multiply by 1e-3 to convert to µs for qugradlab, then × 2π for rad/µs
    dt_fine_us = (config.T / n_fine) * NS_TO_US

    def pulse_function(x):
        # Pack x into (n_steps, 3) coarse grid — one row per time step, 3 J channels
        x_coarse = tf.reshape(convert_dtype(x), (n_steps, config.n_J_channels))
        # Repeat each coarse step n_disc times → piecewise-constant fine grid
        x_fine   = tf.repeat(x_coarse, repeats=n_disc, axis=0)
        return (
            convert_dtype(np.zeros((n_fine, 4))),    # drive amplitudes (4 spins, disabled)
            tf.zeros(4, dtype=tf.complex128),        # drive frequencies (zero; rotating frame)
            x_fine,                                  # J-channel amplitudes on fine grid
            convert_dtype(config.initial_state),     # |ψ₀⟩ (16-dim)
            dt_fine_us * 2 * np.pi,                  # timestep in rad/µs
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

    L(x, λ) = cost(x) + λ·c + (ρ/2)·c²,  c = I(x) − ε

    where cost is the mean-squared multipartite entanglement hinge violation
    (primary objective), I is the state infidelity (constraint I ≤ ε via λ),
    and c is the constraint residual.

    Gradient: ∇L = ∇cost + (λ + ρ·c)·∇I
    """
    # Entanglement cost: sum of entropies over all 7 bipartitions; FD gradient
    S  = _cost_entropy(driven_sys, x, S_max=S_max)
    gS = _entropy_grad_fd(driven_sys, x, S_max=S_max, eps=fd_eps)

    # Infidelity and autodiff gradient from QuGrad
    infidelity, grad_inf = driven_sys.gradient(x, observable)
    infidelity = max(0.0, float(np.real(infidelity)))   # guard numerical noise from complex arithmetic
    grad_inf   = np.asarray(grad_inf, float)
    c = infidelity - eps_infidelity                      # constraint residual I − ε

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
        # Dual ascent: λ ← max(0, λ + α(I−ε)); max(0,·) is the KKT projection
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
