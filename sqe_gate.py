"""
Gate-based SQE optimization for 2-spin systems.

Optimizes piecewise-constant exchange coupling pulses to implement a target
2-qubit gate, with optional operator-entanglement constraints via an augmented
Lagrangian primal-dual method.

entanglement_optimisation flag
-------------------------------
False : Minimize gate infidelity only (uses QuGrad autodiff gradient).
True  : First minimize infidelity, then run the entanglement-constrained
        primal-dual loop from the same x0; baseline stored in results.baseline.

Entanglement measure
--------------------
Set config.gate_ent_measure to any key in entanglement_measures.GATE_MEASURES.
Currently supported: "operator_entanglement" (default), "average_entangling_power",
"max_output_entanglement", "assisted_entangling_power".

Extra measures for plotting
---------------------------
Set config.extra_ent_measures to a list of additional gate measure names. Their
trajectories are pre-computed in _package_gate_results and stored in
SQEGateResults.extra_ent_initial / extra_ent_final.

Performance note
----------------
The 4 basis-state driven systems (used to reconstruct the full unitary) are
built once before the optimisation loop so that the TF computation graphs are
compiled only once, not on every function evaluation.
"""
from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, SQEGateResults, NS_TO_US
from .entanglement_measures import GATE_MEASURES, operator_entanglement
from ._utils import (
    convert_dtype,
    _initialise_x0,
    single_qubit_excitation,
    gate_infidelity,
    get_gate_infidelity_traj,
    make_rotating_frame,
)


# =============================================================================
# System construction
# =============================================================================

def _make_system(config: SimulationConfig) -> SpinChain:
    return SpinChain(
        spins=2,
        zeeman_splittings=config.zeeman_splittings,
        max_drive_strength=config.max_drive_strength,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )


def _make_driven_sys(config: SimulationConfig, sys: SpinChain):
    """
    Create a driven system with piecewise-constant pulse interpolation.

    Optimization parameters live on a coarse grid (n_time_steps); state
    evolution uses a finer grid (n_time_steps × n_discretisation) via
    piecewise-constant interpolation.
    """
    n_steps    = int(config.n_time_steps)
    n_disc     = int(config.n_discretisation)
    n_fine     = n_steps * n_disc
    dt_fine_us = (float(config.T) / n_fine) * NS_TO_US

    def pulse_function(x):
        x_coarse = tf.reshape(tf.convert_to_tensor(x, dtype=tf.float64),
                              (n_steps, int(config.n_J_channels)))
        x_fine   = tf.cast(tf.repeat(x_coarse, repeats=n_disc, axis=0), tf.complex128)
        return (
            convert_dtype(np.zeros((n_fine, 2))),   # drive amplitudes (disabled)
            tf.zeros(2),                            # drive frequencies
            x_fine,                                 # J coupling amplitudes
            convert_dtype(config.initial_state),    # initial state
            dt_fine_us * 2 * np.pi,                 # timestep (rad/μs)
        )

    return sys.pulse_form(pulse_function)


def _build_basis_driven_systems(config: SimulationConfig, sys: SpinChain) -> list:
    """
    Build one driven system per computational basis state of a 2-qubit system.

    The full unitary U(t) is reconstructed from its columns: U(t)|i⟩ for each
    basis vector |i⟩.  Building these systems once before optimisation ensures
    that TF graph compilation occurs only once (not per function evaluation).

    Returns
    -------
    list of 4 driven systems, one per basis state |00⟩, |01⟩, |10⟩, |11⟩.
    """
    dim     = 4
    systems = []
    for i in range(dim):
        basis    = np.zeros(dim, dtype=np.complex128)
        basis[i] = 1.0                     # |i⟩ computational basis state
        cfg_i    = SimulationConfig(**vars(config))
        cfg_i.initial_state = basis
        systems.append(_make_driven_sys(cfg_i, sys))
    return systems


# =============================================================================
# Unitary and observable trajectories
# =============================================================================

def _get_unitary_traj(config: SimulationConfig, sys: SpinChain, x: np.ndarray,
                      basis_driven_systems: list | None = None) -> np.ndarray:
    """
    Propagate each basis state and assemble the full unitary trajectory.

    Returns
    -------
    U_traj : ndarray, shape (n_fine + 1, 4, 4)
        Unitary at each fine time step, with U_traj[t, :, i] = U(t)|i⟩.
    """
    x      = np.asarray(x, dtype=np.float64)
    n_fine = int(config.n_time_steps) * int(config.n_discretisation)
    U_traj = np.zeros((n_fine + 1, 4, 4), dtype=np.complex128)

    if basis_driven_systems is None:
        basis_driven_systems = _build_basis_driven_systems(config, sys)

    for i, ds in enumerate(basis_driven_systems):
        # propagate_all returns shape (4, n_fine+1); each column is U(t)|i⟩.
        # Store in U_traj[:, :, i] so U_traj[t] is the full unitary at time t.
        U_traj[:, :, i] = np.asarray(ds.propagate_all(x)).T

    return U_traj


def get_gate_entanglement_traj(U_traj: np.ndarray,
                               measure_fn=operator_entanglement) -> np.ndarray:
    """
    Gate entanglement measure at each time step along U_traj.

    Parameters
    ----------
    measure_fn : callable
        A gate measure from GATE_MEASURES. Defaults to operator_entanglement.
    """
    return np.array([measure_fn(U_traj[t]) for t in range(U_traj.shape[0])])


def _package_gate_results(*, config, sys, target_gate, x0, xf, success, nit, grad_norm,
                          basis_driven_systems, lam_final=None, history=None,
                          baseline=None) -> SQEGateResults:
    """
    Compute unitary trajectories for x0 and xf and pack into SQEGateResults.

    Accepts pre-built basis_driven_systems to avoid TF graph recompilation.
    Evaluates config.gate_ent_measure as the main entanglement trajectory and
    each name in config.extra_ent_measures as additional stored trajectories.
    """
    U0 = _get_unitary_traj(config, sys, x0, basis_driven_systems)
    Uf = _get_unitary_traj(config, sys, xf, basis_driven_systems)

    ent_fn = GATE_MEASURES[config.gate_ent_measure]

    extra_initial = {name: get_gate_entanglement_traj(U0, GATE_MEASURES[name])
                     for name in config.extra_ent_measures}
    extra_final   = {name: get_gate_entanglement_traj(Uf, GATE_MEASURES[name])
                     for name in config.extra_ent_measures}

    return SQEGateResults(
        config=config,
        x0=np.asarray(x0, float),
        x_final=np.asarray(xf, float),
        infid_initial=get_gate_infidelity_traj(U0, target_gate),
        infid_final=get_gate_infidelity_traj(Uf, target_gate),
        ent_initial=get_gate_entanglement_traj(U0, ent_fn),
        ent_final=get_gate_entanglement_traj(Uf, ent_fn),
        extra_ent_initial=extra_initial,
        extra_ent_final=extra_final,
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

def _cost_entangling_power(basis_driven_systems: list, config: SimulationConfig,
                           x: np.ndarray, *, ep_max: float) -> float:
    """
    Mean squared trajectory violation of the entanglement cap.

    cost = mean((E(t) − E_max)₊²), where E is config.gate_ent_measure.
    """
    ent_fn = GATE_MEASURES[config.gate_ent_measure]
    U_traj = _get_unitary_traj(config, None, x, basis_driven_systems=basis_driven_systems)
    viol   = np.maximum(0.0, get_gate_entanglement_traj(U_traj, ent_fn) - ep_max)
    return float(np.mean(viol**2))


def _ep_grad_fd(basis_driven_systems: list, config: SimulationConfig, x: np.ndarray, *,
                ep_max: float, eps: float) -> np.ndarray:
    """
    Central-difference gradient of the entanglement cost, parallelised over parameters.

    Uses (f(x+eps) − f(x−eps)) / (2·eps) for O(eps²) accuracy.
    """
    x = np.asarray(x, float)

    def _fi(i: int) -> float:
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        return (
            _cost_entangling_power(basis_driven_systems, config, xp, ep_max=ep_max)
            - _cost_entangling_power(basis_driven_systems, config, xm, ep_max=ep_max)
        ) / (2.0 * eps)

    with ThreadPoolExecutor(max_workers=min(len(x), os.cpu_count() or 1)) as pool:
        return np.array(list(pool.map(_fi, range(len(x)))), dtype=float)


def _lagrangian_and_grad(
    driven_sys, basis_driven_systems: list, config: SimulationConfig, x: np.ndarray, *,
    target_gate: np.ndarray, lam: float, ep_max: float, fd_eps: float, rho: float,
    eps_infidelity: float,
) -> tuple[float, np.ndarray]:
    """
    Augmented Lagrangian value and gradient.

    Objective: L = EP_cost + λ(GI − ε) + (ρ/2)(GI − ε)²

    where EP_cost is the mean-squared entanglement violation and GI is the gate
    infidelity (GI ≤ ε enforced via λ).
    """
    # Gate entanglement cost and its parallelised central-difference gradient
    EP  = _cost_entangling_power(basis_driven_systems, config, x, ep_max=ep_max)
    gEP = _ep_grad_fd(basis_driven_systems, config, x, ep_max=ep_max, eps=fd_eps)

    # Gate infidelity and its autodiff gradient (QuGrad gate_gradient method)
    gi, ggi = driven_sys.gate_gradient(x, target_gate)
    gi, ggi = float(gi), np.asarray(ggi, float)
    c = gi - float(eps_infidelity)           # constraint residual GI − ε

    # L = EP + λ(GI−ε) + (ρ/2)(GI−ε)²,  ∇L = ∇EP + (λ + ρc)∇GI
    L  = EP + lam * c + 0.5 * float(rho) * c**2
    gL = gEP + (lam + float(rho) * c) * ggi
    return float(L), gL


def _primal_dual_optimise(
    driven_sys, basis_driven_systems: list, config: SimulationConfig, x0: np.ndarray, *,
    target_gate: np.ndarray, bounds, verbose: bool = True,
) -> tuple[np.ndarray, float, list[dict]]:
    """
    Augmented Lagrangian primal-dual loop.

    Each outer iteration minimises the Lagrangian via L-BFGS-B, then updates
    the dual variable λ ← max(0, λ + α(GI − ε)).

    Returns
    -------
    x_final : np.ndarray
    lam_final : float
    history : list[dict]  – per-outer-iteration diagnostics
    """
    x      = np.asarray(x0, float).copy()
    lam    = 1.0
    ep_max = float(config.S_max)
    hist: list[dict] = []

    for k in range(int(config.n_outer)):
        res = minimize(
            lambda z: _lagrangian_and_grad(
                driven_sys, basis_driven_systems, config, z,
                target_gate=target_gate, lam=lam, ep_max=ep_max,
                fd_eps=float(config.fd_eps), rho=float(config.rho),
                eps_infidelity=float(config.eps_infidelity),
            ),
            x0=x, jac=True, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": int(config.inner_maxiter), "gtol": float(config.gtol)},
        )
        x = np.asarray(res.x, float)

        gi  = float(driven_sys.gate_gradient(x, target_gate)[0])
        c   = gi - float(config.eps_infidelity)
        # Dual ascent: λ ← max(0, λ + α(GI−ε)); max(0,·) is the KKT projection
        lam = max(0.0, lam + float(config.dual_lr) * c)

        hist.append({
            "outer_iter": k, "lam": lam, "GI_final": gi, "constraint_residual": c,
            "inner_success": bool(res.success), "inner_nit": int(res.nit),
            "inner_fun": float(res.fun),
            "inner_grad_norm": float(np.linalg.norm(res.jac)) if res.jac is not None else np.nan,
            "message": str(res.message),
        })
        if verbose:
            print(f"[outer {k:02d}] GI={gi:.3e}, lam={lam:.6e}, "
                  f"inner_nit={res.nit}, inner_success={res.success}, EP="
                  f"{_cost_entangling_power(basis_driven_systems, config, x, ep_max=ep_max):.3e}")

    return x, lam, hist


# =============================================================================
# Public API
# =============================================================================

def sqe_gate_optimisation(config: SimulationConfig) -> SQEGateResults:
    """
    Optimize a 2-qubit gate via piecewise-constant exchange coupling pulses.

    The gate entanglement measure used for the entropy constraint is determined
    by config.gate_ent_measure (default: "operator_entanglement"). Any key in
    entanglement_measures.GATE_MEASURES is valid.

    Additional measures listed in config.extra_ent_measures are evaluated over
    the full unitary trajectory and stored in SQEGateResults.extra_ent_initial
    / extra_ent_final for plotting.

    Parameters
    ----------
    config : SimulationConfig
        Full configuration. Key fields: entanglement_optimisation,
        gate_ent_measure, extra_ent_measures, n_time_steps, n_J_channels,
        T, theta_target, zeeman_splittings, S_max, eps_infidelity.

    Returns
    -------
    SQEGateResults
    """
    if config.optimisation_target == "state":
        raise NotImplementedError("Gate-only module. Use the state optimiser for state mode.")

    if config.gate_ent_measure not in GATE_MEASURES:
        raise ValueError(f"Unknown gate_ent_measure: {config.gate_ent_measure!r}. "
                         f"Valid: {list(GATE_MEASURES)}")
    for name in config.extra_ent_measures:
        if name not in GATE_MEASURES:
            raise ValueError(f"Unknown extra_ent_measure: {name!r}. Valid: {list(GATE_MEASURES)}")

    cfg = SimulationConfig(**vars(config))   # local copy — do not mutate caller's config

    # Build the rotating frame: deltaB = B − mean(B); dress target gate with U_rot
    deltaB, U_rot         = make_rotating_frame(cfg.zeeman_splittings, cfg.T)
    target_gate           = U_rot @ single_qubit_excitation(cfg.theta_target)
    cfg.zeeman_splittings = deltaB               # SpinChain uses relative detunings

    sys        = _make_system(cfg)
    driven_sys = _make_driven_sys(cfg, sys)

    x0     = _initialise_x0(cfg)
    bounds = [(-1.0, 1.0)] * x0.size

    # Build 4 basis-state driven systems once — TF graph compilation happens here
    # and is reused for all entanglement cost evaluations throughout optimisation.
    basis_driven_systems = _build_basis_driven_systems(cfg, sys)

    def _solve_infidelity(x_init: np.ndarray):
        res = minimize(
            driven_sys.gate_gradient, x0=np.asarray(x_init, np.float64),
            args=(target_gate,), jac=True, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": int(cfg.maxiter), "gtol": float(cfg.gtol)},
        )
        gn = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        EP = _cost_entangling_power(basis_driven_systems, cfg, res.x, ep_max=float(cfg.S_max))
        print(f"GI={float(res.fun):.3e}, nit={res.nit}, success={res.success}, "
              f"grad_norm={gn:.3e}, EP={EP:.3e}")
        return res, gn

    def _solve_entropy(x_init: np.ndarray):
        xf, lam, hist = _primal_dual_optimise(
            driven_sys, basis_driven_systems, cfg, x_init,
            target_gate=target_gate, bounds=bounds, verbose=True,
        )
        gn = float(np.linalg.norm(np.asarray(driven_sys.gate_gradient(xf, target_gate)[1], float)))
        ok = bool(hist) and float(hist[-1]["constraint_residual"]) <= 0.0
        EP = _cost_entangling_power(basis_driven_systems, cfg, xf, ep_max=float(cfg.S_max))
        print(f"Final EP={EP:.3e}")
        return xf, lam, hist, gn, ok

    if not cfg.entanglement_optimisation:
        res, gn = _solve_infidelity(x0)
        return _package_gate_results(
            config=cfg, sys=sys, target_gate=target_gate,
            x0=x0, xf=res.x, success=res.success, nit=res.nit, grad_norm=gn,
            basis_driven_systems=basis_driven_systems,
        )

    res, gn0  = _solve_infidelity(x0.copy())
    baseline  = _package_gate_results(
        config=cfg, sys=sys, target_gate=target_gate,
        x0=x0, xf=res.x, success=res.success, nit=res.nit, grad_norm=gn0,
        basis_driven_systems=basis_driven_systems,
    )
    xf, lam, hist, gn, ok = _solve_entropy(x0.copy())
    return _package_gate_results(
        config=cfg, sys=sys, target_gate=target_gate,
        x0=x0, xf=xf, success=ok, nit=len(hist), grad_norm=gn,
        basis_driven_systems=basis_driven_systems,
        lam_final=float(lam), history=hist, baseline=baseline,
    )
