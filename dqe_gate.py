"""
Gate-based DQE optimization for 4-spin systems.

Optimizes piecewise-constant exchange coupling pulses (3 channels for a
nearest-neighbor chain) to implement a target DQE gate — a rotation in the
|0011⟩ ↔ |1100⟩ subspace — with optional gate-entanglement constraints via
an augmented Lagrangian primal-dual method.

entanglement_optimisation flag
-------------------------------
False : Minimize gate infidelity only (autodiff gradient).
True  : First minimize infidelity, then run the entanglement-constrained
        primal-dual loop from the same x0; baseline stored in results.baseline.

Frame transformation
--------------------
Given user-supplied Zeeman splittings [B1, B2, B3, B4], the system is
constructed in the rotating frame of the average Zeeman splitting:

    deltaB = B − mean(B)    (shape 4, per spin)

The frame unitary

    U_rot = expm(−i π Σ_k deltaB_k Z_k · T [ns→µs])

dresses the target gate before forming the gate observable.

Gate entanglement measure
-------------------------
Set config.gate_ent_measure to any key in GATE_MEASURES_4Q
(default: "operator_entanglement_sum_4q" — sum of operator entanglement over
all 7 bipartitions).  Keys in GATE_MEASURES designed for 2-qubit gates will
raise a ValueError.

Performance note
----------------
Building 16 basis-state driven systems takes several seconds on the first run
(TF graph compilation).  They are built once and reused throughout.
"""
from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, DQEGateResults, NS_TO_US
from .entanglement_measures import (
    GATE_MEASURES,
    GATE_MEASURES_4Q,
    operator_entanglement_sum_4q,
)
from ._utils import (
    convert_dtype,
    _initialise_x0,
    dqe_unitary,
    gate_infidelity,
    get_gate_infidelity_traj,
    make_rotating_frame,
)


# =============================================================================
# Utilities
# =============================================================================

def get_gate_entanglement_traj(U_traj: np.ndarray,
                                measure_fn=operator_entanglement_sum_4q) -> np.ndarray:
    """Gate entanglement measure at each time step along U_traj."""
    return np.array([measure_fn(U_traj[t]) for t in range(U_traj.shape[0])])


# =============================================================================
# System construction
# =============================================================================

def _make_system(config: SimulationConfig) -> SpinChain:
    return SpinChain(
        spins=4,
        zeeman_splittings=config.zeeman_splittings,
        max_drive_strength=0.0,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )


def _make_driven_sys(config: SimulationConfig, sys: SpinChain):
    """
    Create a driven system for one initial state with piecewise-constant pulses.

    The J channels are packed as x.reshape(n_time_steps, n_J_channels) on the
    coarse grid and repeated n_discretisation times for the fine grid.
    """
    n_steps    = int(config.n_time_steps)
    n_disc     = int(config.n_discretisation)
    n_fine     = n_steps * n_disc
    dt_fine_us = (float(config.T) / n_fine) * NS_TO_US

    def pulse_function(x):
        x_coarse = tf.reshape(x, (n_steps, config.n_J_channels))
        x_fine   = tf.cast(tf.repeat(x_coarse, repeats=n_disc, axis=0), tf.complex128)
        return (
            convert_dtype(np.zeros((n_fine, 4))),     # drive amplitudes (4 spins, disabled)
            tf.zeros(4, dtype=tf.complex128),         # drive frequencies
            x_fine,                                   # J-channel amplitudes
            convert_dtype(config.initial_state),      # initial state (16-dim)
            dt_fine_us * 2 * np.pi,                   # timestep (rad/µs)
        )

    return sys.pulse_form(pulse_function)


def _build_basis_driven_systems(config: SimulationConfig, sys: SpinChain) -> list:
    """
    Build one driven system per 16-dimensional computational basis state.

    Constructed once before optimisation so TF graph compilation happens only
    once, not on every function evaluation.
    """
    dim     = 16
    systems = []
    for i in range(dim):
        basis    = np.zeros(dim, dtype=np.complex128)
        basis[i] = 1.0
        cfg_i    = SimulationConfig(**vars(config))
        cfg_i.initial_state = basis
        systems.append(_make_driven_sys(cfg_i, sys))
    return systems


def _get_unitary_traj(config: SimulationConfig, sys: SpinChain, x: np.ndarray,
                      basis_driven_systems: list | None = None) -> np.ndarray:
    """
    Propagate each basis state and assemble the full 16×16 unitary trajectory.

    Returns
    -------
    U_traj : ndarray, shape (n_fine + 1, 16, 16)
        U_traj[t, :, i] = U(t)|i⟩.
    """
    x      = np.asarray(x, dtype=np.float64)
    n_fine = int(config.n_time_steps) * int(config.n_discretisation)
    U_traj = np.zeros((n_fine + 1, 16, 16), dtype=np.complex128)

    if basis_driven_systems is None:
        basis_driven_systems = _build_basis_driven_systems(config, sys)

    for i, ds in enumerate(basis_driven_systems):
        U_traj[:, :, i] = np.asarray(ds.propagate_all(x)).T   # (16, n_fine+1).T → (n_fine+1, 16)

    return U_traj


def _package_gate_results(*, config, sys, target_gate, x0, xf, success, nit, grad_norm,
                           basis_driven_systems, lam_final=None, history=None,
                           baseline=None) -> DQEGateResults:
    """Compute unitary trajectories for x0 and xf, pack into DQEGateResults."""
    U0 = _get_unitary_traj(config, sys, x0, basis_driven_systems)
    Uf = _get_unitary_traj(config, sys, xf, basis_driven_systems)

    ent_fn = GATE_MEASURES[config.gate_ent_measure]

    extra_initial = {name: get_gate_entanglement_traj(U0, GATE_MEASURES[name])
                     for name in config.extra_ent_measures}
    extra_final   = {name: get_gate_entanglement_traj(Uf, GATE_MEASURES[name])
                     for name in config.extra_ent_measures}

    return DQEGateResults(
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

def _cost_gate_entanglement(basis_driven_systems: list, config: SimulationConfig,
                             x: np.ndarray, *, ep_max: float) -> float:
    """
    Mean-squared trajectory violation of the gate entanglement cap.

    cost = mean((E(t) − E_max)₊²), where E is config.gate_ent_measure.
    """
    ent_fn = GATE_MEASURES[config.gate_ent_measure]
    U_traj = _get_unitary_traj(config, None, x, basis_driven_systems=basis_driven_systems)
    viol   = np.maximum(0.0, get_gate_entanglement_traj(U_traj, ent_fn) - ep_max)
    return float(np.mean(viol ** 2))


def _ep_grad_fd(basis_driven_systems: list, config: SimulationConfig, x: np.ndarray, *,
                ep_max: float, eps: float) -> np.ndarray:
    """
    Central-difference gradient of the gate entanglement cost, parallelised over parameters.

    Uses (f(x+eps) − f(x−eps)) / (2·eps) for O(eps²) accuracy.
    """
    x = np.asarray(x, float)

    def _fi(i: int) -> float:
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        return (
            _cost_gate_entanglement(basis_driven_systems, config, xp, ep_max=ep_max)
            - _cost_gate_entanglement(basis_driven_systems, config, xm, ep_max=ep_max)
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

    where EP_cost is the mean-squared gate entanglement violation and
    GI is the gate infidelity (GI ≤ ε enforced via λ).
    """
    EP  = _cost_gate_entanglement(basis_driven_systems, config, x, ep_max=ep_max)
    gEP = _ep_grad_fd(basis_driven_systems, config, x, ep_max=ep_max, eps=fd_eps)

    gi, ggi = driven_sys.gate_gradient(x, target_gate)
    gi, ggi = float(gi), np.asarray(ggi, float)
    c = gi - float(eps_infidelity)

    L  = EP + lam * c + 0.5 * float(rho) * c ** 2
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
                  f"inner_nit={res.nit}, inner_success={res.success}")

    return x, lam, hist


# =============================================================================
# Public API
# =============================================================================

def dqe_gate_optimisation(config: SimulationConfig) -> DQEGateResults:
    """
    Optimize a 4-qubit DQE gate via piecewise-constant exchange coupling pulses
    on a nearest-neighbor chain (3 J channels).

    The rotating frame transformation removes free Zeeman precession of the
    |0011⟩ ↔ |1100⟩ transition (see module docstring).

    The gate entanglement measure used for the constraint is determined by
    config.gate_ent_measure (must be a key in GATE_MEASURES_4Q, e.g.
    "operator_entanglement_sum_4q").

    Parameters
    ----------
    config : SimulationConfig
        Key fields: entanglement_optimisation, gate_ent_measure (must be a
        4Q measure), theta_target, zeeman_splittings (length 4), T,
        n_time_steps, S_max, eps_infidelity.

    Returns
    -------
    DQEGateResults
    """
    if config.gate_ent_measure not in GATE_MEASURES_4Q:
        raise ValueError(
            f"gate_ent_measure={config.gate_ent_measure!r} is not a valid 4-qubit measure. "
            f"Valid 4Q measures: {sorted(GATE_MEASURES_4Q)}"
        )
    for name in config.extra_ent_measures:
        if name not in GATE_MEASURES:
            raise ValueError(f"Unknown extra_ent_measure: {name!r}. Valid: {list(GATE_MEASURES)}")

    cfg = SimulationConfig(**vars(config))   # local copy — do not mutate caller's config
    cfg.n_J_channels = 3

    deltaB, U_rot         = make_rotating_frame(cfg.zeeman_splittings, cfg.T)
    target_gate           = U_rot @ dqe_unitary(cfg.theta_target)
    cfg.zeeman_splittings = deltaB

    if cfg.initial_state is None or len(cfg.initial_state) != 16:
        cfg.initial_state = np.zeros(16, dtype=np.complex128)
        cfg.initial_state[3] = 1.0  # |0011⟩ default

    sys        = _make_system(cfg)
    driven_sys = _make_driven_sys(cfg, sys)

    x0     = _initialise_x0(cfg)
    bounds = [(-1.0, 1.0)] * x0.size

    # Build TF graphs once; reused by every entanglement cost evaluation.
    basis_driven_systems = _build_basis_driven_systems(cfg, sys)

    def _solve_infidelity(x_init: np.ndarray):
        res = minimize(
            driven_sys.gate_gradient, x0=np.asarray(x_init, np.float64),
            args=(target_gate,), jac=True, bounds=bounds, method="L-BFGS-B",
            options={"maxiter": int(cfg.maxiter), "gtol": float(cfg.gtol)},
        )
        gn = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        print(f"GI={float(res.fun):.3e}, nit={res.nit}, success={res.success}, grad_norm={gn:.3e}")
        return res, gn

    def _solve_entropy(x_init: np.ndarray):
        xf, lam, hist = _primal_dual_optimise(
            driven_sys, basis_driven_systems, cfg, x_init,
            target_gate=target_gate, bounds=bounds, verbose=True,
        )
        gn = float(np.linalg.norm(
            np.asarray(driven_sys.gate_gradient(xf, target_gate)[1], float)
        ))
        ok = bool(hist) and float(hist[-1]["constraint_residual"]) <= 0.0
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
