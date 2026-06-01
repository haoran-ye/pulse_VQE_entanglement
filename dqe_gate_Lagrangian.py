"""
Gate-based DQE optimisation for 4-spin systems.

Baseline:
    minimise gate infidelity using QuGrad autodiff gradients.

Entanglement optimisation:
    1. First obtain a high-fidelity baseline DQE gate pulse.
    2. Starting from this baseline pulse, minimise gate-entanglement cost
       subject to gate infidelity <= eps_infidelity using trust-constr.
"""

from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize, Bounds, NonlinearConstraint

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
    get_gate_infidelity_traj,
    make_rotating_frame,
)


# =============================================================================
# Utilities
# =============================================================================

def _gate_measure(name: str):
    if name in GATE_MEASURES:
        return GATE_MEASURES[name]

    raise ValueError(
        f"Unknown gate entanglement measure {name!r}. "
        f"Valid measures: {list(GATE_MEASURES)}"
    )


def get_gate_entanglement_traj(
    U_traj: np.ndarray,
    measure_fn=operator_entanglement_sum_4q,
) -> np.ndarray:
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
    n_steps = int(config.n_time_steps)
    n_disc = int(config.n_discretisation)
    n_fine = n_steps * n_disc
    dt_fine_us = (float(config.T) / n_fine) * NS_TO_US

    def pulse_function(x):
        x_coarse = tf.reshape(
            tf.convert_to_tensor(x, dtype=tf.float64),
            (n_steps, int(config.n_J_channels)),
        )
        x_fine = tf.cast(tf.repeat(x_coarse, repeats=n_disc, axis=0), tf.complex128)

        return (
            convert_dtype(np.zeros((n_fine, 4))),
            tf.zeros(4, dtype=tf.complex128),
            x_fine,
            convert_dtype(config.initial_state),
            dt_fine_us * 2 * np.pi,
        )

    return sys.pulse_form(pulse_function)


def _build_basis_driven_systems(config: SimulationConfig, sys: SpinChain) -> list:
    systems = []

    for i in range(16):
        basis = np.zeros(16, dtype=np.complex128)
        basis[i] = 1.0

        cfg_i = SimulationConfig(**vars(config))
        cfg_i.initial_state = basis

        systems.append(_make_driven_sys(cfg_i, sys))

    return systems


def _get_unitary_traj(
    config: SimulationConfig,
    sys: SpinChain,
    x: np.ndarray,
    basis_driven_systems: list | None = None,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n_fine = int(config.n_time_steps) * int(config.n_discretisation)

    U_traj = np.zeros((n_fine + 1, 16, 16), dtype=np.complex128)

    if basis_driven_systems is None:
        basis_driven_systems = _build_basis_driven_systems(config, sys)

    for i, ds in enumerate(basis_driven_systems):
        U_traj[:, :, i] = np.asarray(ds.propagate_all(x)).T

    return U_traj


def _package_gate_results(
    *,
    config,
    sys,
    target_gate,
    x0,
    xf,
    success,
    nit,
    grad_norm,
    basis_driven_systems,
    lam_final=None,
    history=None,
    baseline=None,
) -> DQEGateResults:
    U0 = _get_unitary_traj(config, sys, x0, basis_driven_systems)
    Uf = _get_unitary_traj(config, sys, xf, basis_driven_systems)

    ent_fn = _gate_measure(config.gate_ent_measure)

    extra_initial = {
        name: get_gate_entanglement_traj(U0, _gate_measure(name))
        for name in config.extra_ent_measures
    }

    extra_final = {
        name: get_gate_entanglement_traj(Uf, _gate_measure(name))
        for name in config.extra_ent_measures
    }

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
# Gate-entanglement objective
# =============================================================================

def _cost_gate_entanglement(
    basis_driven_systems: list,
    config: SimulationConfig,
    x: np.ndarray,
    *,
    ep_max: float,
) -> float:
    ent_fn = _gate_measure(config.gate_ent_measure)

    U_traj = _get_unitary_traj(
        config,
        None,
        x,
        basis_driven_systems=basis_driven_systems,
    )

    ent_traj = get_gate_entanglement_traj(U_traj, ent_fn)
    viol = np.maximum(0.0, ent_traj - ep_max)

    return float(np.mean(viol ** 2))


def _ep_grad_fd(
    basis_driven_systems: list,
    config: SimulationConfig,
    x: np.ndarray,
    *,
    ep_max: float,
    eps: float,
) -> np.ndarray:
    x = np.asarray(x, float)

    def _grad_i(i: int) -> float:
        xp = x.copy()
        xm = x.copy()

        xp[i] += eps
        xm[i] -= eps

        return (
            _cost_gate_entanglement(
                basis_driven_systems,
                config,
                xp,
                ep_max=ep_max,
            )
            -
            _cost_gate_entanglement(
                basis_driven_systems,
                config,
                xm,
                ep_max=ep_max,
            )
        ) / (2.0 * eps)

    n_workers = min(len(x), os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return np.array(list(pool.map(_grad_i, range(len(x)))), dtype=float)


def _entropy_objective_and_grad(
    basis_driven_systems: list,
    config: SimulationConfig,
    x: np.ndarray,
    *,
    ep_max: float,
    fd_eps: float,
) -> tuple[float, np.ndarray]:
    EP = _cost_gate_entanglement(
        basis_driven_systems,
        config,
        x,
        ep_max=ep_max,
    )

    gEP = _ep_grad_fd(
        basis_driven_systems,
        config,
        x,
        ep_max=ep_max,
        eps=fd_eps,
    )

    return float(EP), np.asarray(gEP, float)


# =============================================================================
# Gate-infidelity constraint
# =============================================================================

def _gate_infidelity_value(
    driven_sys,
    x: np.ndarray,
    *,
    target_gate: np.ndarray,
) -> float:
    gi = driven_sys.gate_gradient(x, target_gate)[0]
    return float(max(0.0, np.real(gi)))


def _gate_infidelity_jac(
    driven_sys,
    x: np.ndarray,
    *,
    target_gate: np.ndarray,
) -> np.ndarray:
    _, grad = driven_sys.gate_gradient(x, target_gate)
    return np.asarray(grad, float)


def _constrained_entropy_optimise(
    driven_sys,
    basis_driven_systems: list,
    config: SimulationConfig,
    x0: np.ndarray,
    *,
    target_gate: np.ndarray,
    bounds,
    verbose: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """
    Constrained DQE gate-entanglement optimisation:

        minimise_x EP(x)
        subject to GI(x) <= eps_infidelity
    """
    x0 = np.asarray(x0, float).copy()

    ep_max = float(config.S_max)
    fd_eps = float(getattr(config, "fd_eps", 1e-5))

    eps_internal = getattr(config, "eps_internal", None)
    eps_constraint = float(config.eps_infidelity if eps_internal is None else eps_internal)

    scipy_bounds = Bounds(
        lb=np.array([b[0] for b in bounds], dtype=float),
        ub=np.array([b[1] for b in bounds], dtype=float),
    )

    gate_constraint = NonlinearConstraint(
        fun=lambda x: _gate_infidelity_value(
            driven_sys,
            x,
            target_gate=target_gate,
        ),
        lb=-np.inf,
        ub=eps_constraint,
        jac=lambda x: _gate_infidelity_jac(
            driven_sys,
            x,
            target_gate=target_gate,
        ),
    )

    GI0 = _gate_infidelity_value(
        driven_sys,
        x0,
        target_gate=target_gate,
    )

    EP0 = _cost_gate_entanglement(
        basis_driven_systems,
        config,
        x0,
        ep_max=ep_max,
    )

    if verbose:
        print(
            f"[trust-constr DQE gate start] "
            f"GI0={GI0:.3e}, EP0={EP0:.3e}, "
            f"constraint={eps_constraint:.3e}"
        )

    res = minimize(
        lambda x: _entropy_objective_and_grad(
            basis_driven_systems,
            config,
            x,
            ep_max=ep_max,
            fd_eps=fd_eps,
        ),
        x0=x0,
        jac=True,
        method="trust-constr",
        bounds=scipy_bounds,
        constraints=[gate_constraint],
        options={
            "maxiter": int(config.maxiter),
            "gtol": float(config.gtol),
            "xtol": float(getattr(config, "xtol", 1e-10)),
            "barrier_tol": float(getattr(config, "barrier_tol", 1e-10)),
            "verbose": 3 if verbose else 0,
        },
    )

    xf = np.asarray(res.x, float)

    GIf = _gate_infidelity_value(
        driven_sys,
        xf,
        target_gate=target_gate,
    )

    EPf = _cost_gate_entanglement(
        basis_driven_systems,
        config,
        xf,
        ep_max=ep_max,
    )

    gEP = _ep_grad_fd(
        basis_driven_systems,
        config,
        xf,
        ep_max=ep_max,
        eps=fd_eps,
    )

    hist = [{
        "success": bool(res.success),
        "message": str(res.message),
        "status": int(res.status),
        "nit": int(getattr(res, "niter", res.nfev)),
        "nfev": int(res.nfev),
        "fun": float(res.fun),
        "EP": float(EPf),
        "GI_final": float(GIf),
        "constraint_residual": float(GIf - eps_constraint),
        "entropy_grad_norm": float(np.linalg.norm(gEP)),
        "constr_violation": float(getattr(res, "constr_violation", np.nan)),
        "optimality": float(getattr(res, "optimality", np.nan)),
    }]

    if verbose:
        print(
            f"[trust-constr DQE gate final] "
            f"GI={GIf:.3e}, EP={EPf:.3e}, "
            f"GI-eps={GIf - eps_constraint:.3e}, "
            f"success={res.success}, "
            f"nit={hist[-1]['nit']}"
        )

    return xf, hist


# =============================================================================
# Public API
# =============================================================================

def dqe_gate_optimisation(config: SimulationConfig) -> DQEGateResults:
    """
    Optimise a 4-qubit DQE gate using 3 exchange-control channels.

    entanglement_optimisation = False:
        minimise final gate infidelity.

    entanglement_optimisation = True:
        first find a baseline high-fidelity gate pulse, then minimise
        gate-entanglement cost subject to GI <= eps_infidelity.
    """
    if config.gate_ent_measure not in GATE_MEASURES_4Q:
        raise ValueError(
            f"gate_ent_measure={config.gate_ent_measure!r} is not a valid 4-qubit measure. "
            f"Valid 4Q measures: {sorted(GATE_MEASURES_4Q)}"
        )

    for name in config.extra_ent_measures:
        if name not in GATE_MEASURES and name not in GATE_MEASURES_4Q:
            raise ValueError(
                f"Unknown extra_ent_measure: {name!r}. "
                f"Valid: {list(GATE_MEASURES) + list(GATE_MEASURES_4Q)}"
            )

    cfg = SimulationConfig(**vars(config))
    cfg.n_J_channels = 3

    deltaB, U_rot = make_rotating_frame(cfg.zeeman_splittings, cfg.T)
    target_gate = U_rot @ dqe_unitary(cfg.theta_target)
    cfg.zeeman_splittings = deltaB

    if cfg.initial_state is None or len(cfg.initial_state) != 16:
        cfg.initial_state = np.zeros(16, dtype=np.complex128)
        cfg.initial_state[3] = 1.0

    sys = _make_system(cfg)
    driven_sys = _make_driven_sys(cfg, sys)

    x0 = _initialise_x0(cfg)
    bounds = [(-1.0, 1.0)] * x0.size

    basis_driven_systems = _build_basis_driven_systems(cfg, sys)

    def _solve_infidelity(x_init: np.ndarray):
        res = minimize(
            driven_sys.gate_gradient,
            x0=np.asarray(x_init, np.float64),
            args=(target_gate,),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={
                "maxiter": int(cfg.maxiter),
                "gtol": float(cfg.gtol),
            },
        )

        gn = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0

        EP = _cost_gate_entanglement(
            basis_driven_systems,
            cfg,
            res.x,
            ep_max=float(cfg.S_max),
        )

        print(
            f"Baseline DQE gate: GI={float(res.fun):.3e}, "
            f"nit={res.nit}, success={res.success}, "
            f"grad_norm={gn:.3e}, EP={EP:.3e}"
        )

        return res, gn

    def _solve_entropy(x_init: np.ndarray):
        xf, hist = _constrained_entropy_optimise(
            driven_sys,
            basis_driven_systems,
            cfg,
            x_init,
            target_gate=target_gate,
            bounds=bounds,
            verbose=True,
        )

        gi, grad_gi = driven_sys.gate_gradient(xf, target_gate)
        gi = float(max(0.0, np.real(gi)))
        gn = float(np.linalg.norm(np.asarray(grad_gi, float)))

        ok = bool(hist) and float(hist[-1]["constraint_residual"]) <= 0.0

        EP = _cost_gate_entanglement(
            basis_driven_systems,
            cfg,
            xf,
            ep_max=float(cfg.S_max),
        )

        print(
            f"Final constrained DQE gate: GI={gi:.3e}, "
            f"EP={EP:.3e}, success={ok}"
        )

        return xf, hist, gn, ok

    if not cfg.entanglement_optimisation:
        res, gn = _solve_infidelity(x0)

        return _package_gate_results(
            config=cfg,
            sys=sys,
            target_gate=target_gate,
            x0=x0,
            xf=res.x,
            success=bool(res.success),
            nit=int(res.nit),
            grad_norm=gn,
            basis_driven_systems=basis_driven_systems,
        )

    res, gn0 = _solve_infidelity(x0.copy())

    baseline = _package_gate_results(
        config=cfg,
        sys=sys,
        target_gate=target_gate,
        x0=x0,
        xf=res.x,
        success=bool(res.success),
        nit=int(res.nit),
        grad_norm=gn0,
        basis_driven_systems=basis_driven_systems,
    )

    baseline_gi = float(res.fun)

    if baseline_gi > float(cfg.eps_infidelity):
        print(
            f"Warning: baseline DQE gate is not feasible: "
            f"GI={baseline_gi:.3e} > eps={float(cfg.eps_infidelity):.3e}. "
            f"Constrained optimisation may fail."
        )

    xf, hist, gn, ok = _solve_entropy(baseline.x_final.copy())

    return _package_gate_results(
        config=cfg,
        sys=sys,
        target_gate=target_gate,
        x0=x0,
        xf=xf,
        success=ok,
        nit=int(hist[-1]["nit"]) if hist else 0,
        grad_norm=gn,
        basis_driven_systems=basis_driven_systems,
        lam_final=None,
        history=hist,
        baseline=baseline,
    )