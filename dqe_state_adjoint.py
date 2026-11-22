"""DQE state optimisation with an adjoint seven-partition entropy gradient.

The fidelity-only baseline and the entropy-constrained augmented-Lagrangian
run both start from the same initial pulse, matching the SQE adjoint workflow.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ._utils import _initialise_x0
from .dataclasses import SimulationConfig
from .dqe_state import (
    _make_system_and_observable,
    _make_driven_sys,
    _package_results,
)
from .dqe_state_entropy_gradient import entropy_cost_and_grad


def _lagrangian_and_grad(driven_sys, config, x, *, observable, lam):
    """Return augmented-Lagrangian value and analytic control gradient."""
    cost, cost_grad = entropy_cost_and_grad(config, x)
    infidelity, infidelity_grad = driven_sys.gradient(x, observable)
    infidelity = max(0.0, float(np.real(infidelity)))
    infidelity_grad = np.asarray(infidelity_grad, dtype=float)
    residual = infidelity - float(config.eps_infidelity)
    rho = float(config.rho)
    value = cost + lam * residual + 0.5 * rho * residual**2
    gradient = cost_grad + (lam + rho * residual) * infidelity_grad
    return float(value), gradient


def _primal_dual_optimise(driven_sys, x0, *, observable, bounds, config,
                          verbose):
    """Minimise the Lagrangian, then update its nonnegative multiplier."""
    x = np.asarray(x0, dtype=float).copy()
    lam = 1.0
    history = []
    for k in range(int(config.n_outer)):
        result = minimize(
            lambda z: _lagrangian_and_grad(
                driven_sys, config, z, observable=observable, lam=lam),
            x0=x, jac=True, method="L-BFGS-B", bounds=bounds,
            options={
                "maxiter": int(config.inner_maxiter),
                "gtol": float(config.gtol),
                "ftol": 1e-14,
            },
        )
        x = np.asarray(result.x, dtype=float)
        cost, _ = entropy_cost_and_grad(config, x)
        infidelity = float(np.real(driven_sys.gradient(x, observable)[0]))
        residual = infidelity - float(config.eps_infidelity)
        lam = max(0.0, lam + float(config.dual_lr) * residual)
        history.append({
            "outer_iter": k,
            "lam": float(lam),
            "I_final": infidelity,
            "constraint_residual": residual,
            "S": float(cost),
            "inner_success": bool(result.success),
            "inner_nit": int(result.nit),
            "inner_fun": float(result.fun),
            "inner_grad_norm": float(np.linalg.norm(result.jac)),
            "message": str(result.message),
        })
        if verbose:
            print(
                f"[outer {k:02d}] I={infidelity:.3e}, "
                f"S={cost:.3e}, lam={lam:.3e}, "
                f"inner_nit={result.nit}, residual={residual:.3e}"
            )
    return x, lam, history


def dqe_optimisation(config: SimulationConfig, *, verbose: bool = True):
    """Optimise DQE state fidelity and transient entropy with an adjoint.

    The entropy stage is run only when ``entanglement_optimisation`` is true.
    It minimises the seven-partition entropy hinge cost subject to the
    noiseless state-infidelity constraint.
    """
    config = SimulationConfig(**vars(config))
    config.n_J_channels = 3

    sys, observable = _make_system_and_observable(config)
    driven_sys = _make_driven_sys(config, sys)
    x0 = _initialise_x0(config)
    bounds = [(-1.0, 1.0)] * len(x0)

    baseline_res = minimize(
        driven_sys.gradient, x0=x0, args=(observable,), jac=True,
        bounds=bounds, method="L-BFGS-B",
        options={"maxiter": int(config.maxiter), "gtol": float(config.gtol)},
    )
    baseline = _package_results(
        config=config, driven_sys=driven_sys, observable=observable,
        x0=x0, xf=baseline_res.x, success=baseline_res.success,
        nit=baseline_res.nit, grad_norm=np.linalg.norm(baseline_res.jac),
    )
    if verbose:
        baseline_cost, _ = entropy_cost_and_grad(config, baseline_res.x)
        print(
            f"Baseline: I={float(baseline_res.fun):.3e}, "
            f"S={baseline_cost:.3e}, nit={baseline_res.nit}"
        )
    if not config.entanglement_optimisation:
        return baseline

    # Fail early if the small-matrix adjoint and SpinChain ever diverge.
    _, _, model_traj = entropy_cost_and_grad(config, x0, return_trajectory=True)
    simulator_traj = np.asarray(driven_sys.propagate_all(x0))
    mismatch = np.max(np.abs(model_traj - simulator_traj))
    if mismatch > 1e-7:
        raise RuntimeError(
            f"DQE adjoint propagator differs from SpinChain by {mismatch:.3e}; "
            "check Hamiltonian and timestep conventions."
        )

    xf, lam, history = _primal_dual_optimise(
        driven_sys, x0, observable=observable, bounds=bounds,
        config=config, verbose=verbose,
    )
    _, infidelity_grad = driven_sys.gradient(xf, observable)
    return _package_results(
        config=config, driven_sys=driven_sys, observable=observable,
        x0=x0, xf=xf,
        success=bool(history) and history[-1]["constraint_residual"] <= 0.0,
        nit=len(history), grad_norm=np.linalg.norm(infidelity_grad),
        lam_final=float(lam), history=history, baseline=baseline,
    )
