"""
Changes from previous version:
1. Uses local QuGrad-Private and QuGradLab-Private packages instead of
   pip-installed qugradlab/qugrad.
2. Uses the built-in gate_gradient() method from QuantumSystem for computing
   gate infidelity gradients, instead of the manual PySTE switching-function
   approach (_gate_infidelity_and_grad_pyste).

   gate_gradient() uses TF GradientTape + GateInfidelityCustom (which wraps
   PySTE's gate_switching_function under the hood) to automatically
   differentiate through the entire pulse-function chain (coarse->fine,
   rescaling, envelope processing), eliminating the need for manual chain-rule
   computation.
"""
from __future__ import annotations

from .dataclasses import *

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)

# Use local QuGrad-Private and QuGradLab-Private packages instead of
# pip-installed versions. These must be added BEFORE any qugradlab/qugrad
# imports so the local versions shadow any installed ones.
sys.path.insert(0, os.path.join(_PROJECT_DIR, "QuGrad-Private", "src"))
sys.path.insert(0, os.path.join(_PROJECT_DIR, "QuGradLab-Private", "src"))

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain

# ========== Helpers (from sqe_optimisation.py) ==========

def convert_dtype(np_array) -> tf.Tensor:
    """Convert numpy array to TensorFlow complex128 tensor."""
    return tf.cast(tf.convert_to_tensor(np_array), dtype=tf.complex128)


def single_qubit_excitation(theta: float) -> np.ndarray:
    """SQE gate: rotation in |01>↔|10> subspace by angle theta."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta), 0.0],
            [0.0, np.sin(theta),  np.cos(theta), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def get_Sent_traj(traj: np.ndarray) -> np.ndarray:
    """Entanglement entropy along a 2-qubit state trajectory."""
    traj = np.asarray(traj)
    psi = traj.T.reshape(-1, 2, 2)
    rho_A = psi @ np.conj(psi).transpose(0, 2, 1)
    eig = np.linalg.eigvalsh(rho_A)
    eig = np.clip(eig, 1e-15, 1.0)
    return -np.sum(eig * np.log2(eig), axis=1)


def _initialise_seed(config: SimulationConfig) -> None:
    """Initialize random seed from config."""
    if config.init_seed is not None:
        np.random.seed(int(config.init_seed))
        return
    if config.drive_seed is not None or config.J_init_seed is not None:
        seed = config.drive_seed if config.drive_seed is not None else config.J_init_seed
        np.random.seed(int(seed))


def _initialise_x0(config: SimulationConfig) -> np.ndarray:
    """Initialize control parameters in [-1, 1], assuming no drive controls."""
    n_params = config.n_time_steps * config.n_J_channels
    x0 = config.J_init_scale * np.random.uniform(-1.0, 1.0, size=n_params)
    return np.clip(x0, -1.0, 1.0)


def _make_driven_sys(config: SimulationConfig, sys: SpinChain):
    """Create the mapping from x to pulse parameters and make the driven system."""
    n_steps = config.n_time_steps
    n_disc = config.n_discretisation
    T = config.T

    def pulse_function(x):
        x_tf = convert_dtype(x)
        x_coarse = tf.reshape(x_tf, (n_steps, config.n_J_channels))
        x_fine = tf.repeat(x_coarse, repeats=n_disc, axis=0)
        n_fine = n_steps * n_disc
        dt_fine_us = (T / n_fine) * NS_TO_US
        drive_ctrl_amp = convert_dtype(np.zeros((n_fine, 2)))
        J_ctrl_amp = x_fine
        init = convert_dtype(config.initial_state)
        freqs = convert_dtype(config.zeeman_splittings)
        return drive_ctrl_amp, freqs, J_ctrl_amp, init, dt_fine_us

    return sys.pulse_form(pulse_function)


# ========== Gate Infidelity & Entangling Power Utilities ==========

def gate_infidelity(U: np.ndarray, target: np.ndarray) -> float:
    """Average gate infidelity: 1 - (|Tr(target^dag U)|^2 + d) / (d(d+1))"""
    d = U.shape[0]
    overlap = np.trace(target.conj().T @ U)
    return float(1.0 - (np.abs(overlap) ** 2 + d) / (d * (d + 1)))


def get_gate_infidelity_traj(U_traj: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Gate infidelity at each timestep. U_traj: (n_time_steps, dim, dim)."""
    return np.array([gate_infidelity(U_traj[t], target) for t in range(U_traj.shape[0])])


def operator_entanglement(U: np.ndarray) -> float:
    """Operator entanglement of a 2-qubit unitary (bits), via Choi matrix SVD."""
    d = U.shape[0]
    d_A, d_B = 2, 2
    Phi = U / np.sqrt(d)
    Phi_4d = Phi.reshape(d_A, d_B, d_A, d_B)
    Phi_perm = Phi_4d.transpose(0, 2, 1, 3)
    Phi_mat = Phi_perm.reshape(d_A**2, d_B**2)
    s = np.linalg.svd(Phi_mat, compute_uv=False)
    p = s**2
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def get_operator_entanglement_traj(U_traj: np.ndarray) -> np.ndarray:
    """Operator entanglement at each timestep."""
    return np.array([operator_entanglement(U_traj[t]) for t in range(U_traj.shape[0])])


# ========== Internal Helpers ==========

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


def _get_unitary_traj(config: SimulationConfig, sys: SpinChain, x: np.ndarray) -> np.ndarray:
    """
    Returns U_traj (n_fine+1, dim, dim) by propagating each basis vector.
    """
    x = np.asarray(x, dtype=np.float64)
    dim = 4
    n_fine = int(config.n_time_steps) * int(config.n_discretisation)
    U_traj = np.zeros((n_fine + 1, dim, dim), dtype=np.complex128)

    for i in range(dim):
        basis = np.zeros(dim, dtype=np.complex128)
        basis[i] = 1.0
        cfg_i = SimulationConfig(**vars(config))
        cfg_i.initial_state = basis
        driven_sys_i = _make_driven_sys(cfg_i, sys)
        psi_traj = np.asarray(driven_sys_i.propagate_all(x)).T
        U_traj[:, :, i] = psi_traj

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
    lam_final=None,
    history=None,
    baseline=None,
):
    U_traj_0 = _get_unitary_traj(config, sys, x0)
    U_traj_f = _get_unitary_traj(config, sys, xf)

    return SQEGateResults(
        config=config,
        x0=np.asarray(x0, float),
        x_final=np.asarray(xf, float),
        infid_initial=get_gate_infidelity_traj(U_traj_0, target_gate),
        infid_final=get_gate_infidelity_traj(U_traj_f, target_gate),
        ent_initial=get_operator_entanglement_traj(U_traj_0),
        ent_final=get_operator_entanglement_traj(U_traj_f),
        gradient_norm=float(grad_norm),
        n_iterations=int(nit),
        optimization_success=bool(success),
        lam_final=lam_final,
        history=history,
        baseline=baseline,
    )


def _convert_state_results(sqe_results):
    config = sqe_results.config
    ent_initial = get_Sent_traj(sqe_results.initial_traj)
    ent_final = get_Sent_traj(sqe_results.final_traj)

    baseline = None
    if sqe_results.baseline is not None:
        baseline = _convert_state_results(sqe_results.baseline)

    return SQEGateResults(
        config=config,
        x0=sqe_results.x0,
        x_final=sqe_results.x_final,
        infid_initial=sqe_results.val0,
        infid_final=sqe_results.val_final,
        ent_initial=ent_initial,
        ent_final=ent_final,
        gradient_norm=sqe_results.gradient_norm,
        n_iterations=sqe_results.n_iterations,
        optimization_success=sqe_results.optimization_success,
        lam_final=sqe_results.lam_final,
        history=sqe_results.history,
        baseline=baseline,
    )


# ========== Entangling-Power-Constrained Optimization ==========

def _cost_entangling_power(config, sys, x, *, ep_max: float) -> float:
    U_traj = _get_unitary_traj(config, sys, x)
    ep = get_operator_entanglement_traj(U_traj)
    viol = np.maximum(0.0, ep - ep_max)
    return float(np.mean(viol**2))


def _ep_grad_fd(config, sys, x, *, ep_max: float, eps: float) -> np.ndarray:
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    f0 = _cost_entangling_power(config, sys, x, ep_max=ep_max)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_cost_entangling_power(config, sys, xp, ep_max=ep_max) - f0) / eps
    return g


def _lagrangian_and_grad(
    driven_sys, config, sys, x, *, target_gate, lam, ep_max, fd_eps, rho, eps_infidelity
):
    EP = _cost_entangling_power(config, sys, x, ep_max=ep_max)
    gEP = _ep_grad_fd(config, sys, x, ep_max=ep_max, eps=fd_eps)

    gi, grad_gi = driven_sys.gate_gradient(x, target_gate)
    constraint = gi - eps_infidelity

    L = EP + lam * constraint
    grad_L = gEP + lam * grad_gi

    if rho != 0.0:
        L += 0.5 * rho * (constraint**2)
        grad_L += rho * constraint * grad_gi

    return float(L), np.asarray(grad_L, float)


def _primal_dual_optimise(driven_sys, config, sys, x0, *, target_gate, bounds, verbose=True):
    x = np.asarray(x0, float).copy()
    lam = 1e0
    hist = []

    ep_max = float(config.S_max)

    for k in range(int(config.n_outer)):

        def fun_and_jac(z):
            return _lagrangian_and_grad(
                driven_sys, config, sys, z,
                target_gate=target_gate,
                lam=lam,
                ep_max=ep_max,
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

        # Use built-in gate_gradient
        gi, _ = driven_sys.gate_gradient(x, target_gate)
        gi = float(gi)
        constraint = gi - float(config.eps_infidelity)
        lam = max(0.0, lam + float(config.dual_lr) * constraint)

        hist.append(
            {
                "outer_iter": k,
                "lam": lam,
                "GI_final": gi,
                "constraint_residual": constraint,
                "inner_success": bool(res.success),
                "inner_nit": int(res.nit),
                "inner_fun": float(res.fun),
                "inner_grad_norm": float(np.linalg.norm(res.jac)) if res.jac is not None else np.nan,
                "message": str(res.message),
            }
        )

        if verbose:
            print(
                f"[outer {k:02d}] GI={gi:.3e}, lam={lam:.6e}, "
                f"inner_nit={res.nit}, inner_success={res.success}"
            )

    return x, lam, hist


# ========== Public API ==========

def sqe_gate_optimisation(config: SimulationConfig) -> SQEGateResults:
    """
    Gate optimisation using the built-in gate_gradient() method from
    QuantumSystem, which uses TF GradientTape + GateInfidelityCustom
    (PySTE's gate_switching_function under the hood).
    """
    # ---- State optimisation: delegate ----
    if config.optimisation_target == "state":
        raise NotImplementedError(
            "State optimisation mode requires importing sqe_optimisation as a "
            "package module. Use sqe_gate_optimisation.py for state mode, or "
            "add an __init__.py to pulse_vqe_entanglement/ to enable package imports."
        )

    # ---- Gate optimisation ----
    _initialise_seed(config)
    config.n_J_channels = 1

    sys = _make_system(config)
    target_gate = single_qubit_excitation(config.theta_target)

    dummy_init = np.zeros(4, dtype=np.complex128)
    dummy_init[0] = 1.0
    config_copy = SimulationConfig(**vars(config))
    config_copy.initial_state = dummy_init
    driven_sys = _make_driven_sys(config_copy, sys)

    x0 = _initialise_x0(config)
    bounds = [(-1.0, 1.0)] * len(x0)

    def _fun_and_jac_gate(x_opt, target):
        """Gate infidelity and gradient via built-in gate_gradient."""
        gi, grad = driven_sys.gate_gradient(x_opt, target)
        return gi, grad

    def run_baseline(x_init):
        res = minimize(
            _fun_and_jac_gate,
            x0=x_init,
            args=(target_gate,),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": int(config.maxiter), "gtol": float(config.gtol)},
        )

        grad_norm = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        print("Gate infidelity optimisation results:")
        print(f"GI={float(res.fun):.3e}, nit={res.nit}, success={res.success}, grad_norm={grad_norm:.3e}")

        return _package_gate_results(
            config=config,
            sys=sys,
            target_gate=target_gate,
            x0=x_init,
            xf=np.asarray(res.x, float),
            success=bool(res.success),
            nit=int(res.nit),
            grad_norm=grad_norm,
        )

    def run_ep_constrained(x_init):
        x_final, lam_final, hist = _primal_dual_optimise(
            driven_sys,
            config,
            sys,
            x_init,
            target_gate=target_gate,
            bounds=bounds,
            verbose=True,
        )
        # Use built-in gate_gradient
        _, grad_final = driven_sys.gate_gradient(x_final, target_gate)
        grad_norm = float(np.linalg.norm(np.asarray(grad_final, float)))
        success = bool(hist) and (float(hist[-1]["constraint_residual"]) <= 0.0)

        return _package_gate_results(
            config=config,
            sys=sys,
            target_gate=target_gate,
            x0=x_init,
            xf=x_final,
            success=success,
            nit=len(hist),
            grad_norm=grad_norm,
            lam_final=float(lam_final),
            history=hist,
        )

    if config.run_mode == "infidelity":
        return run_baseline(x0)

    if config.run_mode == "entropy":
        return run_ep_constrained(x0)

    if config.run_mode == "both":
        baseline = run_baseline(x0.copy())
        constrained = run_ep_constrained(x0.copy())
        constrained.baseline = baseline
        return constrained

    raise ValueError(f"Unknown run_mode: {config.run_mode!r}")

