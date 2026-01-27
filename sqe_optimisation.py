# pulse_vqe_entanglement/sqe_optimisation.py
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, SQEResults


def single_qubit_excitation(theta: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta), 0.0],
            [0.0, np.sin(theta),  np.cos(theta), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def convert_dtype(np_array) -> tf.Tensor:
    return tf.cast(tf.convert_to_tensor(np_array), dtype=tf.complex128)


def get_Sent_single(state: np.ndarray) -> float:
    psi = state.reshape(2, 2)
    rho_A = psi @ psi.conj().T
    eig = np.linalg.eigvalsh(rho_A)
    eig = np.clip(eig, 1e-15, 1.0)
    return float(-np.sum(eig * np.log2(eig)))


def get_Sent_traj(traj: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj)             # (dim, nT) expected from qugrad
    psi = traj.T.reshape(-1, 2, 2)      # (nT, 2, 2)
    rhoA = psi @ np.conj(psi).transpose(0, 2, 1)
    eig = np.linalg.eigvalsh(rhoA)
    eig = np.clip(eig, 1e-15, 1.0)
    return -np.sum(eig * np.log2(eig), axis=1)


# ---------- internal helpers (same logic as before) ----------

def _initialise_seed(config: SimulationConfig) -> None:
    if config.init_seed is not None:
        np.random.seed(int(config.init_seed))
        return
    if config.drive_seed is not None or config.J_init_seed is not None:
        seed = config.drive_seed if config.drive_seed is not None else config.J_init_seed
        np.random.seed(int(seed))


def _initialise_x0(config: SimulationConfig) -> np.ndarray:
    nT = config.n_time_steps
    if config.max_drive_strength == 0.0:
        x0 = config.J_init_scale * np.random.uniform(-1.0, 1.0, size=(nT * config.n_J_channels,))
        return np.clip(x0, -1.0, 1.0)

    # If you later enable microwave drive again, keep your old packing here.
    raise NotImplementedError("Drive controls not enabled in this project configuration.")


def _make_system_and_observable(config: SimulationConfig):
    target_state = single_qubit_excitation(config.theta_target) @ config.initial_state
    observable = np.eye(4, dtype=np.complex128) - np.outer(target_state, target_state.conj())

    sys = SpinChain(
        spins=2,
        zeeman_splittings=config.zeeman_splittings,
        max_drive_strength=config.max_drive_strength,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )
    return sys, observable


def _make_driven_sys(config: SimulationConfig, sys: SpinChain):
    nT = config.n_time_steps
    n_disc = config.n_discretisation
    T = config.T

    def pulse_function(x):
        # x has shape (nT * n_J_channels,) - coarse optimization parameters
        # Convert to TensorFlow tensor and reshape to coarse grid: (nT, n_J_channels)
        x_tf = convert_dtype(x)
        x_coarse = tf.reshape(x_tf, (nT, config.n_J_channels))

        # Expand to fine grid: repeat each coarse step n_disc times
        # Shape: (nT * n_disc, n_J_channels)
        x_fine = tf.repeat(x_coarse, repeats=n_disc, axis=0)

        # Fine discretization sizes
        nT_fine = nT * n_disc
        dt_fine = T / nT_fine

        drive_ctrl_amp = convert_dtype(np.zeros((nT_fine, 2)))
        J_ctrl_amp = x_fine
        init = convert_dtype(config.initial_state)
        freqs = convert_dtype(config.zeeman_splittings)

        return drive_ctrl_amp, freqs, J_ctrl_amp, init, dt_fine

    return sys.pulse_form(pulse_function)


def _package_results(*, config, driven_sys, observable, x0, xf, success, nit, grad_norm, lam_final=None, history=None, baseline=None):
    val0 = np.real(driven_sys.evolved_expectation_value_all(x0, observable))
    valf = np.real(driven_sys.evolved_expectation_value_all(xf, observable))
    traj0 = driven_sys.propagate_all(x0)
    trajf = driven_sys.propagate_all(xf)

    return SQEResults(
        config=config,
        x0=np.asarray(x0, float),
        x_final=np.asarray(xf, float),
        val0=np.asarray(val0, float),
        val_final=np.asarray(valf, float),
        initial_traj=np.asarray(traj0),
        final_traj=np.asarray(trajf),
        gradient_norm=float(grad_norm),
        n_iterations=int(nit),
        optimization_success=bool(success),
        lam_final=lam_final,
        history=history,
        baseline=baseline,
    )


# ---------- entropy constrained objective (2-qubit entropy) ----------

def _cost_entropy(driven_sys, x, *, S_max: float) -> float:
    traj = np.asarray(driven_sys.propagate_all(x))
    S = np.asarray(get_Sent_traj(traj), float)
    viol = np.maximum(0.0, S - S_max)
    return float(np.mean(viol**2))


def _entropy_grad_fd(driven_sys, x, *, S_max: float, eps: float) -> np.ndarray:
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    f0 = _cost_entropy(driven_sys, x, S_max=S_max)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_cost_entropy(driven_sys, xp, S_max=S_max) - f0) / eps
    return g


def _infidelity_and_grad(driven_sys, x, observable):
    I, gI = driven_sys.gradient(x, observable)
    return float(np.real(I)), np.asarray(gI, float)


def _lagrangian_and_grad(driven_sys, x, *, observable, lam, S_max, fd_eps, rho, eps_infidelity):
    S = _cost_entropy(driven_sys, x, S_max=S_max)
    gS = _entropy_grad_fd(driven_sys, x, S_max=S_max, eps=fd_eps)

    I, gI = _infidelity_and_grad(driven_sys, x, observable)
    c = I - eps_infidelity

    L = S + lam * c
    g = gS + lam * gI

    if rho != 0.0:
        L += 0.5 * rho * (c ** 2)
        g += rho * c * gI

    return float(L), np.asarray(g, float)


def _primal_dual_optimise(driven_sys, x0, *, observable, bounds, config: SimulationConfig, verbose=True):
    x = np.asarray(x0, float).copy()
    lam = 1e-1
    hist = []

    for k in range(int(config.n_outer)):
        def fun_and_jac(z):
            return _lagrangian_and_grad(
                driven_sys, z,
                observable=observable,
                lam=lam,
                S_max=float(config.S_max),
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

        I, _ = _infidelity_and_grad(driven_sys, x, observable)
        c = I - float(config.eps_infidelity)
        lam = max(0.0, lam + float(config.dual_lr) * c)

        hist.append({
            "outer_iter": k,
            "lam": lam,
            "I_final": I,
            "constraint_residual": c,
            "inner_success": bool(res.success),
            "inner_nit": int(res.nit),
            "inner_fun": float(res.fun),
            "inner_grad_norm": float(np.linalg.norm(res.jac)) if res.jac is not None else np.nan,
            "message": str(res.message),
        })

        if verbose:
            print(f"[outer {k:02d}] I={I:.3e}, lam={lam:.3e}, inner_nit={res.nit}, inner_success={res.success}")

    return x, lam, hist


# ---------- public API ----------

def sqe_optimisation(config: SimulationConfig) -> SQEResults:
    _initialise_seed(config)

    # enforce SQE defaults
    config.n_J_channels = 1
    sys, observable = _make_system_and_observable(config)
    driven_sys = _make_driven_sys(config, sys)

    x0 = _initialise_x0(config)
    bounds = [(-1.0, 1.0)] * len(x0)

    def run_baseline(x0_in):
        res = minimize(
            driven_sys.gradient,
            x0=x0_in,
            args=(observable,),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": int(config.maxiter), "gtol": float(config.gtol)},
        )
        grad_norm = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        return _package_results(
            config=config, driven_sys=driven_sys, observable=observable,
            x0=x0_in, xf=np.asarray(res.x, float),
            success=bool(res.success), nit=int(res.nit), grad_norm=grad_norm,
        )

    def run_entropy(x0_in):
        xf, lamf, hist = _primal_dual_optimise(
            driven_sys, x0_in,
            observable=observable,
            bounds=bounds,
            config=config,
            verbose=True,
        )
        I_final, gI_final = driven_sys.gradient(xf, observable)
        grad_norm = float(np.linalg.norm(np.asarray(gI_final, float)))
        success = bool(hist) and (float(hist[-1]["constraint_residual"]) <= 0.0)
        return _package_results(
            config=config, driven_sys=driven_sys, observable=observable,
            x0=x0_in, xf=xf,
            success=success, nit=len(hist), grad_norm=grad_norm,
            lam_final=float(lamf), history=hist,
        )

    if config.run_mode == "infidelity":
        baseline = run_baseline(x0)
        return baseline

    if config.run_mode == "entropy":
        return run_entropy(x0)

    if config.run_mode == "both":
        baseline = run_baseline(x0.copy())
        constrained = run_entropy(x0.copy())
        constrained.baseline = baseline
        return constrained

    raise ValueError(f"Unknown run_mode: {config.run_mode!r}")
