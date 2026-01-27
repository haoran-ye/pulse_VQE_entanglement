# pulse_vqe_entanglement/dqe_optimisation.py
from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, SQEResults


def convert_dtype(np_array) -> tf.Tensor:
    return tf.cast(tf.convert_to_tensor(np_array), dtype=tf.complex128)


def basis_state(n_qubits: int, bitstring: str) -> np.ndarray:
    """Return computational basis |bitstring> as a complex vector."""
    if len(bitstring) != n_qubits:
        raise ValueError("bitstring length must equal n_qubits")
    idx = int(bitstring, 2)
    v = np.zeros((2**n_qubits,), dtype=np.complex128)
    v[idx] = 1.0
    return v


def dqe_unitary(theta: float) -> np.ndarray:
    """
    DQE unitary: identity on 16-dim, except on subspace span{|0011>, |1100>}
    where it applies the same rotation as SQE:
      |0011> -> cosθ|0011> - sinθ|1100>
      |1100> -> sinθ|0011> + cosθ|1100>
    """
    U = np.eye(16, dtype=np.complex128)
    i = int("0011", 2)
    j = int("1100", 2)
    c = np.cos(theta)
    s = np.sin(theta)

    U[i, i] = c
    U[i, j] = -s
    U[j, i] = s
    U[j, j] = c
    return U


# -------- generic bipartite entropy for pure state --------

def _reshape_state_for_partition(state: np.ndarray, subsystem: tuple[int, ...], n_qubits: int):
    """
    Reorder qubits so subsystem A comes first, then B, and reshape into (2^|A|, 2^|B|).
    Convention: computational basis index is big-endian bitstring q0 q1 ... q(n-1).
    """
    state = np.asarray(state, dtype=np.complex128).reshape((2,) * n_qubits)

    A = tuple(subsystem)
    B = tuple(q for q in range(n_qubits) if q not in A)
    perm = A + B
    psi = np.transpose(state, perm)  # move axes
    dimA = 2 ** len(A)
    dimB = 2 ** (n_qubits - len(A))
    return psi.reshape(dimA, dimB)


def entropy_bipartite_single(state: np.ndarray, *, subsystem=(0, 1), n_qubits=4) -> float:
    psiAB = _reshape_state_for_partition(state, tuple(subsystem), n_qubits)
    rhoA = psiAB @ psiAB.conj().T
    eig = np.linalg.eigvalsh(rhoA)
    eig = np.clip(eig, 1e-15, 1.0)
    return float(-np.sum(eig * np.log2(eig)))


def entropy_bipartite_traj(traj: np.ndarray, *, subsystem=(0, 1), n_qubits=4) -> np.ndarray:
    traj = np.asarray(traj)  # expected (dim, nT)
    nT = traj.shape[1]
    out = np.zeros((nT,), dtype=float)
    for t in range(nT):
        out[t] = entropy_bipartite_single(traj[:, t], subsystem=subsystem, n_qubits=n_qubits)
    return out


# -------- shared primal-dual framework (same as SQE, but using bipartite entropy) --------

def _cost_entropy(driven_sys, x, *, S_max: float, subsystem: tuple[int, ...]) -> float:
    traj = np.asarray(driven_sys.propagate_all(x))
    S = np.asarray(entropy_bipartite_traj(traj, subsystem=subsystem, n_qubits=4), float)
    viol = np.maximum(0.0, S - S_max)
    return float(np.mean(viol**2))


def _entropy_grad_fd(driven_sys, x, *, S_max: float, subsystem: tuple[int, ...], eps: float) -> np.ndarray:
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    f0 = _cost_entropy(driven_sys, x, S_max=S_max, subsystem=subsystem)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        g[i] = (_cost_entropy(driven_sys, xp, S_max=S_max, subsystem=subsystem) - f0) / eps
    return g


def _infidelity_and_grad(driven_sys, x, observable):
    I, gI = driven_sys.gradient(x, observable)
    return float(np.real(I)), np.asarray(gI, float)


def _lagrangian_and_grad(driven_sys, x, *, observable, lam, S_max, subsystem, fd_eps, rho, eps_infidelity):
    S = _cost_entropy(driven_sys, x, S_max=S_max, subsystem=subsystem)
    gS = _entropy_grad_fd(driven_sys, x, S_max=S_max, subsystem=subsystem, eps=fd_eps)

    I, gI = _infidelity_and_grad(driven_sys, x, observable)
    c = I - eps_infidelity

    L = S + lam * c
    g = gS + lam * gI

    if rho != 0.0:
        L += 0.5 * rho * (c ** 2)
        g += rho * c * gI

    return float(L), np.asarray(g, float)


def _primal_dual_optimise(driven_sys, x0, *, observable, bounds, config: SimulationConfig, subsystem: tuple[int, ...], verbose=True):
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
                subsystem=subsystem,
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
            print(f"[outer {k:02d}] I={I:.3e}, c=I-eps={c:.3e}, lam={lam:.3e}, inner_nit={res.nit}, inner_success={res.success}")

    return x, lam, hist


# -------- public API --------

def dqe_optimisation(config: SimulationConfig) -> SQEResults:
    """
    Exactly analogous interface to sqe_optimisation, but for DQE on 4 qubits:
      - controls: 3 exchange couplings, piecewise-constant over n_time_steps
      - target unitary: DQE(theta_target)
      - entropy: configurable bipartition via config.entropy_subsystem (default (0,1))
    """
    # seeds (same semantics as SQE)
    if config.init_seed is not None:
        np.random.seed(int(config.init_seed))
    elif config.drive_seed is not None or config.J_init_seed is not None:
        seed = config.drive_seed if config.drive_seed is not None else config.J_init_seed
        np.random.seed(int(seed))

    # defaults for DQE
    config.max_drive_strength = 0.0
    config.n_J_channels = 3

    # subsystem for entropy
    subsystem = tuple(config.entropy_subsystem) if config.entropy_subsystem is not None else (0, 1)

    # target + observable
    U = dqe_unitary(config.theta_target)
    target_state = U @ config.initial_state
    observable = np.eye(16, dtype=np.complex128) - np.outer(target_state, target_state.conj())

    # system
    sys = SpinChain(
        spins=4,
        zeeman_splittings=config.zeeman_splittings,
        max_drive_strength=0.0,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )

    nT = config.n_time_steps
    n_disc = config.n_discretisation
    T = config.T

    def pulse_function(x):
        # x has shape (nT * 3,) - coarse optimization parameters
        # Convert to TensorFlow tensor and reshape to coarse grid: (nT, 3)
        x_tf = convert_dtype(x)
        x_coarse = tf.reshape(x_tf, (nT, 3))

        # Expand to fine grid: repeat each coarse step n_disc times
        # Shape: (nT * n_disc, 3)
        x_fine = tf.repeat(x_coarse, repeats=n_disc, axis=0)

        # Fine discretization sizes
        nT_fine = nT * n_disc
        dt_fine = T / nT_fine

        drive_ctrl_amp = convert_dtype(np.zeros((nT_fine, 4)))      # no drive, 4 spins
        J_ctrl_amp = x_fine                                         # 3 NN couplings, fine sampled
        init = convert_dtype(config.initial_state)
        freqs = convert_dtype(config.zeeman_splittings)

        return drive_ctrl_amp, freqs, J_ctrl_amp, init, dt_fine

    driven_sys = sys.pulse_form(pulse_function)

    # init x0
    x0 = config.J_init_scale * np.random.uniform(-1.0, 1.0, size=(nT * 3,))
    x0 = np.clip(x0, -1.0, 1.0)
    bounds = [(-1.0, 1.0)] * len(x0)

    def package(xf, *, success, nit, grad_norm, lam_final=None, history=None, baseline=None):
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

    # baseline
    def run_baseline():
        res = minimize(
            driven_sys.gradient,
            x0=x0,
            args=(observable,),
            jac=True,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": int(config.maxiter), "gtol": float(config.gtol)},
        )
        grad_norm = float(np.linalg.norm(res.jac)) if res.jac is not None else 0.0
        return package(np.asarray(res.x, float), success=bool(res.success), nit=int(res.nit), grad_norm=grad_norm)

    # constrained
    def run_entropy():
        xf, lamf, hist = _primal_dual_optimise(
            driven_sys, x0,
            observable=observable,
            bounds=bounds,
            config=config,
            subsystem=subsystem,
            verbose=True,
        )
        I_final, gI_final = driven_sys.gradient(xf, observable)
        grad_norm = float(np.linalg.norm(np.asarray(gI_final, float)))
        success = bool(hist) and (float(hist[-1]["constraint_residual"]) <= 0.0)
        return package(xf, success=success, nit=len(hist), grad_norm=grad_norm, lam_final=float(lamf), history=hist)

    if config.run_mode == "infidelity":
        return run_baseline()

    if config.run_mode == "entropy":
        return run_entropy()

    if config.run_mode == "both":
        base = run_baseline()
        ent = run_entropy()
        ent.baseline = base
        return ent

    raise ValueError(f"Unknown run_mode: {config.run_mode!r}")
