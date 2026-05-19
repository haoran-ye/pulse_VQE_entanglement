"""
Single Qubit Excitation (SQE) optimisation for 2-spin systems.

Closed-system optimisation is performed as before.

Optional Lindblad noise evaluation is added after constructing the driven
QuGrad system, since SpinChain itself is from QuGradLab and does not directly
support Lindblad conversion.

config.noise_type:
    "none"
    "dephasing"
    "depolarising"

config.noise_rate:
    Lindblad collapse-operator rate.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from scipy.linalg import expm

from qugradlab.systems.semiconducting.esr import SpinChain
from .dataclasses import SimulationConfig, SQEResults, NS_TO_US
from .entanglement_measures import STATE_MEASURES


# =============================================================================
# Basic helpers
# =============================================================================

def convert_dtype(np_array) -> tf.Tensor:
    return tf.cast(tf.convert_to_tensor(np_array), dtype=tf.complex128)


def single_qubit_excitation(theta: float) -> np.ndarray:
    """
    SQE rotation in the |01> <-> |10> subspace.

    Basis order:
        |00>, |01>, |10>, |11>
    """
    return np.array(
        [[1, 0, 0, 0],
         [0, np.cos(theta), -np.sin(theta), 0],
         [0, np.sin(theta),  np.cos(theta), 0],
         [0, 0, 0, 1]],
        dtype=np.complex128,
    )


def _initialise_seed(config: SimulationConfig) -> None:
    if config.J_init_seed is not None:
        np.random.seed(int(config.J_init_seed))


def _initialise_x0(config: SimulationConfig) -> np.ndarray:
    n = config.n_time_steps * config.n_J_channels
    x0 = config.J_init_scale * np.random.uniform(-1.0, 1.0, size=n) + config.J_init_mean
    return np.clip(x0, -1.0, 1.0)


def _get_noise_type(config: SimulationConfig) -> str:
    return str(getattr(config, "noise_type", "none")).lower()


def _get_noise_rate(config: SimulationConfig) -> float:
    return float(getattr(config, "noise_rate", 0.0))


# =============================================================================
# System construction
# =============================================================================

def _make_system_and_observable(config: SimulationConfig):
    """
    Create the closed-system SpinChain, target state, and state-infidelity
    observable.

    Returns
    -------
    sys : SpinChain
    observable : np.ndarray
        I - |psi_target><psi_target|
    target_state : np.ndarray
    """
    I2 = np.eye(2, dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    Z1 = np.kron(Z, I2)
    Z2 = np.kron(I2, Z)

    B = np.array(config.zeeman_splittings, dtype=np.float64)
    B_avg = np.mean(B)
    deltaB = B - B_avg

    U_rot = expm(
        -1j
        * np.pi
        * (deltaB[0] * Z1 + deltaB[1] * Z2)
        * config.T
        * NS_TO_US
    )

    target_state = U_rot @ single_qubit_excitation(config.theta_target) @ config.initial_state
    observable = np.eye(4, dtype=np.complex128) - np.outer(target_state, target_state.conj())

    sys = SpinChain(
        spins=2,
        zeeman_splittings=deltaB,
        max_drive_strength=config.max_drive_strength,
        J_max=config.J_max,
        J_min=config.J_min,
        feromagnetic=config.feromagnetic,
        use_graph=True,
    )

    return sys, observable, target_state


def _make_driven_sys(
    config: SimulationConfig,
    sys: SpinChain,
    *,
    lindblad_initial_state: bool = False,
):
    """
    Create a driven system with piecewise-constant exchange pulse.

    If lindblad_initial_state=False:
        initial state is |psi0>.

    If lindblad_initial_state=True:
        initial state is rho0 = |psi0><psi0|.

    Important:
        The Lindblad conversion itself is not done here. This function only
        creates the driven system with the correct initial object.
    """
    n_steps = config.n_time_steps
    n_disc = config.n_discretisation
    n_fine = n_steps * n_disc

    dt_fine_us = (config.T / n_fine) * NS_TO_US

    if lindblad_initial_state:
        initial_state = np.outer(config.initial_state, config.initial_state.conj())
    else:
        initial_state = config.initial_state

    def pulse_function(x):
        x_coarse = tf.reshape(convert_dtype(x), (n_steps, config.n_J_channels))
        x_fine = tf.repeat(x_coarse, repeats=n_disc, axis=0)

        return (
            convert_dtype(np.zeros((n_fine, 2))),       # drive amplitudes
            convert_dtype(config.zeeman_splittings),    # drive frequencies
            x_fine,                                     # J amplitudes
            convert_dtype(initial_state),               # |psi0> or rho0
            dt_fine_us * 2 * np.pi,                     # timestep
        )

    return sys.pulse_form(pulse_function)


# =============================================================================
# Lindblad noise construction
# =============================================================================

def _make_lindblad_operators(config: SimulationConfig) -> np.ndarray | None:
    """
    Construct collapse operators for two-qubit Lindblad noise.

    noise_type = "none":
        no Lindblad operators.

    noise_type = "dephasing":
        L = sqrt(gamma) Z_i.

    noise_type = "depolarising":
        L = sqrt(gamma / 3) X_i, Y_i, Z_i.
    """
    noise_type = _get_noise_type(config)
    gamma = _get_noise_rate(config)

    if noise_type in ("none", "no", "false", "") or gamma == 0.0:
        return None

    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    X1, X2 = np.kron(X, I2), np.kron(I2, X)
    Y1, Y2 = np.kron(Y, I2), np.kron(I2, Y)
    Z1, Z2 = np.kron(Z, I2), np.kron(I2, Z)

    if noise_type == "dephasing":
        return np.array(
            [
                np.sqrt(gamma) * Z1,
                np.sqrt(gamma) * Z2,
            ],
            dtype=np.complex128,
        )

    if noise_type == "depolarising":
        prefactor = np.sqrt(gamma / 3.0)
        return np.array(
            [
                prefactor * X1,
                prefactor * Y1,
                prefactor * Z1,
                prefactor * X2,
                prefactor * Y2,
                prefactor * Z2,
            ],
            dtype=np.complex128,
        )

    raise ValueError(
        f"Unknown noise_type={noise_type!r}. "
        "Valid options are 'none', 'dephasing', 'depolarising'."
    )


def _make_driven_lindblad_sys(config: SimulationConfig, sys: SpinChain):
    """
    Correct Lindblad construction.

    Since SpinChain is from QuGradLab, first create a driven QuGrad system using
    pulse_form, with density-matrix initial state, then add Lindblad terms.
    """
    L0 = _make_lindblad_operators(config)

    if L0 is None:
        return None

    driven_lindblad_base = _make_driven_sys(
        config,
        sys,
        lindblad_initial_state=True,
    )

    return driven_lindblad_base.add_lindblad_terms(L0=L0)


# =============================================================================
# Density-matrix fidelity
# =============================================================================

def _density_matrix_fidelity(rho: np.ndarray, target_state: np.ndarray) -> float:
    """
    Fidelity with pure target state:

        F = <psi_target|rho|psi_target>
          = Tr(|psi_target><psi_target| rho)
    """
    rho = np.asarray(rho, dtype=np.complex128)
    target_state = np.asarray(target_state, dtype=np.complex128)

    fidelity = target_state.conj() @ rho @ target_state
    return float(np.real(fidelity))


def _density_matrix_infidelity(rho: np.ndarray, target_state: np.ndarray) -> float:
    return float(1.0 - _density_matrix_fidelity(rho, target_state))


def _noisy_infidelity(driven_lindblad_sys, x: np.ndarray, target_state: np.ndarray) -> float:
    rho_final = np.asarray(driven_lindblad_sys.propagate(x))
    return _density_matrix_infidelity(rho_final, target_state)


def _noisy_infidelity_trajectory(
    driven_lindblad_sys,
    x: np.ndarray,
    target_state: np.ndarray,
) -> np.ndarray:
    """
    Compute noisy infidelity trajectory from rho(t).

    Expected trajectory shape:
        (4, 4, n_fine + 1)
    """
    rho_traj = np.asarray(driven_lindblad_sys.propagate_all(x))

    return np.array(
        [
            _density_matrix_infidelity(rho_traj[:, :, k], target_state)
            for k in range(rho_traj.shape[-1])
        ],
        dtype=float,
    )


# =============================================================================
# Result packaging
# =============================================================================

def _package_results(
    *,
    config,
    driven_sys,
    observable,
    x0,
    xf,
    success,
    nit,
    grad_norm,
    lam_final=None,
    history=None,
    baseline=None,
) -> SQEResults:
    return SQEResults(
        config=config,
        x0=np.asarray(x0, float),
        x_final=np.asarray(xf, float),
        val0=np.asarray(
            np.real(driven_sys.evolved_expectation_value_all(x0, observable)),
            float,
        ),
        val_final=np.asarray(
            np.real(driven_sys.evolved_expectation_value_all(xf, observable)),
            float,
        ),
        initial_traj=np.asarray(driven_sys.propagate_all(x0)),
        final_traj=np.asarray(driven_sys.propagate_all(xf)),
        gradient_norm=float(grad_norm),
        n_iterations=int(nit),
        optimization_success=bool(success),
        lam_final=lam_final,
        history=history,
        baseline=baseline,
    )


def _attach_noisy_diagnostics(
    results: SQEResults,
    *,
    config,
    driven_lindblad_sys,
    x0,
    xf,
    target_state,
) -> SQEResults:
    """
    Attach optional Lindblad diagnostics.

    Requires SQEResults to have optional noisy fields.
    """
    if driven_lindblad_sys is None:
        results.noise_type = "none"
        results.noise_rate = 0.0
        results.noisy_val0 = None
        results.noisy_val_final = None
        results.noisy_val0_traj = None
        results.noisy_val_final_traj = None
        results.noisy_initial_rho_traj = None
        results.noisy_final_rho_traj = None
        return results

    results.noise_type = _get_noise_type(config)
    results.noise_rate = _get_noise_rate(config)

    results.noisy_val0 = _noisy_infidelity(
        driven_lindblad_sys,
        x0,
        target_state,
    )
    results.noisy_val_final = _noisy_infidelity(
        driven_lindblad_sys,
        xf,
        target_state,
    )

    results.noisy_val0_traj = _noisy_infidelity_trajectory(
        driven_lindblad_sys,
        x0,
        target_state,
    )
    results.noisy_val_final_traj = _noisy_infidelity_trajectory(
        driven_lindblad_sys,
        xf,
        target_state,
    )

    results.noisy_initial_rho_traj = np.asarray(driven_lindblad_sys.propagate_all(x0))
    results.noisy_final_rho_traj = np.asarray(driven_lindblad_sys.propagate_all(xf))

    return results


# =============================================================================
# Entropy-constrained optimisation
# =============================================================================

def _cost_entropy(driven_sys, x, *, S_max: float, ent_traj_fn) -> float:
    traj = np.asarray(driven_sys.propagate_all(x))
    viol = np.maximum(0.0, ent_traj_fn(traj) - S_max)
    return float(np.mean(viol**2))


def _entropy_grad_fd(
    driven_sys,
    x,
    *,
    S_max: float,
    eps: float,
    ent_traj_fn,
    f0: float | None = None,
) -> np.ndarray:
    x = np.asarray(x, float)

    if f0 is None:
        f0 = _cost_entropy(
            driven_sys,
            x,
            S_max=S_max,
            ent_traj_fn=ent_traj_fn,
        )

    g = np.empty_like(x)

    for i in range(len(x)):
        xp = x.copy()
        xp[i] += eps
        fp = _cost_entropy(
            driven_sys,
            xp,
            S_max=S_max,
            ent_traj_fn=ent_traj_fn,
        )
        g[i] = (fp - f0) / eps

    return g


def _lagrangian_and_grad(
    driven_sys,
    x,
    *,
    observable,
    lam,
    S_max,
    fd_eps,
    rho,
    eps_infidelity,
    ent_traj_fn,
):
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
        f0=S,
    )

    infidelity, grad_inf = driven_sys.gradient(x, observable)

    infidelity = max(0.0, float(np.real(infidelity)))
    grad_inf = np.asarray(grad_inf, float)

    c = infidelity - eps_infidelity

    L = S + lam * c + 0.5 * rho * c**2
    gL = gS + (lam + rho * c) * grad_inf

    return float(L), gL


def _primal_dual_optimise(
    driven_sys,
    x0,
    *,
    observable,
    bounds,
    config: SimulationConfig,
    ent_traj_fn,
    verbose: bool = True,
):
    x = np.asarray(x0, float).copy()
    lam = 1.0
    hist = []

    for k in range(int(config.n_outer)):
        res = minimize(
            lambda z: _lagrangian_and_grad(
                driven_sys,
                z,
                observable=observable,
                lam=lam,
                S_max=float(config.S_max),
                fd_eps=float(config.fd_eps),
                rho=float(config.rho),
                eps_infidelity=float(config.eps_infidelity),
                ent_traj_fn=ent_traj_fn,
            ),
            x0=x,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(config.inner_maxiter),
                "gtol": float(config.gtol),
            },
        )

        x = np.asarray(res.x, float)

        S = _cost_entropy(
            driven_sys,
            x,
            S_max=float(config.S_max),
            ent_traj_fn=ent_traj_fn,
        )

        infidelity = np.real(driven_sys.gradient(x, observable)[0])
        c = infidelity - float(config.eps_infidelity)

        lam = max(0.0, lam + float(config.dual_lr) * c)

        hist.append(
            {
                "outer_iter": k,
                "lam": lam,
                "I_final": infidelity,
                "constraint_residual": c,
                "S": S,
                "inner_success": bool(res.success),
                "inner_nit": int(res.nit),
                "inner_fun": float(res.fun),
                "inner_grad_norm": (
                    float(np.linalg.norm(res.jac))
                    if res.jac is not None
                    else np.nan
                ),
                "message": str(res.message),
            }
        )

        if verbose:
            print(
                f"[outer {k:02d}] "
                f"I={infidelity:.3e}, "
                f"lam={lam:.6e}, "
                f"inner_nit={res.nit}, "
                f"inner_success={res.success}, "
                f"constraint_residual={c:.3e}, "
                f"S={S:.3e}"
            )

    return x, lam, hist


# =============================================================================
# Public API
# =============================================================================

def sqe_optimisation(config: SimulationConfig) -> SQEResults:
    """
    Optimise SQE pulse using clean state-vector dynamics.

    If config.noise_type != "none", additionally evaluate the obtained pulses
    under Lindblad density-matrix dynamics.
    """
    _initialise_seed(config)
    config.n_J_channels = 1

    if config.state_ent_measure not in STATE_MEASURES:
        raise ValueError(
            f"Unknown state_ent_measure: {config.state_ent_measure!r}. "
            f"Valid options: {list(STATE_MEASURES)}"
        )

    _, ent_traj_fn = STATE_MEASURES[config.state_ent_measure]

    sys, observable, target_state = _make_system_and_observable(config)

    # Clean driven system for optimisation.
    driven_sys = _make_driven_sys(
        config,
        sys,
        lindblad_initial_state=False,
    )

    # Noisy driven system for post-optimisation Lindblad evaluation.
    driven_lindblad_sys = _make_driven_lindblad_sys(config, sys)

    x0 = _initialise_x0(config)
    bounds = [(-1.0, 1.0)] * len(x0)

    def run_baseline(x_init):
        res = minimize(
            driven_sys.gradient,
            x0=x_init,
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
            f"Clean optimisation: "
            f"I={res.fun:.3e}, "
            f"nit={res.nit}, "
            f"success={res.success}, "
            f"grad_norm={gn:.3e}, "
            f"S={S:.3e}"
        )

        results = _package_results(
            config=config,
            driven_sys=driven_sys,
            observable=observable,
            x0=x_init,
            xf=np.asarray(res.x, float),
            success=bool(res.success),
            nit=int(res.nit),
            grad_norm=gn,
        )

        results = _attach_noisy_diagnostics(
            results,
            config=config,
            driven_lindblad_sys=driven_lindblad_sys,
            x0=x_init,
            xf=np.asarray(res.x, float),
            target_state=target_state,
        )

        if driven_lindblad_sys is not None:
            print(
                f"Noisy evaluation "
                f"({results.noise_type}, rate={results.noise_rate:.3e}): "
                f"I_noisy={results.noisy_val_final:.3e}"
            )

        return results

    def run_entropy(x_init):
        xf, lam, hist = _primal_dual_optimise(
            driven_sys,
            x_init,
            observable=observable,
            bounds=bounds,
            config=config,
            ent_traj_fn=ent_traj_fn,
            verbose=True,
        )

        grad = np.asarray(driven_sys.gradient(xf, observable)[1], float)
        gn = float(np.linalg.norm(grad))

        ok = bool(hist) and float(hist[-1]["constraint_residual"]) <= 0.0

        results = _package_results(
            config=config,
            driven_sys=driven_sys,
            observable=observable,
            x0=x_init,
            xf=xf,
            success=ok,
            nit=len(hist),
            grad_norm=gn,
            lam_final=float(lam),
            history=hist,
        )

        results = _attach_noisy_diagnostics(
            results,
            config=config,
            driven_lindblad_sys=driven_lindblad_sys,
            x0=x_init,
            xf=xf,
            target_state=target_state,
        )

        if driven_lindblad_sys is not None:
            print(
                f"Noisy evaluation "
                f"({results.noise_type}, rate={results.noise_rate:.3e}): "
                f"I_noisy={results.noisy_val_final:.3e}"
            )

        return results

    if not config.entanglement_optimisation:
        return run_baseline(x0)

    baseline = run_baseline(x0.copy())
    constrained = run_entropy(x0.copy())
    constrained.baseline = baseline

    return constrained