# pulse_vqe_entanglement/dataclasses.py
"""
Configuration and results dataclasses for quantum gate optimization.

Supports both Single Qubit Excitation (SQE) and Double Qubit Excitation (DQE)
with optional entanglement entropy constraints via primal-dual optimization.

Unit Convention
---------------
- Frequencies (Zeeman splittings, J couplings): MHz
- Time (input/output): nanoseconds (ns)
- Time (internal computation): microseconds (μs)

The qugradlab library expects frequencies in MHz and time in μs. Since user-facing
time is in ns, we convert: dt_μs = dt_ns × NS_TO_US where NS_TO_US = 1e-3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Any, Sequence
import numpy as np


# Unit conversion: nanoseconds to microseconds
NS_TO_US = 1e-3  # 1 ns = 0.001 μs

RunMode = Literal["infidelity", "entropy", "both"]
OptimisationTarget = Literal["state", "gate"]


@dataclass
class SimulationConfig:
    """
    Configuration for quantum gate optimization with piecewise-constant pulses.

    The pulse control is discretized into n_time_steps coarse segments over total
    time T. State evolution uses n_discretisation fine steps per coarse segment
    (piecewise constant interpolation), while optimization parameters remain coarse.

    Parameters
    ----------
    zeeman_splittings : np.ndarray
        Zeeman energy splittings for each spin (MHz).
    initial_state : np.ndarray
        Initial quantum state vector.
    theta_target : float
        Target rotation angle for gate operation (radians). Default: π/4.
    n_time_steps : int
        Number of coarse time segments for optimization. Default: 100.
    n_discretisation : int
        Fine discretization multiplier per coarse step. Total fine steps =
        n_time_steps × n_discretisation. Default: 1 (no extra discretization).
    T : float
        Total pulse duration (ns). Default: 1000.0.

    Control Parameters
    ------------------
    max_drive_strength : float
        Maximum microwave drive amplitude (MHz). Default: 0.0 (no drive).
    drive_init_scale : float
        Initial random scaling for drive controls. Default: 0.0.
    drive_seed : Optional[int]
        Random seed for drive initialization. Default: None.
    J_max : float
        Maximum exchange coupling strength (MHz). Default: 10.0.
    J_min : float
        Minimum exchange coupling strength (MHz). Default: 0.0.
    J_init_scale : float
        Initial random scaling for exchange controls. Default: 1e-3.
    J_init_seed : Optional[int]
        Random seed for exchange initialization. Default: None.
    feromagnetic : bool
        Whether to use ferromagnetic convention (positive J). Default: False.

    Optimization Parameters
    -----------------------
    gtol : float
        Gradient tolerance for L-BFGS-B. Default: 1e-5.
    maxiter : int
        Maximum iterations for baseline optimization. Default: 30.
    run_mode : RunMode
        Optimization mode: "infidelity" (minimize infidelity only),
        "entropy" (entropy-constrained), or "both" (run both). Default: "infidelity".

    Entropy Constraint (Primal-Dual)
    ---------------------------------
    S_max : float
        Maximum allowed entanglement entropy (bits). Default: 0.05.
    eps_infidelity : float
        Target infidelity threshold for constraint. Default: 1e-8.
    rho : float
        Augmented Lagrangian penalty parameter. Default: 10.0.
    dual_lr : float
        Dual variable update learning rate. Default: 1e4.
    n_outer : int
        Number of outer primal-dual iterations. Default: 20.
    inner_maxiter : int
        Maximum inner L-BFGS-B iterations per outer step. Default: 200.
    fd_eps : float
        Finite difference step for entropy gradient. Default: 1e-5.

    System-Specific
    ---------------
    init_seed : Optional[int]
        Master random seed (overrides drive_seed and J_init_seed). Default: None.
    entropy_subsystem : Optional[Sequence[int]]
        Qubit indices for subsystem A in bipartite entropy S(A:B).
        For SQE (2 qubits): (0,) computes S of qubit 0.
        For DQE (4 qubits): (0,1) computes S of qubits 0&1 vs 2&3.
        Default: None (uses system-specific default).
    n_J_channels : int
        Number of independent exchange coupling channels.
        SQE: 1 (single coupling), DQE: 3 (nearest-neighbor chain). Default: 1.
    """

    # System and target
    zeeman_splittings: np.ndarray
    initial_state: Optional[np.ndarray] = None
    theta_target: float = 0.25 * np.pi

    # Time discretization
    n_time_steps: int = 100
    n_discretisation: int = 1
    T: float = 1000.0

    # Microwave drive controls (typically disabled)
    max_drive_strength: float = 0.0
    drive_init_scale: float = 0.0
    drive_seed: Optional[int] = None

    # Exchange coupling controls
    J_max: float = 10.0
    J_min: float = 0.0
    J_init_scale: float = 1e-3
    J_init_seed: Optional[int] = None
    feromagnetic: bool = False

    # Baseline optimization
    gtol: float = 1e-10
    maxiter: int = 30
    run_mode: RunMode = "infidelity"
    optimisation_target: OptimisationTarget = "state"

    # Entropy-constrained optimization
    S_max: float = 0.05
    eps_infidelity: float = 1e-8
    rho: float = 10.0
    dual_lr: float = 1e4
    n_outer: int = 20
    inner_maxiter: int = 200
    fd_eps: float = 1e-5

    # Reproducibility and system configuration
    init_seed: Optional[int] = None
    entropy_subsystem: Optional[Sequence[int]] = None
    n_J_channels: int = 1


@dataclass
class SQEResults:
    """
    Results from quantum gate optimization (SQE or DQE).

    Note: Despite the name "SQE", this dataclass is used for both Single Qubit
    Excitation (SQE) and Double Qubit Excitation (DQE) results for backward
    compatibility.

    Attributes
    ----------
    config : SimulationConfig
        Configuration used for this optimization run.
    x0 : np.ndarray
        Initial control parameters (coarse discretization).
        Shape: (n_time_steps × n_J_channels,)
    x_final : np.ndarray
        Optimized control parameters (coarse discretization).
        Shape: (n_time_steps × n_J_channels,)
    val0 : np.ndarray
        Initial infidelity trajectory (fine discretization).
        Shape: (n_time_steps × n_discretisation + 1,)
    val_final : np.ndarray
        Final infidelity trajectory (fine discretization).
        Shape: (n_time_steps × n_discretisation + 1,)
    initial_traj : np.ndarray
        Initial state trajectory (fine discretization).
        Shape: (dim, n_time_steps × n_discretisation + 1)
    final_traj : np.ndarray
        Final state trajectory (fine discretization).
        Shape: (dim, n_time_steps × n_discretisation + 1)
    gradient_norm : float
        Final gradient norm at convergence.
    n_iterations : int
        Number of optimization iterations performed.
    optimization_success : bool
        Whether optimization converged successfully.
    lam_final : Optional[float]
        Final Lagrange multiplier (entropy-constrained mode only).
    history : Optional[list[dict[str, Any]]]
        Per-iteration history (entropy-constrained mode only).
    baseline : Optional[SQEResults]
        Baseline infidelity-only result (when run_mode="both").
    """

    config: SimulationConfig

    x0: np.ndarray
    x_final: np.ndarray

    val0: np.ndarray
    val_final: np.ndarray

    initial_traj: np.ndarray
    final_traj: np.ndarray

    gradient_norm: float = 0.0
    n_iterations: int = 0
    optimization_success: bool = False

    lam_final: Optional[float] = None
    history: Optional[list[dict[str, Any]]] = None
    baseline: Optional["SQEResults"] = None


@dataclass
class SQEGateResults:
    """
    Results from gate-based quantum gate optimization.

    Supports both gate infidelity and state infidelity optimization modes.
    Field semantics depend on config.optimisation_target:
    - "gate": infid_* contains gate infidelity, ent_* contains operator entanglement.
    - "state": infid_* contains state infidelity, ent_* contains entanglement entropy.

    Attributes
    ----------
    config : SimulationConfig
        Configuration used for this optimization run.
    x0 : np.ndarray
        Initial control parameters, shape (n_time_steps x n_J_channels,).
    x_final : np.ndarray
        Optimized control parameters, shape (n_time_steps x n_J_channels,).
    infid_initial : np.ndarray
        Infidelity trajectory for initial parameters, shape (n_fine + 1,).
    infid_final : np.ndarray
        Infidelity trajectory for optimized parameters, shape (n_fine + 1,).
    ent_initial : np.ndarray
        Entanglement measure trajectory for initial parameters, shape (n_fine + 1,).
    ent_final : np.ndarray
        Entanglement measure trajectory for optimized parameters, shape (n_fine + 1,).
    gradient_norm : float
        Final gradient norm at convergence.
    n_iterations : int
        Number of optimization iterations performed.
    optimization_success : bool
        Whether optimization converged successfully.
    lam_final : Optional[float]
        Final Lagrange multiplier (constrained mode only).
    history : Optional[list[dict[str, Any]]]
        Per-iteration history (constrained mode only).
    baseline : Optional[SQEGateResults]
        Baseline infidelity-only result (when run_mode="both").
    """

    config: SimulationConfig

    x0: np.ndarray
    x_final: np.ndarray

    infid_initial: np.ndarray
    infid_final: np.ndarray

    ent_initial: np.ndarray
    ent_final: np.ndarray

    gradient_norm: float = 0.0
    n_iterations: int = 0
    optimization_success: bool = False

    lam_final: Optional[float] = None
    history: Optional[list[dict[str, Any]]] = None
    baseline: Optional["SQEGateResults"] = None
