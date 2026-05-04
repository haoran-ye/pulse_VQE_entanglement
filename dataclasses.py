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

from dataclasses import dataclass, field
from typing import Optional, Literal, Any, Sequence
import numpy as np


# Unit conversion: nanoseconds to microseconds
NS_TO_US = 1e-3  # 1 ns = 0.001 μs

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
        Gradient tolerance for L-BFGS-B. Default: 1e-12.
    maxiter : int
        Maximum iterations for baseline optimization. Default: 200.
    entanglement_optimisation : bool
        If False, minimize infidelity only. If True, first minimize infidelity,
        then run the entropy-constrained primal-dual loop from the same x0;
        the baseline infidelity result is stored in results.baseline. Default: False.

    Entropy Constraint (Primal-Dual)
    ---------------------------------
    S_max : float
        Maximum allowed entanglement (in units of the chosen measure). Default: 0.05.
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

    Entanglement Measure Selection
    --------------------------------
    state_ent_measure : str
        State entanglement measure used in sqe_state.py optimization.
        Any key in entanglement_measures.STATE_MEASURES. Default: "von_neumann".
    gate_ent_measure : str
        Gate entanglement measure used in sqe_gate.py optimization.
        Any key in entanglement_measures.GATE_MEASURES. Default: "operator_entanglement".
    extra_ent_measures : list[str]
        Additional measures to plot on separate axes (not used in optimization).
        Can mix state or gate measure names depending on the optimisation_target.
        Default: [] (no extra panels).

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
    J_init_mean: float = 0.0
    J_init_seed: Optional[int] = None
    feromagnetic: bool = False

    # Baseline optimization
    gtol: float = 1e-12
    maxiter: int = 200
    entanglement_optimisation: bool = False
    optimisation_target: OptimisationTarget = "state"

    # Entropy-constrained optimization
    S_max: float = 0.05
    eps_infidelity: float = 1e-8
    rho: float = 10.0
    dual_lr: float = 1e4
    n_outer: int = 20
    inner_maxiter: int = 200
    fd_eps: float = 1e-5

    # Entanglement measure selection
    state_ent_measure: str = "von_neumann"
    gate_ent_measure: str = "operator_entanglement"
    extra_ent_measures: list = field(default_factory=list)

    # Reproducibility and system configuration
    entropy_subsystem: Optional[Sequence[int]] = None
    n_J_channels: int = 1

    # DQE plotting: list of bipartitions (tuples of qubit indices) to plot
    # separately. E.g. [(0,1), (0,), (2,3)]. Each generates one extra panel.
    dqe_plot_partitions: list = field(default_factory=list)


@dataclass
class SQEResults:
    """
    Results from state-based SQE/DQE optimization.

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
        Baseline infidelity-only result (when entanglement_optimisation=True).
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
    Results from gate-based SQE optimization.

    Attributes
    ----------
    config : SimulationConfig
        Configuration used for this optimization run.
    x0 : np.ndarray
        Initial control parameters, shape (n_time_steps × n_J_channels,).
    x_final : np.ndarray
        Optimized control parameters, shape (n_time_steps × n_J_channels,).
    infid_initial : np.ndarray
        Infidelity trajectory for initial parameters, shape (n_fine + 1,).
    infid_final : np.ndarray
        Infidelity trajectory for optimized parameters, shape (n_fine + 1,).
    ent_initial : np.ndarray
        Trajectory of config.gate_ent_measure for initial parameters,
        shape (n_fine + 1,).
    ent_final : np.ndarray
        Trajectory of config.gate_ent_measure for optimized parameters,
        shape (n_fine + 1,).
    extra_ent_initial : dict[str, np.ndarray]
        Trajectories for each measure in config.extra_ent_measures (initial x).
    extra_ent_final : dict[str, np.ndarray]
        Trajectories for each measure in config.extra_ent_measures (final x).
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
        Baseline infidelity-only result (when entanglement_optimisation=True).
    """

    config: SimulationConfig

    x0: np.ndarray
    x_final: np.ndarray

    infid_initial: np.ndarray
    infid_final: np.ndarray

    ent_initial: np.ndarray
    ent_final: np.ndarray

    extra_ent_initial: dict = field(default_factory=dict)
    extra_ent_final: dict = field(default_factory=dict)

    gradient_norm: float = 0.0
    n_iterations: int = 0
    optimization_success: bool = False

    lam_final: Optional[float] = None
    history: Optional[list[dict[str, Any]]] = None
    baseline: Optional["SQEGateResults"] = None


@dataclass
class DQEResults:
    """
    Results from Double Qubit Excitation (DQE) state optimization.

    4-spin chain with 3 nearest-neighbor J channels. The control parameter
    vector x has shape (n_time_steps * 3,) — the 3 channels are interleaved
    as x.reshape(n_time_steps, 3).

    Attributes
    ----------
    config : SimulationConfig
        Configuration used for this optimization run.
    x0 : np.ndarray
        Initial control parameters, shape (n_time_steps * 3,).
    x_final : np.ndarray
        Optimized control parameters, shape (n_time_steps * 3,).
    val0 : np.ndarray
        Initial infidelity trajectory (fine grid), shape (n_fine + 1,).
    val_final : np.ndarray
        Final infidelity trajectory (fine grid), shape (n_fine + 1,).
    initial_traj : np.ndarray
        Initial state trajectory, shape (16, n_fine + 1).
    final_traj : np.ndarray
        Final state trajectory, shape (16, n_fine + 1).
    gradient_norm : float
        Final gradient norm at convergence.
    n_iterations : int
        Number of optimization iterations performed.
    optimization_success : bool
        Whether optimization converged successfully.
    lam_final : Optional[float]
        Final Lagrange multiplier (entropy-constrained mode only).
    history : Optional[list[dict]]
        Per-outer-iteration diagnostics (entropy-constrained mode only).
    baseline : Optional[DQEResults]
        Infidelity-only result attached when entanglement_optimisation=True.
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
    baseline: Optional["DQEResults"] = None


@dataclass
class DQEGateResults:
    """
    Results from gate-based DQE optimization.

    4-spin chain with 3 nearest-neighbor J channels.  The control parameter
    vector x has shape (n_time_steps * 3,).

    Attributes
    ----------
    config : SimulationConfig
    x0 : np.ndarray
        Initial control parameters, shape (n_time_steps * 3,).
    x_final : np.ndarray
        Optimized control parameters, shape (n_time_steps * 3,).
    infid_initial : np.ndarray
        Gate infidelity trajectory for initial parameters, shape (n_fine + 1,).
    infid_final : np.ndarray
        Gate infidelity trajectory for optimized parameters, shape (n_fine + 1,).
    ent_initial : np.ndarray
        Trajectory of config.gate_ent_measure for initial parameters,
        shape (n_fine + 1,).
    ent_final : np.ndarray
        Trajectory of config.gate_ent_measure for optimized parameters,
        shape (n_fine + 1,).
    extra_ent_initial : dict[str, np.ndarray]
        Trajectories for each measure in config.extra_ent_measures (initial x).
    extra_ent_final : dict[str, np.ndarray]
        Trajectories for each measure in config.extra_ent_measures (final x).
    gradient_norm : float
    n_iterations : int
    optimization_success : bool
    lam_final : Optional[float]
    history : Optional[list[dict]]
    baseline : Optional[DQEGateResults]
        Infidelity-only result attached when entanglement_optimisation=True.
    """

    config: SimulationConfig

    x0: np.ndarray
    x_final: np.ndarray

    infid_initial: np.ndarray
    infid_final: np.ndarray

    ent_initial: np.ndarray
    ent_final: np.ndarray

    extra_ent_initial: dict = field(default_factory=dict)
    extra_ent_final: dict = field(default_factory=dict)

    gradient_norm: float = 0.0
    n_iterations: int = 0
    optimization_success: bool = False

    lam_final: Optional[float] = None
    history: Optional[list[dict[str, Any]]] = None
    baseline: Optional["DQEGateResults"] = None
