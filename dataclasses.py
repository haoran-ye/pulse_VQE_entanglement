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


NS_TO_US = 1e-3

OptimisationTarget = Literal["state", "gate"]
NoiseType = Literal["none", "dephasing", "depolarising"]


@dataclass
class SimulationConfig:
    zeeman_splittings: np.ndarray
    initial_state: Optional[np.ndarray] = None
    theta_target: float = 0.25 * np.pi

    n_time_steps: int = 100
    n_discretisation: int = 1
    T: float = 1000.0

    max_drive_strength: float = 0.0
    drive_init_scale: float = 0.0
    drive_seed: Optional[int] = None

    J_max: float = 10.0
    J_min: float = 0.0
    J_init_scale: float = 1e-3
    J_init_mean: float = 0.0
    J_init_seed: Optional[int] = None
    feromagnetic: bool = False

    x0_override: np.ndarray | None = None    
    
    gtol: float = 1e-12
    maxiter: int = 200
    entanglement_optimisation: bool = False
    optimisation_target: OptimisationTarget = "state"

    # Optional Lindblad noise evaluation
    noise_type: NoiseType = "none"
    noise_rate: float = 0.0

    S_max: float = 0.05
    eps_infidelity: float = 1e-8
    rho: float = 10.0
    dual_lr: float = 1e4
    n_outer: int = 20
    inner_maxiter: int = 200
    fd_eps: float = 1e-5

    state_ent_measure: str = "von_neumann"
    gate_ent_measure: str = "operator_entanglement"
    extra_ent_measures: list = field(default_factory=list)

    entropy_subsystem: Optional[Sequence[int]] = None
    n_J_channels: int = 1

    dqe_plot_partitions: list = field(default_factory=list)


@dataclass
class SQEResults:
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

    # Optional Lindblad/noisy diagnostics
    noise_type: NoiseType = "none"
    noise_rate: float = 0.0

    noisy_val0: Optional[float] = None
    noisy_val_final: Optional[float] = None

    noisy_val0_traj: Optional[np.ndarray] = None
    noisy_val_final_traj: Optional[np.ndarray] = None

    noisy_initial_rho_traj: Optional[np.ndarray] = None
    noisy_final_rho_traj: Optional[np.ndarray] = None


@dataclass
class SQEGateResults:
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