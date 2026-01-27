# pulse_vqe_entanglement/dataclasses.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Any, Sequence
import numpy as np


RunMode = Literal["infidelity", "entropy", "both"]


@dataclass
class SimulationConfig:
    # --- system / target ---
    zeeman_splittings: np.ndarray
    initial_state: np.ndarray

    theta_target: float = 0.25 * np.pi
    n_time_steps: int = 100
    n_discretisation: int = 1
    T: float = 1000.0

    # --- controls / device ---
    max_drive_strength: float = 0.0
    drive_init_scale: float = 0.0
    drive_seed: Optional[int] = None

    J_max: float = 10.0
    J_min: float = 0.0
    J_init_scale: float = 1e-3
    J_init_seed: Optional[int] = None

    feromagnetic: bool = False

    # --- optimisation ---
    gtol: float = 1e-5
    maxiter: int = 30
    run_mode: RunMode = "infidelity"

    # --- entropy-constrained (primal-dual) ---
    S_max: float = 0.05
    eps_infidelity: float = 1e-8
    rho: float = 10.0
    dual_lr: float = 1e4
    n_outer: int = 20
    inner_maxiter: int = 200
    fd_eps: float = 1e-5

    # --- reproducibility ---
    init_seed: Optional[int] = None

    # --- NEW (DQE/general): which bipartition for entropy ---
    # Example for 2 qubits: (0,) vs (1,) -> subsystem=(0,)
    # Example for 4 qubits: subsystem=(0,1) gives S(12|34) if qubits ordered 0,1,2,3
    entropy_subsystem: Optional[Sequence[int]] = None

    # --- NEW (DQE/general): number of exchange couplings (channels) ---
    # SQE (2 spins): 1, DQE (4 spins NN chain): 3
    n_J_channels: int = 1


@dataclass
class SQEResults:
    # Keep name for backwards compatibility: used for SQE and also DQE.
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

    # entropy constrained extras
    lam_final: Optional[float] = None
    history: Optional[list[dict[str, Any]]] = None

    # optional baseline result (fidelity-only)
    baseline: Optional["SQEResults"] = None
