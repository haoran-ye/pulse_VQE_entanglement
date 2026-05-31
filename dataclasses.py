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
    # ---- Physical system ----
    zeeman_splittings: np.ndarray           # Larmor frequencies per spin (MHz)
    initial_state: Optional[np.ndarray] = None   # Hilbert-space state vector (or ρ₀ for noise)
    theta_target: float = 0.25 * np.pi     # Target rotation angle (radians)

    # ---- Time and discretisation ----
    n_time_steps: int = 100                # Coarse-grid points (one parameter per J channel)
    n_discretisation: int = 1             # Sub-steps per coarse point; n_fine = n_steps × n_disc
    T: float = 1000.0                     # Total gate time (ns)

    # ---- Microwave drive (disabled in J-only experiments) ----
    max_drive_strength: float = 0.0
    drive_init_scale: float = 0.0
    drive_seed: Optional[int] = None

    # ---- Exchange coupling (J) bounds and initialisation ----
    J_max: float = 10.0                   # Physical upper bound (MHz); optimiser works in [-1, 1]
    J_min: float = 0.0                    # Physical lower bound (MHz)
    J_init_scale: float = 1e-3            # Amplitude of random perturbation from J_init_mean
    J_init_mean: float = 0.0             # Centre of initial parameter range in [-1, 1]
    J_init_seed: Optional[int] = None    # RNG seed for reproducible initialisation
    feromagnetic: bool = False            # Sign convention for J (note: intentional library spelling)

    x0_override: np.ndarray | None = None   # Skip random init and use this vector directly

    # ---- Baseline optimisation settings ----
    gtol: float = 1e-12                  # L-BFGS-B gradient norm convergence threshold
    maxiter: int = 200                   # Max iterations for the baseline (infidelity-only) run
    entanglement_optimisation: bool = False  # If True: run primal-dual entropy loop after baseline
    optimisation_target: OptimisationTarget = "state"

    # ---- Lindblad noise (sqe_state_noise only; post-hoc evaluation) ----
    noise_type: NoiseType = "none"       # "none", "dephasing", or "depolarising"
    noise_rate: float = 0.0             # Lindblad collapse rate γ (same units as frequencies, MHz)

    # ---- Augmented Lagrangian / primal-dual hyperparameters ----
    S_max: float = 0.05                  # Entanglement cap (bits for state; gate measure units for gate)
    eps_infidelity: float = 1e-8         # Infidelity tolerance ε; constraint is I ≤ ε
    rho: float = 10.0                    # Penalty coefficient for the augmented term (ρ/2)(I−ε)²
    dual_lr: float = 1e4                 # Dual variable update step size α; λ ← max(0, λ + α(I−ε))
    n_outer: int = 20                    # Number of outer primal-dual iterations
    inner_maxiter: int = 200             # L-BFGS-B iterations per inner Lagrangian minimisation
    fd_eps: float = 1e-5                 # Finite-difference step size for entropy gradient

    # ---- Entanglement measure selection ----
    state_ent_measure: str = "von_neumann"         # Key in STATE_MEASURES registry
    gate_ent_measure: str = "operator_entanglement"  # Key in GATE_MEASURES registry
    extra_ent_measures: list = field(default_factory=list)  # Extra measure names to pre-compute and plot

    entropy_subsystem: Optional[Sequence[int]] = None  # Bipartition subsystem A (DQE; unused if None)
    n_J_channels: int = 1               # Set automatically by each optimiser: 1 (SQE) or 3 (DQE)

    dqe_plot_partitions: list = field(default_factory=list)  # Extra bipartitions to add to DQE plots


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