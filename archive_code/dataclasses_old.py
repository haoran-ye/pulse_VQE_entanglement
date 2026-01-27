from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple


@dataclass
class SimulationConfig:
    theta: float = 0.4 * np.pi
    T_ns: float = 1000.0
    N_SEGMENTS: int = 100
    PTS_PER_SEG: int = 1
    max_iter: int = 200
    snapshot_every: int = 3
    pulse_init_scale: float = 0.2
    seed: Optional[int] = None
    infidelity_threshold: float = 1e-5
    B1: float = 0.0
    B2: float = 8.0
    entanglement_on: bool = False
    Jmax_mhz: Optional[float] = 5.0


@dataclass
class SQEResults:
    config: SimulationConfig
    x0: np.ndarray
    x_opt: np.ndarray
    I_final: float
    res: Any
    snapshot_data: List[Tuple[str, np.ndarray, np.ndarray]]
    pulse_snapshots: List[Tuple[str, np.ndarray]]
    entropy_snapshots: Optional[List[Tuple[str, np.ndarray, np.ndarray]]]
    driven_device: Any
    total_iterations: int = 0
