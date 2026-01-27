import numpy as np
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.met import scan_MET_over_theta

theta_values = np.linspace(0.0, 0.5, 40) * np.pi  
log_infidelity_thresholds = np.arange(-0.5, -13.0, -0.5)
infidelity_thresholds = 10**log_infidelity_thresholds

base_config = SimulationConfig(
    zeeman_splittings=np.array([0.0, 8.0]),
    initial_state=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128),
    theta_target=0.0,  # Will be updated for each theta value
    n_time_steps=100,  # Will be adjusted dynamically to keep dt >= 2ns
    T=1000.0,          # Will be adjusted during MET search
    max_drive_strength=0.0,
    drive_init_scale=0.0,
    J_max=10.0,
    J_min=0.0,
    J_init_scale=1e-2,
    feromagnetic=False,
    gtol=1e-20,  # Initial gradient tolerance (auto-reduces if needed)
)

T_start = 2000.0  # Starting time for MET search (ns)

# Time schedule: [(T_min, step_size), ...]
time_schedule = [
    (100.0, 100.0),   # Coarse: 100ns steps down to 100ns
    (50.0, 20.0),     # Medium: 20ns steps down to 50ns
    (5.0, 5.0),       # Fine: 5ns steps down to 5ns
    (2.0, 2.0),       # Ultra-fine: 2ns steps down to 2ns
]

n_runs = 1        # Number of runs per theta for averaging
n_inits = 2       # Number of initialization attempts per T value
min_dt = 0.1      # Minimum time step (ns) - ensures dt >= 0.1ns

results_dir = Path("/Users/yehaoran/Desktop/VQE_Chem/pulse_vqe_entanglement/results/met_scan_dense")
base_seed = 42

results = scan_MET_over_theta(
    theta_values=theta_values,
    infidelity_thresholds=infidelity_thresholds,
    base_config=base_config,
    T_start=T_start,
    time_schedule=time_schedule,
    n_runs=n_runs,
    n_inits=n_inits,
    min_dt=min_dt,
    results_dir=results_dir,
    base_seed=base_seed,
    suppress_output=True,
)