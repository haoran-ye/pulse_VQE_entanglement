import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

ROOT = Path.cwd().resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pulse_vqe_entanglement.met import METStorage, scan_MET_over_theta

# ----------------------------
# Sweep settings
# ----------------------------
N_THETA = 50
theta_values = np.linspace(0.0, 0.5 * np.pi, N_THETA)

LOG_INFIDELITY_THRESHOLDS = np.arange(-2.0, -16.0, -2.0)
INFIDELITY_THRESHOLDS = 10 ** LOG_INFIDELITY_THRESHOLDS

time_schedule = [
    (1000.0, 100.0),
    (100.0,   50.0),
    (10.0,    10.0),
    (1.0,      1.0),
]
T_start = 3000.0

# Total progress = N_THRESH * N_THETA * N_RUNS
N_RUNS = 4          # average over these runs
N_INITS = 8        # per T, any-success over these random initialisations

# init policy
pulse_init_cutoff = 0.1 * np.pi
pulse_init_scale_zero = 0.0
pulse_init_scale_random = 1e-2

# default: fully random starts for each init
base_seed = None

# shared simulation/optimisation config
cfg_kwargs = dict(
    N_SEGMENTS=100,
    PTS_PER_SEG=1,
    snapshot_every=50,
    max_iter=500,
    B1=0.0,
    B2=8.0,
    Jmax_mhz=5.0,
    entanglement_on=False,
)

# ----------------------------
# Storage
# ----------------------------
storage_root = Path("../results") / "met_scan"
storage = METStorage(storage_root)

storage.write_root_meta({
    "theta_values": theta_values.tolist(),
    "infidelity_thresholds": [float(x) for x in INFIDELITY_THRESHOLDS],
    "time_schedule": [(float(a), float(b)) for a, b in time_schedule],
    "T_start": float(T_start),
    "N_RUNS": int(N_RUNS),
    "N_INITS": int(N_INITS),
    "base_seed": base_seed,
    "cfg_kwargs": cfg_kwargs,
    "pulse_init_policy": {
        "cutoff": float(pulse_init_cutoff),
        "scale_zero": float(pulse_init_scale_zero),
        "scale_random": float(pulse_init_scale_random),
    },
})

# ----------------------------
# One global progress bar
# ----------------------------
TOTAL_RUNS = len(INFIDELITY_THRESHOLDS) * len(theta_values) * int(N_RUNS)

with tqdm(total=TOTAL_RUNS, desc="Total runs", leave=True) as pbar:
    for I_thresh in INFIDELITY_THRESHOLDS:
        scan_MET_over_theta(
            theta_values=theta_values,
            infidelity_threshold=float(I_thresh),
            T_start=float(T_start),
            time_schedule=time_schedule,
            n_runs=int(N_RUNS),
            n_inits=int(N_INITS),
            pulse_init_cutoff=float(pulse_init_cutoff),
            pulse_init_scale_zero=float(pulse_init_scale_zero),
            pulse_init_scale_random=float(pulse_init_scale_random),
            cfg_kwargs=cfg_kwargs,
            base_seed=base_seed,         # None => fully random
            suppress_output=True,        # nothing printed from optimisation
            storage=storage,             # saves per-run + met_table
            pbar=pbar,                   # <-- one bar across thresholds too
        )