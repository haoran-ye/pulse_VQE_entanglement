"""
Example usage of MET (Minimum Excitation Time) scanning.

This script demonstrates how to run a MET scan over theta values and multiple
infidelity thresholds to find the minimum evolution time required.
"""

import numpy as np
from pathlib import Path

from dataclasses import SimulationConfig
from met import scan_MET_over_theta
from pulse_vqe_entanglement.plotting_state import plot_MET_vs_theta, plot_MET_vs_theta_from_dir

# ============================================================================
# Configuration
# ============================================================================

# Theta values to scan
theta_values = np.linspace(0.0, 0.5, 6) * np.pi  # 0 to 0.5π

# Multiple infidelity thresholds to test
infidelity_thresholds = np.array([1e-4, 1e-5, 1e-6])

# Base configuration (theta_target will be updated for each theta value)
base_config = SimulationConfig(
    zeeman_splittings=np.array([0.0, 8.0]),
    initial_state=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128),
    theta_target=0.0,  # Will be updated for each theta value
    n_time_steps=100,  # Will be adjusted dynamically based on dt constraint
    T=1000.0,          # Will be adjusted during MET search
    max_drive_strength=0.0,
    drive_init_scale=0.0,
    J_max=10.0,
    J_min=0.0,
    J_init_scale=1e-2,
    feromagnetic=False,
)

# MET search parameters
T_start = 2000.0  # Starting evolution time (ns)

# Time schedule: [(T_min, step_size), ...]
# Start with coarse steps, refine progressively
time_schedule = [
    (100.0, 100.0),   # From T_start down to 100ns in steps of 100ns
    (50.0, 20.0),     # From last success down to 50ns in steps of 20ns
    (10.0, 5.0),      # From last success down to 10ns in steps of 5ns
    (1.0, 1.0),       # From last success down to 1ns in steps of 1ns
]

n_runs = 3        # Number of runs per theta for averaging
n_inits = 5       # Number of initialization attempts per T value
min_dt = 2.0      # Minimum time step (ns) - ensures dt >= 2ns

# Results directory
results_dir = Path("./results/met_scan")

# Random seed for reproducibility
base_seed = 42

# ============================================================================
# Run MET Scan
# ============================================================================

print("Starting MET scan...")
print(f"Theta values: {theta_values / np.pi}π")
print(f"Infidelity thresholds: {infidelity_thresholds}")
print(f"Number of runs per theta: {n_runs}")
print(f"Number of inits per T: {n_inits}")
print(f"Time schedule: {time_schedule}")
print(f"Results will be saved to: {results_dir}\n")

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

# ============================================================================
# Print Summary
# ============================================================================

print("\n" + "="*60)
print("MET Scan Results Summary:")
print("="*60)
for I_thresh in sorted(results.keys()):
    print(f"\nInfidelity threshold: {I_thresh:.1e}")
    print(f"  Theta values: {results[I_thresh]['theta_values'] / np.pi}π")
    print(f"  MET mean: {results[I_thresh]['MET_mean']} ns")
    print(f"  MET std: {results[I_thresh]['MET_std']} ns")
print("="*60 + "\n")

# ============================================================================
# Plot Results
# ============================================================================

print("Plotting results from scan data...")
plot_MET_vs_theta(results, show_errorbars=True)

print("\nPlotting results from saved files...")
plot_MET_vs_theta_from_dir(
    results_dir=results_dir,
    use="mean",
    show_errorbars=True,
    errorbar_kind="std",
)

print("\nDone!")
