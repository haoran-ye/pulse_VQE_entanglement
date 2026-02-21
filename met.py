"""
Minimum Excitation Time (MET) scanning for single-qubit excitation optimization.

This module implements binary-search-like MET finding with multiple initialization
attempts and optimization schedule control.
"""

from __future__ import annotations

import os
import logging
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

from .dataclasses import SimulationConfig, SQEResults
from .sqe_state import sqe_optimisation


# ============================================================================
# Suppress TensorFlow and optimization noise
# ============================================================================

def _suppress_tensorflow_noise() -> None:
    """Suppress TensorFlow and absl logging."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    try:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("error")
    except Exception:
        pass

    try:
        import tensorflow as tf
        try:
            tf.get_logger().setLevel("ERROR")
            tf.get_logger().propagate = False
            for h in list(tf.get_logger().handlers):
                try:
                    h.setLevel(logging.ERROR)
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass


_suppress_tensorflow_noise()


# ============================================================================
# Silent execution wrapper
# ============================================================================

class _NullWriter:
    """Dummy writer that discards all output."""
    def write(self, _: str) -> int:
        return 0
    def flush(self) -> None:
        pass


def _run_silently(fn):
    """Run function with stdout/stderr suppressed."""
    with redirect_stdout(_NullWriter()), redirect_stderr(_NullWriter()):
        return fn()


# ============================================================================
# Single optimization attempt
# ============================================================================

def _single_optimization_attempt(
    config: SimulationConfig,
    infidelity_threshold: float,
    suppress_output: bool = True,
    min_gtol_factor: float = 100.0,
) -> Tuple[bool, SQEResults]:
    """
    Run optimization with automatic gtol reduction if needed.

    If the optimization fails because gtol is reached but gradient is still large
    (indicating early stopping), this function automatically reduces gtol and retries.

    Parameters
    ----------
    config : SimulationConfig
        Configuration for the optimization. The gtol field will be used and
        potentially modified if retries are needed.
    infidelity_threshold : float
        Target infidelity threshold for success.
    suppress_output : bool
        Whether to suppress optimization output.
    min_gtol_factor : float
        Don't reduce gtol more than this factor below the infidelity threshold.
        Default is 100 (so min_gtol = infidelity_threshold / 100).

    Returns
    -------
    success : bool
        Whether the optimization succeeded (final infidelity <= threshold).
    results : SQEResults
        Full optimization results from the last attempt.
    """
    from dataclasses import replace

    min_gtol = infidelity_threshold / min_gtol_factor
    attempt = 0
    max_attempts = 10  # Prevent infinite loops
    current_config = config

    while attempt < max_attempts:
        attempt += 1

        # Run optimization with current config
        if suppress_output:
            results = _run_silently(lambda: sqe_optimisation(current_config))
        else:
            results = sqe_optimisation(current_config)

        # Check if final infidelity meets threshold
        final_infidelity = float(results.val_final[-1])
        success = final_infidelity <= infidelity_threshold

        if success:
            return True, results

        # Failed - check if we should retry with smaller gtol
        # Only retry if gradient is still large (indicates gtol stopped us early)
        if results.gradient_norm > infidelity_threshold:
            # Reduce gtol by 2 orders of magnitude
            gtol_old = current_config.gtol
            gtol_new = gtol_old / 100.0

            if gtol_new < min_gtol:
                # Can't reduce further, accept failure
                if not suppress_output:
                    print(f"  gtol limit reached ({gtol_new:.1e} < {min_gtol:.1e}), stopping")
                return False, results

            if not suppress_output:
                print(f"  Gradient {results.gradient_norm:.1e} > threshold {infidelity_threshold:.1e}")
                print(f"  Reducing gtol: {gtol_old:.1e} → {gtol_new:.1e}, retrying...")

            # Create new config with reduced gtol
            current_config = replace(current_config, gtol=gtol_new)
            # Continue loop to retry with smaller gtol
        else:
            # Gradient is small but still failed - genuine failure
            # No point retrying with smaller gtol
            return False, results

    # Max attempts reached (shouldn't happen with reasonable settings)
    return False, results


# ============================================================================
# Multiple initialization attempts at fixed T
# ============================================================================

def _try_n_inits(
    base_config: SimulationConfig,
    T: float,
    n_time_steps: int,
    infidelity_threshold: float,
    n_inits: int,
    base_seed: Optional[int] = None,
    suppress_output: bool = True,
) -> Tuple[bool, Optional[SQEResults]]:
    """
    Try n_inits optimization attempts at a given T.

    Success if ANY one of the attempts meets the infidelity threshold.

    Parameters
    ----------
    base_config : SimulationConfig
        Base configuration (will be copied and modified).
    T : float
        Total evolution time.
    n_time_steps : int
        Number of time steps.
    infidelity_threshold : float
        Target infidelity threshold.
    n_inits : int
        Number of initialization attempts.
    base_seed : int, optional
        Base random seed (will increment for each attempt).
    suppress_output : bool
        Whether to suppress optimization output.

    Returns
    -------
    success : bool
        Whether any attempt succeeded.
    best_results : SQEResults or None
        Results from successful attempt, or None if all failed.
    """
    for k in range(n_inits):
        # Create config with updated T, n_time_steps, and seed
        seed_k = None if base_seed is None else base_seed + k

        config = SimulationConfig(
            zeeman_splittings=base_config.zeeman_splittings,
            initial_state=base_config.initial_state,
            theta_target=base_config.theta_target,
            n_time_steps=n_time_steps,
            T=T,
            max_drive_strength=base_config.max_drive_strength,
            drive_init_scale=base_config.drive_init_scale,
            drive_seed=seed_k,
            J_max=base_config.J_max,
            J_min=base_config.J_min,
            J_init_scale=base_config.J_init_scale,
            J_init_seed=seed_k,
            feromagnetic=base_config.feromagnetic,
        )

        success, results = _single_optimization_attempt(
            config, infidelity_threshold, suppress_output
        )

        if success:
            return True, results

    return False, None


# ============================================================================
# MET finding for single (theta, run) combination
# ============================================================================

def _find_MET_single_run(
    base_config: SimulationConfig,
    infidelity_threshold: float,
    T_start: float,
    time_schedule: Sequence[Tuple[float, float]],
    n_inits: int,
    min_dt: float = 2.0,
    base_seed: Optional[int] = None,
    suppress_output: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Find Minimum Excitation Time (MET) for a single run.

    Uses binary-search-like approach with time schedule:
    - Start at T_start
    - For each (T_min, step) in schedule:
      - Decrease T by step while T >= T_min
      - If success: update MET and continue
      - If fail: break to next schedule stage
    - When fail, go back to last successful MET

    Parameters
    ----------
    base_config : SimulationConfig
        Base configuration (theta_target is already set).
    infidelity_threshold : float
        Target infidelity threshold.
    T_start : float
        Starting evolution time.
    time_schedule : list of (T_min, step)
        Optimization schedule. Each tuple specifies (minimum_time, step_size).
    n_inits : int
        Number of initialization attempts per T.
    min_dt : float
        Minimum time step in ns (default 2.0 ns).
    base_seed : int, optional
        Base random seed for this run.
    suppress_output : bool
        Whether to suppress optimization prints.
    verbose : bool
        Whether to print T → success/fail messages.

    Returns
    -------
    dict
        MET finding results including MET value and success status.
    """
    last_success_T: Optional[float] = None
    last_success_results: Optional[SQEResults] = None
    T = float(T_start)

    for T_min, step in time_schedule:
        T_min = float(T_min)
        step = float(step)

        while T >= T_min - 1e-12:
            # Adjust n_time_steps to ensure dt >= min_dt
            n_time_steps = int(np.floor(T / min_dt))
            n_time_steps = max(n_time_steps, 1)  # At least 1 step

            # Try n_inits at this T
            success, results = _try_n_inits(
                base_config=base_config,
                T=T,
                n_time_steps=n_time_steps,
                infidelity_threshold=infidelity_threshold,
                n_inits=n_inits,
                base_seed=base_seed,
                suppress_output=suppress_output,
            )

            if verbose:
                status = "success" if success else "fail"
                print(f"T={T:.1f}ns → {status}")

            if success:
                last_success_T = float(T)
                last_success_results = results
                T -= step
            else:
                # Failed at this T, break to next schedule stage
                break

        # If we never succeeded, stop entirely
        if last_success_T is None:
            break

        # Reset T to last successful value for next schedule stage
        T = float(last_success_T)

    return {
        "MET": last_success_T,
        "success": last_success_T is not None,
        "theta": float(base_config.theta_target),
        "infidelity_threshold": float(infidelity_threshold),
        "best_results": last_success_results,
    }


# ============================================================================
# MET scan over theta values
# ============================================================================

def scan_MET_over_theta(
    theta_values: np.ndarray,
    infidelity_thresholds: np.ndarray,
    base_config: SimulationConfig,
    T_start: float,
    time_schedule: Sequence[Tuple[float, float]],
    n_runs: int,
    n_inits: int,
    min_dt: float = 2.0,
    results_dir: Optional[Path] = None,
    base_seed: Optional[int] = None,
    suppress_output: bool = True,
) -> Dict[float, Dict[str, Any]]:
    """
    Scan MET over multiple theta values and infidelity thresholds.

    Parameters
    ----------
    theta_values : np.ndarray
        Array of theta target angles to scan.
    infidelity_thresholds : np.ndarray
        Array of infidelity thresholds to test.
    base_config : SimulationConfig
        Base configuration (theta_target will be updated for each theta).
    T_start : float
        Starting evolution time for MET search.
    time_schedule : list of (T_min, step)
        Optimization schedule for binary search.
    n_runs : int
        Number of independent runs for each theta (for averaging).
    n_inits : int
        Number of initialization attempts per T value.
    min_dt : float
        Minimum time step in ns (adjusts n_time_steps automatically).
    results_dir : Path, optional
        Directory to save results.
    base_seed : int, optional
        Base random seed (will be offset for different runs/thetas/thresholds).
    suppress_output : bool
        Whether to suppress single-optimization prints.

    Returns
    -------
    dict
        Dictionary mapping infidelity threshold to MET scan results.
        Each entry contains: theta_values, MET_runs, MET_mean, MET_std.
    """
    theta_values = np.asarray(theta_values, dtype=float)
    infidelity_thresholds = np.asarray(infidelity_thresholds, dtype=float)
    n_theta = len(theta_values)
    n_thresholds = len(infidelity_thresholds)
    n_runs = int(n_runs)

    # Total number of runs across all thresholds
    total_runs = n_theta * n_runs * n_thresholds
    pbar = tqdm(total=total_runs, desc="MET Scan", unit="run")

    print(f"\n{'='*60}")
    print(f"MET Scan:")
    print(f"  {n_theta} theta values × {n_runs} runs × {n_thresholds} thresholds")
    print(f"  = {total_runs} total optimizations")
    print(f"Infidelity thresholds: {infidelity_thresholds}")
    print(f"{'='*60}\n")

    # Dictionary to store results for each threshold
    all_results = {}

    for threshold_idx, infidelity_threshold in enumerate(infidelity_thresholds):
        print(f"\n{'='*60}")
        print(f"Threshold {threshold_idx+1}/{n_thresholds}: I = {infidelity_threshold:.1e}")
        print(f"{'='*60}\n")

        # Initialize results arrays for this threshold
        MET_runs = np.full((n_theta, n_runs), np.nan, dtype=float)

        for i, theta in enumerate(theta_values):
            for r in range(n_runs):
                # Generate unique seed for this (threshold, theta, run) combination
                run_seed = None if base_seed is None else (
                    base_seed + 10_000 * r + 100_000 * i + 1_000_000 * threshold_idx
                )

                # Create config with current theta
                config = SimulationConfig(
                    zeeman_splittings=base_config.zeeman_splittings,
                    initial_state=base_config.initial_state,
                    theta_target=theta,
                    n_time_steps=base_config.n_time_steps,  # Will be adjusted in MET search
                    T=base_config.T,  # Will be adjusted in MET search
                    max_drive_strength=base_config.max_drive_strength,
                    drive_init_scale=base_config.drive_init_scale,
                    drive_seed=run_seed,
                    J_max=base_config.J_max,
                    J_min=base_config.J_min,
                    J_init_scale=base_config.J_init_scale,
                    J_init_seed=run_seed,
                    feromagnetic=base_config.feromagnetic,
                )

                print(f"\n--- I={infidelity_threshold:.1e}, Theta {i+1}/{n_theta} (θ={theta/np.pi:.3f}π), Run {r+1}/{n_runs} ---")

                # Find MET for this (theta, run)
                met_result = _find_MET_single_run(
                    base_config=config,
                    infidelity_threshold=infidelity_threshold,
                    T_start=T_start,
                    time_schedule=time_schedule,
                    n_inits=n_inits,
                    min_dt=min_dt,
                    base_seed=run_seed,
                    suppress_output=suppress_output,
                    verbose=True,
                )

                met_value = met_result["MET"]
                MET_runs[i, r] = np.nan if met_value is None else float(met_value)

                # Print conclusion
                if met_value is not None:
                    print(f"✓ MET found: {met_value:.1f} ns")
                else:
                    print(f"✗ MET not found (failed at all T values)")

                # Update progress bar
                pbar.update(1)
                met_str = f"{met_value:.1f}" if met_value is not None else "nan"
                pbar.set_postfix_str(
                    f"I={infidelity_threshold:.1e}, θ={theta/np.pi:.3f}π, MET={met_str}ns"
                )

        # Compute statistics for this threshold
        MET_mean = np.nanmean(MET_runs, axis=1)
        MET_std = np.nanstd(MET_runs, axis=1)

        # Save results for this threshold
        all_results[float(infidelity_threshold)] = {
            "theta_values": theta_values,
            "MET_runs": MET_runs,
            "MET_mean": MET_mean,
            "MET_std": MET_std,
            "infidelity_threshold": float(infidelity_threshold),
        }

        # Save to file if directory provided
        if results_dir is not None:
            results_dir = Path(results_dir).expanduser().resolve()
            results_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectory for this infidelity threshold
            threshold_name = f"I_{infidelity_threshold:.0e}".replace("+", "")
            threshold_dir = results_dir / threshold_name
            threshold_dir.mkdir(parents=True, exist_ok=True)

            # Save MET table
            np.savez_compressed(
                threshold_dir / "met_table.npz",
                theta_values=theta_values,
                MET_runs=MET_runs,
                MET_mean=MET_mean,
                MET_std=MET_std,
                infidelity_threshold=infidelity_threshold,
            )

            print(f"\n✓ Results saved to: {threshold_dir / 'met_table.npz'}")

    pbar.close()

    print(f"\n{'='*60}")
    print("MET Scan Complete!")
    print(f"{'='*60}\n")

    return all_results
