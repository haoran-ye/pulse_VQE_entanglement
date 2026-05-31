"""
thesis_utils.py — shared utilities for all Thesis pulse-database generation scripts.

This module centralises helpers that were previously copy-pasted into every
generate_*.py script, covering:

  Config helpers
  --------------
  dataclass_field_names   -- set of valid SimulationConfig field names
  filter_kwargs_for_config -- strip unknown keys before constructing a config
  safe_replace_config     -- replace config fields, silently ignoring unknowns

  Low-level array helpers
  -----------------------
  get_attr_any   -- try a list of attribute names and return the first that exists
  as_1d_float_array -- coerce any array-like to a flat float64 1D ndarray
  fixed_width_x  -- NaN-pad or truncate a pulse vector to a fixed width

  Statistics
  ----------
  nanmean_safe  -- mean over finite values; NaN if none finite
  nanmax_safe   -- max over finite values; NaN if none finite

  Result extraction (called on optimisation result objects)
  ---------------------------------------------------------
  extract_state_result_arrays -- SQE / DQE state results  (val_final, final_traj)
  extract_gate_result_arrays  -- SQE / DQE gate results   (infid_final, ent_final)

  Physics
  -------
  target_sqe_entropy    -- von Neumann entropy of U_SQE(θ)|01⟩ reduced state
  target_bell_entropy   -- von Neumann entropy of U_SQE(θ)|Bell⟩ reduced state

Physics notes
-------------
SQE gate U(θ) acts on the {|01⟩, |10⟩} subspace:
    U(θ)|01⟩ = cos(θ)|01⟩ + sin(θ)|10⟩

Starting from |01⟩, the reduced state of qubit 1 has eigenvalues cos²(θ) and
sin²(θ).  The entropy is H(cos²(θ)) = -cos²(θ)log₂cos²(θ) - sin²(θ)log₂sin²(θ).

Starting from |Bell⟩ = (|01⟩+|10⟩)/√2, the output is
    [(cosθ-sinθ)|01⟩ + (cosθ+sinθ)|10⟩]/√2.
Reduced state eigenvalues: (1-sin2θ)/2 and (1+sin2θ)/2.
Entropy is H((1+sin2θ)/2).
"""
from __future__ import annotations

from dataclasses import replace, fields, is_dataclass
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import SimulationConfig so callers can import it from here without needing
# the ROOT sys.path hack.  pulse_vqe_entanglement must be pip-installed.
# ---------------------------------------------------------------------------
from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.entanglement_measures import (
    von_neumann_entropy_traj,
    entropy_sum_traj,
)


# ===========================================================================
# Config helpers
# ===========================================================================

def dataclass_field_names(cls: type) -> set[str]:
    """Return the set of field names of a dataclass, or empty set otherwise."""
    if not is_dataclass(cls):
        return set()
    return {f.name for f in fields(cls)}


# Pre-computed set of valid SimulationConfig field names.
CONFIG_FIELDS: set[str] = dataclass_field_names(SimulationConfig)


def filter_kwargs_for_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop any keys that are not valid SimulationConfig fields."""
    return {k: v for k, v in kwargs.items() if k in CONFIG_FIELDS}


def safe_replace_config(config: SimulationConfig, **kwargs: Any) -> SimulationConfig:
    """
    Return a new SimulationConfig with the given fields replaced.

    Unknown keyword arguments are silently ignored so callers can pass a
    superset of fields without raising.
    """
    return replace(config, **filter_kwargs_for_config(kwargs))


# ===========================================================================
# Low-level array helpers
# ===========================================================================

def get_attr_any(obj: Any, names: list[str], default: Any = None) -> Any:
    """
    Return the value of the first matching attribute in *names*.

    Useful for extracting fields from optimisation result objects whose
    attribute names may vary across library versions.
    """
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def as_1d_float_array(x: Any) -> np.ndarray:
    """Coerce *x* to a 1D float64 ndarray, or return an empty array if None."""
    if x is None:
        return np.array([], dtype=float)
    return np.asarray(x, dtype=float).reshape(-1)


def fixed_width_x(x: Any, width: int) -> np.ndarray:
    """
    Return a float64 array of exactly *width* elements.

    If *x* is shorter than *width*, the tail is NaN-padded.
    If *x* is longer, it is truncated.
    If *x* is None, a full-NaN array is returned.
    """
    out = np.full(width, np.nan, dtype=float)
    if x is None:
        return out
    arr = np.asarray(x, dtype=float).reshape(-1)
    n = min(width, arr.size)
    out[:n] = arr[:n]
    return out


# ===========================================================================
# Statistics
# ===========================================================================

def nanmean_safe(values: Any) -> float:
    """
    Mean of finite values; returns NaN if there are no finite values.

    Replaces the separate average_entropy / average_entanglement helpers that
    appeared verbatim in every generate script.
    """
    arr = as_1d_float_array(values)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.nan
    return float(np.nanmean(arr))


def nanmax_safe(values: Any) -> float:
    """
    Max of finite values; returns NaN if there are no finite values.

    Replaces max_entropy / max_entanglement helpers.
    """
    arr = as_1d_float_array(values)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return np.nan
    return float(np.nanmax(arr))


# ===========================================================================
# Result extraction
# ===========================================================================

def extract_state_result_arrays(result: Any) -> dict[str, Any]:
    """
    Pull standard fields out of an SQE / DQE state optimisation result.

    Tries several candidate attribute names to be resilient against minor API
    changes in the optimisation library.

    Returns
    -------
    dict with keys:
        infidelities    -- 1D infidelity trajectory (time axis)
        entropies       -- 1D entropy trajectory (computed from final_traj if absent)
        x0              -- initial pulse vector (or None)
        x_final         -- optimised pulse vector
        final_infid     -- scalar final infidelity
        n_iterations    -- number of optimiser iterations
        gradient_norm   -- final gradient norm
        optimization_success -- bool
    """
    infidelities = as_1d_float_array(
        get_attr_any(result, ["val_final", "infidelities", "infidelity_evolution",
                              "state_infidelities"])
    )

    final_infid = get_attr_any(result, ["final_infidelity", "best_infidelity"],
                               default=None)
    if final_infid is None:
        final_infid = float(infidelities[-1]) if infidelities.size else np.nan

    entropies = as_1d_float_array(
        get_attr_any(result, ["entropies", "entropy_evolution", "von_neumann_entropies",
                              "von_neumann_entropy", "S_evolution", "S_values",
                              "entropy_sum"])
    )

    # Fall back to computing entropy from the state trajectory if needed.
    if entropies.size == 0:
        final_traj = get_attr_any(result, ["final_traj"])
        if final_traj is not None:
            traj = np.asarray(final_traj)
            # DQE trajectories: use sum of bipartite entropies over 7 partitions.
            # SQE trajectories: use single bipartite entropy.
            # Both are handled by trying entropy_sum_traj first; for 2-qubit states
            # it falls back to von_neumann_entropy_traj automatically.
            try:
                entropies = entropy_sum_traj(traj)
            except Exception:
                entropies = von_neumann_entropy_traj(traj)

    return dict(
        infidelities=infidelities,
        entropies=entropies,
        x0=get_attr_any(result, ["x0", "x_initial", "initial_x"]),
        x_final=get_attr_any(result, ["x_final", "x_opt", "x", "optimised_x",
                                      "optimized_x"]),
        final_infid=float(final_infid),
        n_iterations=get_attr_any(result, ["n_iterations", "nit", "iterations"],
                                  default=np.nan),
        gradient_norm=get_attr_any(result, ["gradient_norm", "grad_norm"],
                                   default=np.nan),
        optimization_success=bool(get_attr_any(result,
                                               ["optimization_success", "success"],
                                               default=False)),
    )


def extract_gate_result_arrays(result: Any) -> dict[str, Any]:
    """
    Pull standard fields out of an SQE / DQE gate optimisation result.

    Returns
    -------
    dict with keys:
        infidelities    -- 1D gate-infidelity trajectory
        entanglements   -- 1D gate-entanglement trajectory (e.g. operator entanglement)
        x0, x_final, final_infid, n_iterations, gradient_norm, optimization_success
    """
    infidelities = as_1d_float_array(
        get_attr_any(result, ["infid_final", "infidelities", "infidelity_evolution",
                              "gate_infidelities", "val_final"])
    )

    final_infid = get_attr_any(result, ["final_infidelity", "best_infidelity",
                                        "infidelity"], default=None)
    if final_infid is None:
        final_infid = float(infidelities[-1]) if infidelities.size else np.nan

    entanglements = as_1d_float_array(
        get_attr_any(result, ["ent_final", "entanglements", "gate_entanglements",
                              "S_final", "entropy_final", "entanglement_final"])
    )

    return dict(
        infidelities=infidelities,
        entanglements=entanglements,
        x0=get_attr_any(result, ["x0", "x_initial", "initial_x"]),
        x_final=get_attr_any(result, ["x_final", "x_opt", "x", "optimised_x",
                                      "optimized_x"]),
        final_infid=float(final_infid),
        n_iterations=get_attr_any(result, ["n_iterations", "nit", "iterations"],
                                  default=np.nan),
        gradient_norm=get_attr_any(result, ["gradient_norm", "grad_norm"],
                                   default=np.nan),
        optimization_success=bool(get_attr_any(result,
                                               ["optimization_success", "success"],
                                               default=False)),
    )


# ===========================================================================
# Physics: entropy targets
# ===========================================================================

def target_sqe_entropy(theta: float) -> float:
    """
    von Neumann entropy of the SQE target state starting from |01⟩.

    U_SQE(θ)|01⟩ = cos(θ)|01⟩ + sin(θ)|10⟩

    Reduced state of qubit 1 has eigenvalues cos²(θ) and sin²(θ), so
        S = -cos²(θ) log₂ cos²(θ) - sin²(θ) log₂ sin²(θ)  =  H(cos²(θ)).

    This is the binary entropy function evaluated at p = cos²(θ), NOT at
    p = cos²(θ/2) (which would be the Bloch-sphere polar-angle formula).
    """
    # Eigenvalues of the reduced single-qubit state after SQE gate on |01>.
    p = float(np.cos(theta) ** 2)          # probability of qubit-1 being in |0>
    q = 1.0 - p                             # probability of qubit-1 being in |1>
    eps = 1e-15                             # clip to avoid log(0)
    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    return float(-(p * np.log2(p) + q * np.log2(q)))


def target_bell_entropy(theta: float) -> float:
    """
    von Neumann entropy of the SQE target state starting from |Bell⟩ = (|01⟩+|10⟩)/√2.

    U_SQE(θ)|Bell⟩ = [(cosθ−sinθ)|01⟩ + (cosθ+sinθ)|10⟩] / √2

    Reduced state of qubit 1 has eigenvalues:
        p₁ = (1 − sin 2θ) / 2
        p₂ = (1 + sin 2θ) / 2

    So S = H((1 + sin 2θ) / 2).

    Special values:
        θ = 0      → S = 1 bit  (Bell state unchanged)
        θ = π/4   → S = 0      (state collapses to |10⟩, fully entanglement-free)
        θ = π/2   → S = 1 bit  (Bell state again, up to global phase)
    """
    p = float((1.0 + np.sin(2.0 * theta)) / 2.0)  # larger eigenvalue
    q = 1.0 - p
    eps = 1e-15
    p = float(np.clip(p, eps, 1.0 - eps))
    q = float(np.clip(q, eps, 1.0 - eps))
    return float(-(p * np.log2(p) + q * np.log2(q)))
