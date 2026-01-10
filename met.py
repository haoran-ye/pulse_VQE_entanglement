from __future__ import annotations

import json
import os
import logging
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np


# ======================================================
# Global suppression of TensorFlow / absl warnings
#   - Works even if TF was imported earlier (we reconfigure loggers)
#   - Also blocks C++ backend logs via TF_CPP_MIN_LOG_LEVEL
# ======================================================

def _suppress_tensorflow_noise() -> None:
    # C++/backend TF logs (0=all, 1=INFO, 2=WARNING, 3=ERROR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    # Python logging channels
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("absl").setLevel(logging.ERROR)

    # Some TF builds use absl.logging directly
    try:
        from absl import logging as absl_logging  # type: ignore
        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("error")
    except Exception:
        pass

    # If tensorflow is available (or already imported), clamp its logger too
    try:
        import tensorflow as tf  # type: ignore

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
        # If TF isn't installed in some environments, fine.
        pass


_suppress_tensorflow_noise()


from .dataclasses import SimulationConfig
from .optimisation import SQEOptimisation


# ======================================================
# Helper: pulse init policy
# ======================================================

def pulse_init_scale_for_theta(
    theta: float,
    cutoff: float,
    scale_zero: float,
    scale_random: float,
) -> float:
    return scale_zero if abs(theta) <= cutoff else scale_random


# ======================================================
# Helper: safe extraction for saving
# ======================================================

def _maybe_to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None


def _results_to_save_payload(results: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    # Small JSON-safe fields
    for k in ["I_final", "n_iter", "converged", "success"]:
        if hasattr(results, k):
            v = getattr(results, k)
            if isinstance(v, (int, float, bool, str)) or v is None:
                payload[k] = v

    # Candidate arrays saved to npz (best-effort)
    array_candidates = [
        "pulse",
        "pulse_x",
        "pulse_y",
        "pulse_z",
        "t",
        "time",
        "times",
        "psi_t",
        "state_traj",
        "state_trajectory",
        "rho_t",
        "density_traj",
        "entropy_t",
        "S_t",
        "entanglement_entropy_t",
    ]

    arrays: Dict[str, np.ndarray] = {}
    for k in array_candidates:
        if hasattr(results, k):
            arr = _maybe_to_numpy(getattr(results, k))
            if arr is not None:
                arrays[k] = arr

    payload["_arrays_present"] = sorted(arrays.keys())
    payload["_arrays"] = arrays  # caller must pop before json dump
    return payload


def _jsonable_cfg(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {"repr": repr(cfg)}


# ======================================================
# Storage manager
# ======================================================

class METStorage:
    """
    Layout:
      root/
        meta.json
        I_1e-06/
          met_table.npz
          theta_000_theta0.000000/
            run_000/
              summary.json
              cfg.json
              arrays.npz
            ...
    """

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _thr_dir(self, infidelity_threshold: float) -> Path:
        name = f"I_{infidelity_threshold:.0e}".replace("+", "")
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _theta_dir(self, thr_dir: Path, theta_idx: int, theta: float) -> Path:
        p = thr_dir / f"theta_{theta_idx:03d}_theta{theta:.6f}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _run_dir(self, theta_dir: Path, run_idx: int) -> Path:
        p = theta_dir / f"run_{run_idx:03d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_root_meta(self, meta: Dict[str, Any]) -> None:
        with (self.root / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

    def write_run_result(
        self,
        *,
        infidelity_threshold: float,
        theta_idx: int,
        theta: float,
        run_idx: int,
        summary: Dict[str, Any],
        cfg: Optional[Any] = None,
        arrays: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        thr_dir = self._thr_dir(infidelity_threshold)
        theta_dir = self._theta_dir(thr_dir, theta_idx, theta)
        run_dir = self._run_dir(theta_dir, run_idx)

        with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        if cfg is not None:
            with (run_dir / "cfg.json").open("w", encoding="utf-8") as f:
                json.dump(_jsonable_cfg(cfg), f, indent=2, sort_keys=True)

        if arrays:
            np.savez_compressed(run_dir / "arrays.npz", **arrays)

    def write_threshold_table(
        self,
        *,
        infidelity_threshold: float,
        theta_values: np.ndarray,
        MET_runs: np.ndarray,     # (n_theta, n_runs)
        MET_mean: np.ndarray,     # (n_theta,)
        MET_std: np.ndarray,      # (n_theta,)
    ) -> None:
        thr_dir = self._thr_dir(infidelity_threshold)
        np.savez_compressed(
            thr_dir / "met_table.npz",
            theta_values=np.asarray(theta_values, dtype=float),
            MET_runs=np.asarray(MET_runs, dtype=float),
            MET_mean=np.asarray(MET_mean, dtype=float),
            MET_std=np.asarray(MET_std, dtype=float),
        )


# ======================================================
# Hard silence (stdout/stderr) for optimisation
# ======================================================

class _NullWriter:
    def write(self, _: str) -> int:
        return 0
    def flush(self) -> None:
        return None


def _run_optimisation_silently(fn):
    with redirect_stdout(_NullWriter()), redirect_stderr(_NullWriter()):
        return fn()


# ======================================================
# One optimisation attempt (silent by default)
# ======================================================

def _single_attempt(
    *,
    theta: float,
    T_ns: float,
    infidelity_threshold: float,
    attempt_idx: int,
    seed: Optional[int],
    cfg_kwargs: dict,
    pulse_init_scale: float,
    suppress_output: bool = True,
) -> Dict[str, Any]:
    cfg = SimulationConfig(
        theta=theta,
        T_ns=T_ns,
        infidelity_threshold=infidelity_threshold,
        pulse_init_scale=pulse_init_scale,
        seed=seed,
        **cfg_kwargs,
    )

    opt = SQEOptimisation(cfg)
    if suppress_output:
        results = _run_optimisation_silently(opt.optimise)
    else:
        results = opt.optimise()

    I_final = float(getattr(results, "I_final", np.inf))
    ok = (I_final <= infidelity_threshold)

    return dict(
        ok=ok,
        I_final=I_final,
        results=results,
        cfg=cfg,
        seed=seed,
        attempt_idx=int(attempt_idx),
    )


# ======================================================
# Any-success at fixed (theta, T) using n_inits
# ======================================================

def any_success_for_T(
    *,
    theta: float,
    T_ns: float,
    infidelity_threshold: float,
    n_inits: int,
    pulse_init_scale: float,
    cfg_kwargs: dict,
    base_seed: Optional[int],
    suppress_output: bool = True,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    for k in range(int(n_inits)):
        seed_k = None if base_seed is None else int(base_seed) + int(k)

        attempt = _single_attempt(
            theta=theta,
            T_ns=T_ns,
            infidelity_threshold=infidelity_threshold,
            attempt_idx=k,
            seed=seed_k,
            pulse_init_scale=pulse_init_scale,
            cfg_kwargs=cfg_kwargs,
            suppress_output=suppress_output,
        )
        if attempt["ok"]:
            return True, attempt

    return False, None


# ======================================================
# MET for ONE theta, ONE run
# ======================================================

def find_MET_for_theta_single_run(
    *,
    theta: float,
    infidelity_threshold: float,
    T_start: float,
    time_schedule: Sequence[Tuple[float, float]],
    n_inits: int,
    pulse_init_cutoff: float,
    pulse_init_scale_zero: float,
    pulse_init_scale_random: float,
    cfg_kwargs: dict,
    base_seed: Optional[int] = None,
    suppress_output: bool = True,
) -> Dict[str, Any]:
    pulse_init_scale = pulse_init_scale_for_theta(
        theta,
        cutoff=pulse_init_cutoff,
        scale_zero=pulse_init_scale_zero,
        scale_random=pulse_init_scale_random,
    )

    last_success_T: Optional[float] = None
    last_winner: Optional[Dict[str, Any]] = None
    T = float(T_start)

    for T_min, step in time_schedule:
        T_min = float(T_min)
        step = float(step)

        while T >= T_min - 1e-12:
            ok, winner = any_success_for_T(
                theta=theta,
                T_ns=T,
                infidelity_threshold=infidelity_threshold,
                n_inits=n_inits,
                base_seed=base_seed,
                pulse_init_scale=pulse_init_scale,
                cfg_kwargs=cfg_kwargs,
                suppress_output=suppress_output,
            )
            if ok:
                last_success_T = float(T)
                last_winner = winner
                T -= step
            else:
                break

        if last_success_T is None:
            break

        T = float(last_success_T)

    return dict(
        MET=last_success_T,
        success=(last_success_T is not None),
        theta=float(theta),
        infidelity_threshold=float(infidelity_threshold),
        winner=last_winner,
        pulse_init_scale=float(pulse_init_scale),
        n_inits=int(n_inits),
        base_seed=base_seed,
    )


# ======================================================
# Scan theta WITHOUT creating progress bars.
# Caller passes a single global pbar (optional).
# ======================================================

def scan_MET_over_theta(
    *,
    theta_values: np.ndarray,
    infidelity_threshold: float,
    T_start: float,
    time_schedule: Sequence[Tuple[float, float]],
    n_runs: int,
    n_inits: int,
    pulse_init_cutoff: float,
    pulse_init_scale_zero: float,
    pulse_init_scale_random: float,
    cfg_kwargs: dict,
    base_seed: Optional[int] = None,
    suppress_output: bool = True,
    storage: Optional[METStorage] = None,
    pbar=None,  # external global tqdm
) -> Dict[str, Any]:
    theta_values = np.asarray(theta_values, dtype=float)
    n_theta = int(theta_values.size)
    n_runs = int(n_runs)
    n_inits = int(n_inits)

    MET_runs = np.full((n_theta, n_runs), np.nan, dtype=float)

    for i, theta in enumerate(theta_values):
        for r in range(n_runs):
            run_base_seed = None if base_seed is None else int(base_seed) + 10_000 * int(r) + 100_000 * int(i)

            out = find_MET_for_theta_single_run(
                theta=float(theta),
                infidelity_threshold=float(infidelity_threshold),
                T_start=float(T_start),
                time_schedule=time_schedule,
                n_inits=n_inits,
                pulse_init_cutoff=float(pulse_init_cutoff),
                pulse_init_scale_zero=float(pulse_init_scale_zero),
                pulse_init_scale_random=float(pulse_init_scale_random),
                cfg_kwargs=cfg_kwargs,
                base_seed=run_base_seed,
                suppress_output=suppress_output,
            )

            met = out["MET"]
            MET_runs[i, r] = np.nan if met is None else float(met)

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(
                    f"I={infidelity_threshold:.0e}  "
                    f"theta={theta/np.pi:.3f}π  "
                    f"run={r+1}/{n_runs}  "
                    f"MET={('nan' if met is None else f'{met:.1f}ns')}"
                )

            if storage is not None:
                summary = {
                    "theta": float(theta),
                    "theta_idx": int(i),
                    "run_idx": int(r),
                    "infidelity_threshold": float(infidelity_threshold),
                    "MET": None if met is None else float(met),
                    "success": bool(out["success"]),
                    "n_inits": int(n_inits),
                    "n_runs": int(n_runs),
                    "pulse_init_scale": float(out["pulse_init_scale"]),
                    "run_base_seed": run_base_seed,
                }

                arrays_to_save: Dict[str, np.ndarray] = {}
                cfg_to_save = None

                winner = out.get("winner", None)
                if winner is not None:
                    cfg_to_save = winner.get("cfg", None)
                    results = winner.get("results", None)

                    summary["winner_attempt_idx"] = int(winner.get("attempt_idx", -1))
                    summary["winner_I_final"] = float(winner.get("I_final", np.nan))
                    summary["winner_seed"] = winner.get("seed", None)

                    if results is not None:
                        payload = _results_to_save_payload(results)
                        arrays = payload.pop("_arrays", {})
                        for k, v in arrays.items():
                            arrays_to_save[k] = v
                        summary["saved_arrays"] = payload.get("_arrays_present", [])

                storage.write_run_result(
                    infidelity_threshold=float(infidelity_threshold),
                    theta_idx=int(i),
                    theta=float(theta),
                    run_idx=int(r),
                    summary=summary,
                    cfg=cfg_to_save,
                    arrays=arrays_to_save if len(arrays_to_save) > 0 else None,
                )

    MET_mean = np.nanmean(MET_runs, axis=1)
    MET_std = np.nanstd(MET_runs, axis=1)

    if storage is not None:
        storage.write_threshold_table(
            infidelity_threshold=float(infidelity_threshold),
            theta_values=theta_values,
            MET_runs=MET_runs,
            MET_mean=MET_mean,
            MET_std=MET_std,
        )

    return dict(
        theta_values=theta_values,
        MET_runs=MET_runs,
        MET_mean=MET_mean,
        MET_std=MET_std,
        infidelity_threshold=float(infidelity_threshold),
    )
