from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from typing import List, Optional, Tuple

from .dataclasses_old import SimulationConfig, SQEResults
from .device import (
    build_device,
    build_hilbert_space,
    get_observable,
    build_pulse_form_and_initial,
    trajectory_metrics_once,
)


class TargetReached(Exception):
    pass


class SQEOptimisation:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results: Optional[SQEResults] = None
        self.driven_device = None
        self.observable = None
        self.bounds = None
        self.best_x = None
        self.best_I = float('inf')
        self.snapshot_data: List[Tuple[str, np.ndarray, np.ndarray]] = []
        self.pulse_snapshots: List[Tuple[str, np.ndarray]] = []
        self.entropy_snapshots: Optional[List[Tuple[str, np.ndarray, np.ndarray]]] = None
        self.current_iter = 0
        self.next_snapshot = 0
        self.total_iterations = 0

    def _infidelity_T(self, x: np.ndarray) -> float:
        return float(np.real(self.driven_device.evolved_expectation_value(x, self.observable)))

    def _cost_grad(self, x: np.ndarray):
        val, grad = self.driven_device.gradient(x, self.observable)
        return float(np.real(val)), np.real(grad).astype(np.float64)

    def _make_snapshot(self, label: str, x: np.ndarray):
        t_ns, I_t, S_t = trajectory_metrics_once(
            self.driven_device, x_mhz=x, observable=self.observable,
            compute_entanglement=self.config.entanglement_on
        )
        self.snapshot_data.append((label, t_ns, I_t))
        self.pulse_snapshots.append((label, x.copy()))
        if self.config.entanglement_on and self.entropy_snapshots is not None and S_t is not None:
            self.entropy_snapshots.append((label, t_ns, S_t))

    def _projected_grad_norm(self, x: np.ndarray) -> float:
        _, g = self._cost_grad(x)
        if self.bounds is None:
            return float(np.linalg.norm(g))
        lb, ub = self.bounds[0]
        pg = g.copy()
        at_lb = np.isclose(x, lb, atol=1e-12)
        at_ub = np.isclose(x, ub, atol=1e-12)
        pg[at_lb & (g > 0)] = 0.0
        pg[at_ub & (g < 0)] = 0.0
        return float(np.linalg.norm(pg))

    def _update_best(self, x: np.ndarray):
        I = self._infidelity_T(x)
        if I < self.best_I:
            self.best_I = I
            self.best_x = x.copy()

    def _callback(self, xk: np.ndarray, stage: int):
        self.current_iter += 1
        self.total_iterations += 1
        self._update_best(xk)

        if self.best_I <= self.config.infidelity_threshold:
            raise TargetReached

        if self.current_iter >= self.next_snapshot:
            self._make_snapshot(f"iter {self.total_iterations}", xk)
            self.next_snapshot += self.config.snapshot_every

    def optimise(self, config: Optional[SimulationConfig] = None) -> SQEResults:
        if config is not None:
            self.config = config
        cfg = self.config

        hilbert_space, DIM = build_hilbert_space()
        device = build_device(cfg, hilbert_space)
        x0, pulse_form = build_pulse_form_and_initial(cfg, DIM)
        self.driven_device = device.pulse_form(pulse_form)
        self.observable = get_observable(cfg, DIM)

        if cfg.Jmax_mhz is None:
            self.bounds = None
            method = "BFGS"
        else:
            Jmax = float(cfg.Jmax_mhz)
            if Jmax <= 0:
                raise ValueError("Jmax_mhz must be positive or None")
            self.bounds = [(-Jmax, +Jmax)] * cfg.N_SEGMENTS
            method = "L-BFGS-B"
            x0 = np.clip(x0, -Jmax, +Jmax)

        self.snapshot_data = []
        self.pulse_snapshots = []
        self.entropy_snapshots = [] if cfg.entanglement_on else None
        self.best_x = x0.copy()
        self.best_I = self._infidelity_T(self.best_x)
        self.total_iterations = 0

        self._make_snapshot("init", self.best_x)

        gtol = 1e-8
        gtol_shrink = 1e-2
        gtol_min = cfg.infidelity_threshold / 100  # 2 orders of magnitude smaller
        max_stages = 100
        res_last = None
        success = False

        for stage in range(max_stages):
            self.current_iter = 0
            self.next_snapshot = cfg.snapshot_every

            try:
                res = minimize(
                    self._cost_grad,
                    x0=self.best_x,
                    jac=True,
                    method=method,
                    bounds=self.bounds,
                    callback=lambda xk: self._callback(xk, stage),
                    options={"gtol": gtol, "maxiter": cfg.max_iter, "disp": False},
                )
                res_last = res
                self._update_best(res.x)
            except TargetReached:
                success = True
                break

            # Condition 1: Check if infidelity threshold reached
            if self.best_I <= cfg.infidelity_threshold:
                success = True
                break

            # Check termination reason
            msg = str(res_last.message if res_last else "").upper()
            gtol_reached = any(x in msg for x in ["GRADIENT", "PROJECTED GRADIENT", "PGTOL"])
            maxiter_reached = "ITERATION" in msg or res_last.nit >= cfg.max_iter

            # Condition 2: Max iterations reached without success
            if maxiter_reached:
                success = False
                break

            # Condition 3: gtol reached, try reducing it
            if gtol_reached:
                new_gtol = gtol * gtol_shrink
                if new_gtol >= gtol_min:
                    gtol = new_gtol
                    continue  # Continue to next stage with smaller gtol
                else:
                    # Cannot reduce gtol further
                    success = False
                    break

            # Other stopping condition (shouldn't normally reach here)
            success = False
            break

        self._make_snapshot("final", self.best_x)
        self._print_results(gtol, res_last, success)

        out = SQEResults(
            config=cfg,
            x0=x0,
            x_opt=self.best_x,
            I_final=self.best_I,
            res=res_last,
            snapshot_data=self.snapshot_data,
            pulse_snapshots=self.pulse_snapshots,
            entropy_snapshots=self.entropy_snapshots,
            driven_device=self.driven_device,
            total_iterations=self.total_iterations,
        )
        self.results = out
        return out

    def _print_results(self, gtol: float, res_last, success: bool):
        grad_n = self._projected_grad_norm(self.best_x)
        I_thr = self.config.infidelity_threshold
        gtol_min = I_thr / 100

        print(f"\n{'='*60}")
        print(f"Total iterations: {self.total_iterations}")
        print(f"I_final = {self.best_I:.6e}  (threshold {I_thr:.1e})")
        print(f"||grad|| = {grad_n:.6e}  (final gtol {gtol:.1e}, min {gtol_min:.1e})")

        if success:
            print("RESULT: SUCCESS - Infidelity threshold reached")
        else:
            if res_last:
                msg = str(res_last.message)
                if "ITERATION" in msg.upper() or res_last.nit >= self.config.max_iter:
                    print("RESULT: FAIL - Maximum iterations reached")
                elif gtol < gtol_min * 1.1:  # Account for floating point
                    print("RESULT: FAIL - Minimum gtol reached, cannot improve further")
                else:
                    print(f"RESULT: FAIL - {msg}")
            else:
                print("RESULT: FAIL - Optimization stopped")
        print(f"{'='*60}\n")

