"""Numerical checks against SpinChain's independent DQE propagation path."""
import unittest

import numpy as np

from pulse_vqe_entanglement.dataclasses import SimulationConfig
from pulse_vqe_entanglement.dqe_state import (
    _cost_entropy, _entropy_grad_fd, _make_driven_sys,
    _make_system_and_observable,
)
from pulse_vqe_entanglement.dqe_state_entropy_gradient import entropy_cost_and_grad


class DQEStateAdjointTest(unittest.TestCase):
    def test_trajectory_cost_and_gradient_match_simulator(self):
        rng = np.random.default_rng(4)
        basis_initial = np.eye(16, dtype=np.complex128)[:, 3]
        random_initial = rng.normal(size=16) + 1j * rng.normal(size=16)
        random_initial /= np.linalg.norm(random_initial)
        for feromagnetic, j_min in ((False, 0.0), (True, -4.0)):
            with self.subTest(feromagnetic=feromagnetic, j_min=j_min):
                config = SimulationConfig(
                    zeeman_splittings=np.array([1000., 1008., 1016., 1024.]),
                    initial_state=(random_initial if feromagnetic else basis_initial),
                    n_time_steps=3,
                    n_discretisation=2, n_J_channels=3, T=70.,
                    J_min=j_min, J_max=10., S_max=0.05,
                    feromagnetic=feromagnetic,
                )
                x = rng.uniform(-0.5, 0.5, 9)
                system, _ = _make_system_and_observable(config)
                driven = _make_driven_sys(config, system)
                cost, gradient, trajectory = entropy_cost_and_grad(
                    config, x, return_trajectory=True
                )
                simulator_trajectory = np.asarray(driven.propagate_all(x))
                simulator_cost = _cost_entropy(driven, x, S_max=config.S_max)
                finite_difference = _entropy_grad_fd(
                    driven, x, S_max=config.S_max, eps=1e-5
                )
                self.assertGreater(cost, 0.0)
                np.testing.assert_allclose(trajectory, simulator_trajectory, atol=1e-7)
                self.assertAlmostEqual(cost, simulator_cost, places=8)
                np.testing.assert_allclose(
                    gradient, finite_difference, atol=1e-5, rtol=1e-4
                )


if __name__ == "__main__":
    unittest.main()
