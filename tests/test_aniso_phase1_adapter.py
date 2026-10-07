import unittest
from unittest.mock import patch

import numpy as np

from engine.aniso_phase1 import AnisotropicCenterAdapter, AnisotropicMaterialParams, query_gpu_memory, select_lowest_memory_device


class TestAnisoPhase1Adapter(unittest.TestCase):
    def test_device_selector_accepts_explicit_cpu_and_auto(self):
        self.assertEqual(select_lowest_memory_device("cpu"), "cpu")
        self.assertIn(select_lowest_memory_device("auto").split(":")[0], ("cpu", "cuda"))

    def test_device_selector_picks_lowest_used_memory_gpu(self):
        completed = type("Result", (), {"stdout": "0, 8000, 24576\n1, 1000, 24576\n", "returncode": 0})()
        with patch("engine.aniso_phase1.device.subprocess.run", return_value=completed):
            self.assertEqual(select_lowest_memory_device("auto"), "cuda:1")
            self.assertEqual(query_gpu_memory()["cuda:1"]["used_mib"], 1000)

    def test_particle_directions_are_averaged_as_structure_tensors(self):
        params = AnisotropicMaterialParams(1.0, 2.0, 8.0, [1.0, 0.0, 0.0])
        adapter = AnisotropicCenterAdapter(2, params)
        adapter.initialize_from_particles(
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            np.array([1.0, 2.0, 1.0]),
            np.array([0, 0, 1]),
        )
        self.assertTrue(np.allclose(adapter.A0[0], np.diag([1.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(adapter.A0[1], np.diag([0.0, 1.0, 0.0])))

    def test_failed_trial_does_not_change_committed_state(self):
        params = AnisotropicMaterialParams(1.0, 2.0, 8.0)
        adapter = AnisotropicCenterAdapter(1, params)
        old = adapter.committed_F.copy()
        adapter.begin_trial(
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]),
            0.1,
        )
        adapter.commit(False)
        self.assertTrue(np.allclose(adapter.committed_F, old))
        self.assertTrue(np.allclose(adapter.trial_F, old))

    def test_successful_trial_commits_once(self):
        params = AnisotropicMaterialParams(1.0, 2.0, 8.0)
        adapter = AnisotropicCenterAdapter(1, params)
        grad_weights = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        adapter.begin_trial(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), grad_weights, 0.1)
        trial = adapter.trial_F.copy()
        adapter.commit(True)
        self.assertTrue(np.allclose(adapter.committed_F, trial))
        self.assertTrue(adapter.valid.all())

    def test_newton_pcg_commits_converged_center_step(self):
        params = AnisotropicMaterialParams(2.0, 3.0, 12.0, [1.0, 0.0, 0.0])
        adapter = AnisotropicCenterAdapter(1, params)
        adapter.set_center_measurements(np.array([0.2]), np.array([1.0]))
        weights = np.array([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        old_F = adapter.committed_F.copy()
        velocity, logs = adapter.solve_newton(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.ones(2),
            np.array([0.0, 0.0, -1.0]),
            1.0e-2,
            weights,
            max_newton=4,
            max_cg=40,
            velocity_tol=1.0e-8,
        )
        self.assertTrue(np.isfinite(velocity).all())
        self.assertTrue(logs[-1]["converged"])
        self.assertTrue(adapter.valid.all())
        self.assertTrue(np.allclose(adapter.committed_F, old_F))
        self.assertLess(velocity[0, 2], 0.0)

    def test_uniform_affine_center_state_matches_closed_form_steps(self):
        params = AnisotropicMaterialParams(2.0, 3.0, 12.0, [1.0, 0.0, 0.0])
        adapter = AnisotropicCenterAdapter(1, params)
        G = np.array([[0.10, 0.02, 0.00], [0.00, -0.05, 0.01], [0.00, 0.00, 0.02]])
        grad_weights = np.eye(3, dtype=np.float64)[None, :, :]
        dt = 1.0e-2
        expected = np.eye(3)
        for _ in range(4):
            adapter.begin_trial(G.T, grad_weights, dt)
            expected = (np.eye(3) + dt * G) @ expected
            adapter.commit(True)
        self.assertTrue(np.allclose(adapter.committed_F[0], expected, rtol=1.0e-13, atol=1.0e-13))


if __name__ == "__main__":
    unittest.main()
