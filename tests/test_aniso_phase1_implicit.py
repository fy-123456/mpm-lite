import unittest

import numpy as np

from engine.aniso_phase1 import (
    AnisotropicMaterialParams,
    center_matvec,
    center_residual,
    center_trial_F,
)


class TestAnisoPhase1Implicit(unittest.TestCase):
    def setUp(self):
        self.params = AnisotropicMaterialParams(3.0, 5.0, 17.0, [1.0, 2.0, 0.5])
        self.F0 = np.stack([np.diag([1.1, 0.95, 1.03]), np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.05], [0.0, 0.0, 1.0]])])
        self.A0 = np.stack([self.params.A0, self.params.A0])
        self.dw = np.array([
            [[-0.4, -0.2, -0.1], [0.4, 0.2, 0.1]],
            [[-0.3, 0.1, -0.2], [0.3, -0.1, 0.2]],
        ])
        self.mass = np.array([2.0, 1.5])
        self.volume = np.array([0.7, 0.4])
        self.vold = np.zeros((2, 3))
        self.g = np.array([0.0, 0.0, -9.81])
        self.dt = 2.0e-3

    def test_trial_is_frozen_from_committed_state(self):
        G = np.array([np.diag([0.2, 0.0, 0.0]), np.diag([0.0, 0.1, 0.0])])
        got = center_trial_F(self.F0, G, self.dt)
        expected = np.stack([(np.eye(3) + self.dt * G[c]) @ self.F0[c] for c in range(2)])
        self.assertTrue(np.allclose(got, expected))

    def test_matrix_free_action_matches_residual_finite_difference(self):
        # The matrix-free operator is the frozen Newton/CG Jacobian at the
        # committed state, so the residual finite difference is taken at v=0.
        v = np.zeros((2, 3))
        p = np.array([[0.3, 0.2, -0.1], [-0.2, 0.1, 0.25]])
        eps = 2.0e-6
        numerical = (center_residual(v + eps * p, self.vold, self.mass, self.g, self.dt, self.F0, self.A0, self.volume, self.dw, self.params) - center_residual(v - eps * p, self.vold, self.mass, self.g, self.dt, self.F0, self.A0, self.volume, self.dw, self.params)) / (2.0 * eps)
        applied = center_matvec(p, self.mass, self.dt, self.F0, self.A0, self.volume, self.dw, self.params)
        self.assertTrue(np.allclose(applied, numerical, rtol=3.0e-4, atol=3.0e-6))

    def test_nonzero_newton_iterate_uses_trial_material_tangent(self):
        dt = 0.2
        v = np.array([[0.3, -0.2, 0.15], [-0.2, 0.1, 0.05]])
        p = np.array([[0.3, 0.2, -0.1], [-0.2, 0.1, 0.25]])
        Ftrial = center_trial_F(self.F0, np.einsum('ni,cnj->cij', v, self.dw), dt)
        args = (self.vold, self.mass, self.g, dt, self.F0, self.A0, self.volume, self.dw, self.params)
        eps = 1.0e-5
        numerical = (center_residual(v + eps * p, *args) - center_residual(v - eps * p, *args)) / (2 * eps)
        applied = center_matvec(p, self.mass, dt, self.F0, self.A0, self.volume, self.dw, self.params, trial_F=Ftrial)
        np.testing.assert_allclose(applied, numerical, rtol=1.0e-8, atol=1.0e-9)
        stale = center_matvec(p, self.mass, dt, self.F0, self.A0, self.volume, self.dw, self.params)
        self.assertGreater(np.linalg.norm(stale - numerical), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
