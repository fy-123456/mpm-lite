"""Closed-loop invariants, including failure cases, of the carried Q1 prototype."""
import unittest
import numpy as np
from engine.aniso_phase1.consistent_transfer import MaterialQ1, material_response, bent_nodes, minimize_with_backtracking
from engine.aniso_phase1.constitutive import energy, pk1
from engine.aniso_phase1.beam_reference import beam_matrices


class ConsistentTransferTests(unittest.TestCase):
    def setUp(self):
        self.m = MaterialQ1(grid=9)
        self.rng = np.random.default_rng(17)

    def test_positive_mass_and_rank_failure(self):
        m = self.m
        self.assertGreater(m.mass_eigenvalues[0], 0.)
        self.assertGreater(m.lumped.min(), 0.)
        self.assertAlmostEqual(m.lumped.sum(), .5*.125**2, places=14)
        with self.assertRaisesRegex(ValueError, 'full-rank'):
            MaterialQ1(grid=9, ppc_axis=1)
        np.testing.assert_allclose(m.particles.N.sum(axis=1), 1., atol=1e-14)
        np.testing.assert_allclose(m.particles.N @ m.X, m.particles.X, atol=1e-14)

    def test_particle_projection_momentum_angular_momentum_and_energy(self):
        m = self.m
        vp = self.rng.normal(size=(len(m.mass), 3))
        grid = m.project(vp)
        out = m.particles.N @ grid
        np.testing.assert_allclose((m.mass[:, None]*out).sum(axis=0),
                                   (m.mass[:, None]*vp).sum(axis=0), atol=1e-15)
        np.testing.assert_allclose((m.mass[:, None]*np.cross(m.particles.X, out)).sum(axis=0),
                                   (m.mass[:, None]*np.cross(m.particles.X, vp)).sum(axis=0), atol=1e-15)
        self.assertLessEqual(np.sum(m.mass[:, None]*out*out), np.sum(m.mass[:, None]*vp*vp)+1e-15)
        # Already represented nodal fields have an exact round trip, including K.
        v = self.rng.normal(size=m.X.shape)
        np.testing.assert_allclose(m.project(m.particles.N @ v), v, atol=1e-12)
        self.assertAlmostEqual(float(np.sum(v*(m.M@v))),
                               float(np.sum(m.mass[:, None]*(m.particles.N@v)**2)), places=14)

    def test_random_resolved_modes_survive_nonuniform_history_loop(self):
        m = self.m
        x = bent_nodes(m)
        xp, F = m.particles.N @ x, m.gradient(m.particles, x)
        v = self.rng.normal(size=x.shape)*.02
        v[m.fixed] = 0
        restored, xp1, F1, _, info = m.commit(x, xp, F, v, .001)
        np.testing.assert_allclose(restored, x+.001*v, atol=1e-12)
        np.testing.assert_allclose(F1, m.gradient(m.particles, restored), atol=1e-12)
        self.assertLess(info['increment_recovery_relative'], 1e-8)
        self.assertLess(abs(info['history_energy_relative']), 1e-9)
        face = m.sample(np.array([[.25, .45, .48], [.25, .54, .51]]), np.ones(2))
        np.testing.assert_allclose(face.N @ v, 0, atol=1e-15)

    def test_incompatible_history_is_rejected_not_reset(self):
        m = self.m
        F = m.gradient(m.particles, m.X)
        F[0, 0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'incompatible particle history'):
            m.recover(m.particles.X, F)
        with self.assertRaises(ValueError):
            m.commit(m.X, m.particles.X, F, np.zeros_like(m.X), -.01)

    def test_energy_gradient_derivative_and_material_match(self):
        m = self.m
        x = bent_nodes(m)
        F = m.gradient(m.quadrature, x)
        psi, P = material_response(F, m.quadrature.A, m.params)
        self.assertAlmostEqual(psi[0], energy(F[0], m.quadrature.A[0], m.params), places=12)
        np.testing.assert_allclose(P[0], pk1(F[0], m.quadrature.A[0], m.params), atol=1e-12)
        v = self.rng.normal(size=x.shape)*.03
        p = self.rng.normal(size=x.shape)
        vhat = np.zeros_like(v)
        dt, eps = .005, 1e-5
        _, r = m.potential(v, x, vhat, dt)
        fd = (m.potential(v+eps*p, x, vhat, dt)[0]-m.potential(v-eps*p, x, vhat, dt)[0])/(2*eps)
        self.assertAlmostEqual(fd, float(np.sum(r*p)), delta=1e-9)
        q = self.rng.normal(size=x.shape)
        Jp = (m.potential(v+eps*p, x, vhat, dt)[1]-m.potential(v-eps*p, x, vhat, dt)[1])/(2*eps)
        Jq = (m.potential(v+eps*q, x, vhat, dt)[1]-m.potential(v-eps*q, x, vhat, dt)[1])/(2*eps)
        self.assertAlmostEqual(float(np.sum(q*Jp)), float(np.sum(p*Jq)), delta=1e-9)

    def test_exact_rigid_pose_preserves_energy_and_rotates_stress(self):
        m = self.m
        x = bent_nodes(m)
        angle = .73
        Q = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        F = m.gradient(m.quadrature, x)
        E, P = material_response(F, m.quadrature.A, m.params)
        rotated = (x-.5) @ Q.T+.5+np.array([.21, -.03, .01])
        E1, P1 = material_response(m.gradient(m.quadrature, rotated), m.quadrature.A, m.params)
        np.testing.assert_allclose(E1, E, atol=1e-11)
        np.testing.assert_allclose(P1, Q@P, atol=1e-10)
        xp = m.particles.N @ rotated
        restored, _ = m.recover(xp, m.gradient(m.particles, rotated))
        np.testing.assert_allclose(restored, rotated, atol=1e-12)

    def test_no_clamped_zero_stiffness_modes_without_mass(self):
        # Full quadrature Q1 is an independent same-law small-strain reference.
        _, _, free, _, _, matrices = beam_matrices(9)
        eig = np.linalg.eigvalsh(matrices[1][free][:, free].toarray())
        self.assertGreater(eig[0], 1e-9*eig[-1])
        m = MaterialQ1(grid=9, field='uniform')
        p = self.rng.normal(size=m.X.shape)
        eps = 1e-7
        fd = (m.elastic(m.X+eps*p)[1]-m.elastic(m.X-eps*p)[1])/(2*eps)
        np.testing.assert_allclose(fd.ravel(), matrices[1]@p.ravel(), rtol=1e-7, atol=1e-7)

    def test_short_implicit_closed_loop(self):
        m = self.m
        x = bent_nodes(m)
        xp, F = m.particles.N @ x, m.gradient(m.particles, x)
        v = np.zeros_like(xp)
        previous, _ = m.elastic(x)
        for _ in range(3):
            xp, F, v, info = m.step(xp, F, v, .001)
            self.assertLessEqual(info['mechanical'], previous+1e-10)
            self.assertLess(abs(info['history_energy_relative']), 1e-8)
            self.assertGreater(info['min_det'], .9)
            self.assertEqual(info['clamp_speed'], 0.)
            self.assertLess(abs(info['energy_budget_residual']), 1e-16)
            self.assertLess(abs(info['projection_delta']), 1e-16)
            previous = info['mechanical']

    def test_inadmissible_trial_backtracks_instead_of_false_convergence(self):
        def objective(y):
            if np.any(y < 0) or np.any(y >= .5):
                return np.inf, np.zeros_like(y)
            return float(100*np.sum((y-.4)**2)), 200*(y-.4)
        y, _, rejected = minimize_with_backtracking(objective, np.array([.1]))
        self.assertGreater(rejected, 0)
        np.testing.assert_allclose(y, .4, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
