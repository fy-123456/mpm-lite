"""History/template separation and variational checks independent of dynamics."""
import unittest
import numpy as np

from engine.aniso_phase1.consistent_transfer import MaterialQ1, bent_nodes, material_response
from engine.aniso_phase1.template_remap import RemappedQ1, axes_for
from engine.aniso_phase1.history_increment import HistoryState, HistoryIncrementalQ1, material_tangent
from engine.aniso_phase1.beam_reference import reference_hessian


class HistoryIncrementTests(unittest.TestCase):
    def setUp(self):
        self.old = MaterialQ1(grid=9)
        self.x = bent_nodes(self.old)
        self.state = HistoryState.from_nodal(self.old, self.x)
        self.axes = axes_for(self.old, 'identity')
        self.axes[0][2] += .2*self.old.h
        self.model = HistoryIncrementalQ1(self.old, self.state, self.axes)
        self.rng = np.random.default_rng(851)

    def test_switch_keeps_sites_shape_stress_direction_and_strict_control(self):
        s, m = self.state, self.model
        for mapped, sites in ((m.particles, s.particles), (m.quadrature, s.quadrature)):
            self.assertIs(mapped.X, sites.X)
            self.assertIs(mapped.A, sites.A)
            self.assertIs(mapped.weight, sites.weight)
        self.assertIs(m.mass, s.mass)
        before = HistoryIncrementalQ1(self.old, s)
        self.assertEqual(before.elastic(s)[0], m.elastic(s)[0])
        self.assertGreater(np.linalg.norm(m.elastic(s)[1]), 1e-5)
        with self.assertRaisesRegex(ValueError, 'incompatible particle history'):
            RemappedQ1(self.old, self.axes).recover(s.xp, s.Fp)
        with self.assertRaises(ValueError):
            s.Fq[0, 0, 0] = 5

    def test_analytic_material_tangent_with_repeated_singular_values(self):
        Q, _ = np.linalg.qr(self.rng.normal(size=(3, 3)))
        F = np.stack([np.eye(3), 1.12*Q, Q @ np.diag([1., 1.+1e-11, .94]),
                      Q @ np.diag([.85, 1.1, 1.3])])
        A = self.state.quadrature.A[:4]
        direction = self.rng.normal(size=F.shape)
        exact = material_tangent(F, A, direction, self.old.params)
        eps = 1e-6
        fd = (material_response(F+eps*direction, A, self.old.params)[1]-
              material_response(F-eps*direction, A, self.old.params)[1])/(2*eps)
        np.testing.assert_allclose(exact, fd, rtol=2e-7, atol=1e-6)
        a = np.array([1., 0., 0.])
        tangent = material_tangent(np.eye(3)[None], np.outer(a, a)[None], direction[:1], self.old.params)
        np.testing.assert_allclose(tangent.ravel(), reference_hessian() @ direction[0].ravel(), atol=1e-12)

    def test_potential_residual_analytic_tangent_and_symmetry(self):
        m, s = self.model, self.state
        v, p, q, predictor = [self.rng.normal(size=m.X.shape)*.03 for _ in range(4)]
        for a in (v, p, q, predictor):
            a[m.fixed] = 0
        dt, eps = .003, 1e-4
        _, residual = m.potential(v, s, predictor, dt)
        plus, rp = m.potential(v+eps*p, s, predictor, dt)
        minus, rm = m.potential(v-eps*p, s, predictor, dt)
        self.assertAlmostEqual((plus-minus)/(2*eps), np.sum(residual*p), delta=1e-11)
        Jp = m.tangent(v, s, p, dt)
        Jq = m.tangent(v, s, q, dt)
        np.testing.assert_allclose(Jp, (rp-rm)/(2*eps), rtol=1e-7, atol=1e-11)
        self.assertAlmostEqual(np.sum(q*Jp), np.sum(p*Jq), delta=1e-15)
        K = m.stiffness(s)
        np.testing.assert_allclose((K @ p.ravel()).reshape(p.shape),
                                   m.tangent(np.zeros_like(v), s, p, 1., include_mass=False), atol=1e-12)
        self.assertLess(np.linalg.norm((K-K.T).toarray()), 1e-11)

    def test_compatible_field_after_additive_commit_off_sample_and_clamp(self):
        m, s = self.model, self.state
        v = self.rng.normal(size=m.X.shape)*.01
        v[m.fixed] = 0
        next_state = m.commit(s, v, .002)
        np.testing.assert_allclose(next_state.Fq, s.Fq+.002*m.gradient(m.quadrature, v), atol=1e-15)
        for sites, x, F in ((s.particles, next_state.xp, next_state.Fp), (s.quadrature, next_state.xq, next_state.Fq)):
            xf, Ff = next_state.field.evaluate(sites.X)
            np.testing.assert_allclose(xf, x, atol=1e-15)
            np.testing.assert_allclose(Ff, F, atol=1e-14)
        X = np.array([[.31, .46, .48], [.58, .51, .54]])
        _, F = next_state.field.evaluate(X)
        eps = 1e-6
        fd = np.stack([(next_state.field.evaluate(X+eps*e)[0]-next_state.field.evaluate(X-eps*e)[0])/(2*eps)
                       for e in np.eye(3)], axis=2)
        np.testing.assert_allclose(F, fd, atol=1e-10)
        boundary = X.copy(); boundary[:, 0] = .25
        np.testing.assert_array_equal(next_state.field.evaluate(boundary)[0], s.field.evaluate(boundary)[0])
        twice = m.commit(next_state, v, .002)
        self.assertEqual(len(twice.field.terms), len(next_state.field.terms))
        with self.assertRaisesRegex(ValueError, 'clamp'):
            m.commit(s, np.ones_like(v), .001)
        bad = np.zeros_like(v); bad[:, 0] = -1000*(m.X[:, 0]-.25)
        with self.assertRaisesRegex(ValueError, 'Jacobian'):
            m.commit(s, bad, .01)
        with self.assertRaisesRegex(ValueError, 'same authoritative'):
            m.elastic(HistoryState.from_nodal(self.old, self.x))

    def test_common_physical_virtual_work_and_stiffness(self):
        m, s = self.model, self.state
        old = HistoryIncrementalQ1(self.old, s)
        def mode(X):
            v = np.zeros_like(X)
            v[:, 1] = (X[:, 0]-.25)*(X[:, 2]-.48)
            return v
        a, b = mode(old.X), mode(m.X)
        np.testing.assert_allclose(old.quadrature.N @ a, m.quadrature.N @ b, atol=1e-16)
        np.testing.assert_allclose(old.gradient(old.quadrature, a), m.gradient(m.quadrature, b), atol=1e-15)
        self.assertAlmostEqual(np.sum(old.elastic(s)[1]*a), np.sum(m.elastic(s)[1]*b), delta=1e-13)
        self.assertAlmostEqual(a.ravel() @ old.stiffness(s) @ a.ravel(),
                               b.ravel() @ m.stiffness(s) @ b.ravel(), delta=1e-13)
        np.testing.assert_allclose(np.sum(s.mass[:, None]*(m.particles.N @ b)**2), np.sum(b*(m.M @ b)), atol=1e-16)

    def test_mass_free_stiffness_and_rigid_rotation_limitation(self):
        m, s = self.model, self.state
        dofs = np.flatnonzero(np.repeat(m.free, 3))
        eigen = np.linalg.eigvalsh(m.stiffness(s)[dofs][:, dofs].toarray())
        self.assertGreater(eigen[0], 1e-9*eigen[-1])
        theta = .45
        Q = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        exact_du = (s.xp-.5) @ (Q-np.eye(3)).T
        fit = m.project(exact_du)
        self.assertGreater(np.linalg.norm(m.particles.N @ fit-exact_du), 1e-7)
        old = HistoryIncrementalQ1(self.old, s)
        old_fit = old.project(exact_du)
        np.testing.assert_allclose(old.particles.N @ old_fit, exact_du, atol=1e-13)
        E, P = material_response(s.Fq, s.quadrature.A, m.params)
        E1, P1 = material_response(Q @ s.Fq, s.quadrature.A, m.params)
        np.testing.assert_allclose(E1, E, atol=1e-11)
        np.testing.assert_allclose(P1, Q @ P, atol=1e-10)

    def test_short_release_identity_control_and_switched_history(self):
        s = self.state
        m = HistoryIncrementalQ1(self.old, s)
        vp = np.zeros_like(s.xp)
        strict = (s.xp.copy(), s.Fp.copy(), vp.copy())
        for _ in range(2):
            s, vp, info = m.step(s, vp, .001)
            xp, Fp, vstrict, _ = self.old.step(*strict, .001)
            strict = xp, Fp, vstrict
            np.testing.assert_allclose(s.xp, xp, atol=1e-11)
            np.testing.assert_allclose(s.Fp, Fp, atol=1e-10)
        m = HistoryIncrementalQ1(self.old, s, self.axes)
        previous = info['mechanical']
        for _ in range(3):
            s, vp, info = m.step(s, vp, .001)
            self.assertLessEqual(info['mechanical'], previous+1e-10)
            self.assertLess(abs(info['energy_budget_residual']), 1e-16)
            self.assertGreater(info['min_det'], .9)
            self.assertEqual(info['clamp_speed'], 0)
            previous = info['mechanical']


if __name__ == '__main__':
    unittest.main()
