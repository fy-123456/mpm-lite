import unittest
import numpy as np
import scipy.sparse as sp

from engine.aniso_phase1.consistent_transfer import MaterialQ1, bent_nodes
from engine.aniso_phase1.history_increment import HistoryState, HistoryIncrementalQ1
from engine.aniso_phase1.residual_enrichment import ResidualEnrichedQ1
from engine.aniso_phase1.template_remap import axes_for


class ResidualEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.source = MaterialQ1(17, ppc_axis=3)
        self.state = HistoryState.from_nodal(self.source, bent_nodes(self.source))
        self.axes = axes_for(self.source, 'identity')
        self.axes[0][4] += .2*self.source.h
        self.model = ResidualEnrichedQ1(self.source, self.state, self.axes)
        self.rng = np.random.default_rng(711)

    def test_rank_removal_and_extended_consistent_mass(self):
        m, s = self.model, self.state
        self.assertEqual(m.rank, 1)
        self.assertEqual(ResidualEnrichedQ1(self.source, s).rank, 0)
        rest = HistoryState.from_nodal(self.source, self.source.X)
        self.assertEqual(ResidualEnrichedQ1(self.source, rest, self.axes).rank, 0)
        u = self.rng.normal(size=(m.ncoeff, 3))
        np.testing.assert_allclose(np.sum(u*(m.M @ u)), np.sum(s.mass[:, None]*(m.particles.N @ u)**2), atol=1e-15)
        np.testing.assert_allclose(m.project(m.particles.N @ u, clamped=False), u, atol=2e-12)
        cross = m.M[:m.nnode, m.nnode:].toarray()
        self.assertLess(np.max(abs(cross)), 1e-12)
        self.assertGreater(m.rank_info['min_scaled_mass_eigenvalue'], 1e-3)
        for x, y in [(m.particles.X, s.particles.X), (m.quadrature.A, s.quadrature.A)]:
            self.assertIs(x, y)

    def test_exact_finite_rotations_and_committed_history(self):
        from scipy.spatial.transform import Rotation
        s = self.state
        m = ResidualEnrichedQ1(self.source, s, self.axes, clamped=False)
        U0 = m.elastic(s)[0]
        for axis, angle in [([0, 0, 1], 45), ([1, 2, 3], 90)]:
            axis = np.array(axis)/np.linalg.norm(axis)
            Q = Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()
            target = (s.xp-.5) @ (Q-np.eye(3)).T
            fit = m.project(target)
            np.testing.assert_allclose(m.particles.N @ fit, target, atol=1e-12)
            np.testing.assert_allclose(m.gradient(m.quadrature, fit), (Q-np.eye(3)) @ s.Fq, atol=1e-11)
            self.assertLess(abs(m.elastic(s, fit)[0]/U0-1), 1e-9)
            next_state = m.commit(s, fit, 1.)
            x, F = next_state.field.evaluate(s.quadrature.X)
            np.testing.assert_allclose(x, (s.xq-.5) @ Q.T+.5, atol=1e-12)
            np.testing.assert_allclose(F, Q @ s.Fq, atol=1e-11)

    def test_constraints_apply_to_total_field_and_whole_boundary(self):
        m = self.model
        v = m.R @ self.rng.normal(size=(m.R.shape[1], 3))
        face = self.rng.uniform([.25, .4375, .4375], [.25, .5625, .5625], size=(23, 3))
        np.testing.assert_allclose(m.map_points(face).N @ v, 0, atol=1e-10)
        # The nodal part alone need not vanish; enrichment cancels its trace.
        self.assertGreater(np.max(abs(v[:m.nnode][m.fixed])), 1e-6)
        bad = v.copy(); bad[np.flatnonzero(m.fixed)[0], 0] += 1
        with self.assertRaisesRegex(ValueError, 'total clamp'):
            m.commit(self.state, bad, .001)

    def test_nonrepresentable_boundary_modes_are_removed(self):
        x = bent_nodes(self.source)
        x[self.source.fixed, 1] += .002*((x[self.source.fixed, 1]-.4375)/.125)**2
        s = HistoryState.from_nodal(self.source, x)
        axes = [a.copy() for a in self.axes]; axes[1][1] += .2*self.source.h
        m = ResidualEnrichedQ1(self.source, s, axes)
        self.assertLess(m.R.shape[1]-m.free.sum(), m.rank)
        v = m.R @ self.rng.normal(size=(m.R.shape[1], 3))
        face = self.rng.uniform([.25, .4375, .4375], [.25, .5625, .5625], size=(31, 3))
        np.testing.assert_allclose(m.map_points(face).N @ v, 0, atol=1e-10)

    def test_variational_derivatives_symmetry_and_static_rank(self):
        m, s = self.model, self.state
        v, p, q, predictor = [m.R @ (self.rng.normal(size=(m.R.shape[1], 3))*.01) for _ in range(4)]
        dt, eps = .001, 1e-4
        _, r = m.potential(v, s, predictor, dt)
        Ep, rp = m.potential(v+eps*p, s, predictor, dt)
        Em, rm = m.potential(v-eps*p, s, predictor, dt)
        self.assertAlmostEqual((Ep-Em)/(2*eps), np.sum(p*r), delta=1e-11)
        Jp, Jq = m.tangent(v, s, p, dt), m.tangent(v, s, q, dt)
        np.testing.assert_allclose(Jp, (rp-rm)/(2*eps), rtol=1e-6, atol=1e-11)
        self.assertAlmostEqual(np.sum(q*Jp), np.sum(p*Jq), delta=1e-14)
        K = m.stiffness(s)
        np.testing.assert_allclose(K @ p.ravel(), m.tangent(np.zeros_like(v), s, p, 1, False).ravel(), atol=1e-11)
        R = sp.kron(m.R, np.eye(3))
        spectrum = np.linalg.eigvalsh((R.T @ K @ R).toarray())
        self.assertGreater(spectrum[0], 1e-9*spectrum[-1])

    def test_flattened_history_consistency_and_frozen_step(self):
        m, s = self.model, self.state
        for _ in range(3):
            v = m.R @ (self.rng.normal(size=(m.R.shape[1], 3))*.002)
            next_state = m.commit(s, v, .001)
            with self.assertRaisesRegex(ValueError, 'rebuild frozen'):
                m.elastic(next_state)
            np.testing.assert_allclose(next_state.field.evaluate(s.particles.X)[1], next_state.Fp, atol=1e-11)
            self.assertEqual(len(next_state.field.terms), 2)
            s = next_state
            m = ResidualEnrichedQ1(self.source, s, self.axes)
        X = np.array([[.31, .46, .48], [.58, .51, .54]])
        _, F = s.field.evaluate(X)
        eps = 1e-6
        fd = np.stack([(s.field.evaluate(X+eps*e)[0]-s.field.evaluate(X-eps*e)[0])/(2*eps) for e in np.eye(3)], axis=2)
        np.testing.assert_allclose(F, fd, atol=1e-9)

    def test_all_nine_added_dofs_and_off_sample_rotation(self):
        from scipy.spatial.transform import Rotation
        x = bent_nodes(self.source)
        noise = self.rng.normal(size=x.shape)*1e-5
        noise[self.source.fixed] = 0
        s = HistoryState.from_nodal(self.source, x+noise)
        m = ResidualEnrichedQ1(self.source, s, self.axes, clamped=False)
        self.assertEqual(m.rank, 3)
        Q = Rotation.from_rotvec(np.array([.3, .5, -.7])).as_matrix()
        fit = m.project((s.xp-.5) @ (Q-np.eye(3)).T)
        X = self.rng.uniform([.25, .4375, .4375], [.75, .5625, .5625], size=(35, 3))
        sample = m.map_points(X)
        xhist, Fhist = s.field.evaluate(X)
        np.testing.assert_allclose(sample.N @ fit, (xhist-.5) @ (Q-np.eye(3)).T, atol=1e-11)
        np.testing.assert_allclose(m.gradient(sample, fit), (Q-np.eye(3)) @ Fhist, atol=1e-10)
        v, p = [self.rng.normal(size=(m.ncoeff, 3))*.002 for _ in range(2)]
        eps, dt = 1e-4, .0005
        rp = m.potential(v+eps*p, s, v*0, dt)[1]
        rm = m.potential(v-eps*p, s, v*0, dt)[1]
        np.testing.assert_allclose((rp-rm)/(2*eps), m.tangent(v, s, p, dt), rtol=1e-6, atol=1e-11)

    def test_identity_control_and_short_enriched_release(self):
        s = self.state
        m = ResidualEnrichedQ1(self.source, s)
        baseline = HistoryIncrementalQ1(self.source, s)
        vp = np.zeros_like(s.xp)
        probe = self.rng.normal(size=self.source.X.shape)*.01
        probe[m.fixed] = 0
        E0, r0 = baseline.potential(probe, s, probe*0, .0005)
        E1, r1 = m.potential(probe, s, probe*0, .0005)
        self.assertEqual(E0, E1)
        np.testing.assert_array_equal(r0, r1)
        a, va, _ = m.step(s, vp, .0005)
        b, vb, _ = baseline.step(s, vp, .0005)
        np.testing.assert_allclose(a.xp, b.xp, atol=1e-10)
        # Original solver uses a different diagonal scale and 1e-9 stopping
        # tolerance; compare dynamics within that solve accuracy.
        self.assertLess(np.linalg.norm(va-vb)/np.linalg.norm(vb), 1e-4)
        s, vp = a, va
        previous = m.elastic(self.state)[0]
        for _ in range(3):
            m = ResidualEnrichedQ1(self.source, s, self.axes)
            s, vp, info = m.step(s, vp, .0005)
            self.assertLessEqual(info['mechanical'], previous+1e-10)
            self.assertLess(abs(info['energy_budget_residual']), 1e-15)
            self.assertGreater(info['min_det'], .99)
            self.assertLess(info['clamp_speed'], 1e-10)
            previous = info['mechanical']


if __name__ == '__main__':
    unittest.main()
