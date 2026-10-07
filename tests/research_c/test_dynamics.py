import copy
import unittest

import numpy as np
import scipy.linalg as la

from engine.aniso_phase1.research_c.model import Model, EnrichedSpace, quadrature
from engine.aniso_phase1.research_c.dynamics import AVF, StepRejected
from engine.aniso_phase1.research_c.transfer import encode, roundtrip
from engine.aniso_phase1.research_c.migration import migrate


class DynamicsTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.rng = np.random.default_rng(220930)

    def test_affine_quadratic_rotation_and_virtual_work(self):
        m = self.model; s = m.rest(); A = self.rng.normal(size=(3, 3))*.02
        s.q[:m.space.carriers] = m.space.Y0@A.T + [.01, -.01, .02]
        s.velocity[:] = self.rng.normal(size=s.q.shape)*.003
        k = m.space.kinematics(m.X, s.q, s.velocity)
        np.testing.assert_allclose(k['F'], np.broadcast_to(np.eye(3)+A, k['F'].shape), atol=1e-10)
        # Local Q2 exactly represents t(1-t) on the free span.
        q = np.zeros_like(s.q); q[m.space.carriers, 0] = .002
        xx = m.X[:, 0]; expected = np.where((xx > .25) & (xx < .75), .008*((xx-.25)/.5)*(1-(xx-.25)/.5), 0)
        np.testing.assert_allclose((m.N@q)[:, 0], expected, atol=1e-12)
        dual = self.rng.normal(size=k['C'].shape)
        force = self.rng.normal(size=k['v'].shape)
        lhs = np.sum(k['v']*force)+np.sum(k['C']*dual)
        adj = k['T'].T@force+np.einsum('pnj,pij->ni', k['L'], dual)
        self.assertAlmostEqual(lhs, float(np.sum(s.velocity*adj)), places=10)
        theta = .7; R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        before = m.evaluate(s.q)
        rotated = s.q@R.T
        rotated[:m.space.carriers] += m.space.Y0@(R-np.eye(3)).T
        after = m.evaluate(rotated)
        self.assertAlmostEqual(before['U'], after['U'], places=10)
        np.testing.assert_allclose(after['P'], R@before['P'], atol=1e-9)

    def test_energy_tangent_and_full_mass(self):
        m = self.model; q = self.rng.normal(scale=1e-4, size=(m.space.n, 3)); d = self.rng.normal(size=q.shape)
        out = m.evaluate(q, tangent=True); eps = 1e-6
        plus, minus = m.evaluate(q+eps*d), m.evaluate(q-eps*d)
        derivative = (plus['U']-minus['U'])/(2*eps)
        self.assertLess(abs(derivative-np.sum(out['force']*d)), 1e-6)
        fd = (plus['force']-minus['force'])/(2*eps)
        np.testing.assert_allclose(m.tangent_action(q, d), fd, rtol=2e-5, atol=1e-5)
        np.testing.assert_allclose((out['K']@d.ravel()).reshape(d.shape), fd, rtol=2e-5, atol=1e-5)
        self.assertGreater(la.eigvalsh(m.M)[0], 0.)
        self.assertGreater(la.norm(m.M[:m.space.carriers, m.space.carriers:]), 1e-5)
        v = self.rng.normal(size=q.shape)
        self.assertAlmostEqual(m.kinetic(v), .5*np.sum(m.mass[:, None]*(m.N@v)**2), places=12)

    def test_v22_high_order_potential_displacement_adapter(self):
        from engine.aniso_phase1.high_order_space import HighOrderPotential
        from engine.aniso_phase1.tensor_reference import coordinates
        from engine.aniso_phase1.research_c.model import EDGES
        import itertools
        m = self.model; space = m.space
        X = np.array(list(itertools.product(*coordinates(EDGES, 4))))
        T, _ = space.basis(X)
        potential = HighOrderPotential(EDGES, 4, T[:, :space.carriers], T[:, space.carriers:],
                                       np.zeros((space.carriers, space.carriers)), m.params, m.params.A0)
        q = self.rng.normal(scale=1e-4, size=(space.n, 3)); Y = q.copy(); Y[:space.carriers] += space.Y0
        exact = Model(orders=(5, 5, 5)).evaluate(q); old = potential.evaluate(Y, order=5)
        self.assertAlmostEqual(old['U'], exact['U'], places=10)
        np.testing.assert_allclose(old['force'], exact['force'], atol=1e-9)

    def test_boundary_impulses_cycle_and_rollback(self):
        solver = AVF(self.model, moving_grid=True)
        solver.child_states = {'B': {'version': 1, 'history': [1.]}}
        for _ in range(4):
            row = solver.step(.0005)
        self.assertLess(abs(row['budget_defect_J']), 1e-9)
        self.assertLess(row['linear_momentum_error'], 1e-9)
        self.assertGreater(row['min_det_F'], .9)
        before = copy.deepcopy(solver.__dict__)
        def bad(state, children):
            children['B']['history'][0] = 999
            state.q[:] = 0
            raise ValueError('injected child failure')
        with self.assertRaises(StepRejected):
            solver.step(.0005, prepare_children=bad)
        np.testing.assert_array_equal(solver.state.q, before['state'].q)
        np.testing.assert_array_equal(solver.previous, before['previous'])
        np.testing.assert_array_equal(solver.packet.residual_C, before['packet'].residual_C)
        self.assertEqual(solver.child_states, before['child_states'])
        self.assertEqual(solver.state.time, before['state'].time)
        with self.assertRaises(StepRejected):
            solver.step(.0005, max_iters=0)
        self.assertEqual(solver.state.time, before['state'].time)
        with self.assertRaises(StepRejected):
            solver.step(float('nan'))

    def test_grid_affine_local_history_and_actual_crossings(self):
        m = self.model; s = m.rest(); s.velocity[m.space.carriers:, :] = .02
        _, packet, _ = roundtrip(m, s)
        prior = packet.cells.copy()
        s.q[:m.space.carriers, 0] = .11
        for _ in range(20):
            s.velocity, packet, row = roundtrip(m, s, prior)
        self.assertGreater(row['crossed_particles'], 0)
        np.testing.assert_allclose(s.velocity[m.space.carriers:], .02, atol=1e-9)
        x = m.X; C = np.broadcast_to(np.eye(3)*.02, (len(x), 3, 3)); v = x*.02+[.1, .2, .3]
        pkt = encode(x, v, C, m.mass)
        np.testing.assert_allclose(pkt.residual_v, 0, atol=1e-12)
        np.testing.assert_allclose(pkt.residual_C, 0, atol=1e-12)

    def test_nested_expansion_rejects_lossy_coarsening(self):
        a, b = Model(local_modes=1), self.model; state = a.rest()
        state.q[a.space.carriers, 0] = .002
        state.velocity[a.space.carriers, 1] = .003
        migrated, report = migrate(a, b, state)
        self.assertTrue(report['accepted'])
        restored, back = migrate(b, a, migrated)
        self.assertTrue(back['accepted'])
        np.testing.assert_allclose(restored.q, state.q, atol=1e-9)
        migrated.q[-1, 0] = .002
        rejected, report = migrate(b, a, migrated)
        self.assertIsNone(rejected); self.assertFalse(report['accepted'])
        np.testing.assert_allclose(migrated.q[-1, 0], .002)


if __name__ == '__main__':
    unittest.main()
