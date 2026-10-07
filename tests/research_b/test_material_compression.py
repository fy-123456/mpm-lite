import unittest
import numpy as np
from scipy.linalg import expm
from benchmarks.research_b.scenes import source_rule, operator, directions, PARAMS
from engine.aniso_phase1.research_b.material import response
from engine.aniso_phase1.research_b.rules import MaterialRule, compress, local_fallback, moment_audit
from engine.aniso_phase1.research_b.operator import MaterialSession
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.history_increment import material_tangent


class MaterialCompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = source_rule(order=3)
        cls.rule = compress(cls.full, 4)
        cls.op = operator(cls.rule)
        cls.q = np.zeros((9, 3)); cls.q[0, 0] = .035; cls.q[3, 1] = .12; cls.q[6, 0] = -.12
        cls.d = directions()['random']

    def test_full_operator_matches_existing_law_and_independent_diagonal_formula(self):
        F = np.broadcast_to(np.diag([1.1, .98, 1.03]), (len(self.full.weight), 3, 3)).copy()
        dF = np.broadcast_to(np.eye(3)*.07, F.shape)
        a = response(F, self.full.A2, self.full.A4, PARAMS, dF)
        b = material_response(F, self.full.A2, PARAMS)
        np.testing.assert_allclose(a[0], b[0], rtol=2e-10, atol=1e-12)
        np.testing.assert_allclose(a[1], b[1], rtol=2e-10, atol=1e-12)
        np.testing.assert_allclose(a[2], material_tangent(F, self.full.A2, dF, PARAMS), rtol=2e-9, atol=1e-10)
        logs = np.log([1.1, .98, 1.03])
        exact = 10*np.sum(logs**2)+10*logs.sum()**2+100*((1.1**2+.98**2)/2-1)**2
        self.assertAlmostEqual(a[0][0], exact, places=11)

    def test_positive_rules_and_spatial_moments_for_three_budgets(self):
        full = source_rule(mixture=True, order=3)
        for method in ('conditional', 'representatives'):
            for budget in (1, 2, 4):
                rule = compress(full, budget, method)
                audit = moment_audit(full, rule)
                self.assertLess(audit['volume_absolute'], 1e-12)
                self.assertLess(audit['first_moment_absolute'], 1e-12)
                self.assertGreater(audit['min_weight'], 0.)
                self.assertLessEqual(len(rule.weight), len(full.weight))

    def test_fourth_moment_common_F_and_equal_mean_direction_counterexample(self):
        # Both distributions have zero vector mean and identical A2, but
        # different A4. Averaging either directions or A2 loses fiber energy.
        axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
        diag = np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]])/np.sqrt(2)
        energies = []
        F = np.array([[1.13, .1, 0], [0, .96, 0], [0, 0, 1.]])
        for a in (axes, diag):
            full = MaterialRule.from_directions(np.zeros((4, 3)), np.ones(4)/4, a)
            compact = compress(full, 1, direction_jump=2.)
            A = np.broadcast_to(F, (4, 3, 3))
            full_values = response(A, full.A2, full.A4, PARAMS, A*.02)
            small_values = response(F[None], compact.A2, compact.A4, PARAMS, F[None]*.02)
            for x, y in zip(full_values, small_values):
                np.testing.assert_allclose(np.mean(x, axis=0), y[0], rtol=1e-8, atol=1e-10)
            energies.append(float(np.mean(full_values[0])))
        self.assertGreater(abs(energies[0]-energies[1]), .1)

    def test_position_direction_pairing_is_not_replaced_by_mean_F(self):
        X = np.array([[.25, .5, .5], [.75, .5, .5]])
        a = np.array([[1., 0, 0], [0, 1., 0]])
        rules = [MaterialRule.from_directions(X, [1., 1.], aa) for aa in (a, a[::-1])]
        q = np.zeros((9, 3)); q[0, 0] = .05; q[3, 0] = .3
        values = [operator(r).energy(q) for r in rules]
        self.assertGreater(abs(values[0]-values[1]), .1)
        for rule in rules:
            # Mandatory direction splitting keeps the two associations.
            self.assertAlmostEqual(operator(compress(rule, 1)).energy(q), operator(rule).energy(q))

    def test_energy_force_tangent_and_symmetry_multiple_steps(self):
        base = self.op.evaluate(self.q, self.d)
        for eps in (1e-4, 1e-5, 1e-6):
            a, b = self.op.evaluate(self.q+eps*self.d), self.op.evaluate(self.q-eps*self.d)
            self.assertLess(abs((a['U']-b['U'])/(2*eps)-np.sum(base['force']*self.d)), 2e-7)
            np.testing.assert_allclose((a['force']-b['force'])/(2*eps), base['tangent_action'], rtol=2e-4, atol=2e-7)
        v = directions()['bending']
        self.assertLess(abs(np.sum(v*base['tangent_action'])-np.sum(self.d*self.op.tangent_action(self.q, v))), 1e-9)

    def test_rigid_rotation_covariance(self):
        R = expm(np.array([[0., -.6, .2], [.6, 0, -.1], [-.2, .1, 0]]))
        q = self.q @ R.T
        q[:3] += (R-np.eye(3)).T
        a, b = self.op.evaluate(self.q), self.op.evaluate(q)
        self.assertAlmostEqual(a['U'], b['U'], places=10)
        np.testing.assert_allclose(b['force'], a['force'] @ R.T, rtol=2e-8, atol=1e-9)

    def test_invalid_trial_cannot_commit_previous_success_or_mutate_caller(self):
        session = MaterialSession(self.op, 1e-4, 1e-3)
        q = self.q.copy()
        session.trial(q)
        with self.assertRaises(RuntimeError):
            session.propose(operator(self.full), q, self.d)
        bad = q.copy(); bad[0, 0] = -2
        with self.assertRaises(ValueError):
            session.trial(bad)
        with self.assertRaises(RuntimeError):
            session.commit(accepted_step=True)
        session.rollback()
        np.testing.assert_array_equal(q, self.q)
        self.assertEqual(session.cache_version, 0)
        session.trial(q); session.commit(accepted_step=True)
        self.assertEqual(session.cache_version, 1)

    def test_finite_but_overflowing_state_is_rejected_without_tangent(self):
        session = MaterialSession(self.op, 1e-4, 1e-3)
        session.trial(self.q)
        bad = self.q.copy(); bad[0, 0] = 1e200
        with np.errstate(all='ignore'):
            with self.assertRaises(ValueError):
                session.trial(bad)
        with self.assertRaises(RuntimeError):
            session.commit(accepted_step=True)
        session.rollback()
        self.assertEqual(session.cache_version, 0)

    def test_rule_budget_rejection_stale_state_and_absolute_jump_accounting(self):
        full = operator(self.full)
        session = MaterialSession(self.op, 0., 0.)
        p = session.propose(full, self.q, self.d)
        self.assertFalse(session.commit_rule(p, self.q, accepted_step=True))
        self.assertIs(session.operator, self.op)
        session = MaterialSession(self.op, 1., 1.)
        p = session.propose(full, self.q, self.d)
        with self.assertRaises(RuntimeError):
            session.commit_rule(p, self.q*.9, accepted_step=True)
        self.assertTrue(session.commit_rule(p, self.q, accepted_step=True))
        back = session.propose(self.op, self.q, self.d)
        self.assertTrue(session.commit_rule(back, self.q, accepted_step=True))
        self.assertAlmostEqual(session.cumulative_absolute, 2*abs(p.delta_U_rule))
        self.assertGreater(session.cumulative_absolute, 0.)

    def test_local_fallback_preserves_unaffected_regions_and_immutable_rule(self):
        rule = local_fallback(self.full, self.rule, [0, 5])
        moment_audit(self.full, rule)
        self.assertEqual(sum(rule.partition == 0), sum(self.full.partition == 0))
        self.assertEqual(sum(rule.partition == 1), sum(self.rule.partition == 1))
        with self.assertRaises(ValueError):
            rule.weight.setflags(write=True)
        with self.assertRaises(ValueError):
            rule.X[0, 0] = 0


if __name__ == '__main__':
    unittest.main()
