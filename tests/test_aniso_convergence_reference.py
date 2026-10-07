import tempfile
import unittest
from pathlib import Path

import numpy as np

from engine.aniso_phase1.consistent_transfer import bent_nodes
from engine.aniso_phase1.history_increment import HistoryField, ReferenceBasis, frozen
from engine.aniso_phase1.residual_enrichment import ResidualEnrichedQ1
from engine.aniso_phase1.convergence_reference import (
    FrozenResidualQ1, geometry, knots, union_knots, tensor_rule, sites,
    state_from_field, save_fields, load_fields, rms, compare_fields)


class ConvergenceReferenceTests(unittest.TestCase):
    def setUp(self):
        self.source = geometry(17)
        self.axes = knots(self.source, True)
        points, weights = tensor_rule(union_knots(knots(self.source), self.axes), 3)
        self.sites = sites(points, weights)
        field = HistoryField(((ReferenceBasis(self.source), frozen(bent_nodes(self.source))),))
        self.state = state_from_field(field, self.sites, self.sites)
        self.model = FrozenResidualQ1(self.source, self.state, self.axes)

    def test_fixed_anchor_commits_are_compatible_after_multiple_steps(self):
        m, s = self.model, self.state
        rng = np.random.default_rng(651)
        for _ in range(3):
            v = m.R @ (rng.normal(size=(m.R.shape[1], 3))*.003)
            s = m.commit(s, v, .001)
            x, F = s.field.evaluate(s.particles.X)
            np.testing.assert_allclose(x, s.xp, atol=1e-11)
            np.testing.assert_allclose(F, s.Fp, atol=1e-10)
        self.assertIs(m.state, self.state)
        self.assertEqual(len(s.field.terms), 2)

    def test_mass_whitening_keeps_the_same_discrete_solution(self):
        m, s = self.model, self.state
        other = ResidualEnrichedQ1(self.source, s, self.axes)
        vp = np.zeros_like(s.xp)
        a, va, info = m.step(s, vp, .00025)
        b, vb, _ = other.step(s, vp, .00025)
        self.assertLess(rms(va-vb, s.mass)/rms(vb, s.mass), 1e-4)
        np.testing.assert_allclose(a.Fp, b.Fp, atol=1e-8)
        self.assertLess(abs(info['energy_budget_residual']), 1e-16)
        np.testing.assert_allclose(m.last_velocity.evaluate(s.particles.X)[0], va, atol=1e-11)
        L = np.tril(m.reduced_factor[0])
        np.testing.assert_allclose(L @ L.T, m.Mr.toarray(), atol=1e-15)

    def test_common_initial_field_is_not_refitted_on_finer_grid(self):
        fine = geometry(33)
        q = sites(*tensor_rule(knots(fine), 3))
        s = state_from_field(self.state.field, self.sites, q)
        np.testing.assert_array_equal(s.xp, self.state.xp)
        np.testing.assert_array_equal(s.Fp, self.state.Fp)
        np.testing.assert_allclose(q.weight.sum(), .5*.125**2, atol=1e-15)

    def test_fixed_space_agrees_with_refreshed_residual_span(self):
        m = self.model
        fixed, refreshed = self.state, self.state
        vf = np.zeros_like(fixed.xp)
        vr = vf.copy()
        for _ in range(3):
            fixed, vf, _ = m.step(fixed, vf, .00025)
            current = ResidualEnrichedQ1(self.source, refreshed, self.axes)
            refreshed, vr, _ = current.step(refreshed, vr, .00025)
        self.assertLess(rms(vf-vr, fixed.mass)/rms(vr, fixed.mass), 2e-4)
        np.testing.assert_allclose(fixed.Fp, refreshed.Fp, atol=2e-8)

    def test_field_roundtrip_includes_velocity_and_gradients(self):
        m, s = self.model, self.state
        s, _, _ = m.step(s, np.zeros_like(s.xp), .00025)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'fields.npz'
            save_fields(path, position=s.field, velocity=m.last_velocity)
            restored = load_fields(path)
        X = np.array([[.31, .47, .51], [.53, .46, .49]])
        for key, field in [('position', s.field), ('velocity', m.last_velocity)]:
            for a, b in zip(field.evaluate(X), restored[key].evaluate(X)):
                np.testing.assert_allclose(a, b, atol=1e-12)

    def test_common_probe_norms_and_region_error_shares(self):
        source = geometry(65)
        X, w = tensor_rule(knots(source), 2)
        ref = self.state.field
        translated = HistoryField(tuple((b, frozen(c+np.array([.01, 0, 0]))) for b, c in ref.terms))
        zero = HistoryField(((ReferenceBasis(self.source), frozen(np.zeros_like(self.source.X))),))
        result = compare_fields(translated, zero, ref, zero, X, w, source.params)
        self.assertAlmostEqual(result['all']['x_rms'], .01, places=13)
        self.assertLess(result['all']['F_rms'], 1e-13)
        self.assertAlmostEqual(sum(result[k]['x_squared_error_share'] for k in ('clamp', 'switch', 'bulk')), 1., places=12)
        self.assertAlmostEqual(result['clamp']['x_squared_error_share'], .125, places=12)


if __name__ == '__main__':
    unittest.main()
