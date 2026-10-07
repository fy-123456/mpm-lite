import tempfile
from pathlib import Path
import unittest

import numpy as np

from engine.aniso_phase1.smooth_history import fit_smooth_history
from engine.aniso_phase1.convergence_reference import (
    geometry, knots, tensor_rule, sites, state_from_field, sum_fields, load_fields, save_fields)
from engine.aniso_phase1.history_increment import HistoryField, ReferenceBasis, frozen
from engine.aniso_phase1.consistent_transfer import bent_nodes
from engine.aniso_phase1.refinement import TensorReferenceQ1


class SmoothHistoryTests(unittest.TestCase):
    def setUp(self):
        self.source = geometry(17)
        self.original = HistoryField(((ReferenceBasis(self.source), frozen(bent_nodes(self.source))),))
        self.X, self.w = tensor_rule(knots(self.source), 5)
        self.smooth, _ = fit_smooth_history(self.original, self.X, self.w)

    def test_analytic_gradient_clamp_and_continuous_traces(self):
        rng = np.random.default_rng(718)
        X = rng.uniform([.26,.44,.44],[.74,.56,.56],size=(100,3))
        _, F = self.smooth.evaluate(X)
        for d in range(3):
            delta = np.eye(3)[d]*1e-6
            finite = (self.smooth.evaluate(X+delta)[0]-self.smooth.evaluate(X-delta)[0])/2e-6
            np.testing.assert_allclose(finite, F[:,:,d], atol=2e-9, rtol=1e-8)
        boundary = self.source.X[self.source.fixed]
        np.testing.assert_allclose(self.smooth.evaluate(boundary)[0], boundary, atol=1e-15)
        X[:,0] = .5
        jumps = [np.linalg.norm(self.smooth.evaluate(X+[eps,0,0])[1]-
                                self.smooth.evaluate(X-[eps,0,0])[1]) for eps in (1e-6,1e-7)]
        self.assertAlmostEqual(jumps[0]/jumps[1], 10, places=5)

    def test_restart_and_commit_preserve_analytic_history(self):
        q = sites(self.X,self.w)
        state = state_from_field(self.smooth,q,q)
        model = TensorReferenceQ1(self.source,state)
        vp = np.zeros_like(state.xp)
        for _ in range(3):
            state,vp,_ = model.step(state,vp,.00003125)
        x,F = state.field.evaluate(q.X)
        np.testing.assert_allclose(x,state.xq,atol=2e-15)
        np.testing.assert_allclose(F,state.Fq,atol=2e-14)
        self.assertEqual(len(state.field.terms),2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.npz'
            save_fields(path,position=state.field,velocity=model.last_velocity)
            fields=load_fields(path)
            for a,b in zip(fields['position'].evaluate(q.X),(x,F)):
                np.testing.assert_array_equal(a,b)
        model.close()

    def test_distinct_polynomial_coordinates_are_not_merged(self):
        from engine.aniso_phase1.smooth_history import PolynomialHistoryBasis
        b,c=self.smooth.terms[0]
        changed=PolynomialHistoryBasis(b.degrees,origin=(.2,.5,.5))
        other=HistoryField(((changed,c),))
        combined=sum_fields(self.smooth,other)
        self.assertEqual(len(combined.terms),2)
        for a,b in zip(combined.evaluate(self.X),
                       (a+b for a,b in zip(self.smooth.evaluate(self.X),other.evaluate(self.X)))):
            np.testing.assert_allclose(a,b)

    def test_analytic_coefficients_are_not_spatial_interfaces(self):
        from benchmarks.aniso_refinement import field_axes
        axes = field_axes(dict(position=self.smooth), dict(position=self.original))
        for a,b in zip(axes,knots(self.source)):
            np.testing.assert_array_equal(a,b)
        with self.assertRaisesRegex(ValueError,'explicit integration axes'):
            field_axes(dict(position=self.smooth))


if __name__ == '__main__': unittest.main()
