import unittest
import numpy as np
from engine.aniso_phase1.research_d.common_kinetic import PointInertia


class PolynomialSpace:
    p = 2
    edges = [np.array([0., 1.])]*3
    ndof = 2
    def __init__(self):
        x = np.stack(np.meshgrid(*([np.linspace(0, 1, 3)]*3), indexing="ij"), axis=-1).reshape(-1, 3)
        self.T = np.column_stack((np.ones(len(x)), x[:, 0]*(1-x[:, 0])))
    def nodes(self, v): return self.T @ v
    def adjoint(self, f): return self.T.T @ f


class ConsistentPointMass(unittest.TestCase):
    def test_analytic_mass_with_cross_inertia(self):
        s = PolynomialSpace()
        m = PointInertia(s)
        M = np.array([[1., 1/6], [1/6, 1/30]])
        v = np.array([[.2, -.3, .7], [.8, .1, -.5]])
        np.testing.assert_allclose(m.apply(v), M @ v, atol=2e-16)
        self.assertAlmostEqual(m.energy(v), .5*np.sum(v*(M@v)), places=14)
        self.assertGreater(np.linalg.eigvalsh(M).min(), 0)
        self.assertNotEqual(m.apply(np.array([[0.,0.,0.],[1.,0.,0.]]))[0,0], 0.)
        np.testing.assert_allclose(PointInertia(s, order=4).apply(v), m.apply(v), atol=3e-16)

    def test_underintegration_is_detectable_and_inputs_rejected(self):
        s = PolynomialSpace()
        v = np.array([[0.,0.,0.],[1.,0.,0.]])
        self.assertGreater(abs(PointInertia(s, order=2).energy(v)-1/60), 1e-4)
        for kwargs in ({"density":0}, {"density":float("nan")}, {"order":0}, {"order":2.5}):
            with self.assertRaises(ValueError): PointInertia(s, **kwargs)
        with self.assertRaises(ValueError): PointInertia(s).apply(np.full((2,3),np.nan))
