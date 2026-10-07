import unittest
import numpy as np
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import balanced_solve

class ScalingTests(unittest.TestCase):
    def test_nonsymmetric_ill_scaled(self):
        B=np.array([[4.,1.,-2.],[2.,5.,1.],[-1.,3.,7.]])
        A=np.array([1e-10,1.,1e6])[:,None]*B*np.array([1e-3,1.,1e4])[None,:]
        expected=np.array([.2,-.3,.1]);original=A.copy();rhs=A@expected
        x,info=balanced_solve(A,rhs,diagnose=True)
        np.testing.assert_allclose(x,expected,rtol=1e-6,atol=1e-8)
        np.testing.assert_array_equal(A,original)
        self.assertLess(info['original_linear_residual_budget_fraction'],1)
        self.assertGreater(info['scaled_rcond_1'],info['raw_rcond_1'])
    def test_invalid_systems(self):
        for A in (np.zeros((2,2)),np.array([[1.,0.],[2.,0.]]),np.array([[np.nan,0.],[0.,1.]])):
            with self.assertRaises(ValueError):balanced_solve(A,np.ones(2))
    def test_same_unscaled_solution(self):
        A=np.array([[2.,1.],[-3.,0.1]]);b=np.array([3.,-.2]);x,_=balanced_solve(A,b)
        np.testing.assert_allclose(x,np.linalg.solve(A,b),atol=1e-12)
