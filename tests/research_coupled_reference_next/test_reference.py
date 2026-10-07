import unittest
import numpy as np
from engine.aniso_phase1.research_coupled_reference_next.reference import affine_operator,exact,mass_basis,interval_average

class ReferenceContracts(unittest.TestCase):
    def test_non_equilibrium_affine_drift(self):
        out=exact(np.zeros((2,2)),np.array([2.,-3.]),[0,.5],scale=np.array([.01,20.]))
        np.testing.assert_allclose(out,[[0,0],[1,-1.5]],atol=1e-12)

    def test_coupling_pressure_work_cancels(self):
        A,b=affine_operator(np.array([[4.]]),np.array([[2.]]),np.array([3.]),np.zeros((1,1)),np.zeros((1,1)),np.zeros(1),np.zeros(1))
        y=exact(A,b,[0,.1,.5],initial=np.array([.2,.3,.4]))
        energy=.5*(4*y[:,0]**2+y[:,1]**2+3*y[:,2]**2)
        np.testing.assert_allclose(energy,energy[0],rtol=1e-10,atol=1e-12)

    def test_darcy_geometry_sign_and_velocity_exchange(self):
        A,_=affine_operator(np.array([[4.]]),np.array([[2.]]),np.array([3.]),np.array([[5.]]),np.array([[7.]]),np.zeros(1),np.zeros(1))
        np.testing.assert_allclose(A[2],[-7/3,-1.6/3,-5/3])

    def test_mass_basis_keeps_cross_terms(self):
        M=np.array([[2.,.4],[.4,1.]])
        U=mass_basis(M,np.eye(2),2)
        np.testing.assert_allclose(U.T@M@U,np.eye(2),atol=1e-12)
        x=np.array([.3,-.2]);np.testing.assert_allclose(U@U.T@M@x,x,atol=1e-12)

    def test_flux_common_interval_is_integral_average(self):
        fine_t=np.array([0,.25,1.]);cum=np.array([[0],[.5],[2.]])
        np.testing.assert_allclose(interval_average(cum,fine_t),2.)
        np.testing.assert_allclose(interval_average(cum[[0,2]],fine_t[[0,2]]),2.)
        with self.assertRaises(ValueError):interval_average(cum,[0,1,1])

    def test_fixed_skeleton_relaxation_and_scale_rejection(self):
        out=exact(np.array([[-2.]]),np.array([6.]),[0,.5],initial=np.array([1.]))[:,0]
        np.testing.assert_allclose(out,3-2*np.exp(-2*np.array([0,.5])))
        with self.assertRaises(ValueError):exact(np.eye(1),np.ones(1),[1],scale=np.zeros(1))

if __name__=='__main__':unittest.main()
