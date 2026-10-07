"""Check theta pressure blocks against a small physical residual derivative."""
import unittest
from types import SimpleNamespace
import numpy as np
from engine.aniso_phase1.research_pressure_startup_next.coupled import StartupAVF

class CoupledMatrix(unittest.TestCase):
    def fixture(self):
        c=StartupAVF.__new__(StartupAVF);c.cells=2;c.nflux=3;c.fixed_solid=False;c.alpha=.8;c.capacity=np.array([2.,3.]);c.B=np.array([[1.,-1.,0.],[0.,1.,-1.]])
        c.model=SimpleNamespace(ids=np.arange(3),free=np.array([0]),M3ff=np.diag([1.,2.,3.]),rest_K=np.diag([2.,3.,4.]))
        return c
    def test_pressure_blocks_match_residual_derivative(self):
        c=self.fixture();g=np.arange(6,dtype=float).reshape(2,1,3)/7;h=.1;H=np.diag([2.,3.,4.]);p0=np.array([.2,.3]);x=np.arange(8)/17;d=np.arange(8)[::-1]/11
        for theta in (.5,1.):
            def residual(x):
                v=x[:3];p=x[3:5];z=x[5:];pt=(1-theta)*p0+theta*p
                solid=2*c.model.M3ff@v+.5*h*h*c.model.rest_K@v-h*c.alpha*g.reshape(2,3).T@pt
                mass=h*c.alpha*g.reshape(2,3)@v+c.capacity*(p-p0)+h*c.B@z
                return np.r_[solid,mass,H@z-c.B.T@pt]
            eps=1e-5;actual=(residual(x+eps*d)-residual(x-eps*d))/(2*eps)
            np.testing.assert_allclose(c.theta_matrix(h,g,g,H,theta)@d,actual,rtol=1e-8,atol=1e-9)
    def test_midpoint_reproduces_existing_matrix(self):
        c=self.fixture();g=np.ones((2,1,3));H=np.eye(3)
        np.testing.assert_array_equal(c.theta_matrix(.1,g,g,H,.5),c.matrix(.1,g,g,H))

if __name__=='__main__':unittest.main()
