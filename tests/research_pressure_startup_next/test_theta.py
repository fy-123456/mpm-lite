import unittest
from types import SimpleNamespace
import numpy as np
from engine.aniso_phase1.research_pressure_startup_next.theta import step,schedule,integrate

def scalar(rate=3.,reservoir=.2):
    return dict(C=np.ones(1),D=np.ones(1),A=np.array([[rate]]),L=np.array([[rate]]),H=np.array([[1/rate]]),Z=np.array([[rate]]),z0=np.array([-rate*reservoir]),rhs=np.array([rate*reservoir]),top=SimpleNamespace(B=np.ones((1,1)),nflux=1))

class ThetaTests(unittest.TestCase):
    def test_stiff_decay(self):
        a=scalar(100.,0.);p,z,_=step(a,np.ones(1),1.,1.)
        self.assertAlmostEqual(p[0],1/101);self.assertGreater(p[0],0)
        mid,_,_=step(a,np.ones(1),1.,.5);self.assertLess(mid[0],0)
    def test_energy_and_mass(self):
        for theta in (.5,1.):
            _,_,r=step(scalar(),np.array([.7]),.12,theta)
            self.assertLess(abs(r['energy_balance_J']),1e-12);self.assertLess(r['mass_defect_m3'],1e-12)
            self.assertGreaterEqual(r['numerical_dissipation_J'],0)
    def test_equilibrium(self):
        for theta in (.5,1.):
            p,z,_=step(scalar(),np.array([.2]),.4,theta)
            np.testing.assert_allclose(p,.2);np.testing.assert_allclose(z,0,atol=1e-14)
    def test_same_physical_switch(self):
        a=schedule(np.linspace(0,2e-4,17),'startup');b=schedule(np.linspace(0,2e-4,33),'startup')
        self.assertEqual(np.count_nonzero(a==1),2);self.assertEqual(np.count_nonzero(b==1),4)
    def test_invalid_grid_and_method(self):
        for t in ([0,0],[1,2],[0,float('nan')]):
            with self.assertRaises(ValueError):schedule(t,'midpoint')
        with self.assertRaises(ValueError):schedule([0,1],'unknown')
        with self.assertRaises(ValueError):schedule([0,1],'startup')
    def test_time_convergence_and_integrated_flow(self):
        a=scalar(3.,0.);p=dict(pressure0_Pa=1.)
        coarse=integrate(a,p,np.linspace(0,1,9),'backward-euler');fine=integrate(a,p,np.linspace(0,1,17),'backward-euler')
        self.assertLess(abs(fine['pressure'][-1,0]-np.exp(-3)),abs(coarse['pressure'][-1,0]-np.exp(-3)))
        self.assertAlmostEqual(fine['cumulative'][-1,0]+fine['pressure'][-1,0],1.)

if __name__=='__main__':unittest.main()
