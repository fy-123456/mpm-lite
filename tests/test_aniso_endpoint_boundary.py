import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_carrier_joint import load_case,cube,spectrum
from benchmarks.aniso_v17_modes import controlled_case
from engine.aniso_phase1.carrier_driven import FastGeometry,projection,DrivenAVF
from engine.aniso_phase1.endpoint_boundary import EndpointAVF,endpoint_impulse,prescribed_speed
from engine.aniso_phase1.unresolved_velocity import pack

class EndpointTests(unittest.TestCase):
    def test_constraint_impulse_preserves_free_and_null_history(self):
        s,e,m,h,_=load_case(1.6);g=FastGeometry(s,e,m,h);z=pack(s.v,s.C);new,d=endpoint_impulse(g,z,0.,h)
        self.assertLess(d['energy_change_J'],1e-18);self.assertAlmostEqual(d['energy_change_J'],-d['kinetic_loss_J'],places=16)
        self.assertLess(la.norm((g.J@g.Q).T@(g.metric[:,None]*(new-z))),1e-13)
        p0,_=projection(g.J,g.metric,z);p1,_=projection(g.J,g.metric,new)
        np.testing.assert_allclose(z-p0,new-p1,atol=1e-10)
        nn,dd=endpoint_impulse(g,new,0.,h);np.testing.assert_allclose(nn,new,atol=1e-10);self.assertLess(dd['kinetic_loss_J'],1e-20)
    def test_driven_actual_endpoint_and_total_work(self):
        s,e,m,h,_=controlled_case();s.v[:]=0;s.C[:]=0;a=EndpointAVF(s,e,m,h,mode='driven')
        for _ in range(40):
            r=a.step(.0005);self.assertLess(r['endpoint_velocity_constraint'],1e-12);self.assertLess(abs(r['budget_defect_J']),1e-18)
            self.assertLess(abs(r['endpoint_speed']-prescribed_speed(a.state.time)),1e-15)
            self.assertLess(abs(r['reaction_N']-r['midpoint_reaction_N']-r['endpoint_reaction_N']),1e-15)
        self.assertTrue(spectrum(e.tangent(a.state.Y,a.endpoint_geometry.Q))['passed'])
    def test_free_rigid_motion_no_boundary_dissipation(self):
        s,e,m,h=cube(velocity=np.array([.1,.025,-.02]));a=EndpointAVF(s,e,m,h,clamped=False)
        for _ in range(12):r=a.step(.005);self.assertEqual(r['constraint_kinetic_loss_J'],0.)
        np.testing.assert_allclose(a.state.x,s.x+.06*s.v,atol=1e-10)
    def test_nonzero_work_linear_and_angular_impulse(self):
        s,e,m,h,_=load_case(1.1);g=FastGeometry(s,e,m,h);z=pack(s.v,s.C)
        new,d=endpoint_impulse(g,z,.012,h);delta=d['delta'];n=len(m)
        np.testing.assert_allclose(d['impulse'].sum(0),m@delta[:n],atol=1e-13)
        angular=np.sum(np.cross(s.x,m[:,None]*delta[:n]),axis=0)
        for j in range(3):angular+=np.sum(np.cross(np.eye(3)[j],g.metric[(j+1)*n:(j+2)*n,None]*delta[(j+1)*n:(j+2)*n]),axis=0)
        np.testing.assert_allclose(np.sum(np.cross(s.Y,d['impulse']),axis=0),angular,atol=1e-13)
        self.assertLess(abs(d['energy_change_J']-d['external_work_J']+d['kinetic_loss_J']),1e-16)
    def test_failure_atomic(self):
        s,e,m,h,_=load_case();a=EndpointAVF(s,e,m,h,mode='driven');old=a.state.clone()
        with self.assertRaises(RuntimeError):a.step(.001,max_iters=0)
        for k in ('x','Y','v','C'):np.testing.assert_array_equal(getattr(a.state,k),getattr(old,k))
        self.assertEqual(a.steps,0)
if __name__=='__main__':unittest.main()
