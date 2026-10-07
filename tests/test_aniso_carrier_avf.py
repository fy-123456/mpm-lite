"""Time-centered force/kinetic consistency; no added physical damping."""
import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_carrier_joint import load_case,cube,spectrum
from engine.aniso_phase1.carrier_joint import gradient
from engine.aniso_phase1.carrier_avf import CarrierAVFSolver,trial_terms
from engine.aniso_phase1.unresolved_velocity import pack


class CarrierAVFTests(unittest.TestCase):
    def test_incremental_potential_residual_exact_tangent(self):
        s,e,m,h,_=load_case();so=CarrierAVFSolver(s,e,m,h);g=so.geometry;dt=.001;rng=np.random.default_rng(162)
        u=.003*rng.normal(size=(g.Q.shape[1],3));d=rng.normal(size=u.shape);d/=la.norm(d);z=pack(s.v,s.C);eps=1e-5
        E,r,H,_,_=trial_terms(e,s.Y,z,g,u,dt,True)
        ep,rp,*_=trial_terms(e,s.Y,z,g,u+eps*d,dt);em,rm,*_=trial_terms(e,s.Y,z,g,u-eps*d,dt)
        expected=float(np.sum(r*d));self.assertLess(abs((ep-em)/(2*eps)-expected)/max(abs(expected),1e-12),1e-5)
        self.assertLess(la.norm((rp-rm).T.ravel()/(2*eps)-H@d.T.ravel())/la.norm(H@d.T.ravel()),1e-6)
        self.assertLess(la.norm(H-H.T),1e-12)

    def test_fixed_geometry_energy_exchange_and_no_starting_projection_loss(self):
        s,e,m,h,_=load_case();so=CarrierAVFSolver(s,e,m,h);total0=e.evaluate(s.Y)['U']+.5*np.sum(so.geometry.metric[:,None]*pack(s.v,s.C)**2)
        for _ in range(12):
            row=so.step(.0005)
            self.assertLess(abs(row['kinetic_force_work_defect_J']),1e-13)
            self.assertLess(abs(row['potential_quadrature_error_J']),1e-13)
            self.assertLess(abs(row['total_J']-total0),1e-12)
        so=CarrierAVFSolver(s,e,m,h);self.assertLess(abs(so.step(1e-7)['delta_total_J']),1e-12)

    def test_rigid_translation_moving_support_and_static_gate(self):
        v=np.array([.1,.025,-.02]);s,e,m,h=cube(velocity=v);so=CarrierAVFSolver(s,e,m,h,moving=True,clamped=False)
        for _ in range(30):row=so.step(.005)
        np.testing.assert_allclose(so.state.x,s.x+so.state.time*v,atol=1e-11)
        np.testing.assert_allclose(so.state.Y,s.Y+so.state.time*v,atol=1e-11)
        np.testing.assert_allclose(gradient(e.B,so.state.Y),np.broadcast_to(np.eye(3),(len(s.x),3,3)),atol=1e-11)
        self.assertTrue(spectrum(e.tangent(so.state.Y),6)['passed'])

    def test_failed_solve_keeps_all_history(self):
        s,e,m,h,_=load_case();so=CarrierAVFSolver(s,e,m,h,moving=True);old=so.state.clone()
        with self.assertRaises(RuntimeError):so.step(.001,max_iters=0)
        for key in ('x','Y','v','C'):np.testing.assert_array_equal(getattr(so.state,key),getattr(old,key))
        self.assertEqual(so.steps,0);self.assertEqual(so.state.time,old.time)

if __name__=='__main__':unittest.main()
