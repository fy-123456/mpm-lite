import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_carrier_joint import load_case,cube,spectrum
from benchmarks.aniso_v17_modes import controlled_case
from engine.aniso_phase1.carrier_joint import Geometry,gradient
from engine.aniso_phase1.carrier_avf import CarrierAVFSolver
from engine.aniso_phase1.carrier_driven import DrivenAVF,FastGeometry,EquivalentEnergy,position_map,kinetic_metric,projection,displacement
from engine.aniso_phase1.unresolved_velocity import maps,pack

class DrivenTests(unittest.TestCase):
    def test_maps_energy_and_tangent_equivalence(self):
        for t in (1.1,1.6):
            s,e,m,h,_=load_case(t);g=Geometry(s,e,m,h);f=FastGeometry(s,e,m,h);fast=EquivalentEnergy(e)
            np.testing.assert_array_equal(g.nodes,f.nodes)
            np.testing.assert_allclose(g.N,f.N,atol=1e-14)
            np.testing.assert_allclose(g.E,f.E,atol=1e-12)
            np.testing.assert_allclose(g.J,f.J,atol=1e-11)
            rng=np.random.default_rng(18);Y=s.Y+rng.normal(size=s.Y.shape)*1e-5
            a,b=e.evaluate(Y),fast.evaluate(Y)
            for k in ('U','Us','Um','force','F','P'):np.testing.assert_allclose(a[k],b[k],rtol=1e-9,atol=1e-12)
            z=pack(s.v,s.C);p,_=projection(f.J,f.metric,z)
            U,sv,_=la.svd(np.sqrt(g.metric)[:,None]*g.J,full_matrices=False);U=U[:,sv>1e-12*sv[0]]
            ref=U@(U.T@(np.sqrt(g.metric)[:,None]*z))/np.sqrt(g.metric)[:,None]
            np.testing.assert_allclose(p,ref,atol=1e-10)
            self.assertTrue(spectrum(e.tangent(Y,f.Q))['passed'])

    def test_original_moving_hold_and_exact_newton_agree(self):
        s,e,m,h,_=load_case(1.6);a=CarrierAVFSolver(s,e,m,h,True);b=DrivenAVF(s,e,m,h);c=DrivenAVF(s,e,m,h,chord=False)
        for _ in range(12):
            a.step(.00003125);b.step(.00003125);c.step(.00003125)
            for k in ('x','Y','v','C'):
                np.testing.assert_allclose(getattr(a.state,k),getattr(b.state,k),atol=2e-9)
                np.testing.assert_allclose(getattr(c.state,k),getattr(b.state,k),atol=2e-9)

    def test_driven_work_and_kinematic_constraints(self):
        s,e,m,h,_=controlled_case();s.v[:]=0;s.C[:]=0;a=DrivenAVF(s,e,m,h,mode='driven');b=DrivenAVF(s,e,m,h,mode='driven',chord=False)
        for _ in range(30):
            r=a.step(.000125);b.step(.000125)
            self.assertLess(abs(r['delta_total_J']-r['boundary_work_J']-r['metric_change_J']-r['kinetic_force_work_defect_J']-r['potential_quadrature_error_J']),1e-18)
            self.assertLess(abs(r['kinetic_force_work_defect_J']),5e-13);self.assertLess(r['grip_velocity_error'],1e-10)
            for k in ('x','Y','v','C'):np.testing.assert_allclose(getattr(a.state,k),getattr(b.state,k),atol=1e-8)
        self.assertGreater(r['boundary_work_J'],0)
        self.assertLess(abs(r['boundary_work_J']-r['reaction_N']*r['loading_speed']*r['dt']),1e-18)
        self.assertTrue(spectrum(e.tangent(a.state.Y,a.geometry.Q))['passed'])

    def test_rigid_motion_and_failure_atomicity(self):
        v=np.array([.1,.025,-.02]);s,e,m,h=cube(velocity=v);a=DrivenAVF(s,e,m,h,clamped=False)
        for _ in range(30):a.step(.005)
        np.testing.assert_allclose(a.state.x,s.x+.15*v,atol=1e-10)
        np.testing.assert_allclose(a.state.Y,s.Y+.15*v,atol=1e-10)
        np.testing.assert_allclose(gradient(e.B,a.state.Y),np.tile(np.eye(3),(len(m),1,1)),atol=1e-10)
        s,e,m,h,_=load_case();a=DrivenAVF(s,e,m,h,mode='driven');old=a.state.clone()
        with self.assertRaises(RuntimeError):a.step(.001,max_iters=0)
        for k in ('x','Y','v','C'):np.testing.assert_array_equal(getattr(a.state,k),getattr(old,k))
        self.assertEqual(a.steps,0)

if __name__=='__main__':unittest.main()
