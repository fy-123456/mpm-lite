import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v20_common import snapshot,lift_state,gauss_sites
from benchmarks.aniso_v17_modes import controlled_case
from engine.aniso_phase1.integrated_avf import IntegratedAVF,material_null_condensation
from engine.aniso_phase1.separate_kinetic import KineticGeometry
from engine.aniso_phase1.carrier_driven import FastGeometry
from engine.aniso_phase1.endpoint_boundary import EndpointAVF
from benchmarks.aniso_carrier_joint import spectrum

class IntegratedTests(unittest.TestCase):
    def test_original_maps_exactly_reproduced(self):
        s,e,m,h,_=snapshot(1.4);a=KineticGeometry(s,e,m,h);b=FastGeometry(s,e,m,h)
        for name in ('N','E','Q','T','J','metric'):np.testing.assert_allclose(getattr(a,name),getattr(b,name),atol=1e-12)
    def test_lift_retains_old_values_and_original_elastic_history(self):
        s,e,m,h,meta=snapshot(1.4);a,ee,mm,_=lift_state(s,e,m,h,meta,meta['particle_reference'],m)
        for k in ('x','Y','v','C'):np.testing.assert_allclose(getattr(a,k),getattr(s,k),atol=1e-14)
        q,eq,mq,_=lift_state(s,e,m,h,meta,*gauss_sites(h,3));np.testing.assert_array_equal(eq.B,e.B)
        for Y in (s.Y,s.Y+1e-5*np.random.default_rng(12).normal(size=s.Y.shape)):
            self.assertEqual(e.evaluate(Y)['U'],eq.evaluate(Y)['U']);np.testing.assert_array_equal(e.tangent(Y),eq.tangent(Y))
    def test_condensation_is_original_energy_minimum_and_static_schur(self):
        s,e,m,h,_=controlled_case();H,R,Z,C=material_null_condensation(e,s.Y,h);self.assertEqual(Z.shape[1],3)
        Y=s.Y+1e-4*np.random.default_rng(12).normal(size=s.Y.shape);Yr=R@Y
        for b in e.B:np.testing.assert_allclose(b@Y,b@Yr,atol=1e-12)
        self.assertLessEqual(e.evaluate(Yr)['U'],e.evaluate(Y)['U']);np.testing.assert_allclose(C@Yr,0,atol=1e-12)
        # R is the exact stationary solution along Z, not a stiffness coefficient change.
        for k in range(3):
            d=Z[:,k,None]*np.array([.01,-.02,.03]);self.assertGreater(e.evaluate(Yr+d)['U'],e.evaluate(Yr)['U'])
        e.carrier_reference=s.Y.copy();so=IntegratedAVF(s,e,m,h,condense=True);self.assertTrue(spectrum(e.tangent(s.Y,so.geometry.Q))['passed'])
        # Degree-two fields, hence rigid and normal bending fields, are untouched.
        x=s.Y;pol=np.column_stack([np.ones(len(x)),x,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]])
        np.testing.assert_allclose(R@pol,pol,atol=1e-12)
    def test_unsupported_old_history_rejected(self):
        s,e,m,h,meta=snapshot(1.4);e.carrier_reference=meta['carrier_reference']
        with self.assertRaisesRegex(ValueError,'history'):IntegratedAVF(s,e,m,h,condense=True)
    def test_motion_work_history_and_static_gate(self):
        s,e,m,h,meta=controlled_case();s.v[:]=0;s.C[:]=0;s,e,m,h=lift_state(s,e,m,h,meta,*gauss_sites(h,3));e.carrier_reference=meta['carrier_reference'];so=IntegratedAVF(s,e,m,h,condense=True)
        for _ in range(12):
            row=so.step(.001);self.assertLess(abs(row['budget_defect_J']),1e-13);self.assertLess(row['history_commit_max'],1e-12);self.assertLess(row['material_null_constraint'],1e-10)
        self.assertTrue(spectrum(e.tangent(so.state.Y,so.endpoint_geometry.Q))['passed'])
    def test_baseline_trajectory_bridge(self):
        s,e,m,h,_=controlled_case();s.v[:]=0;s.C[:]=0;a=IntegratedAVF(s,e,m,h);b=EndpointAVF(s,e,m,h,mode='driven')
        for _ in range(20):ra=a.step(.001);rb=b.step(.001)
        np.testing.assert_allclose(a.state.Y,b.state.Y,atol=1e-11);self.assertLess(abs(ra['reaction_N']-rb['reaction_N']),1e-9)
if __name__=='__main__':unittest.main()
