"""Regression checks for sub-nanosecond time identity and cache ownership."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from benchmarks import research_pressure_window_next
import numpy as np
from engine.aniso_phase1.research_pressure_window_next.coupled import WindowCoupling,OwnedGeometry,time_tolerance,residual_budgets
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianGeometry
from benchmarks.research_observable_boundary_next.pressure_correction import internal_schedule


class RuntimeTests(unittest.TestCase):
    def fixture(self):
        c=WindowCoupling.__new__(WindowCoupling);c.times=np.array([0.,1.65e-11,4.45e-11]);c.signature='owned';c.source=np.zeros(1)
        state=SimpleNamespace(step=1,time=c.times[1],child_states={'explicit_pressure_grid':'owned'})
        c.core=SimpleNamespace(validate=lambda s:None,state=state,step=lambda h,**kw:h)
        return c,state

    def test_early_wrong_time_is_rejected(self):
        c,s=self.fixture();s.time+=1e-13
        with self.assertRaises(ValueError):c.validate(s)

    def test_one_ulp_roundoff_is_accepted(self):
        c,s=self.fixture();s.time=np.nextafter(s.time,np.inf);c.validate(s)

    def test_wrong_source_grid_is_rejected(self):
        c,s=self.fixture();s.child_states['explicit_pressure_grid']='foreign'
        with self.assertRaises(ValueError):c.validate(s)

    def test_requested_dt_cannot_change_frozen_step(self):
        c,s=self.fixture();h=c.times[2]-c.times[1]
        with self.assertRaises(ValueError):c.step(h*1.01)
        self.assertEqual(c.step(h),h)

    def test_cached_return_cannot_mutate_future_value(self):
        g=OwnedGeometry.__new__(OwnedGeometry);cache={'H':np.eye(2),'volume':np.array([1.,2.]),'min_detF':1.}
        with patch.object(CartesianGeometry,'evaluate',return_value=cache):
            a=g.evaluate(np.zeros(1));a['H'][0,0]=-5;a['volume'][:]=0;a['min_detF']=-1
            b=g.evaluate(np.zeros(1));np.testing.assert_array_equal(b['H'],np.eye(2));np.testing.assert_array_equal(b['volume'],[1.,2.]);self.assertEqual(b['min_detF'],1.)

    def test_observation_nodes_survive_internal_refinement(self):
        obs=np.array([0.,1e-8,3e-8,1e-6]);times=internal_schedule(obs,3e10)
        self.assertTrue(all(x in times for x in obs));self.assertTrue(np.all(np.diff(times)>0));self.assertLessEqual(len(times)-1,64)
        self.assertLess(time_tolerance(times[1]),times[1]*1e-12)

    def test_impulse_budget_retains_force_resolution(self):
        m=SimpleNamespace(M=np.eye(1),free=np.array([0]));z=np.zeros((1,3));w=dict(W=z,v0=z,path={'force':z},pressure_force=np.ones((1,3))*.001,p1=np.array([.01]),z=np.zeros(2))
        a=residual_budgets(m,1e-11,np.array([.0002]),np.array([1.]),np.array([.01]),w,z,1e-7)
        b=residual_budgets(m,1e-9,np.array([.0002]),np.array([1.]),np.array([.01]),w,z,1e-7)
        self.assertLess(a[0],1e-14);self.assertAlmostEqual(a[0]/1e-11,b[0]/1e-9);self.assertTrue(np.isfinite(a).all())

if __name__=='__main__':unittest.main()
