import unittest
import benchmarks.research_observable_boundary_next
import numpy as np
from benchmarks.research_observable_boundary_next.events import events,difference
from benchmarks.research_observable_boundary_next.pressure_study import restrict
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology
from benchmarks.research_phase_reference_next.coupling_study import quadrature
from benchmarks.research_observable_pressure_next.coupling_study import affine_check
from benchmarks.research_observable_boundary_next import config

class RuntimeContracts(unittest.TestCase):
    def test_latest_defaults(self):
        import inspect
        p=inspect.signature(config.make).parameters
        for name in ('shared_reduction','coarse_instrumentation','reuse_transpose_buffers','vectorized_segment_metadata'):self.assertTrue(p[name].default)
    def test_boundary_grid_and_source(self):
        c=CartesianTopology([[0,.001,.02,.5,1],[0,.3,1],[0,1]])
        f=CartesianTopology([[0,.0005,.001,.01,.02,.2,.5,.8,1],[0,.15,.3,.6,1],[0,1]])
        self.assertEqual(f.cells,32);self.assertTrue(np.all(f.B[:,f.internal].sum(axis=0)==0));self.assertAlmostEqual(f.source().sum(),.0005)
        v=np.ones((2,f.cells))*np.array([[3],[7]]);np.testing.assert_allclose(restrict(v,c,f),np.ones((2,c.cells))*np.array([[3],[7]]))
        # A non-uniform child-volume mean differs from reshape/unweighted mean.
        small=CartesianTopology([[0,.1,1],[0,1],[0,1]]);large=CartesianTopology([[0,1],[0,1],[0,1]])
        self.assertAlmostEqual(restrict(np.array([[10.,0.]]),large,small)[0,0],1.)
    def test_invalid_grids(self):
        for cuts in ([[0,1,1],[0,1],[0,1]],[[0,float('nan'),1],[0,1],[0,1]],[np.linspace(0,1,34),[0,1],[0,1]]):
            with self.assertRaises(ValueError):CartesianTopology(cuts)
    def test_tensor_patch_and_negative_j(self):
        top=CartesianTopology([[0,.03,.2,.5,1],[0,.4,1],[0,1]])
        self.assertLess(affine_check(top,np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]]))['error'],1e-9)
        X,w,ids=quadrature(top)
        with self.assertRaises(ValueError):top.assemble(X,w,ids,np.broadcast_to(np.diag([-1.,1,1]),(len(X),3,3)))
    def test_resolved_pressure_start_keeps_observations(self):
        from benchmarks.research_observable_boundary_next.pressure_correction import internal_schedule
        obs=np.r_[0.,np.geomspace(6.75e-9,7.6e-6,24)];lam=3.04e10;ts=internal_schedule(obs,lam)
        self.assertLessEqual(len(ts)-1,64);self.assertTrue(np.isin(obs,ts).all());self.assertLessEqual(ts[1]*lam,.5+1e-12)
    def test_invalid_pressure_schedule(self):
        from benchmarks.research_observable_boundary_next.pressure_correction import internal_schedule
        for obs,rate in (([0,1,1],1.),([1,2],1.),([0,1],-1.),([0,1],float('nan'))):
            with self.assertRaises(ValueError):internal_schedule(obs,rate)
    def test_unobserved_is_not_pass(self):
        x=events([0,.1,.2],[1,1,1]);self.assertNotEqual(difference(x,x)['status'],'passed_scoped')
    def test_wrong_cycle_not_aliased(self):
        ts=np.linspace(0,.003,61);a=events(ts,np.sin(ts*2*np.pi/.00075));b=events(ts,np.sin(ts*2*np.pi/.00075+np.pi))
        self.assertNotEqual(difference(a,b,.000075)['status'],'passed_scoped')

if __name__=='__main__':unittest.main()
