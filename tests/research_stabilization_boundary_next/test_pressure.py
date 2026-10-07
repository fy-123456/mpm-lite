import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import unittest
import numpy as np
from benchmarks.research_stabilization_boundary_next.pressure import algebra,exact,midpoint,graded,bisect
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction

PARAMS=dict(mobility_scale=.05,storage=.0002,reservoir_Pa=.002,pressure0_Pa=.2,source_density_s_inv=.001)
CUTS=[np.linspace(0,1,17).tolist(),[0,.2],[0,.3]]

class PressureReferenceTest(unittest.TestCase):
    def test_bound_is_local_and_original_guard_remains(self):
        cuts=bisect(bisect(CUTS));self.assertEqual(ReferenceTopology(cuts).cells,64)
        with self.assertRaises(ValueError):CartesianTopology(cuts)
        with self.assertRaises(ValueError):ReferenceTopology([np.linspace(0,1,130),[0,1],[0,1]])

    def test_invalid_grid_is_rejected(self):
        for x in ([0,.5,.5,1],[0,float('nan'),1],[1,0]):
            with self.assertRaises(ValueError):ReferenceTopology([x,[0,1],[0,1]])

    def test_conservative_nested_transfer(self):
        a=ReferenceTopology(graded(CUTS,4));b=ReferenceTopology(bisect(graded(CUTS,4)));P,C,Z=restriction(a,b)
        np.testing.assert_allclose(a.B@Z,C@b.B,atol=1e-12)
        np.testing.assert_allclose(P@np.ones(b.cells),1,atol=1e-12)
        np.testing.assert_allclose(C@b.V0,a.V0,atol=1e-12)
        self.assertTrue(np.all(a.B[:,a.internal].sum(axis=0)==0))

    def test_full_tensor_capacity_and_conservation(self):
        a=algebra(CUTS,PARAMS);self.assertGreater(np.linalg.eigvalsh(a['H'])[0],0)
        times=np.array([0,1e-6,4e-6,2e-5]);v=exact(a,PARAMS,times)
        residual=a['C']*(v['pressure']-PARAMS['pressure0_Pa'])+v['cumulative']@a['top'].B.T-times[:,None]*a['top'].source(PARAMS['source_density_s_inv'])
        np.testing.assert_allclose(residual,0,atol=1e-12)
        np.testing.assert_allclose(a['groups'].sum(axis=0),a['top'].boundary_sign,atol=0)
        axes=a['top'].axes;self.assertGreater(np.max(abs(a['H'][axes[:,None]!=axes[None,:]])),0)

    def test_equilibrium_is_stationary(self):
        p=dict(PARAMS,pressure0_Pa=.002,source_density_s_inv=0);a=algebra(CUTS,p);t=np.linspace(0,.01,5)
        for v in (exact(a,p,t),midpoint(a,p,t)):
            np.testing.assert_allclose(v['pressure'],p['reservoir_Pa'],atol=1e-10)
            np.testing.assert_allclose(v['flux'],0,atol=1e-10)

    def test_midpoint_reduces_time_error_in_resolved_regime(self):
        a=algebra(CUTS,PARAMS);end=.2/a['lam'][-1];t=np.linspace(0,end,5);tt=np.linspace(0,end,9);truth=exact(a,PARAMS,[0,end])['pressure'][-1]
        e0=np.linalg.norm(midpoint(a,PARAMS,t)['pressure'][-1]-truth);e1=np.linalg.norm(midpoint(a,PARAMS,tt)['pressure'][-1]-truth)
        self.assertLess(e1,.4*e0)

if __name__=='__main__':unittest.main()
