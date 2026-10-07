import unittest
import numpy as np
from benchmarks.aniso_v20_common import snapshot,lift_state,gauss_sites
from benchmarks.aniso_v20_runs import setup
from engine.aniso_phase1.thin_projection import ThinProjector
from engine.aniso_phase1.carrier_driven import projection
from engine.aniso_phase1.integrated_avf import IntegratedAVF
from engine.aniso_phase1.fast_integrated_avf import FastIntegratedAVF

class ProjectionTests(unittest.TestCase):
    def test_same_weighted_rank_and_projection_without_tall_Q(self):
        rng=np.random.default_rng(11);J=rng.normal(size=(450,35));J[:,-1]=J[:,0]+2*J[:,3];q=np.exp(rng.normal(size=450));z=rng.normal(size=(450,3));a,rank=projection(J,q,z);b=ThinProjector(np.sqrt(q)[:,None]*J);self.assertEqual(rank,b.rank);np.testing.assert_allclose(b.project(np.sqrt(q)[:,None]*z)/np.sqrt(q)[:,None],a,atol=1e-13)
    def test_same_moving_condensed_loading_trajectory(self):
        s,e,m,h,_=setup('gauss3-condensed');a=IntegratedAVF(s,e,m,h,condense=True);b=FastIntegratedAVF(s,e,m,h,condense=True)
        for k in range(20):
            ra=a.step(.0005);rb=b.step(.0005);self.assertLess(abs(ra['reaction_N']-rb['reaction_N']),1e-10)
        for name in ('x','Y','v','C'):np.testing.assert_allclose(getattr(a.state,name),getattr(b.state,name),atol=1e-10)
    def test_same_real_hold_and_unload_snapshots(self):
        for t in (.85,1.4):
            s,e,m,h,meta=snapshot(t);s,e,m,h=lift_state(s,e,m,h,meta,*gauss_sites(h,3));e.carrier_reference=meta['carrier_reference'];a=IntegratedAVF(s,e,m,h);b=FastIntegratedAVF(s,e,m,h)
            for _ in range(5):
                ra=a.step(.0000625);rb=b.step(.0000625);self.assertLess(abs(ra['reaction_N']-rb['reaction_N']),1e-8)
            np.testing.assert_allclose(a.state.Y,b.state.Y,atol=1e-11)
if __name__=='__main__':unittest.main()
