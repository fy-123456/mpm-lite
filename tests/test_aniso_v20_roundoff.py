import unittest
import numpy as np
from benchmarks.aniso_v20_common import snapshot
from engine.aniso_phase1.fast_integrated_avf import FastIntegratedAVF
from engine.aniso_phase1.robust_integrated_avf import RobustIntegratedAVF

class RoundoffTests(unittest.TestCase):
    def test_real_hold_same_solution_and_unchanged_final_gates(self):
        s,e,m,h,meta=snapshot(1.4);e.carrier_reference=meta['carrier_reference'];a=FastIntegratedAVF(s,e,m,h);b=RobustIntegratedAVF(s,e,m,h)
        for _ in range(24):
            ra=a.step(.0000625);rb=b.step(.0000625);self.assertLess(abs(rb['budget_defect_J']),1e-13);self.assertLessEqual(rb['newton_residual'],1e-17);self.assertLess(rb['history_commit_max'],1e-12);self.assertLess(abs(ra['reaction_N']-rb['reaction_N']),1e-8)
        np.testing.assert_allclose(a.state.Y,b.state.Y,atol=1e-11)
    def test_failed_attempt_does_not_commit_physical_state(self):
        s,e,m,h,meta=snapshot(.85);e.carrier_reference=meta['carrier_reference'];so=RobustIntegratedAVF(s,e,m,h);old=so.state.clone()
        with self.assertRaises(RuntimeError):so.step(.0005,max_iters=0)
        for name in ('x','Y','v','C'):np.testing.assert_array_equal(getattr(so.state,name),getattr(old,name))
        self.assertEqual(so.state.time,old.time)
if __name__=='__main__':unittest.main()
