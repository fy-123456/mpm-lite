"""Risk-focused checks for explicit space, sharing and pressure time positivity."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
from benchmarks.research_cost_phase_next import config
from engine.aniso_phase1.research_cost_phase_next.shared import SharedReduction
from tests.research_post_release.test_runtime import RuntimeTests

class EntryTests(unittest.TestCase):
    def test_missing_space_is_not_implicit_old_model(self):
        with self.assertRaisesRegex(ValueError,'explicit qualified'):config.make(1.)
        cfg=config.make(1.,space={'path':'test','sha256':'x'})
        self.assertEqual(cfg['mass_order'],7)

    def test_shared_arrays_readonly_and_identity_scoped(self):
        r=SimpleNamespace(signature='r',M=np.eye(2),parent=SimpleNamespace(raw=np.eye(2)))
        s=SharedReduction({'sha256':'one'},r,{'mass_order':7},'cuda:0')
        self.assertIs(s.acquire({'sha256':'one'},'cuda:0')[0],r)
        with self.assertRaises(ValueError):r.M[0,0]=2.
        for entry,device in [({'sha256':'two'},'cuda:0'),({'sha256':'one'},'cuda:1')]:
            with self.assertRaisesRegex(ValueError,'foreign shared'):s.acquire(entry,device)

    def test_unseen_time_path_uses_full_rule(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from benchmarks.research_cost_phase_next import run
        from benchmarks.research_cost_phase_next.provenance import write
        with TemporaryDirectory() as d,patch.object(run,'verify',return_value={'energy_scale_J':1.}):
            p=Path(d)
            write(p/'selected-space.json',dict(package={'path':'test','sha256':'x'},mass_order=7,full_order=7))
            write(p/'P5/qualification-final.json',dict(qualified=True,scope={'max_dt_s':.0125,'peak_m':.005,'time_grid_s':[0.,.0125,.025]}))
            cfg=run.create_config(p,'new-path',end=.025,dt=.00625,rule_policy='q5_with_full_retry')
            self.assertEqual(cfg['post_release']['rule_policy'],'full_only')

    def test_monotone_spatial_operator_does_not_allow_arbitrary_midpoint_dt(self):
        B=np.array([[1.,0.],[-1.,1.]])
        L=B@np.diag([1/60.,1/30.])@B.T
        C=.0046875*np.eye(2)
        G=np.linalg.solve(C+.005*L,C-.005*L)
        self.assertTrue(np.all(G>=0))
        small=C*.001
        bad=np.linalg.solve(small+.005*L,small-.005*L)
        self.assertLess(np.min(bad),0)

if __name__=='__main__':unittest.main()
