"""Regression coverage for actual pressure grids and scoped runtime selection."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from engine.aniso_phase1.research_phase_stress_next.coupled import ExplicitGridCoupling
from engine.aniso_phase1.research_d.common_state import CommonState,StateTransaction
from benchmarks.research_phase_stress_next import run,config
from benchmarks.research_phase_stress_next.provenance import write


class ToyCore:
    def __init__(self,model,cfg,*,state=None,**kwargs):
        self.cells=2;self.identity={'physics':'toy'};self.geometry=SimpleNamespace(cache={})
        self._transaction=StateTransaction(state or CommonState(np.zeros((1,3)),np.zeros((1,3))),validator=self.validate)
    @property
    def state(self):return self._transaction.snapshot()
    def validate(self,state):pass
    def step(self,h,*,source_m3_s,external_force=None,inject=None):
        trial=self._transaction.begin_trial()
        try:
            s=trial.state;s.time+=h;s.step+=1;s.q+=h*sum(source_m3_s)
            if inject:inject('before_commit',s)
            self._transaction.commit(trial);return {'dt':h,'time':s.time,'step':s.step}
        except Exception:self._transaction.rollback(trial);raise


class GridTests(unittest.TestCase):
    def test_real_interval_reaches_step_and_resume(self):
        with patch('engine.aniso_phase1.research_phase_stress_next.coupled.MultiCellAVF',ToyCore):
            c=ExplicitGridCoupling(None,{},[0.,1e-5,3e-5],source_m3_s=[1.,2.])
            self.assertEqual(c.step()['dt'],1e-5)
            d=ExplicitGridCoupling(None,{},[0.,1e-5,3e-5],source_m3_s=[1.,2.],state=c.state)
            self.assertAlmostEqual(d.step()['dt'],2e-5)
            self.assertAlmostEqual(d.state.q[0,0],9e-5)
            with self.assertRaisesRegex(ValueError,'exhausted'):d.step()
    def test_foreign_grid_source_and_requested_dt_rejected(self):
        with patch('engine.aniso_phase1.research_phase_stress_next.coupled.MultiCellAVF',ToyCore):
            c=ExplicitGridCoupling(None,{},[0.,1e-5,2e-5],source_m3_s=[1.,0.]);before=c.state.digest()
            with self.assertRaisesRegex(ValueError,'dt differs'):c.step(.01)
            self.assertEqual(c.state.digest(),before);c.step()
            for ts,src in [([0.,1e-5,4e-5],[1.,0.]),([0.,1e-5,2e-5],[2.,0.])]:
                with self.assertRaisesRegex(ValueError,'foreign pressure'):ExplicitGridCoupling(None,{},ts,source_m3_s=src,state=c.state)
    def test_grid_identity_corruption_rolls_back(self):
        with patch('engine.aniso_phase1.research_phase_stress_next.coupled.MultiCellAVF',ToyCore):
            c=ExplicitGridCoupling(None,{},[0.,1e-5]);before=c.state.digest()
            def corrupt(where,state):state.child_states['explicit_pressure_grid']='foreign'
            with self.assertRaisesRegex(ValueError,'foreign pressure'):c.step(inject=corrupt)
            self.assertEqual(c.state.digest(),before)


class EntryTests(unittest.TestCase):
    def test_unseen_grid_full_only_preserves_existing_shared_optimization(self):
        with TemporaryDirectory() as d,patch.object(run,'verify',return_value={'energy_scale_J':1.}):
            p=Path(d);write(p/'selected-space.json',dict(package={'path':'test','sha256':'x'},mass_order=7,full_order=7))
            write(p/'S3/performance-decision.json',dict(warm_adopted=False))
            write(p/'S5/inherited-main.json',dict(qualified=True,scope={'max_dt_s':.0125,'peak_m':.005,'time_grid_s':[0.,.0125,.025]}))
            cfg=run.create_config(p,'unseen',end=.025,dt=.00625,rule_policy='q5_with_full_retry')
            self.assertEqual(cfg['post_release']['rule_policy'],'full_only')
            self.assertTrue(cfg['cost_phase']['shared_reduction']);self.assertFalse(cfg['implementation']['coarse_instrumentation'])


if __name__=='__main__':unittest.main()
