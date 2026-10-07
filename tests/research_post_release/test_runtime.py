"""Risk tests for explicit schedules, real CLI dispatch and atomic rule state."""
import contextlib
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from tests.research_sequential_next.test_rules import RuleToy
from benchmarks.research_sequential_next.config import protocol
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_post_release.config import make,validate
from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError,advance_publish
from engine.aniso_phase1.research_sequential_next.integrator import StepRejected


class ScheduleTests(unittest.TestCase):
    def test_breaks_and_monotonicity(self):
        good=make(1.,times=[0.,.25,.5,.6,1.1,1.6])
        for times in ([0.,.5,.5,1.6],[0.,.6,.5,1.6],[0.,.5,1.1,1.6],[0.,float('nan'),1.6]):
            bad=copy.deepcopy(good);bad['times']=times
            with self.assertRaises(ValueError):validate(bad)
    def test_declared_backend_cannot_be_silently_ignored(self):
        good=make(1.)
        for key,value in [('operator','original'),('linearization_cache',True),('preconditioner','reuse_static')]:
            bad=copy.deepcopy(good);bad['implementation'][key]=value
            with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(good);bad['scenario']={'peak_m':.0075}
        with self.assertRaises(ValueError):validate(bad)
    def test_sealed_case_cannot_be_configured(self):
        from benchmarks.research_post_release.run import create_config
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'release.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'sealed release'):create_config(d,'x')
            self.assertFalse((Path(d)/'cases').exists())
    def test_sealed_case_cannot_be_advanced(self):
        from benchmarks.research_post_release.run import run_case
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'release.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'sealed release'):run_case(d,'x')
            self.assertFalse((Path(d)/'cases').exists())
    def test_stop_after_reaches_real_dispatch(self):
        from benchmarks.research_post_release import run
        result=dict(status='in_progress',case='x',steps=2,end_s=.025,min_detF=1.,this_segment={})
        with patch('sys.argv',['run','cycle','--run','/tmp/x','--case','x','--stop-after','2']),patch.object(run,'serial_lock',return_value=contextlib.nullcontext()),patch.object(run,'run_case',return_value=result) as call:
            run.main()
        call.assert_called_once_with(Path('/tmp/x'),'x',stop_after=2)


class RuntimeTests(unittest.TestCase):
    def setup_controller(self):
        a,b=RuleToy(5),RuleToy(7)
        return RuntimeRetry(a,b,protocol(1.,device='cpu'))
    @staticmethod
    def reject_first(attempt,where,state):
        if attempt==0 and where=='after_prepare':raise RetryableRuleError('registered rejection')
    def test_rule_retry_and_owned_history_survive_serialization(self):
        c=self.setup_controller();row=c.step(.025,inject=self.reject_first)
        self.assertEqual(row['material_attempts'],2)
        self.assertEqual(c.state.child_states['identity'],c.full.identity)
        with tempfile.TemporaryDirectory() as d:
            store=GenerationStore(d,c.identity);store.save(c.compressed.rest(),[]);store.save(c.state,[row])
            loaded=store.load(validator=c.validate)['state']
        resumed=RuntimeRetry(c.compressed,c.full,c.config,loaded)
        self.assertEqual(len(resumed.failures),1)
        self.assertEqual(resumed.step(.025)['material_attempts'],1)
        self.assertEqual(len(resumed.state.child_states['material_failure_history']),1)
    def test_programming_error_is_not_a_rule_retry(self):
        c=self.setup_controller();before=c.state.digest()
        def fault(attempt,where,state):
            if where=='after_prepare':raise KeyError('programming error')
        with self.assertRaises(StepRejected):c.step(.025,inject=fault)
        self.assertEqual(c.state.digest(),before);self.assertEqual(len(c.failures),1)
        self.assertFalse(c.failures[0]['retryable'])
    def test_double_failure_discards_children_and_predictor(self):
        c=self.setup_controller();before=c.state.digest()
        def fault(attempt,where,state):
            if where=='after_prepare':raise RetryableRuleError('both rules reject')
        def child(state,values):values['pressure']=[4.,5.];return values
        with self.assertRaises(StepRejected):c.step(.025,inject=fault,prepare_children=child)
        self.assertEqual(c.state.digest(),before);self.assertEqual(len(c.failures),2)
    def test_rule_migration_cannot_hide_energy_jump(self):
        c=self.setup_controller();old=c.full.evaluate
        def shifted(*args,**kwargs):
            r=old(*args,**kwargs);r=dict(r);r['U']+=1.;return r
        c.full.evaluate=shifted;before=c.state.digest()
        with self.assertRaises(StepRejected):c.step(.025,inject=self.reject_first)
        self.assertEqual(c.state.digest(),before)
    def test_prepublication_failure_restores_memory_and_disk(self):
        c=self.setup_controller();base=c.state.digest()
        with tempfile.TemporaryDirectory() as d:
            store=GenerationStore(d,c.identity);store.save(c.state,[])
            def fail(where):
                if where=='before_pointer':raise OSError('write failure')
            with self.assertRaises(OSError):advance_publish(c,store,[],.025,inject_store=fail)
            self.assertEqual(c.state.digest(),base);self.assertEqual(store.load()['state'].digest(),base)
            self.assertEqual(len(store.history()),1)
    def test_after_pointer_exception_does_not_duplicate_commit(self):
        c=self.setup_controller()
        with tempfile.TemporaryDirectory() as d:
            store=GenerationStore(d,c.identity);store.save(c.state,[])
            def fail(where):
                if where=='after_pointer':raise OSError('observation failed after commit')
            row=advance_publish(c,store,[],.025,inject_store=fail)
            self.assertEqual(row['step'],1);self.assertEqual(len(store.history()),2)
            advance_publish(c,store,[row],.025)
            self.assertEqual(len(store.history()),3)


if __name__=='__main__':unittest.main()
