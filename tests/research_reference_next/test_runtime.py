"""Focused public-entry, scope and immutable publication regressions."""
import copy
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from benchmarks.research_reference_next.config import make,validate
from benchmarks.research_reference_next import run
from benchmarks.research_reference_next.provenance import write
# Retain the six inherited transaction failure cases used by this controller.
from tests.research_post_release.test_runtime import RuntimeTests


class EntryTests(unittest.TestCase):
    def test_supported_peak_and_rejected_silent_declarations(self):
        for peak in (.005,.0075):self.assertEqual(make(1.,peak_m=peak)['scenario']['peak_m'],peak)
        for peak in (.006,float('nan')):
            with self.assertRaises(ValueError):make(1.,peak_m=peak)
        bad=make(1.);bad['scenario']['fiber_angle_degrees']=30
        with self.assertRaises(ValueError):validate(bad)
        for key,value in [('operator','original'),('linearization_cache',True),('preconditioner','reuse_static'),('force_only_responses','yes')]:
            bad=make(1.);bad['implementation'][key]=value
            with self.assertRaises(ValueError):validate(bad)
    def test_explicit_schedule_cannot_skip_breaks_or_repeat_time(self):
        good=make(1.,times=[0.,.5,.6,1.1,1.6])
        for times in ([0.,.5,.5,1.6],[0.,.6,.5,1.6],[0.,.5,1.1,1.6],[0.,float('nan'),1.6]):
            bad=copy.deepcopy(good);bad['times']=times
            with self.assertRaises(ValueError):validate(bad)
    def test_sealed_cases_cannot_write(self):
        from benchmarks.research_reference_next.coupling_study import cycle
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'release.json').write_text('{}')
            for call in (lambda:run.create_config(d,'x'),lambda:run.run_case(d,'x'),lambda:cycle(d,'x')):
                with self.assertRaisesRegex(ValueError,'sealed'):call()
            self.assertFalse((Path(d)/'cases').exists())
    def test_actual_cli_stop_after_dispatch(self):
        result=dict(status='in_progress',case='x',steps=2,end_s=.025,min_detF=1.,this_segment={})
        with patch('sys.argv',['run','cycle','--run','/tmp/x','--case','x','--stop-after','2']),patch.object(run,'serial_lock',return_value=contextlib.nullcontext()),patch.object(run,'run_case',return_value=result) as call:
            run.main()
        call.assert_called_once_with(Path('/tmp/x'),'x',stop_after=2)
    def test_qualification_scope_falls_back_and_sensitive_requires_certificate(self):
        with tempfile.TemporaryDirectory() as d,patch.object(run,'verify',return_value={'energy_scale_J':1.}):
            write(Path(d)/'Q4/qualification.json',dict(qualified=True,scope=dict(max_dt_s=.0125)))
            cfg=run.create_config(d,'outside',dt=.025,end=.05,rule_policy='q5_with_full_retry')
            self.assertEqual(cfg['material_order'],7);self.assertEqual(cfg['post_release']['rule_policy'],'full_only')
            with self.assertRaisesRegex(ValueError,'qualification'):run.create_config(d,'sensitive',end=.05,peak_m=.0075,rule_policy='q5_with_full_retry')
    def test_fork_rejects_unsealed_input_before_creating_output(self):
        from benchmarks.research_reference_next import publication
        with tempfile.TemporaryDirectory() as d,patch.object(publication,'freeze') as freeze:
            with self.assertRaises((FileNotFoundError,ValueError)):publication.fork(d)
            freeze.assert_not_called()


if __name__=='__main__':unittest.main()
