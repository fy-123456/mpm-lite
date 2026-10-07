"""Checks for the new package, sufficient-rule binding and immutable CLI."""
import copy,contextlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from benchmarks.research_spatial_phase_next import config,run,publication
from benchmarks.research_spatial_phase_next.provenance import write,sha
from engine.aniso_phase1.research_spatial_phase_next.performance import load as load_cache
from tests.research_post_release.test_runtime import RuntimeTests


class EntryTests(unittest.TestCase):
    def test_mass_and_full_rule_types(self):
        self.assertEqual(config.make(1.,mass_order=7)['mass_order'],7)
        for bad in (True,0,2.5):
            with self.assertRaises(ValueError):config.make(1.,mass_order=bad)
            with self.assertRaises(ValueError):config.make(1.,full_order=bad)

    def test_loaded_mass_must_match_declared_order(self):
        cfg=config.make(1.,mass_order=5)
        with tempfile.TemporaryDirectory() as d,patch.object(run,'verify',return_value={}),patch('benchmarks.research_spatial_phase_next.spaces.load_selected',return_value=(None,{'mass_order':7,'full_order':7})):
            with self.assertRaisesRegex(ValueError,'declared mass'):run.load_model(d,cfg)

    def test_full_retry_must_match_material_certificate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'qualification.json';write(p,dict(full_model={'wrong':'rule'}))
            cfg=config.make(1.,rule_policy='q5_with_full_retry',qualification=dict(path=str(p),sha256=sha(p)))
            with patch.object(run,'load_model',return_value=(SimpleNamespace(identity={'actual':'model'}),{})):
                with self.assertRaisesRegex(ValueError,'full retry model'):run.make_stepper(d,cfg,None,None)

    def test_unqualified_sensitive_scope_uses_sufficient_rule(self):
        with tempfile.TemporaryDirectory() as d,patch.object(run,'verify',return_value={'energy_scale_J':1.}):
            write(Path(d)/'selected-space.json',dict(package=None,mass_order=7,full_order=7))
            write(Path(d)/'N3/qualification-peak0075.json',dict(qualified=False,scope=dict(max_dt_s=.0125)))
            cfg=run.create_config(d,'sensitive',peak_m=.0075,end=.05,rule_policy='q5_with_full_retry')
            self.assertEqual(cfg['post_release']['rule_policy'],'full_only');self.assertEqual(cfg['material_order'],7)

    def test_corrupt_or_foreign_cache_rejected_before_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cache.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest'):load_cache(dict(path=str(p),sha256='bad'),dict(sha256='x'))
            write(p,dict(schema='qualified-space-array-cache-v1',source_package_sha256='foreign'))
            with self.assertRaisesRegex(ValueError,'another space'):load_cache(dict(path=str(p),sha256=sha(p)),dict(sha256='x'))

    def test_sealed_entry_and_unsealed_fork(self):
        from benchmarks.research_spatial_phase_next.coupling_study import cycle
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'release.json').write_text('{}')
            for f in (lambda:run.create_config(d,'x'),lambda:run.run_case(d,'x'),lambda:cycle(d)):
                with self.assertRaisesRegex(ValueError,'sealed'):f()
            self.assertFalse((Path(d)/'cases').exists())
        with tempfile.TemporaryDirectory() as d,patch.object(publication,'freeze') as freeze:
            with self.assertRaises((ValueError,FileNotFoundError)):publication.fork(d)
            freeze.assert_not_called()

    def test_stop_after_cli_dispatch(self):
        result=dict(status='in_progress',case='x',steps=2,end_s=.025,min_detF=1.,this_segment={})
        with patch('sys.argv',['run','cycle','--run','/tmp/x','--case','x','--stop-after','2']),patch.object(run,'serial_lock',return_value=contextlib.nullcontext()),patch.object(run,'run_case',return_value=result) as call:
            run.main()
        call.assert_called_once_with(Path('/tmp/x'),'x',stop_after=2)

if __name__=='__main__':unittest.main()
