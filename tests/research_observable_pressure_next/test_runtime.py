"""Observable scope, independent full-tensor conservation and current entry guards."""
import unittest
import numpy as np
from engine.aniso_phase1.research_observable_pressure_next.rt0 import BoundedTopology,BoundedAVF
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import ScaledTensorCellAVF
from benchmarks.research_observable_pressure_next.coupling_study import affine_check
from benchmarks.research_observable_pressure_next.time_study import grid,WINDOWS,H
from benchmarks.research_observable_pressure_next.base_config import time_grid
from benchmarks.research_observable_pressure_next import run
from benchmarks.research_observable_pressure_next.provenance import write,sha
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

class CurrentTests(unittest.TestCase):
    def test_four_cell_manufactured_finite_shear(self):
        t=BoundedTopology([[0,2],[-.1,.1],[-.05,.05]],4);r=affine_check(t,np.array([[1.03,.08,.01],[.01,.98,.03],[.02,0.,1.01]]))
        self.assertLess(r['error'],1e-9);self.assertEqual(t.nflux,21);self.assertEqual(len(t.internal),3)
        np.testing.assert_array_equal(t.B[:,t.internal].sum(axis=0),0)
        self.assertAlmostEqual(sum(t.V0),.04)
    def test_no_new_mixed_residual(self):
        self.assertIs(BoundedAVF._compute,ScaledTensorCellAVF._compute)
    def test_unsupported_topology_rejected(self):
        for n in (True,2.,3,8):
            with self.assertRaises(ValueError):BoundedTopology([[0,1],[0,1],[0,1]],n)
    def test_four_cell_start_requires_complete_grid_gate(self):
        from benchmarks.research_observable_pressure_next import coupling_study as study
        def fake_read(path):
            return {'four_cell_fixed':False} if str(path).endswith('pressure-scope-decision.json') else {'status':'passed_scoped'}
        with patch.object(study,'read',side_effect=fake_read),patch.object(study,'coupled') as build:
            with self.assertRaisesRegex(ValueError,'complete fixed-grid'):study.start(Path('/unused'),4)
            build.assert_not_called()
    def test_fast_tensor_preserves_finite_shear_and_mixed_residual(self):
        from engine.aniso_phase1.research_observable_pressure_next.fast_rt0 import FastTopology,FastAVF
        from benchmarks.research_observable_pressure_next.coupling_study import quadrature
        a=BoundedTopology([[0,2],[0,1],[0,1]],4);b=FastTopology(a.bounds,4);X,w,c=quadrature(a);F=np.broadcast_to(np.array([[1.02,.08,0.],[.01,.99,.03],[0.,.04,1.01]]),(len(w),3,3))
        np.testing.assert_allclose(a.assemble(X,w,c,F)[0],b.assemble(X,w,c,F)[0],rtol=1e-10,atol=1e-10)
        self.assertIs(FastAVF._compute,ScaledTensorCellAVF._compute)
    def test_schedule_preserves_load_boundaries_and_budget(self):
        times=sorted(set(time_grid(.0125)+[t for a,b in WINDOWS for t in grid(a,b,H)]))
        self.assertEqual(len(times)-1,252);self.assertTrue(np.all(np.diff(times)>0))
        for t in (0.,.5,.6,1.075,1.1,1.225,1.25,1.6):self.assertIn(t,times)
    def test_current_entry_rejects_stale_grid_permission(self):
        with TemporaryDirectory() as d,patch.object(run,'verify',return_value={'energy_scale_J':1.}),patch.object(run,'source_files',return_value={'code':'new'}):
            p=Path(d);write(p/'selected-space.json',dict(package={'path':'x','sha256':'space'},mass_order=7,full_order=7))
            cert=dict(schema='basis-allocation-q5-qualification-v1',qualified=True,numerical_source_sha256={'code':'new'},scope=dict(physical_space_sha256='space',mass_order=7,full_order=7,peak_m=.005,fiber_angle_degrees=45.,time_grid_s=[0.,.0125,.025],rest_start=True,initial_states=[]))
            write(p/'cert.json',cert)
            cfg=run.create_config(p,'test',start=0.,end=.025,times=[0.,.00625,.0125,.025],qualification_path=p/'cert.json',rule_policy='q5_with_full_retry')
            self.assertEqual(cfg['post_release']['rule_policy'],'full_only')
if __name__=='__main__':unittest.main()
