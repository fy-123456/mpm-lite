"""Physical topology, model scope and exact-grid permission regressions."""
import unittest
import numpy as np
from engine.aniso_phase1.research_cross_direction_next.rt0 import BoundedTopology
from engine.aniso_phase1.research_observable_pressure_next.rt0 import BoundedTopology as OldTopology
from benchmarks.research_observable_pressure_next.coupling_study import affine_check
from benchmarks.research_cross_direction_next import config
from engine.aniso_phase1.research_basis_allocation_next.rules import permission

class TopologyChecks(unittest.TestCase):
    bounds=[[.25,.75],[-.125,.125],[-.1875,.1875]]
    def test_eight_cell_conservation(self):
        t=BoundedTopology(self.bounds,8)
        self.assertEqual(t.B.shape,(8,41));self.assertEqual(len(t.internal),7)
        np.testing.assert_array_equal(t.B[:,t.internal].sum(axis=0),0.)
        self.assertAlmostEqual(sum(t.V0),.046875)
    def test_eight_cell_affine_shear_and_work(self):
        for F in (np.eye(3),np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]])):
            self.assertLess(affine_check(BoundedTopology(self.bounds,8),F)['error'],1e-9)
    def test_old_topology_unchanged(self):
        for c in (2,4):
            a,b=BoundedTopology(self.bounds,c),OldTopology(self.bounds,c)
            for k in ('B','V0','faces','axes','areas','centres'):np.testing.assert_array_equal(getattr(a,k),getattr(b,k))
    def test_invalid_grid_rejected(self):
        for c in (True,3,8.0):
            with self.assertRaises(ValueError):BoundedTopology(self.bounds,c)
    def test_inverted_geometry_rejected(self):
        from benchmarks.research_phase_reference_next.coupling_study import quadrature
        t=BoundedTopology(self.bounds,8);X,w,ids=quadrature(t);F=np.broadcast_to(np.diag([-1.,1,1]),(len(X),3,3))
        with self.assertRaises(ValueError):t.assemble(X,w,ids,F)
    def test_source_same_physical_half(self):
        for c in (2,4,8):
            t=BoundedTopology(self.bounds,c);self.assertAlmostEqual(sum(t.V0[:c//2]*.001),.046875*.001/2)

class ScopeChecks(unittest.TestCase):
    def test_coupling_requires_pressure_content_and_outflow_gate(self):
        from unittest.mock import patch
        from pathlib import Path
        from benchmarks.research_cross_direction_next import pressure_runtime as runtime
        with patch.object(runtime,'read',return_value={'eligible_coupled':False}),patch.object(runtime,'baseline_model') as load:
            with self.assertRaisesRegex(ValueError,'complete fixed'):runtime.build(Path('/unused'),8)
            load.assert_not_called()
    def test_mixed_equations_still_use_general_solver(self):
        from engine.aniso_phase1.research_cross_direction_next.rt0 import BoundedAVF
        from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import ScaledTensorCellAVF
        self.assertIs(BoundedAVF._compute,ScaledTensorCellAVF._compute)
    def test_new_entry_downgrades_stale_grid_license(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from benchmarks.research_cross_direction_next import run
        from benchmarks.research_cross_direction_next.provenance import write
        with TemporaryDirectory() as d,patch.object(run,'verify',return_value={'energy_scale_J':1.}),patch.object(run,'source_files',return_value={'code':'new'}):
            p=Path(d);write(p/'selected-space.json',dict(package={'path':'x','sha256':'space'},mass_order=7,full_order=7))
            cert=dict(schema='basis-allocation-q5-qualification-v1',qualified=True,numerical_source_sha256={'code':'new'},scope=dict(physical_space_sha256='space',mass_order=7,full_order=7,peak_m=.005,fiber_angle_degrees=45.,time_grid_s=[0.,.0125,.025],rest_start=True,initial_states=[]))
            write(p/'cert.json',cert);cfg=run.create_config(p,'test',end=.025,times=[0.,.00625,.0125,.025],qualification_path=p/'cert.json',rule_policy='q5_with_full_retry')
            self.assertEqual(cfg['post_release']['rule_policy'],'full_only')

class DeterminantChecks(unittest.TestCase):
    def test_full_tensor_variable_deformation_matches_general_assembly(self):
        from engine.aniso_phase1.research_cross_direction_next.det_rt0 import FastTopology
        from engine.aniso_phase1.research_observable_pressure_next.rt0 import BoundedTopology as Slow
        from benchmarks.research_phase_reference_next.coupling_study import quadrature
        t=FastTopology([[0,2],[-.1,.1],[-.05,.05]],8);X,w,cells=quadrature(t)
        F=np.broadcast_to(np.eye(3),(len(X),3,3)).copy();F[:,0,1]=.12*X[:,0];F[:,1,2]=.04;F[:,2,0]=.02;F[:,0,0]+=.01*X[:,0]
        actual,J=t.assemble(X,w,cells,F);expected,_=Slow.assemble(t,X,w,cells,F)
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(actual)[0],0.);self.assertGreater(J,.1)
