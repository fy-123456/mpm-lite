"""Contracts of the optional grouped integration and its diagnostic gates."""
import os,unittest
import numpy as np
import warp as wp
from engine.aniso_phase1.grouped_quadrature import freeze_material,freeze_maps,GroupedQuadratureImplicitSolver
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,center_support,response,particle_response
from engine.aniso_phase1 import AnisotropicMaterialParams
from benchmarks.aniso_material_snapshot import make_snapshot


class GroupedMapTests(unittest.TestCase):
    def test_frozen_rule_positive_affine_complete_and_conditional_response(self):
        from itertools import product
        s=make_snapshot(field='smooth');h=1/16;p=AnisotropicMaterialParams(10,20,200)
        F=np.array([[1.04,.02,0],[0,.98,0],[0,0,1.01]])
        s=MaterialSnapshot((s.X-.5)@F.T+.5,s.X,np.tile(F,(len(s.x),1,1)),s.A,s.volume)
        records,E=freeze_material(s,h,p);nodes=np.unique(np.concatenate([key+np.array(list(product((0,1),repeat=3))) for key,_,_ in center_support(s,h)]),axis=0)*h
        ids,B,error=freeze_maps(records,nodes,h);valid=ids>=0;xn=nodes[np.maximum(ids,0)]
        F0=np.concatenate([r[5] for r in records]);V=np.concatenate([r[2] for r in records])
        self.assertLess(error,1e-10);self.assertGreater(V.min(),0);self.assertAlmostEqual(V.sum(),s.volume.sum(),places=14)
        np.testing.assert_allclose(B.sum(axis=1),0,atol=1e-12)
        np.testing.assert_allclose(np.einsum('qni,qnj->qij',xn,B),F0,atol=1e-12)
        ep,_,_=particle_response(s,p);self.assertLess(abs(E/(s.volume@ep)-1),1e-10)
        np.testing.assert_array_equal(B[~valid],0)

    def test_reject_incompatible_history_before_solver_allocation(self):
        for options in ({'history_mode':'grid_locked'},{'force_discretization':'kirchhoff'}):
            with self.assertRaisesRegex(ValueError,'requires variational'):
                GroupedQuadratureImplicitSolver((9,)*3,AnisotropicMaterialParams(10,20,200),**options)


class GroupedSparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        cls.device=os.environ.get('ANISO_TEST_DEVICE','cpu')

    def test_actual_mixed_sparse_gradient_hessian_projection_and_objectivity(self):
        from benchmarks.aniso_grouped import operator_audit
        r=operator_audit(self.device)
        self.assertEqual(r['blocks'],8)
        self.assertLess(r['potential_fd'],2e-4);self.assertLess(r['tangent_fd'],2e-5)
        for eig in r['spectra'].values():self.assertLess(eig['symmetry'],1e-10)
        self.assertGreater(r['spectra']['projected']['min_eigenvalue'],0)
        self.assertLess(r['material_objectivity']['energy_error'],1e-10)
        self.assertLess(r['material_objectivity']['force_error'],1e-10)
        self.assertLess(r['compressed']['exact_min_eigenvalue'],0)
        self.assertEqual(r['compressed']['exact_pcg_status'],'nonpositive_curvature')
        self.assertGreater(r['compressed']['projected_min_eigenvalue'],0)
        self.assertEqual(r['compressed']['projected_pcg_status'],'converged')

    def test_short_solve_rebuild_budget_and_reaction_use_group_forces(self):
        from demos.aniso import Scene,Config
        scene=Scene(Config('tensile',9,.005,quadrature='group4x8',fiber_field='smooth',loading_time=.04,loading_speed=.025,smooth_loading=True),self.device)
        for _ in range(3):self.assertTrue(scene.step(),scene.solver.last_step_stats)
        s=scene.solver
        for row in s.energy_ledger.rows[1:]:
            self.assertLess(abs(row['budget_closure']),1e-12)
            self.assertLess(abs(row['momentum_balance_error']),1e-9)
            self.assertIn('group_rebuild_delta',row)
            self.assertAlmostEqual(row['elastic'],row['group_solved_energy'],places=13)
        self.assertLess(np.max(np.abs(s.enhancements.extra.numpy())),1e-20)
        self.assertEqual(s.last_step_stats['linear_solver'],'pcg')
        # Group map/weights must remain exactly frozen during line searches.
        old=[a.numpy().copy() for a in (s.group_F0,s.group_B,s.group_V)]
        s.evaluate_residual();s.incremental_potential()
        for a,b in zip(old,(s.group_F0,s.group_B,s.group_V)):np.testing.assert_array_equal(a,b.numpy())


if __name__=='__main__':unittest.main()
