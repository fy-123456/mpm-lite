"""Short qualitative and numerical regressions for transported center history."""
import gc,os,unittest
import numpy as np
import warp as wp
from engine.aniso_phase1.controlled import KinematicScene
from engine.aniso_phase1.direction_moments import audit_direction_mixture
from engine.aniso_phase1.beam_reference import beam_audit
from engine.aniso_phase1.tensile import loading_displacement


class HistoryReferenceTests(unittest.TestCase):
    def setUp(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        self.device=os.environ.get('ANISO_TEST_DEVICE','cpu')

    def test_prestretched_body_crosses_sparse_blocks_without_reset(self):
        for policy in ('grid_locked','particle_resample'):
            scene=KinematicScene('translate',65,.05,.1,policy,self.device)
            scene.step();scene.step();r=scene.rows[-1]
            if policy=='grid_locked':self.assertLess(r['elastic_relative_error'],-.5)
            else:
                self.assertLess(r['F_relative_error'],1e-5)
                self.assertLess(abs(r['elastic_relative_error']),1e-5)
                self.assertGreater(scene.solver.energy_ledger.rows[0]['elastic'],0.)
                for row in scene.solver.energy_ledger.rows[1:]:self.assertLess(abs(row['budget_closure']),1e-9)
            del scene;gc.collect()

    def test_rotation_and_nonuniform_direction_transport(self):
        for mode in ('rotate','directions'):
            scene=KinematicScene(mode,33,.05,.1,device=self.device)
            scene.step();scene.step();r=scene.rows[-1]
            self.assertLess(r['F_relative_error'],1e-5)
            self.assertLess(r['A_rms_error'],1e-5)
            self.assertGreater(r['min_det_F'],.5)
            self.assertLess(r['particle_position_max_error'],1e-5)
            if mode=='rotate':self.assertLess(abs(r['integration_energy_relative']),1e-5)
            del scene;gc.collect()

    def test_same_law_beam_reference_exposes_static_modes(self):
        r=beam_audit(17,device=self.device)
        self.assertGreater(r['center_soft_modes'],0)
        self.assertEqual(r['full_soft_modes'],0)
        self.assertGreater(r['center_load_in_soft_space'],.01)
        self.assertLess(r['production_stiffness_relative_error'],1e-5)
        self.assertLess(r['full_tip_displacement'],0.)

    def test_direction_variance_and_fourth_moment_response(self):
        r=audit_direction_mixture(np.diag([1.1,1.,1.]),np.array([[1,0,0],[0,1,0]]))
        self.assertAlmostEqual(r['energy_mean_A_relative_error'],.5,places=5)
        self.assertAlmostEqual(r['missing_energy'],r['predicted_missing_energy'],places=7)
        for key in ('energy','stress','tangent'):self.assertLess(r[key+'_fourth_moment_relative_error'],1e-6)
        r=audit_direction_mixture(np.diag(np.sqrt([1.1,.9,1.])),np.array([[1,0,0],[0,1,0]]))
        self.assertAlmostEqual(r['energy_mean_A_relative_error'],1.,places=5)

    def test_smooth_reload_returns_to_same_command(self):
        T=.2
        for t in (.03,.1,.19):
            self.assertAlmostEqual(loading_displacement(t,.01,T,2,True),loading_displacement(t+2*T,.01,T,2,True))
        self.assertEqual(loading_displacement(4*T,.01,T,2,True),0.)


if __name__=='__main__':unittest.main()
