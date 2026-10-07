import gc
import unittest
import numpy as np
import warp as wp
from demos.aniso import Config, Scene
from engine.aniso_phase1.transfer_audit import TransferAuditLedger
from engine.aniso_phase1.tensile import set_grip_velocity,grip_reactions,loading_displacement

wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'


class ForceAccuracyTests(unittest.TestCase):
    def tearDown(self):gc.collect()

    def test_force_cap_overrides_loose_stopping_and_scales_with_dt(self):
        targets=[]
        for dt in (.001,.0005):
            scene=Scene(Config('tensile',9,dt,smooth_loading=True),'cpu');s=scene.solver
            set_grip_velocity(s,scene.boundary_indices,.01)
            self.assertTrue(s.step(max_iters=16,print_every=0,reaction_force_atol=1e-7,
                newton_atol=1.,newton_rtol=1.,cg_atol=1.,cg_tol=.99,v_tol=1.,max_cg_iters=500))
            stats=s.last_step_stats;targets.append(stats['newton_residual_target'])
            self.assertLessEqual(stats['reaction_force_residual_bound'],.5e-7)
            self.assertLessEqual(stats['linear_residual_target'],.1*stats['newton_residual_target']*(1+1e-12))
            self.assertLess(abs(grip_reactions(s)['momentum_balance_error']),1e-7)
            del scene,s;gc.collect()
        self.assertAlmostEqual(targets[1]/targets[0],.5,places=12)

    def test_invalid_force_tolerance_and_exhaustion_do_not_commit(self):
        scene=Scene(Config('tensile',9,.001,smooth_loading=True),'cpu');s=scene.solver
        before=[a.numpy().copy() for a in (s.ptc_x,s.ptc_v,s.ptc_F)]
        for value in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):s.step(reaction_force_atol=value)
        with self.assertRaises(ValueError):s.step(reaction_force_atol=1e-7,damping=.9)
        set_grip_velocity(s,scene.boundary_indices,.01)
        self.assertFalse(s.step(max_iters=0,print_every=0,reaction_force_atol=1e-7))
        for a,b in zip((s.ptc_x,s.ptc_v,s.ptc_F),before):np.testing.assert_array_equal(a.numpy(),b)
        self.assertEqual((s.sim_steps,s.sim_time),(0,0.))

    def test_audit_is_read_only_and_frozen_identity_closes(self):
        outputs=[]
        for audited in (False,True):
            scene=Scene(Config('tensile',9,.001,45.,smooth_loading=True,reaction_force_atol=1e-7),'cpu')
            if audited:scene.solver.energy_ledger=TransferAuditLedger()
            for _ in range(3):
                if audited:scene.solver.energy_ledger.audit_next=True
                self.assertTrue(scene.step())
            outputs.append([getattr(scene.solver,f).numpy().copy() for f in ('ptc_x','ptc_v','ptc_F')])
            if audited:
                r=scene.solver.energy_ledger.audit
                self.assertLess(r['resample_oracle_F_max'],1e-13)
                self.assertLess(r['resample_oracle_volume_max'],1e-14)
                self.assertLess(r['particle_update_max_error'],1e-13)
                self.assertLess(r['frozen_decomposition_max_error'],1e-13)
                self.assertEqual(r['matched_previous_centers'],r['centers'])
            del scene;gc.collect()
        for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)

    def test_slow_loading_preserves_endpoint_and_phase(self):
        for phase in (0.,.1,.5,1.):
            self.assertAlmostEqual(loading_displacement(.5*phase,.01,.5,1,True),
                                   loading_displacement(phase,.005,1.,1,True),places=15)
        self.assertAlmostEqual(loading_displacement(1.,.005,1.,1,True),.005)


if __name__=='__main__':unittest.main()
