"""Independent affine-return coefficient: real kernels, unchanged physics, rollback."""
from dataclasses import replace
import gc,unittest
from unittest.mock import patch
import numpy as np
from demos.aniso import Config,Scene
from engine.aniso_phase1.affine_transfer import finish_transfer
from benchmarks.aniso_apic_frequency import initial_field
import tests.test_aniso_affine_transfer as old


class SplitAffineTests(unittest.TestCase):
    def tearDown(self):gc.collect()
    def test_split_real_kernel_against_independent_oracle(self):
        for boundary in (0,1,2):
            ref=old.transfer_probe(beta=1.,boundary=boundary)
            for bv,bc in ((.9,1.),(1.,.9),(.3,0.),(0.,.7)):
                class SplitScene(Scene):
                    def __init__(self,cfg,device):super().__init__(replace(cfg,affine_flip_ratio=bc),device)
                with patch.object(old,'Scene',SplitScene):r=old.transfer_probe(beta=bv,boundary=boundary)
                expected=ref['L']+bc*(ref['expected_C']-ref['L'])
                np.testing.assert_allclose(r['C'],expected,rtol=0,atol=1e-13)
                for k in ('L','F','v'):np.testing.assert_allclose(r[k],r['expected_'+k],rtol=0,atol=1e-13)
                self.assertLess(r['momentum_error'],1e-14)

    def test_public_api_equals_isolated_v11_control(self):
        # The exploration only substituted beta during finish_transfer.
        # Confirm the public option reproduces this exact operation.
        for bv,bc in ((.9,1.),(.9**.125,1.),(1.,.9**.125)):
            class SplitScene(Scene):
                def __init__(self,cfg,device):super().__init__(replace(cfg,affine_flip_ratio=bc),device)
            with patch.object(old,'Scene',SplitScene):a=old.transfer_probe(beta=bv,boundary=1)
            def control(s):
                save=s.flip_ratio
                try:s.flip_ratio=bc;finish_transfer(s)
                finally:s.flip_ratio=save
            with patch.object(old,'finish_transfer',control):b=old.transfer_probe(beta=bv,boundary=1)
            for k in ('C','L','v','F'):np.testing.assert_array_equal(a[k],b[k])

    def test_same_input_solve_energy_F_and_positions_unchanged(self):
        cfg=Config('tensile',9,.001,45.,smooth_loading=True,history_consistency='residual_center',stabilization='selective_patch',boundary_impulse_transfer=True,apic_transfer='incremental',reaction_force_atol=1e-7)
        results=[]
        for bc in (None,1.):
            scene=Scene(replace(cfg,affine_flip_ratio=bc),'cpu');s=scene.solver
            v,C=initial_field(s.ptc_x.numpy(),'sine_45');s.ptc_v.assign(v);s.ptc_C.assign(C)
            self.assertTrue(scene.step(),s.last_step_stats)
            results.append(({k:getattr(s,k).numpy().copy() for k in ('ptc_x','ptc_F','ptc_v','ptc_L','ptc_C','grid_v_new')},s.energy_ledger.rows[-1]['elastic'],s.incremental_potential()))
        a,b=results
        for k in ('ptc_x','ptc_F','ptc_v','ptc_L','grid_v_new'):np.testing.assert_array_equal(a[0][k],b[0][k])
        self.assertEqual(a[1:],b[1:]);self.assertGreater(np.max(abs(a[0]['ptc_C']-b[0]['ptc_C'])),1e-7)

    def test_history_momentum_energy_and_failure_rollback(self):
        cfg=Config('tensile',9,.001,45.,smooth_loading=True,history_consistency='residual_center',stabilization='selective_patch',boundary_impulse_transfer=True,apic_transfer='incremental',affine_flip_ratio=1.,reaction_force_atol=1e-7)
        scene=Scene(cfg,'cpu');s=scene.solver
        for _ in range(12):
            self.assertTrue(scene.step(),s.last_step_stats)
            np.testing.assert_allclose(s.ptc_F.numpy(),s.local_trial.numpy(),atol=1e-13,rtol=0)
            np.testing.assert_allclose((s.mapped_W@s.ptc_F.numpy().reshape(s.n_ptc,9)).reshape(-1,3,3),s.mapped_trial.numpy(),atol=1e-13,rtol=0)
            self.assertLess(scene.loading_rows[-1]['particle_momentum_balance_error_norm'],1e-7)
            self.assertLess(abs(s.energy_ledger.rows[-1]['budget_closure']),1e-14)
        oldstate={k:getattr(s,k).numpy().copy() for k in ('ptc_x','ptc_F','ptc_C','ptc_L','ptc_v')};clock=(s.sim_time,s.sim_steps)
        self.assertFalse(s.step(max_iters=0,print_every=0))
        for k,a in oldstate.items():np.testing.assert_array_equal(getattr(s,k).numpy(),a)
        self.assertEqual(clock,(s.sim_time,s.sim_steps))

    def test_invalid_affine_control_rejected_before_state_change(self):
        for bc,mode in ((-.1,'incremental'),(1.1,'incremental'),(float('nan'),'incremental'),(1.,'overwrite')):
            with self.assertRaises(ValueError):Scene(Config('tensile',9,.001,apic_transfer=mode,boundary_impulse_transfer=True,affine_flip_ratio=bc),'cpu')
        scene=Scene(Config('tensile',9,.001,apic_transfer='incremental',boundary_impulse_transfer=True,affine_flip_ratio=1.),'cpu');s=scene.solver
        s.affine_flip_ratio=float('inf');F=s.ptc_F.numpy().copy()
        with self.assertRaises(ValueError):s.step()
        np.testing.assert_array_equal(F,s.ptc_F.numpy());self.assertEqual(s.sim_steps,0)

if __name__=='__main__':unittest.main()
