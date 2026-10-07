"""Focused regressions for prescribed rotation and true particle diagnostics."""
import gc,os,unittest
import numpy as np
import warp as wp
from demos.aniso import Config,Scene
from engine.aniso_phase1.rotation_probe import BentRotation,reference_frame_audit
from engine.aniso_phase1.diagnostics import energy_density,ParticleEnergyLedger


class RotationDissipationTests(unittest.TestCase):
    def setUp(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        self.device=os.environ.get('ANISO_TEST_DEVICE','cpu')

    def tearDown(self):gc.collect()

    def test_corotated_potential_and_full_force_rotate(self):
        rows=reference_frame_audit()
        for r in rows:
            if r['corotated']:
                self.assertLess(abs(r['relative_energy_change']),1e-7)
                self.assertLess(r['force_covariance_error'],1e-5)
        old=[r for r in rows if r['stabilization']=='supplemental' and not r['corotated'] and r['angle']==60][0]
        self.assertGreater(abs(old['relative_energy_change']),.1)

    def test_exact_rotation_preserves_nonuniform_history(self):
        scene=BentRotation(grid=17,dt=.01,duration=.02,mode='supplemental',device=self.device)
        for _ in range(2):
            r=scene.step()
            self.assertLess(r['position_max_error'],1e-7)
            self.assertLess(r['particle_F_max_error'],1e-7)
            self.assertLess(r['stress_rotation_relative_error'],1e-6)
            self.assertLess(r['fiber_rotation_max_error'],1e-7)
            self.assertLess(abs(r['particle_energy_relative_change']),1e-6)

    def test_particle_energy_and_reaction_use_particle_quadrature(self):
        scene=Scene(Config('tensile',9,.005,quadrature='particle',fiber_field='crossed',stabilization='supplemental',
            smooth_loading=True,loading_time=.08,loading_speed=.025),self.device)
        s=scene.solver
        self.assertIsInstance(s.energy_ledger,ParticleEnergyLedger)
        for _ in range(3):self.assertTrue(scene.step())
        row=s.energy_ledger.rows[-1]
        direct=float(np.dot(s.ptc_vol0.numpy(),energy_density(s.ptc_F.numpy(),s.ptc_A0.numpy(),s.aniso_params)))
        self.assertAlmostEqual(row['elastic']-row['stabilization_energy'],direct,places=12)
        self.assertLess(abs(row['budget_closure']),1e-10)
        self.assertLess(scene.loading_rows[-1]['free_force_residual_norm'],1e-6)
        self.assertLess(abs(scene.loading_rows[-1]['momentum_balance_error']),1e-6)
        # An outer product of the averaged A would incorrectly mix x/y fibers.
        M=s.enhancements.M.numpy()
        self.assertLess(abs(M[:,0,4]).max(),1e-10)
        self.assertGreater(M[:,0,0].max(),.1)
        self.assertGreater(M[:,4,4].max(),.1)


if __name__=='__main__':unittest.main()
