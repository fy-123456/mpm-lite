"""Missing acceptance checks for the single-material Lite delivery preset."""
import gc
import unittest
import numpy as np
from demos.aniso import Config, Scene
from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams, energy, pk1


class MainlineTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def test_energy_gradient_direction_ratios_and_isotropic_limit(self):
        F = np.array([[1.12, .07, 0.], [.02, .94, .03], [0., .01, 1.04]])
        H = np.array([[.2, -.1, .3], [.05, .1, 0.], [.1, -.2, .15]])
        eps = 1e-6
        iso = AnisotropicMaterialParams(10., 20., 0.)
        for angle in (0., 45., 90.):
            p = Config(fiber_angle=angle).params
            fd = (energy(F+eps*H, p.A0, p)-energy(F-eps*H, p.A0, p))/(2*eps)
            exact = np.sum(pk1(F, p.A0, p)*H)
            self.assertLess(abs(fd-exact)/max(1., abs(exact)), 2e-7)
            stretch = np.diag([1.1, 1., 1.])
            increment = (pk1(stretch, p.A0, p)-pk1(stretch, iso.A0, iso))[0, 0]
            expected = 2*p.k_f*1.1*(1.1**2-1)*np.cos(np.deg2rad(angle))**4
            self.assertAlmostEqual(increment, expected, delta=1e-11)
            off = Config(kf=0., fiber_angle=angle).params
            np.testing.assert_allclose(pk1(F, off.A0, off), pk1(F, iso.A0, iso), atol=1e-13)
            self.assertAlmostEqual(energy(F, off.A0, off), energy(F, iso.A0, iso), delta=1e-13)
        angle = .73
        R = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        np.testing.assert_allclose(pk1(R@F, p.A0, p), R@pk1(F, p.A0, p), atol=1e-10)

    def test_invalid_inputs_do_not_append_or_replace_material(self):
        s = AnisotropicLiteImplicitSolver((9,)*3, Config().params, device='cpu', gravity=0.)
        x = np.array([[.5, .5, .5]])
        s.seed_particles(x)
        fields = ('ptc_x', 'ptc_v', 'ptc_F', 'ptc_G', 'ptc_A0', 'ptc_reference_x')
        before = [getattr(s, f).numpy().copy() for f in fields]
        for options in ({'fiber_directions': [[0.,0.,0.]]}, {'deformation_gradient': np.diag([-1.,1.,1.])},
                        {'velocity_gradient': np.full((3,3), np.nan)}, {'use_material_k': 1}, {'density': -1.}):
            with self.assertRaises(ValueError):
                s.seed_particles(x, **options)
            self.assertEqual(s.n_ptc, 1)
            for f, old in zip(fields, before):
                np.testing.assert_array_equal(getattr(s, f).numpy(), old)
        with self.assertRaisesRegex(ValueError, 'single elastic material'):
            s.add_material(0, 2)
        for values in ((-1.,20.,200.), (10.,20.,-1.), (10.,20.,float('nan'))):
            with self.assertRaises(ValueError):
                AnisotropicMaterialParams(*values)

    def test_recommended_isotropic_trajectory_is_direction_independent(self):
        results = []
        for angle in (0., 73.):
            scene = Scene(Config('tensile', 9, .001, angle, 0., smooth_loading=True), 'cpu')
            for _ in range(3):
                self.assertTrue(scene.step())
            results.append([getattr(scene.solver, f).numpy().copy() for f in ('ptc_x','ptc_v','ptc_F')]
                           + [np.array([r['right_force'] for r in scene.loading_rows])])
            del scene
            gc.collect()
        for a, b in zip(*results):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)

    def test_particle_resample_affine_step(self):
        scene = Scene(Config('affine', 9, .001), 'cpu')
        s = scene.solver
        G = np.array([[.08,.02,.01],[0.,-.03,.015],[0.,0.,.04]])
        A = s.ptc_A0.numpy().copy()
        self.assertTrue(scene.step())
        np.testing.assert_allclose(s.ptc_F.numpy(), np.broadcast_to(np.eye(3)+.001*G, (s.n_ptc,3,3)), atol=5e-8, rtol=0.)
        np.testing.assert_array_equal(s.ptc_A0.numpy(), A)

    def test_particle_resample_failed_step_preserves_particle_history_and_clock(self):
        scene = Scene(Config('tensile', 9, .001, smooth_loading=True), 'cpu')
        self.assertTrue(scene.step())
        s = scene.solver
        s.prepare_centers()
        n = int(s.bcn)
        fields = ('ptc_x','ptc_v','ptc_F','ptc_G','ptc_A0')
        before = [getattr(s,f).numpy().copy() for f in fields]
        center = s.aniso_committed_F[:,:n].numpy().copy()
        clock = (s.sim_time, s.sim_steps, len(s.energy_ledger.rows))
        self.assertFalse(s.step(max_iters=0, print_every=0))
        self.assertEqual(clock, (s.sim_time, s.sim_steps, len(s.energy_ledger.rows)))
        for f, old in zip(fields, before):
            np.testing.assert_array_equal(getattr(s,f).numpy(), old)
        np.testing.assert_array_equal(s.aniso_committed_F[:,:n].numpy(), center)
        np.testing.assert_array_equal(s.aniso_trial_F[:,:n].numpy(), center)


if __name__ == '__main__':
    unittest.main()
