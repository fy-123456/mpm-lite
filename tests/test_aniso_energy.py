"""Energy representation and telescoping diagnostics regressions."""
import itertools
import unittest
import numpy as np
import warp as wp
from demos.aniso import Config, Scene
from engine.aniso_phase1.diagnostics import particle_kinetic, energy_density
from engine.aniso_phase1 import energy


class EnergyTests(unittest.TestCase):
    def test_apic_matches_explicit_two_level_quadrature(self):
        rng = np.random.default_rng(11)
        dx = .1
        x = rng.uniform(.2, .7, (3, 3))
        v, G = rng.normal(size=(3, 3)), rng.normal(size=(3, 3, 3))
        mass = np.array([.2, .7, 1.3])
        expected = 0.
        for p in range(3):
            base = np.floor(x[p]/dx-.5)
            f = x[p]/dx-.5-base
            for a in itertools.product((0, 1), repeat=3):
                w = np.prod(np.where(a, f, 1-f))
                for b in itertools.product((0, 1), repeat=3):
                    xi = (base + a + b)*dx
                    vi = v[p] + G[p]@(xi-x[p])
                    expected += .5*mass[p]*w/8*np.dot(vi, vi)
        self.assertAlmostEqual(sum(particle_kinetic(x, v, G, mass, dx)), expected, places=13)
        self.assertEqual(particle_kinetic(x, v, G, mass, dx, False)[1], 0.)

    def test_vectorized_elastic_matches_material(self):
        params = Config(fiber_angle=45).params
        F = np.array([np.eye(3), [[1.1, .1, 0], [0, .97, 0], [0, 0, 1.02]]])
        A = np.broadcast_to(params.A0, F.shape)
        np.testing.assert_allclose(energy_density(F, A, params), [energy(f, a, params) for f, a in zip(F, A)])

    def test_budget_and_diagnostics_do_not_change_physics(self):
        wp.config.kernel_cache_dir = '/tmp/mpm-lite-warp-cache'
        scene = Scene(Config(), 'cpu')
        self.assertTrue(scene.step())
        self.assertTrue(scene.step())
        for row in scene.solver.energy_ledger.rows[1:]:
            self.assertLess(abs(row['budget_closure']), 1e-14)
            self.assertLessEqual(row['boundary_projection_delta'], 0.)
            self.assertLessEqual(row['p2c_delta'], 1e-14)
            self.assertLessEqual(row['c2g_delta'], 1e-14)
        reference = scene.solver.ptc_x.numpy(), scene.solver.ptc_v.numpy()
        del scene
        scene = Scene(Config(), 'cpu')
        scene.solver.energy_ledger = None
        self.assertTrue(scene.step())
        self.assertTrue(scene.step())
        np.testing.assert_allclose(scene.solver.ptc_x.numpy(), reference[0], atol=1e-14)
        np.testing.assert_allclose(scene.solver.ptc_v.numpy(), reference[1], atol=1e-14)


if __name__ == '__main__':
    unittest.main()


class ReactivationBudgetTests(unittest.TestCase):
    def test_reactivation_reset_is_separate_from_volume_arrival(self):
        from types import SimpleNamespace
        from engine.sp_grid import B
        from engine.types import real, vec3, mat33
        from engine.aniso_phase1.diagnostics import EnergyLedger
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        params=Config().params
        shape=(1,1,B,B*B)
        vol=np.zeros(shape);vol[0,0,3,3*B+3]=.01
        state=np.zeros(shape+(3,3));state[0,0,3,3*B+3]=np.eye(3)
        tensors=np.zeros_like(state);tensors[0,0,3,3*B+3]=params.A0
        mass=vol.reshape(1,B,B,B)
        s=SimpleNamespace(bcn=1,dx=.1,grid_size=wp.vec3i(9,9,9),enable_apic=True,aniso_params=params,
            center_vol=wp.array(vol,dtype=real),center_m=wp.array(mass,dtype=real),
            center_v=wp.zeros((1,B,B,B),dtype=vec3),center_G=wp.zeros((1,B,B,B),dtype=mat33),
            aniso_committed_F=wp.array(state,dtype=mat33),aniso_A0=wp.array(tensors,dtype=mat33),
            block_xyz_by_id=wp.array([[0,0,0]],dtype=wp.vec3i))
        ledger=EnergyLedger();ledger.k0=0.
        ledger.history[(3,3,3)]=2.
        ledger.previous_volumes={}
        ledger.previous_elastic=0.
        ledger.p2c(s)
        self.assertEqual(ledger.current['reactivated_centers'],1)
        self.assertAlmostEqual(ledger.current['state_reset_delta'],-.02)
        self.assertAlmostEqual(ledger.current['volume_remap_delta'],.02)
