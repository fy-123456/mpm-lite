import unittest
import warp as wp
from engine.aniso_phase1.operator_probe import SparseProbe


class SparseOperatorTests(unittest.TestCase):
    def test_actual_multiblock_tangent_and_projected_boundary(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        result=SparseProbe().check()
        self.assertTrue(result['linear_converged'])
        self.assertEqual(result['active_blocks'],8)
        self.assertLess(min(result['fd_errors'].values()),1e-8)
        self.assertEqual(result['repeat_max_error'],0.)
        self.assertLess(result['boundary_projection_error'],1e-14)
        self.assertTrue(result['frozen_trial'] and result['frozen_committed'])
        self.assertLess(result['max_symmetry_error'],1e-12)

    def test_legacy_operator_remains_nonsymmetric(self):
        result=SparseProbe(force_discretization='legacy_kirchhoff').check()
        self.assertGreater(result['max_symmetry_error'],1e-4)
        self.assertLess(min(result['fd_errors'].values()),1e-8)


if __name__=='__main__':unittest.main()


class NonzeroGpuSolverTests(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available() and wp.get_cuda_device_count()>1, 'requires two CUDA GPUs')
    def test_nonsymmetric_krylov_uses_selected_ordinal(self):
        from demos.aniso import Config, Scene
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        scene=Scene(Config(), 'cuda:1')
        self.assertTrue(scene.step())
        self.assertGreater(scene.solver.last_step_stats['linear_iterations'],0)
        self.assertLess(scene.solver.last_step_stats['last_residual_norm'],1e-9)
