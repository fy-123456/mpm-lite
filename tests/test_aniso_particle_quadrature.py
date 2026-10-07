import gc
import unittest
import numpy as np
import warp as wp
from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams
from engine.aniso_phase1.particle_quadrature import ParticleQuadratureImplicitSolver
from engine.aniso_phase1.operator_probe import SparseProbe
from benchmarks.aniso_fair_comparison import sample_lattice


class ParticleQuadratureTests(unittest.TestCase):
    def test_identical_quadrature_when_particles_are_at_centers(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        outputs=[]
        params=AnisotropicMaterialParams(10,20,200,[1,1,0])
        axis=np.array([2.5,3.5])/7
        x=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
        G=np.array([[.08,.02,0],[0,-.03,.01],[0,0,.04]])
        for cls in (AnisotropicLiteImplicitSolver,ParticleQuadratureImplicitSolver):
            s=cls((8,)*3,params,dx=1/7,gravity=0,device='cpu')
            s.seed_particles(x,density=1,vol0=.001,velocity=(x-.5)@G.T,velocity_gradient=G)
            s.set_dt(.001)
            self.assertTrue(s.step(max_iters=12,print_every=0,cg_tol=1e-6,cg_atol=1e-13,newton_atol=1e-12))
            outputs.append([a.numpy().copy() for a in (s.ptc_x,s.ptc_v,s.ptc_F)])
            del s
            gc.collect()
        for a,b in zip(*outputs):np.testing.assert_allclose(a,b,atol=2e-10,rtol=1e-8)

    def test_particle_production_tangent_finite_difference(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        result=SparseProbe(solver_cls=ParticleQuadratureImplicitSolver).check()
        self.assertLess(min(result['fd_errors'].values()),1e-8)
        self.assertLess(result['boundary_projection_error'],1e-14)
        self.assertTrue(result['frozen_trial'] and result['frozen_committed'])

    def test_fair_lattice_volume_and_mass(self):
        for grid in (9,17,33):
            for samples in (1,2,3):
                x=sample_lattice(grid,samples)
                self.assertAlmostEqual(len(x)*(.125/len(x)),.125)
                np.testing.assert_allclose(x.mean(axis=0),.5,atol=1e-12)
                self.assertTrue(np.all(x>.25) and np.all(x<.75))


if __name__=='__main__':unittest.main()
