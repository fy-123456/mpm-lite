"""Production variational identities, full spectra, and nonlinear safeguards."""
import os
import unittest
import numpy as np
import warp as wp
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.particle_quadrature import ParticleQuadratureImplicitSolver
from engine.aniso_phase1.linear import guarded_pcg
from engine.types import vec3
from benchmarks.aniso_variational import preconditioner


class VariationalTests(unittest.TestCase):
    def setUp(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        self.device=os.environ.get('ANISO_TEST_DEVICE','cpu')

    def test_potential_gradient_exact_hessian_and_full_spectrum(self):
        p=SparseProbe(self.device,200.,.005)
        # Includes the known external-force predictor and mixed sticky/slip constraints.
        p.s.gravity=-9.81
        result=p.variational_check()
        self.assertEqual(result['free_dofs'],45)
        self.assertLess(min(result['potential_fd_errors'].values()),1e-7)
        self.assertLess(min(result['tangent_fd_errors'].values()),1e-8)
        self.assertLess(result['exact']['symmetry_relative'],1e-12)
        self.assertGreater(result['exact']['min_eigenvalue'],0.)
        # Input as well as output must be projected, even for non-feasible directions.
        rng=np.random.default_rng(5)
        a,b=(rng.normal(size=(p.n,3)) for _ in range(2))
        Aa,Ab=p.tangent(a),p.tangent(b)
        self.assertAlmostEqual(float(np.sum(a*Ab)),float(np.sum(b*Aa)),delta=1e-14)
        Z,matrices,_=p.last_dense
        rhs=wp.zeros_like(p.s.node_residual);x=wp.zeros_like(rhs)
        expected=p.project(a)
        wp.copy(rhs,wp.array(p.tangent(expected),dtype=vec3,device=self.device),count=p.n)
        _,error,tol,status=guarded_pcg(p.s.apply_tangent,rhs,x,preconditioner(p.s),1e-10,1e-15,200)
        self.assertEqual(status,'converged')
        self.assertLessEqual(error,tol*1.01)
        np.testing.assert_allclose(x[:p.n].numpy(),expected,atol=1e-8)

    def test_compression_indefinite_exact_and_spd_modified_tangent(self):
        p=SparseProbe(self.device,200.,.005)
        r=p.variational_check(np.diag([.65,.85,1.]))
        self.assertLess(r['exact']['min_eigenvalue'],0.)
        self.assertGreater(r['modified']['min_eigenvalue'],0.)
        self.assertGreater(r['tangent_modification_relative'],.01)
        self.assertLess(r['modified']['symmetry_relative'],1e-12)
        self.assertLess(min(r['potential_fd_errors'].values()),1e-7)
        Z,matrices,_=p.last_dense
        _,vectors=np.linalg.eigh(matrices['exact'])
        rhs=wp.zeros_like(p.s.node_residual);x=wp.zeros_like(rhs)
        wp.copy(rhs,wp.array((Z@vectors[:,0]).reshape(p.n,3),dtype=vec3,device=self.device),count=p.n)
        _,_,_,reason=guarded_pcg(p.s.apply_tangent,rhs,x,preconditioner(p.s),1e-8,1e-14,200)
        self.assertEqual(reason,'nonpositive_curvature')
        apply=lambda a,b:p.s.apply_tangent(a,b,project_pd=True)
        _,error,tol,reason=guarded_pcg(apply,rhs,x,preconditioner(p.s),1e-8,1e-14,200)
        self.assertEqual(reason,'converged')
        self.assertLessEqual(error,tol*1.01)

    def test_particle_potential_has_same_variational_identities(self):
        p=SparseProbe(self.device,200.,.005,solver_cls=ParticleQuadratureImplicitSolver)
        r=p.variational_check()
        self.assertLess(min(r['potential_fd_errors'].values()),1e-7)
        self.assertLess(min(r['tangent_fd_errors'].values()),1e-8)
        self.assertLess(r['exact']['symmetry_relative'],1e-12)

    def test_original_energy_armijo_and_nonlinear_fallback(self):
        p=SparseProbe(self.device,200.,.005)
        p.set_deformation(np.diag([.65,.85,1.]))
        self.assertTrue(p.s.step(max_iters=80,max_cg_iters=500,cg_tol=1e-5,
                               cg_atol=1e-12,newton_atol=1e-10,print_every=0))
        self.assertGreater(p.s.last_step_stats['curvature_failures'],0)
        self.assertGreater(p.s.last_step_stats['projected_tangent_solves'],0)
        self.assertLess(p.s.last_step_stats['last_residual_norm'],1e-9)
        self.assertTrue(p.s.last_newton_trace)
        for t in p.s.last_newton_trace:
            self.assertLess(t['slope'],0.)
            self.assertLessEqual(t['potential_after'],t['potential_before']+1e-4*t['alpha']*t['slope']+t['armijo_slack'])
            if t['roundoff_acceptance']:
                self.assertLess(t['residual_after'],t['residual_before'])

    def test_small_strain_line_search_reaches_strict_residual_tolerance(self):
        from demos.aniso import Config, Scene
        scene=Scene(Config(dt=.0005),self.device)
        for _ in range(40):
            self.assertTrue(scene.step(),scene.solver.last_step_stats)
            self.assertLessEqual(scene.solver.last_step_stats['last_residual_norm'],1e-10)

    def test_reject_spectral_clamp_region_without_commit(self):
        p=SparseProbe(self.device)
        F=np.diag([1e-8,1.,1.]);p.set_deformation(F)
        before=p.s.ptc_x.numpy().copy()
        self.assertFalse(p.s.step(max_iters=4,print_every=0))
        np.testing.assert_array_equal(p.s.ptc_x.numpy(),before)

    def test_pcg_refuses_legacy_nonsymmetric_equation(self):
        p=SparseProbe(self.device,force_discretization='legacy_kirchhoff')
        with self.assertRaisesRegex(ValueError,'symmetric'):
            p.s.step(linear_solver='pcg',print_every=0)


if __name__=='__main__':unittest.main()
