"""Polynomial completeness, production derivatives and short moving histories."""
import gc,os,unittest
from functools import partial
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from engine.types import mat33
from engine.aniso_phase1.quadratic import polynomial,node_patch,history_polynomial,mapped_samples,MODES,GAUSS
from engine.aniso_phase1.solver import AnisotropicLiteImplicitSolver
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.rotation_probe import axis_rotation,BentRotation
import tests.test_aniso_corotated as corotated_tests


class QuadraticTests(unittest.TestCase):
    def setUp(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache';self.device=os.environ.get('ANISO_TEST_DEVICE','cpu')

    def tearDown(self):gc.collect()

    def probe(self,boundary=True):
        p=SparseProbe(device=self.device,dt=.005,boundary=boundary,
            solver_cls=partial(AnisotropicLiteImplicitSolver,stabilization=self.mode,direction_model='fourth_moment'))
        A=np.zeros((p.s.n_ptc,3,3));A[::2,0,0]=1;A[1::2,1,1]=1
        p.s.ptc_A0.assign(wp.array(A,dtype=mat33,device=self.device));p.s.step(max_iters=0,print_every=0)
        return p

    def test_complete_quadratic_velocity_and_reference_under_rotation(self):
        rng=np.random.default_rng(37);h=.1;center=np.array([.48,.51,.43]);nodes=rng.uniform(-2,2,(100,3))*h+center
        coef=rng.normal(size=(10,3))*.002
        for angle in (0.,.37,.9):
            R=axis_rotation(angle);ids,g,_,_=node_patch(nodes,center,h,cKDTree(nodes),R)
            P,_=polynomial((nodes-center)/h);_,G=polynomial(GAUSS@R.T)
            measured=np.einsum('nm,qnd->qmd',P[ids]@coef,g)
            expected=np.einsum('qdk,km->qmd',G,coef)/h
            np.testing.assert_allclose(measured,expected,atol=1e-10)
            np.testing.assert_allclose(g.sum(axis=1),0,atol=1e-10)
            # An exactly quadratic inverse map, with consistent inverse-F data.
            x=nodes;_,Gp=polynomial((x-center)/h)
            X=x+P@coef;J=np.eye(3)+np.einsum('ndk,km->nmd',Gp,coef)/h;F=np.linalg.inv(J)
            fit,_=history_polynomial(x,X,F,np.ones(len(x)),center,h,cKDTree(x))
            np.testing.assert_allclose(mapped_samples(fit,h,R),np.linalg.inv(np.eye(3)+expected),atol=1e-10)
            # Translate particles/centers across cells; material references stay fixed.
            shift=np.array([.21,-.08,.05]);fit,_=history_polynomial(x+shift,X,F,np.ones(len(x)),center+shift,h,cKDTree(x+shift))
            np.testing.assert_allclose(mapped_samples(fit,h,R),np.linalg.inv(np.eye(3)+expected),atol=1e-10)

    def test_rank_and_inverted_map_fail_explicitly(self):
        x=np.stack((np.linspace(-1,1,15),np.zeros(15),np.zeros(15)),axis=1)
        with self.assertRaisesRegex(ValueError,'full-rank'):node_patch(x,np.zeros(3),1,cKDTree(x))
        coef=np.zeros((10,3));coef[1,0]=-2
        with self.assertRaisesRegex(ValueError,'inverted'):mapped_samples(coef,1,np.eye(3))

    def test_actual_sparse_gradient_hessian_and_boundary_projection(self):
        for self.mode in MODES:
            with self.subTest(mode=self.mode):
                corotated_tests.CorotatedTests.test_actual_sparse_gradient_exact_hessian_symmetry_and_gn(self)
            gc.collect()

    def test_stabilizer_exact_derivative_objectivity_and_gn(self):
        for self.mode in MODES:
            with self.subTest(mode=self.mode):
                corotated_tests.CorotatedTests.test_isolated_stabilizer_gradient_rotation_covariance_and_gn_psd(self)
            gc.collect()

    def test_prebent_history_rotates_without_orientation_energy_collapse(self):
        for mode in MODES:
            scene=BentRotation(grid=17,dt=.01,duration=.02,mode=mode,device=self.device)
            for _ in range(2):
                row=scene.step();self.assertLess(row['particle_F_max_error'],1e-7)
            energy=scene.solver.energy_ledger.rows
            # Includes 45-degree rebuilding, where the old Q1 map loses bending.
            self.assertLess(abs(energy[-1]['elastic']/energy[0]['elastic']-1),.05)
            for row in energy[1:]:self.assertLess(abs(row['stabilization_solve_delta'])/max(row['stabilization_start_energy'],1e-15),1e-7)
            del scene;gc.collect()

    def test_modified_pcg_line_search_uses_original_potential(self):
        from demos.aniso import Scene,Config
        for mode in MODES:
            scene=Scene(Config('beam',17,.001,stabilization=mode,direction_model='fourth_moment',
                linear_solver='pcg_projected',residual_atol=1e-9),self.device)
            self.assertTrue(scene.step(),scene.solver.last_step_stats)
            self.assertGreater(scene.solver.last_step_stats['projected_tangent_solves'],0)
            for trace in scene.solver.last_newton_trace:
                self.assertLess(trace['slope'],0.)
                self.assertLessEqual(trace['potential_after'],trace['potential_before']+1e-4*trace['alpha']*trace['slope']+trace['armijo_slack'])
            del scene;gc.collect()


if __name__=='__main__':unittest.main()
