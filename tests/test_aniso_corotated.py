"""Exact polar derivatives, objective forces, PSD fallback and sparse plumbing."""
import gc,os,unittest
from functools import partial
import numpy as np
import warp as wp
from engine.types import vec3,mat33
from engine.aniso_phase1.solver import AnisotropicLiteImplicitSolver
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.stabilization_probe import node_coordinates,static_audit
from engine.aniso_phase1.rotation_probe import axis_rotation,BentRotation
from demos.aniso import Scene,Config


class CorotatedTests(unittest.TestCase):
    def setUp(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        self.device=os.environ.get('ANISO_TEST_DEVICE','cpu')

    def tearDown(self):gc.collect()

    def probe(self,boundary=True):
        p=SparseProbe(device=self.device,dt=.005,boundary=boundary,
            solver_cls=partial(AnisotropicLiteImplicitSolver,stabilization='corotated',direction_model='fourth_moment'))
        s=p.s;A=np.zeros((s.n_ptc,3,3));A[::2,0,0]=1;A[1::2,1,1]=1
        s.ptc_A0.assign(wp.array(A,dtype=mat33,device=self.device));s.step(max_iters=0,print_every=0)
        return p

    def test_actual_sparse_gradient_exact_hessian_symmetry_and_gn(self):
        p=self.probe();rng=np.random.default_rng(92)
        v=p.project(rng.normal(size=(p.n,3))*.15)
        d=p.project(rng.normal(size=v.shape));d/=np.linalg.norm(d)
        r=p.residual(v);H=p.tangent(d);eps=1e-5
        rp=p.residual(v+eps*d);ep=p.s.incremental_potential()
        rm=p.residual(v-eps*d);em=p.s.incremental_potential()
        self.assertLess(abs((ep-em)/(2*eps)-np.sum(r*d))/max(abs(np.sum(r*d)),1e-12),1e-4)
        self.assertLess(np.linalg.norm((rp-rm)/(2*eps)-H)/np.linalg.norm(H),1e-4)
        p.residual(v);q=rng.normal(size=v.shape) # Includes infeasible boundary components.
        Hq=p.tangent(q)
        self.assertLess(abs(np.sum(q*H)-np.sum(d*Hq))/max(np.linalg.norm(H)*np.linalg.norm(q),1e-15),1e-7)
        Gn=p.tangent(d,True);Gq=p.tangent(q,True)
        self.assertGreater(np.sum(d*Gn),0.)
        self.assertLess(abs(np.sum(q*Gn)-np.sum(d*Gq))/max(np.linalg.norm(Gn)*np.linalg.norm(q),1e-15),1e-7)

    def test_isolated_stabilizer_gradient_rotation_covariance_and_gn_psd(self):
        p=self.probe(False);s=p.s;e=s.enhancements
        rng=np.random.default_rng(4);v=rng.normal(size=(p.n,3))*.2
        direction=rng.normal(size=v.shape);direction/=np.linalg.norm(direction)
        def evaluate(values):
            e.values.zero_();wp.copy(e.values,wp.array(values,dtype=vec3,device=self.device),count=p.n)
            out=wp.zeros_like(e.extra);energy=wp.zeros_like(e.hg_energy);s.aniso_invalid_trial.zero_()
            e._hg(e.values,out,energy,False)
            self.assertEqual(int(s.aniso_invalid_trial.numpy()[0]),0)
            return float(energy.numpy()[0]),out[:p.n].numpy()
        E,g=evaluate(v);eps=1e-5
        ep,gp=evaluate(v+eps*direction);em,gm=evaluate(v-eps*direction)
        self.assertLess(abs((ep-em)/(2*eps)-np.sum(g*direction))/abs(np.sum(g*direction)),1e-5)
        evaluate(v);d=wp.zeros_like(e.values);wp.copy(d,wp.array(direction,dtype=vec3,device=self.device),count=p.n)
        out=wp.zeros_like(d);e._hg(d,out,e.hg_energy,True)
        exact=out[:p.n].numpy()
        self.assertLess(np.linalg.norm((gp-gm)/(2*eps)-exact)/np.linalg.norm(exact),1e-5)
        out.zero_();e._hg(d,out,e.hg_energy,True,True)
        self.assertGreaterEqual(float(np.sum(direction*out[:p.n].numpy())),-1e-14)
        x=node_coordinates(s)*s.dx
        for axis in ('z','y'):
            Q=axis_rotation(.9,axis)
            rotated=((x+s.dt*v-.5)@Q.T+.5-x)/s.dt
            Er,gr=evaluate(rotated)
            self.assertLess(abs(Er/E-1),1e-7)
            self.assertLess(np.linalg.norm(gr-g@Q.T)/np.linalg.norm(g),1e-7)

    def test_reference_mapping_identity_and_invalid_map_guard(self):
        from engine.aniso_phase1.types import AnisotropicMaterialParams
        with self.assertRaisesRegex(ValueError,'bulk modulus'):
            AnisotropicLiteImplicitSolver((8,)*3,AnisotropicMaterialParams(10,-9,0),stabilization='corotated')
        from engine.aniso_phase1.corotated import prepare_reference
        p=self.probe(False);e=p.s.enhancements
        np.testing.assert_allclose(e.reference_F0.numpy(),np.broadcast_to(np.eye(3),e.reference_F0.numpy().shape),atol=1e-10)
        x=node_coordinates(p.s)*p.s.dx;u=np.zeros_like(x);u[:,0]=2*x[:,0]
        wp.copy(e.u,wp.array(u,dtype=vec3,device=self.device),count=p.n)
        invalid=wp.zeros(1,dtype=int,device=self.device)
        wp.launch(prepare_reference,dim=(len(e.ids),9),inputs=[e.ids,e.u,e.reference_g,e.reference_F0,e.reference_B,invalid],device=self.device)
        self.assertEqual(int(invalid.numpy()[0]),1)

    def test_modified_solver_uses_original_energy_line_search(self):
        scene=Scene(Config('beam',17,.001,stabilization='corotated',direction_model='fourth_moment',
            linear_solver='pcg_projected',residual_atol=1e-9),self.device)
        self.assertTrue(scene.step(),scene.solver.last_step_stats)
        self.assertGreater(scene.solver.last_step_stats['projected_tangent_solves'],0)
        for t in scene.solver.last_newton_trace:
            self.assertLess(t['slope'],0.)
            self.assertLessEqual(t['potential_after'],t['potential_before']+1e-4*t['alpha']*t['slope']+t['armijo_slack'])

    def test_prebent_exact_rotation_and_short_dynamic_scenes(self):
        scene=BentRotation(grid=17,dt=.01,duration=.02,mode='corotated',device=self.device)
        for _ in range(2):
            r=scene.step();row=scene.solver.energy_ledger.rows[-1]
            self.assertLess(r['particle_F_max_error'],1e-7)
            self.assertLess(abs(row['stabilization_solve_delta'])/max(row['stabilization_start_energy'],1e-15),1e-7)
        del scene;gc.collect()
        for kind in ('center','particle'):
            scene=Scene(Config('tensile',9,.005,quadrature=kind,stabilization='corotated',
                direction_model='fourth_moment' if kind=='center' else 'mean_tensor',fiber_field='crossed',
                loading_time=.16,loading_speed=.0125,smooth_loading=True),self.device)
            for _ in range(3):self.assertTrue(scene.step(),scene.solver.last_step_stats)
            self.assertLess(abs(scene.solver.energy_ledger.rows[-1]['budget_closure']),1e-9)
            self.assertLess(scene.loading_rows[-1]['free_force_residual_norm'],1e-6)
            del scene;gc.collect()


if __name__=='__main__':unittest.main()
