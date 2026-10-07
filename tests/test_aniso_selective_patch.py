"""v11 exact derivatives, polynomial/objective gates and actual massless operator."""
import gc,unittest
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.types import vec3
from engine.sp_grid import B
from engine.aniso_phase1.selective_patch import SelectiveHistoryLiteSolver,scalar_matrix,polynomial,projectors
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.stabilization_probe import node_coordinates
from benchmarks.aniso_residual_gate import geometry,matrix,rank_gate
from benchmarks.aniso_boundary_reference import hessian
from demos.aniso import Config,Scene
wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'


class WideProbeSolver(SelectiveHistoryLiteSolver):
    def seed_particles(self,positions,*args,**kwargs):
        # Existing SparseProbe has only three node layers. This candidate
        # explicitly needs four; enlarge the physical probe, not its gate.
        return super().seed_particles(.5+3*(np.asarray(positions)-.5),*args,**kwargs)


class SelectivePatchTests(unittest.TestCase):
    def tearDown(self):gc.collect()
    def probe(self,boundary=True):return SparseProbe('cpu',200.,.005,boundary=boundary,solver_cls=WideProbeSolver)
    def test_normal_quadratic_modes_zero_and_checkerboard_positive(self):
        g=geometry(9);K,ids,P=scalar_matrix(g['nodes'],g['centers'],g['volume'],.125)
        Q=polynomial(g['nodes']);self.assertLess(np.max(abs(P@Q[ids])),1e-14)
        self.assertLess(np.linalg.norm(K@Q),1e-12)
        x=g['nodes'];bend=np.column_stack((-x[:,0]*x[:,1],.5*x[:,0]**2,np.zeros(len(x))))
        self.assertLess(abs(np.sum(bend*(K@bend))),1e-14)
        v=(-1.)**np.rint((x[:,0]+x[:,1])/.125).astype(int)
        self.assertGreater(float(v@(K@v)),.01)
    def test_energy_residual_tangent_symmetry_nonuniform_history(self):
        p=self.probe();s=p.s;F=s.ptc_F.numpy();F[:,0,0]=.94;F[:,0,1]=np.linspace(-.02,.02,len(F))
        s.ptc_F.assign(F);s.step(max_iters=0,print_every=0)
        rng=np.random.default_rng(71);v=p.project(.03*rng.normal(size=(p.n,3)));d=p.project(rng.normal(size=v.shape));d/=np.linalg.norm(d)
        q=p.project(rng.normal(size=v.shape));q/=np.linalg.norm(q);r=p.residual(v);H=p.tangent(d);Hq=p.tangent(q);eps=1e-5
        rp=p.residual(v+eps*d);ep=s.incremental_potential();rm=p.residual(v-eps*d);em=s.incremental_potential()
        self.assertLess(abs((ep-em)/(2*eps)-np.sum(r*d))/max(abs(np.sum(r*d)),1e-12),1e-6)
        self.assertLess(np.linalg.norm((rp-rm)/(2*eps)-H)/np.linalg.norm(H),1e-6)
        self.assertLess(abs(np.sum(q*H)-np.sum(d*Hq)),1e-11)
        p.residual(v);self.assertGreater(float(np.sum(d*p.tangent(d,True))),0.)
    def test_finite_rotation_affine_and_objectivity(self):
        p=self.probe(False);s=p.s;x=node_coordinates(s)*s.dx;a=.7;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        def energy(v):
            p.residual(v);s._potential_sum.zero_();s._elastic_potential()
            return float(s._potential_sum.numpy()[0]),s.projected_internal_force()+s.enhancements.extra[:p.n].numpy()/s.dt
        E,f=energy(((x-.5)@Q.T+.5-x)/s.dt);self.assertLess(abs(E),1e-20);self.assertLess(np.linalg.norm(f),1e-10)
        A=np.array([[.03,.01,0],[-.01,-.02,.003],[0,.005,.01]]);energy((x-.5)@A.T/s.dt)
        self.assertLess(abs(s.enhancements.hg_energy.numpy()[0]),1e-20)
        np.testing.assert_allclose(s.local_trial.numpy(),np.broadcast_to(np.eye(3)+A,(s.n_ptc,3,3)),atol=1e-13)
        v=.1*np.random.default_rng(9).normal(size=x.shape);E,f=energy(v);Er,fr=energy(((x+s.dt*v-.5)@Q.T+.5-x)/s.dt)
        self.assertLess(abs(Er-E)/abs(E),1e-8);self.assertLess(np.linalg.norm(fr-f@Q.T)/np.linalg.norm(f),1e-8)
    def test_reference_rotation_does_not_change_polynomial_space(self):
        g=geometry(9);_,ids,P=scalar_matrix(g['nodes'],g['centers'],g['volume'],.125)
        rng=np.random.default_rng(122);Q,_=np.linalg.qr(rng.normal(size=(3,3)));X=g['nodes']@Q.T+np.array([1.,-.3,.8])
        Pr=projectors(X,ids,.125);self.assertLess(np.max(abs(P-Pr)),1e-13)
    def test_actual_massless_tangent_all_directions(self):
        g=geometry(9);S,_,_=scalar_matrix(g['nodes'],g['centers'],g['volume'],.125)
        for label,kf,angle in [('ISO',0.,0.),('F0',200.,0.),('F45',200.,45.),('F90',200.,90.)]:
            K=matrix(g,hessian(label),'residual_center')+sp.block_diag([S]*3,format='csr');self.assertTrue(rank_gate(g,K,True)['passed'])
            scene=Scene(Config('tensile',9,.005,angle,kf=kf,history_consistency='residual_center',stabilization='selective_patch'),'cpu');s=scene.solver;s.step(max_iters=0,print_every=0);s.evaluate_residual()
            coords=node_coordinates(s);n=len(coords);lookup={tuple(np.rint(x/s.dx).astype(int)):i for i,x in enumerate(g['nodes'])};perm=np.array([lookup[tuple(x)] for x in coords])
            u=np.random.default_rng(3).normal(size=(len(g['nodes']),3));u[g['fixed']]=0.;p=wp.zeros_like(s.node_residual);out=wp.zeros_like(p);wp.copy(p,wp.array(u[perm],dtype=vec3,device='cpu'),count=n);s.apply_tangent(p,out)
            addr=s.ndof2bijk[:n].numpy();l=addr[:,1];mass=s.grid_m[:s.bcn].numpy()[addr[:,0],l//B**2,(l//B)%B,l%B]
            actual=(out[:n].numpy()-mass[:,None]*u[perm])/s.dt**2;expected=(K@u.T.ravel()).reshape(3,-1).T[perm];expected[g['fixed'][perm]]=0.
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-10)
    def test_commit_momentum_history_and_rollback(self):
        scene=Scene(Config('tensile',9,.001,45.,smooth_loading=True,history_consistency='residual_center',stabilization='selective_patch',boundary_impulse_transfer=True,apic_transfer='incremental',reaction_force_atol=1e-7),'cpu');s=scene.solver
        for _ in range(8):
            self.assertTrue(scene.step(),s.last_step_stats);np.testing.assert_allclose(s.ptc_F.numpy(),s.local_trial.numpy(),atol=1e-13)
            np.testing.assert_allclose((s.mapped_W@s.ptc_F.numpy().reshape(s.n_ptc,9)).reshape(-1,3,3),s.mapped_trial.numpy(),atol=1e-13)
            self.assertLess(scene.loading_rows[-1]['particle_momentum_balance_error_norm'],1e-7)
            self.assertLess(abs(s.energy_ledger.rows[-1]['budget_closure']),1e-14)
        fields=('ptc_x','ptc_v','ptc_C','ptc_L','ptc_F');old={k:getattr(s,k).numpy().copy() for k in fields};clock=s.sim_time,s.sim_steps
        self.assertFalse(s.step(max_iters=0,print_every=0))
        for k in fields:np.testing.assert_array_equal(old[k],getattr(s,k).numpy())
        self.assertEqual(clock,(s.sim_time,s.sim_steps))
    def test_no_coefficient_tuning_or_incomplete_support(self):
        with self.assertRaises(ValueError):SelectiveHistoryLiteSolver((9,)*3,Config().params,stabilization_strength=.1)
        from engine.aniso_phase1.selective_patch import patches
        nodes=np.array([[i,j,k] for i in range(3) for j in range(4) for k in range(4)])
        with self.assertRaises(ValueError):patches(nodes,[[1,1,1]])


if __name__=='__main__':unittest.main()
