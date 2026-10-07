"""Material stabilization history: variational, static and remap gates."""
import gc,unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.types import vec3
from engine.sp_grid import B
from engine.aniso_phase1.material_patch import MaterialPatchLiteSolver,carrier_map
from engine.aniso_phase1.selective_patch import scalar_matrix,polynomial
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.stabilization_probe import node_coordinates
from benchmarks.aniso_residual_gate import geometry,matrix,rank_gate
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_unresolved_history import StageLedger
from demos.aniso import Config,Scene
wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
CFG=Config('tensile',9,.001,45.,smooth_loading=True,history_consistency='residual_center',stabilization='material_patch',boundary_impulse_transfer=True,apic_transfer='incremental',affine_flip_ratio=1.,velocity_dissipation='null',reaction_force_atol=1e-7)

class WideMaterial(MaterialPatchLiteSolver):
    def seed_particles(self,positions,*a,**k):return super().seed_particles(.5+3*(np.asarray(positions)-.5),*a,**k)

class MaterialPatchTests(unittest.TestCase):
    def tearDown(self):gc.collect()

    def test_interpolation_affine_partition_renumber_and_support(self):
        g=geometry(9);nodes=np.rint(g['nodes']/.125).astype(int);Y=g['nodes']+np.array([.004,-.002,.001]);ids,w,N=carrier_map(Y,nodes,.125)
        np.testing.assert_allclose(N@np.ones(len(nodes)),1,atol=1e-14)
        np.testing.assert_allclose(N@g['nodes'],Y,atol=1e-14)
        p=np.random.default_rng(31).permutation(len(nodes));_,_,R=carrier_map(Y,nodes[p],.125)
        np.testing.assert_allclose(R@g['nodes'][p],Y,atol=1e-14)
        with self.assertRaises(ValueError):carrier_map(Y+2,nodes,.125)
        with self.assertRaises(ValueError):carrier_map(Y,nodes[1:],.125)

    def test_energy_force_exact_tangent_nonuniform_history(self):
        p=SparseProbe('cpu',200.,.005,solver_cls=WideMaterial);s=p.s;e=s.enhancements;rng=np.random.default_rng(113)
        F=s.ptc_F.numpy().copy();F[:,0,0]=.96;F[:,0,1]=np.linspace(-.015,.015,len(F));s.ptc_F.assign(F)
        e.origin.assign(e.origin.numpy()+rng.normal(size=e.origin.numpy().shape)*1e-4);s.step(max_iters=0,print_every=0)
        v=p.project(.02*rng.normal(size=(p.n,3)));d=p.project(rng.normal(size=v.shape));d/=np.linalg.norm(d);q=p.project(rng.normal(size=v.shape));q/=np.linalg.norm(q)
        r=p.residual(v);H=p.tangent(d);Hq=p.tangent(q);eps=1e-5
        rp=p.residual(v+eps*d);ep=s.incremental_potential();rm=p.residual(v-eps*d);em=s.incremental_potential()
        self.assertLess(abs((ep-em)/(2*eps)-np.sum(r*d))/max(abs(np.sum(r*d)),1e-12),1e-6)
        self.assertLess(np.linalg.norm((rp-rm)/(2*eps)-H)/np.linalg.norm(H),1e-6)
        self.assertLess(abs(np.sum(q*H)-np.sum(d*Hq)),1e-11)

    def test_rigid_affine_quadratic_and_rotation_covariance(self):
        p=SparseProbe('cpu',200.,.005,boundary=False,solver_cls=WideMaterial);s=p.s;e=s.enhancements;x=node_coordinates(s)*s.dx
        A=np.array([[.02,.01,0],[-.01,-.01,.005],[0,0,.01]])
        p.residual((x-.5)@A.T/s.dt);self.assertLess(abs(e.hg_energy.numpy()[0]),1e-20)
        bend=np.column_stack((-(x[:,0]-.5)*(x[:,1]-.5),.5*(x[:,0]-.5)**2,np.zeros(len(x))))
        p.residual(.1*bend/s.dt);self.assertLess(abs(e.hg_energy.numpy()[0]),1e-20)
        a=.6;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        p.residual(((x-.5)@Q.T+.5-x)/s.dt)
        self.assertLess(abs(e.hg_energy.numpy()[0]),1e-20)
        v=.05*np.random.default_rng(8).normal(size=x.shape);p.residual(v);E=e.hg_energy.numpy()[0];force=e.extra[:p.n].numpy().copy()/s.dt
        p.residual(((x+s.dt*v-.5)@Q.T+.5-x)/s.dt);Er=e.hg_energy.numpy()[0];fr=e.extra[:p.n].numpy()/s.dt
        self.assertLess(abs(Er-E)/max(E,1e-30),1e-8);self.assertLess(np.linalg.norm(fr-force@Q.T)/max(np.linalg.norm(force),1e-30),1e-8)

    def test_actual_massless_operator_four_directions(self):
        g=geometry(9);S,_,_=scalar_matrix(g['nodes'],g['centers'],g['volume'],.125)
        from dataclasses import replace
        for label,kf,angle in [('ISO',0.,0.),('F0',200.,0.),('F45',200.,45.),('F90',200.,90.)]:
            K=matrix(g,hessian(label),'residual_center')+sp.block_diag([S]*3,format='csr');self.assertTrue(rank_gate(g,K,True)['passed'])
            scene=Scene(replace(CFG,kf=kf,fiber_angle=angle),'cpu');s=scene.solver;s.step(max_iters=0,print_every=0);s.evaluate_residual()
            coords=node_coordinates(s);n=len(coords);lookup={tuple(np.rint(x/s.dx).astype(int)):i for i,x in enumerate(g['nodes'])};perm=np.array([lookup[tuple(x)] for x in coords]);u=np.random.default_rng(3).normal(size=(len(g['nodes']),3));u[g['fixed']]=0
            p=wp.zeros_like(s.node_residual);out=wp.zeros_like(p);wp.copy(p,wp.array(u[perm],dtype=vec3,device='cpu'),count=n);s.apply_tangent(p,out)
            addr=s.ndof2bijk[:n].numpy();l=addr[:,1];mass=s.grid_m[:s.bcn].numpy()[addr[:,0],l//B**2,(l//B)%B,l%B]
            actual=(out[:n].numpy()-mass[:,None]*u[perm])/s.dt**2;expected=(K@u.T.ravel()).reshape(3,-1).T[perm];expected[g['fixed'][perm]]=0
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-10)

    def test_moving_commit_rebuild_energy_history_and_failure(self):
        scene=Scene(CFG,'cpu');s=scene.solver;s.energy_ledger=StageLedger()
        for _ in range(20):
            self.assertTrue(scene.step(),s.last_step_stats);r=scene.metrics()
            self.assertLess(abs(r['stabilization_rebuild_delta']),1e-16);self.assertLess(abs(r['stage_budget_error']),1e-14)
            self.assertLess(r['particle_momentum_balance_error_norm'],1e-7)
            np.testing.assert_allclose(s.ptc_F.numpy(),s.local_trial.numpy(),atol=1e-13)
            np.testing.assert_allclose(s.enhancements.origin.numpy(),s.enhancements.last_origin+s.dt*s.enhancements.carrier_velocity.numpy(),atol=1e-14)
        before={k:getattr(s,k).numpy().copy() for k in ('ptc_x','ptc_F','ptc_v','ptc_C')};Y=s.enhancements.origin.numpy().copy();clock=s.sim_time,s.sim_steps
        with patch('engine.aniso_phase1.unresolved_velocity.prepare',side_effect=ValueError('failure test')):self.assertFalse(scene.step())
        np.testing.assert_array_equal(Y,s.enhancements.origin.numpy());self.assertEqual(clock,(s.sim_time,s.sim_steps))
        for k,a in before.items():np.testing.assert_array_equal(a,getattr(s,k).numpy())
        self.assertTrue(scene.step())

    def test_same_state_remap_energy_and_history_restart(self):
        scene=Scene(CFG,'cpu');s=scene.solver
        for _ in range(4):self.assertTrue(scene.step())
        e=s.enhancements;state=e.state();e.correction(False);E=float(e.hg_energy.numpy()[0])
        for _ in range(5):
            e.resample();e.prepare();e.correction(False)
            self.assertEqual(E,float(e.hg_energy.numpy()[0]));np.testing.assert_array_equal(state['Y'],e.origin.numpy())
        from engine.aniso_phase1.material_patch import MaterialPatchEnhancements
        new=MaterialPatchEnhancements(s);new.restart_state=state;new.prepare();new.correction(False)
        self.assertEqual(E,float(new.hg_energy.numpy()[0]))
        self.assertEqual(new.weights.shape,e.weights.shape)

if __name__=='__main__':unittest.main()
