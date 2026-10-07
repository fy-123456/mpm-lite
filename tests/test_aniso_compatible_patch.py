"""Actual solver gates for common carrier/particle history."""
import gc
import copy
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.types import vec3
from engine.sp_grid import B
from engine.aniso_phase1.compatible_patch import CompatiblePatchLiteSolver, carrier_gradient, unpack_reference
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.stabilization_probe import node_coordinates
from engine.aniso_phase1.selective_patch import scalar_matrix
from benchmarks.aniso_residual_gate import geometry, matrix, rank_gate
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_unresolved_history import StageLedger
from demos.aniso import Config, Scene
wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
CFG=Config('tensile',9,.001,45.,smooth_loading=True,history_consistency='residual_center',stabilization='compatible_patch',boundary_impulse_transfer=True,apic_transfer='incremental',affine_flip_ratio=1.,velocity_dissipation='null',reaction_force_atol=1e-7)

class WideCompatible(CompatiblePatchLiteSolver):
    def seed_particles(self, positions, *args, **kwargs):
        return super().seed_particles(.5+3*(np.asarray(positions)-.5), *args, **kwargs)

class CompatibleTests(unittest.TestCase):
    def tearDown(self):gc.collect()

    def test_potential_residual_exact_tangent_deformed_local_history(self):
        p=SparseProbe('cpu',200.,.005,solver_cls=WideCompatible);s=p.s;e=s.enhancements
        p.residual(np.zeros((p.n,3)))
        rng=np.random.default_rng(208);R=np.broadcast_to(np.eye(3),(s.n_ptc,3,3)).copy()
        R[:,0,0]=.97;R[:,0,1]=np.linspace(-.015,.015,len(R));s.compatible_R=R
        e.origin.assign(e.origin.numpy()+rng.normal(size=e.origin.numpy().shape)*1e-5)
        s.ptc_F.assign(carrier_gradient(s.compatible_G,e.origin.numpy())@R)
        s.step(max_iters=0,print_every=0)
        v=p.project(.01*rng.normal(size=(p.n,3)));d=p.project(rng.normal(size=v.shape));d/=np.linalg.norm(d)
        q=p.project(rng.normal(size=v.shape));q/=np.linalg.norm(q)
        r=p.residual(v);H=p.tangent(d);Hq=p.tangent(q);eps=1e-5
        rp=p.residual(v+eps*d);ep=s.incremental_potential();rm=p.residual(v-eps*d);em=s.incremental_potential()
        self.assertLess(abs((ep-em)/(2*eps)-np.sum(r*d))/max(abs(np.sum(r*d)),1e-12),1e-6)
        self.assertLess(np.linalg.norm((rp-rm)/(2*eps)-H)/np.linalg.norm(H),1e-6)
        self.assertLess(abs(np.sum(q*H)-np.sum(d*Hq)),1e-11)

    def test_affine_quadratic_and_finite_rotation_objectivity(self):
        p=SparseProbe('cpu',200.,.005,boundary=False,solver_cls=WideCompatible);s=p.s;e=s.enhancements
        x=node_coordinates(s)*s.dx;A=np.array([[.02,.01,0],[-.01,-.01,.005],[0,0,.01]])
        p.residual((x-.5)@A.T/s.dt)
        np.testing.assert_allclose(s.local_trial.numpy(),np.broadcast_to(np.eye(3)+A,(s.n_ptc,3,3)),atol=2e-14)
        self.assertLess(abs(e.hg_energy.numpy()[0]),1e-20)
        bend=np.column_stack((-(x[:,0]-.5)*(x[:,1]-.5),.5*(x[:,0]-.5)**2,np.zeros(len(x))))
        p.residual(.1*bend/s.dt);self.assertLess(abs(e.hg_energy.numpy()[0]),1e-20)
        a=.6;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        p.residual(((x-.5)@Q.T+.5-x)/s.dt)
        np.testing.assert_allclose(s.local_trial.numpy(),np.broadcast_to(Q,(s.n_ptc,3,3)),atol=2e-14)
        self.assertLess(abs(e.hg_energy.numpy()[0]),1e-20)
        v=.03*np.random.default_rng(81).normal(size=x.shape);p.residual(v)
        F=s.local_trial.numpy().copy();from engine.aniso_phase1.diagnostics import energy_density
        energy=lambda:float(s.local_V.numpy()@energy_density(s.local_trial.numpy(),s.ptc_A0.numpy(),s.aniso_params)+e.hg_energy.numpy()[0])
        E=energy();fm=s.projected_internal_force()+e.extra[:p.n].numpy()/s.dt
        p.residual(((x+s.dt*v-.5)@Q.T+.5-x)/s.dt)
        np.testing.assert_allclose(s.local_trial.numpy(),Q@F,atol=3e-14)
        self.assertLess(abs(energy()-E)/max(E,1e-30),1e-8)
        fr=s.projected_internal_force()+e.extra[:p.n].numpy()/s.dt
        self.assertLess(np.linalg.norm(fr-fm@Q.T)/np.linalg.norm(fm),1e-8)

    def test_initial_actual_massless_four_directions(self):
        g=geometry(9);S,_,_=scalar_matrix(g['nodes'],g['centers'],g['volume'],.125)
        for label,kf,angle in [('ISO',0.,0.),('F0',200.,0.),('F45',200.,45.),('F90',200.,90.)]:
            K=matrix(g,hessian(label),'residual_center')+sp.block_diag([S]*3,format='csr')
            self.assertTrue(rank_gate(g,K,True)['passed'])
            scene=Scene(replace(CFG,kf=kf,fiber_angle=angle),'cpu');s=scene.solver;s.step(max_iters=0,print_every=0);s.evaluate_residual()
            coords=node_coordinates(s);n=len(coords);lookup={tuple(np.rint(x/s.dx).astype(int)):i for i,x in enumerate(g['nodes'])};perm=np.array([lookup[tuple(x)] for x in coords])
            u=np.random.default_rng(3).normal(size=(len(g['nodes']),3));u[g['fixed']]=0
            p=wp.zeros_like(s.node_residual);out=wp.zeros_like(p);wp.copy(p,wp.array(u[perm],dtype=vec3,device='cpu'),count=n);s.apply_tangent(p,out)
            addr=s.ndof2bijk[:n].numpy();l=addr[:,1];m=s.grid_m[:s.bcn].numpy()[addr[:,0],l//B**2,(l//B)%B,l%B]
            actual=(out[:n].numpy()-m[:,None]*u[perm])/s.dt**2
            expected=(K@u.T.ravel()).reshape(3,-1).T[perm];expected[g['fixed'][perm]]=0
            self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected),1e-10)

    def test_moving_closure_budget_remap_and_failure_atomicity(self):
        scene=Scene(CFG,'cpu');s=scene.solver;s.energy_ledger=StageLedger()
        for _ in range(12):
            self.assertTrue(scene.step(),s.last_step_stats);r=scene.metrics();e=s.enhancements
            self.assertLess(abs(r['stabilization_rebuild_delta']),1e-16)
            self.assertLess(abs(r['stage_budget_error']),1e-13)
            np.testing.assert_allclose(s.ptc_F.numpy(),carrier_gradient(s.compatible_G,e.origin.numpy())@s.compatible_R,atol=2e-13)
            np.testing.assert_array_equal(s.ptc_F.numpy(),s.local_trial.numpy())
        before={k:getattr(s,k).numpy().copy() for k in ('ptc_x','ptc_F','ptc_v','ptc_C')};Y=e.origin.numpy().copy();clock=s.sim_time,s.sim_steps
        with patch('engine.aniso_phase1.unresolved_velocity.prepare',side_effect=ValueError('failure test')):self.assertFalse(scene.step())
        np.testing.assert_array_equal(Y,e.origin.numpy());self.assertEqual(clock,(s.sim_time,s.sim_steps))
        for k,a in before.items():np.testing.assert_array_equal(a,getattr(s,k).numpy())
        self.assertTrue(scene.step())
        e.correction(False);E=float(e.hg_energy.numpy()[0]);F=s.ptc_F.numpy().copy()
        for _ in range(3):e.resample();e.prepare();e.correction(False);self.assertEqual(E,float(e.hg_energy.numpy()[0]))
        np.testing.assert_array_equal(s.ptc_F.numpy(),F)

    def test_support_expansion_rejected_before_physical_commit(self):
        import itertools
        h=1/16;dt=.005;v=np.array([.1,.025,-.02])
        x=np.array(list(itertools.product(.34375+np.arange(4)*h/2,repeat=3)))
        s=CompatiblePatchLiteSolver((17,)*3,Config('tensile',17,dt,45.).params,dx=h,device='cpu',gravity=0.,
            energy_diagnostics=True,boundary_impulse_transfer=True,apic_transfer='incremental',affine_flip_ratio=1.,velocity_dissipation='null')
        s.seed_particles(x,density=1.,vol0=h**3/8,velocity=np.broadcast_to(v,x.shape));s.set_dt(dt)
        options=dict(max_iters=16,print_every=0,v_tol=1e-10,cg_tol=1e-4,cg_atol=1e-12,newton_atol=1e-10,max_cg_iters=1000)
        self.assertTrue(s.step(**options));before={k:getattr(s,k).numpy().copy() for k in ('ptc_x','ptc_F','ptc_v','ptc_C')}
        Y=s.enhancements.origin.numpy().copy();clock=s.sim_time,s.sim_steps
        for _ in range(2):
            self.assertFalse(s.step(**options))
            self.assertTrue(s.last_step_stats['compatible_support_rejected'])
            self.assertEqual((s.last_step_stats['carrier_capacity'],s.last_step_stats['active_grid_nodes']),(64,80))
            self.assertEqual(clock,(s.sim_time,s.sim_steps));np.testing.assert_array_equal(Y,s.enhancements.origin.numpy())
            for k,a in before.items():np.testing.assert_array_equal(a,getattr(s,k).numpy())

    def test_saved_frames_remain_independent_after_real_steps(self):
        from benchmarks.aniso_compatible_history import capture_frame
        scene=Scene(CFG,'cpu');self.assertTrue(scene.step());first=capture_frame(scene.solver)
        expected={k:np.array(v,copy=True) for k,v in first.items()}
        for _ in range(4):self.assertTrue(scene.step())
        last=capture_frame(scene.solver)
        for k,v in expected.items():np.testing.assert_array_equal(first[k],v)
        self.assertGreater(np.max(abs(last['F']-first['F'])),1e-8)
        self.assertGreater(np.max(abs(last['Y']-first['Y'])),1e-9)
        for k in ('x','F','v','C','Y'):
            stack=np.array([first[k],last[k]])
            np.testing.assert_array_equal(stack[0],expected[k])

    def test_complete_restart_preserves_next_step_and_local_history(self):
        a=Scene(CFG,'cpu')
        for _ in range(4):self.assertTrue(a.step())
        s=a.solver;state=s.enhancements.state();G,R=unpack_reference(state)
        np.testing.assert_allclose(carrier_gradient(G,state['Y'])@R,s.ptc_F.numpy(),atol=1e-13)
        b=Scene(CFG,'cpu');t=b.solver
        for key in ('ptc_x','ptc_F','ptc_v','ptc_C'):getattr(t,key).assign(getattr(s,key).numpy())
        t.enhancements.restart_state=state;t.sim_time=s.sim_time;t.sim_steps=s.sim_steps
        t.energy_ledger=copy.deepcopy(s.energy_ledger);b.loading_work=a.loading_work
        self.assertTrue(a.step());self.assertTrue(b.step())
        for key in ('ptc_x','ptc_F','ptc_v','ptc_C'):np.testing.assert_allclose(getattr(s,key).numpy(),getattr(t,key).numpy(),atol=1e-12)
        np.testing.assert_allclose(s.enhancements.origin.numpy(),t.enhancements.origin.numpy(),atol=1e-13)

if __name__=='__main__':unittest.main()
