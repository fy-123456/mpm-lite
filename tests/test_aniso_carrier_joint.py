"""Physical and transactional gates for the bounded carrier joint prototype."""
import unittest
from unittest.mock import patch
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_carrier_joint import cube,load_case,spectrum
from engine.aniso_phase1.carrier_joint import CarrierJointSolver,Geometry,gradient,support_lift
from engine.aniso_phase1.selective_patch import polynomial


class CarrierJointTests(unittest.TestCase):
    def test_expanded_support_preserves_quadratic_fields_and_static_energy(self):
        s,e,m,h=cube();K=e.tangent(s.Y);s.x+=h*.008*np.array([1,.25,-.2]);s.Y+=h*.008*np.array([1,.25,-.2])
        g=Geometry(s,e,m,h,False);self.assertGreater(len(g.nodes),len(s.Y))
        S=polynomial((g.nodes-g.nodes.mean(0)))
        np.testing.assert_allclose(g.E@g.N@S,S,atol=2e-12)
        N=la.block_diag(g.N,g.N,g.N);E=la.block_diag(g.E,g.E,g.E)
        raw=N.T@K@N;new=E.T@raw@E
        self.assertGreater(spectrum(raw,6)['extra_modes'],0)
        self.assertTrue(spectrum(new,6)['passed'])
        np.testing.assert_allclose(new,K,atol=1e-12)
        with self.assertRaises(ValueError):support_lift(np.ones((4,5)),np.zeros((5,3)),h)

    def test_exact_potential_force_tangent_and_rotation_with_local_history(self):
        s,e,m,h,_=load_case();g=Geometry(s,e,m,h,True);rng=np.random.default_rng(161)
        d=g.Q@rng.normal(size=(g.Q.shape[1],3));d/=la.norm(d);eps=2e-6
        v=e.evaluate(s.Y);vp=e.evaluate(s.Y+eps*d);vm=e.evaluate(s.Y-eps*d)
        expected=float(np.sum(v['force']*d));self.assertLess(abs((vp['U']-vm['U'])/(2*eps)-expected)/max(abs(expected),1e-10),1e-5)
        K=e.tangent(s.Y);fd=(vp['force']-vm['force'])/(2*eps)
        self.assertLess(la.norm(fd.T.ravel()-K@d.T.ravel())/la.norm(fd),1e-6)
        a=.53;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        rotated=e.evaluate(s.Y@Q.T+np.array([.04,-.02,.01]))
        np.testing.assert_allclose(rotated['F'],Q@v['F'],atol=1e-13)
        self.assertLess(abs(rotated['U']-v['U']),1e-17)
        np.testing.assert_allclose(rotated['force'],v['force']@Q.T,atol=2e-12)

    def test_affine_velocity_and_normal_bending_preserved(self):
        s,e,m,h=cube();g=Geometry(s,e,m,h,False)
        A=np.array([[.02,.03,0],[-.03,.01,.01],[0,-.01,-.02]])
        W=s.Y@A.T+np.array([.1,-.2,.03]);z=g.J@W;n=len(s.x)
        np.testing.assert_allclose(z[:n],s.x@A.T+np.array([.1,-.2,.03]),atol=1e-13)
        for k in range(3):np.testing.assert_allclose(z[(k+1)*n:(k+2)*n],np.broadcast_to(A[:,k],s.x.shape),atol=1e-13)
        x=s.Y-.4;bend=np.column_stack((-x[:,0]*x[:,1],.5*x[:,0]**2,np.zeros(len(x))))
        before=e.evaluate(s.Y);after=e.evaluate(s.Y+.1*bend)
        self.assertLess(abs(after['Us']-before['Us']),1e-22)
        self.assertGreater(after['Um'],0.)

    def test_joint_fixed_geometry_energy_history_and_input_preservation(self):
        s,e,m,h,_=load_case();before={k:getattr(s,k).copy() for k in ('x','Y','v','C')}
        solver=CarrierJointSolver(s,e,m,h,'joint',False)
        for _ in range(5):
            r=solver.step(.0005)
            self.assertLess(abs(r['kinetic_force_work_defect_J']),1e-14)
            self.assertLess(r['delta_total_J'],1e-14)
            self.assertLess(r['history_commit_max'],1e-12)
        for k,v in before.items():np.testing.assert_array_equal(getattr(s,k),v)

    def test_moving_rigid_translation_crosses_support_without_extra_stiffness(self):
        velocity=np.array([.1,.025,-.02]);s,e,m,h=cube(velocity=velocity)
        solver=CarrierJointSolver(s,e,m,h,'joint',True,False);sizes=[]
        for _ in range(180):sizes.append(solver.step(.005)['grid_nodes'])
        self.assertGreater(max(sizes),len(s.Y));t=solver.state.time
        np.testing.assert_allclose(solver.state.x,s.x+t*velocity,atol=1e-11)
        np.testing.assert_allclose(solver.state.Y,s.Y+t*velocity,atol=1e-11)
        np.testing.assert_allclose(gradient(e.B,solver.state.Y),np.broadcast_to(np.eye(3),(len(s.x),3,3)),atol=1e-11)
        np.testing.assert_allclose(solver.state.v,np.broadcast_to(velocity,s.x.shape),atol=1e-11)
        self.assertTrue(spectrum(e.tangent(solver.state.Y),6)['passed'])

    def test_small_dt_retains_admissible_null_velocity_history(self):
        s,e,m,h,_=load_case()
        a=CarrierJointSolver(s,e,m,h,'joint');b=CarrierJointSolver(s,e,m,h,'joint_projected')
        ra=a.step(1e-7);rb=b.step(1e-7)
        self.assertLess(abs(ra['delta_total_J']),1e-9)
        self.assertGreater(-rb['delta_total_J'],1e-7)
        np.testing.assert_allclose(a.state.Y,b.state.Y,atol=1e-13)
        self.assertLess(abs(ra['kinetic_force_work_defect_J']),1e-14)

    def test_legacy_step_matches_actual_warp_without_velocity_filters(self):
        import os,json,warp as wp
        from pathlib import Path
        from demos.aniso import Scene,Config
        from benchmarks.aniso_carrier_joint import BASE
        from benchmarks.aniso_compatible_history import restore
        from benchmarks.aniso_unresolved_history import displacement
        cache=os.environ.get('MPM_LITE_TEST_CACHE','/tmp/mpm-lite-warp-cache')
        wp.config.kernel_cache_dir=cache
        cfg=json.loads((BASE/'v14/protocol.json').read_text())['configs']['material-fourth']
        cfg.update(dt=.001,flip_ratio=1.,velocity_dissipation='none',stabilization='compatible_patch')
        scene=Scene(Config(**cfg),'cpu')
        restore(scene,dict(source_case='material-fourth',source_step=12800,start=1.6))
        s,e,m,h,_=load_case();probe=CarrierJointSolver(s,e,m,h,'legacy_split',True)
        with patch('engine.aniso_phase1.tensile.loading_displacement',displacement):self.assertTrue(scene.step())
        row=probe.step(.001);actual=scene.solver
        np.testing.assert_allclose(probe.state.Y,actual.enhancements.origin.numpy(),atol=1e-10)
        np.testing.assert_allclose(probe.state.x,actual.ptc_x.numpy(),atol=1e-10)
        np.testing.assert_allclose(gradient(e.B,probe.state.Y),actual.ptc_F.numpy(),atol=1e-9)
        np.testing.assert_allclose(probe.state.v,actual.ptc_v.numpy(),atol=1e-9)
        np.testing.assert_allclose(probe.state.C,actual.ptc_C.numpy(),atol=1e-8)

    def test_failure_is_atomic_and_joint_force_is_not_an_added_damping(self):
        s,e,m,h,_=load_case();solver=CarrierJointSolver(s,e,m,h,'joint',True);before=solver.state.clone()
        with self.assertRaises(RuntimeError):solver.step(.001,max_iters=0)
        for k in ('x','Y','v','C'):np.testing.assert_array_equal(getattr(solver.state,k),getattr(before,k))
        self.assertEqual(solver.steps,0);self.assertEqual(solver.state.time,before.time)
        # Same Y -> exactly same material/stabilization energy and force, for
        # every velocity mode. No coefficient change is hidden in joint mode.
        for mode in ('legacy_split','common_split','adjoint_split','joint'):
            q=CarrierJointSolver(s,e,m,h,mode);self.assertEqual(q.energy.evaluate(q.state.Y)['U'],e.evaluate(s.Y)['U'])

if __name__=='__main__':unittest.main()
