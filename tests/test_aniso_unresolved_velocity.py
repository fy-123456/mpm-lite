"""Joint v/C dissipation: independent transfer, energy, invariance and rollback."""
from dataclasses import replace
import gc,unittest
from unittest.mock import patch
import numpy as np
from demos.aniso import Config,Scene
from engine.aniso_phase1.unresolved_velocity import VelocityFilter,maps,pack
from engine.aniso_phase1.diagnostics import particle_kinetic
from engine.aniso_phase1.tensile import grid_values
from benchmarks.aniso_apic_frequency import initial_field

CFG=Config('tensile',9,.001,45.,smooth_loading=True,history_consistency='residual_center',stabilization='selective_patch',boundary_impulse_transfer=True,apic_transfer='incremental',affine_flip_ratio=1.,reaction_force_atol=1e-7)

def geometry():
    axes=[.125+(np.arange(12)+.5)*.75/12,.375+(np.arange(4)+.5)*.25/4,.375+(np.arange(4)+.5)*.25/4]
    x=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    return x,np.full(len(x),.046875/len(x))


def angular(x,m,D,v,C):
    return np.sum(m[:,None]*(np.cross(x,v)+sum(D[:,k,None]*np.cross(np.eye(3)[k],C[:,:,k]) for k in range(3))),axis=0)


class UnresolvedVelocityTests(unittest.TestCase):
    def tearDown(self):gc.collect()
    def test_map_matches_actual_two_level_kernel_and_kinetic_metric(self):
        scene=Scene(CFG,'cpu');s=scene.solver;x=s.ptc_x.numpy();m=s.ptc_m.numpy();rng=np.random.default_rng(982)
        v=.01*rng.normal(size=x.shape);C=.1*rng.normal(size=(len(x),3,3));s.ptc_v.assign(v);s.ptc_C.assign(C)
        self.assertFalse(s.step(max_iters=0,print_every=0));a=maps(x,m,s.dx,s.grid_size)
        expected=a['momentum_map']@pack(v,C)/a['mass'][:,None]
        np.testing.assert_allclose(grid_values(s,s.grid_v_raw,a['nodes']),expected,atol=1e-14,rtol=0)
        self.assertAlmostEqual(.5*np.sum(a['metric'][:,None]*pack(v,C)**2),sum(particle_kinetic(x,v,C,m,s.dx)),places=18)
        np.testing.assert_allclose(np.sum(np.cross(a['nodes']*s.dx,a['momentum_map']@pack(v,C)),axis=0),angular(x,m,a['D'],v,C),atol=1e-14)

    def test_dissipation_momentum_angular_and_visible_modes(self):
        x,m=geometry();rng=np.random.default_rng(221);x+=rng.uniform(-.005,.005,x.shape);m*=rng.uniform(.5,1.5,len(m));v=rng.normal(size=x.shape);C=rng.normal(size=(len(x),3,3))
        for mode in ('null','weak'):
            p=VelocityFilter(x,m,.125,.003,25.,mode);a,b,d=p.apply(v,C)
            self.assertLessEqual(d['dissipation_delta'],1e-15);self.assertLess(d['dissipation_identity_error'],1e-14)
            self.assertLess(d['protected_affine_change_max'],1e-14);self.assertLess(d['preserved_visible_mode_error'],1e-14)
            np.testing.assert_allclose(m@a,m@v,atol=1e-14,rtol=0)
            np.testing.assert_allclose(angular(x,m,p.data['D'],a,b),angular(x,m,p.data['D'],v,C),atol=1e-14,rtol=0)
            if mode=='null':np.testing.assert_allclose(p.data['momentum_map']@pack(a,b),p.data['momentum_map']@pack(v,C),atol=1e-14,rtol=0)

    def test_affine_rigid_velocity_and_superposition_preserved(self):
        x,m=geometry();rng=np.random.default_rng(123);x+=rng.uniform(-.003,.003,x.shape)
        for A in (np.zeros((3,3)),np.array([[0.,-.4,.2],[.4,0.,-.3],[-.2,.3,0.]]),rng.normal(size=(3,3))):
            v=x@A.T+np.array([.2,-.1,.4]);C=np.broadcast_to(A,(len(x),3,3)).copy()
            for mode in ('null','weak'):
                p=VelocityFilter(x,m,.125,.005,25.,mode);a,b,d=p.apply(v,C)
                np.testing.assert_allclose(a,v,atol=1e-13,rtol=0);np.testing.assert_allclose(b,C,atol=1e-13,rtol=0)
                vr=rng.normal(size=x.shape);Cr=rng.normal(size=C.shape);u,G,_=p.apply(vr,Cr);u1,G1,_=p.apply(vr+v,Cr+C)
                np.testing.assert_allclose(u1,u+v,atol=1e-13,rtol=0);np.testing.assert_allclose(G1,G+C,atol=1e-13,rtol=0)

    def test_frozen_time_semigroup_and_null_idempotence(self):
        x,m=geometry();rng=np.random.default_rng(11);v=rng.normal(size=x.shape);C=rng.normal(size=(len(x),3,3))
        p=VelocityFilter(x,m,.125,.001,25.,'weak');a,b,_=p.apply(v,C);a,b,_=p.apply(a,b)
        c,d,_=VelocityFilter(x,m,.125,.002,25.,'weak').apply(v,C)
        np.testing.assert_allclose(a,c,atol=1e-13,rtol=0);np.testing.assert_allclose(b,d,atol=1e-13,rtol=0)
        p=VelocityFilter(x,m,.125,.001,25.,'null');a,b,_=p.apply(v,C);c,d,_=p.apply(a,b)
        np.testing.assert_allclose(a,c,atol=1e-13,rtol=0);np.testing.assert_allclose(b,d,atol=1e-13,rtol=0)

    def test_same_solve_F_positions_L_and_potential_unchanged(self):
        states=[]
        for mode in ('none','weak','null'):
            scene=Scene(replace(CFG,velocity_dissipation=mode),'cpu');s=scene.solver;v,C=initial_field(s.ptc_x.numpy(),'sine_45');s.ptc_v.assign(v);s.ptc_C.assign(C)
            self.assertTrue(scene.step(),s.last_step_stats)
            states.append(({k:getattr(s,k).numpy().copy() for k in ('ptc_x','ptc_F','ptc_L','grid_v_new')},s.incremental_potential(),s.energy_ledger.rows[-1]['elastic']))
        for a in states[1:]:
            for k in states[0][0]:np.testing.assert_array_equal(a[0][k],states[0][0][k])
            self.assertEqual(a[1:],states[0][1:])

    def test_moving_particles_history_momentum_and_energy_budget(self):
        scene=Scene(replace(CFG,velocity_dissipation='weak'),'cpu');s=scene.solver
        for _ in range(20):
            self.assertTrue(scene.step(),s.last_step_stats)
            np.testing.assert_allclose(s.ptc_F.numpy(),s.local_trial.numpy(),atol=1e-13,rtol=0)
            np.testing.assert_allclose((s.mapped_W@s.ptc_F.numpy().reshape(s.n_ptc,9)).reshape(-1,3,3),s.mapped_trial.numpy(),atol=1e-13,rtol=0)
            self.assertLess(s.dissipation_stats['dissipation_delta'],1e-15)
            v0,C0=s._dissipation_unfiltered
            self.assertFalse(np.shares_memory(v0,s.ptc_v.numpy()))
            self.assertFalse(np.shares_memory(C0,s.ptc_C.numpy()))
            k0=sum(particle_kinetic(s.ptc_x.numpy(),v0,C0,s.ptc_m.numpy(),s.dx))
            k1=sum(particle_kinetic(s.ptc_x.numpy(),s.ptc_v.numpy(),s.ptc_C.numpy(),s.ptc_m.numpy(),s.dx))
            self.assertAlmostEqual(k1-k0,s.dissipation_stats['dissipation_delta'],places=17)
            self.assertLess(scene.loading_rows[-1]['particle_momentum_balance_error_norm'],1e-7)
            self.assertLess(abs(s.energy_ledger.rows[-1]['budget_closure']),1e-14)

    def test_decomposition_failure_does_not_commit_any_particle(self):
        scene=Scene(replace(CFG,velocity_dissipation='weak'),'cpu');s=scene.solver;self.assertTrue(scene.step())
        before={k:getattr(s,k).numpy().copy() for k in ('ptc_x','ptc_v','ptc_C','ptc_L','ptc_F')};clock=s.sim_time,s.sim_steps
        with patch('engine.aniso_phase1.unresolved_velocity.prepare',side_effect=np.linalg.LinAlgError('test decomposition failure')):
            self.assertFalse(scene.step())
        for k,a in before.items():np.testing.assert_array_equal(a,getattr(s,k).numpy())
        self.assertEqual(clock,(s.sim_time,s.sim_steps));self.assertIn('velocity_dissipation_failure',s.last_step_stats)

    def test_invalid_inputs_and_unsupported_mode_rejected(self):
        x,m=geometry()
        with self.assertRaises(ValueError):VelocityFilter(x,m,.125,.001,25.,'bad')
        with self.assertRaises(ValueError):maps(x,-m,.125)
        with self.assertRaises(ValueError):maps(x+1,m,.125,(9,9,9))
        with self.assertRaises(ValueError):Scene(replace(CFG,velocity_dissipation='invalid'),'cpu')
        scene=Scene(replace(CFG,history_consistency='standard',stabilization='none',velocity_dissipation='weak'),'cpu')
        with self.assertRaises(ValueError):scene.step()
        self.assertEqual(scene.solver.sim_steps,0)

if __name__=='__main__':unittest.main()
