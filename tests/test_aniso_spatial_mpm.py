import os
import unittest
from unittest.mock import patch
import numpy as np
from benchmarks.aniso_spatial_mpm import initial,origin_at,H
from engine.aniso_phase1.spatial_mpm import SpatialGrid,SpatialMPM,ParticleHistory


class SpatialMPMTests(unittest.TestCase):
    def test_spatial_affine_reproduction_full_mass_and_projection(self):
        state=initial();g=SpatialGrid(state.x,state.mass,H,[.2*H,.1*H,-.1*H])
        np.testing.assert_allclose(g.N.sum(axis=1),1.,atol=1e-14)
        np.testing.assert_allclose(g.N@g.x,state.x,atol=1e-14)
        A=np.array([[.1,.02,-.03],[.01,.2,.03],[.01,-.02,.15]])
        velocity=g.x@A.T+[1.,.2,-.1]
        np.testing.assert_allclose(g.gradient(velocity),np.broadcast_to(A,(len(state.x),3,3)),atol=1e-13)
        vp=g.N@velocity
        np.testing.assert_allclose(g.project(vp),velocity,atol=2e-11)
        self.assertAlmostEqual(np.sum(velocity*(g.M@velocity)),np.sum(state.mass[:,None]*vp**2),delta=1e-17)
        arbitrary=np.random.default_rng(163).normal(size=state.v.shape)
        projected=g.N@g.project(arbitrary)
        np.testing.assert_allclose((state.mass[:,None]*projected).sum(0),(state.mass[:,None]*arbitrary).sum(0),atol=1e-15)
        self.assertLessEqual(np.sum(state.mass[:,None]*projected**2),np.sum(state.mass[:,None]*arbitrary**2)+1e-15)

    def test_potential_gradient_tangent_and_history_freeze(self):
        state=initial();model=SpatialMPM();system=model.frozen_step(state,[.25*H,.125*H,-.125*H])
        rng=np.random.default_rng(88);v=system.grid.project(state.v)
        a=rng.normal(size=v.shape)*.03;b=rng.normal(size=v.shape)*.02;dt=.0005;eps=1e-4
        _,r=system.potential(v,v,dt)
        ep,rp=system.potential(v+eps*a,v,dt);em,rm=system.potential(v-eps*a,v,dt)
        self.assertAlmostEqual((ep-em)/(2*eps),np.sum(r*a),delta=2e-13)
        Ja=system.tangent(v,a,dt);Jb=system.tangent(v,b,dt)
        np.testing.assert_allclose((rp-rm)/(2*eps),Ja,rtol=2e-6,atol=2e-12)
        self.assertAlmostEqual(np.sum(b*Ja),np.sum(a*Jb),delta=1e-16)
        Hessian=system.whitened_hessian(v,dt)
        np.testing.assert_allclose(Hessian@system.grid.to_y(a).ravel(),system.grid.force_to_y(Ja).ravel(),rtol=2e-10,atol=1e-14)
        Fdirect=(np.eye(3)+dt*system.grid.gradient(v))@state.F
        np.testing.assert_allclose(system.trial_F(v,dt),Fdirect,atol=5e-16)
        for attr in ('x','v','F','mass','volume','A'):self.assertFalse(getattr(state,attr).flags.writeable)

    def test_rigid_advection_rebuilds_real_spatial_grid(self):
        old=initial();identity=np.broadcast_to(np.eye(3),old.F.shape)
        state=ParticleHistory(old.X,old.X,np.tile([1.,0.,0.],(len(old.x),1)),identity,old.mass,old.volume,old.A)
        start=state.x.copy();model=SpatialMPM();grid_sizes=[]
        for step in range(24):
            state,r=model.step(state,.0005,origin_at(step*.0005,True));grid_sizes.append(r['nodes'])
            self.assertLess(r['kinetic_relative'],1e-12)
            self.assertLess(r['prediction_commit_F_absolute'],1e-12)
        np.testing.assert_allclose(state.x,start+[.012,0,0],atol=2e-12)
        np.testing.assert_allclose(state.F,identity,atol=2e-11)
        self.assertTrue(np.any(np.floor(state.x/H)!=np.floor(start/H)))
        self.assertGreater(len(set(grid_sizes)),1)

    def test_failed_solve_and_bad_trial_do_not_commit_history(self):
        state=initial();model=SpatialMPM();before=[getattr(state,k).copy() for k in ('x','v','F')]
        system=model.frozen_step(state)
        with self.assertRaises(ValueError):system.elastic(-2*system.grid.x,.75)
        with patch('engine.aniso_phase1.spatial_mpm.minimize_newton',side_effect=RuntimeError('forced solve failure')):
            with self.assertRaisesRegex(RuntimeError,'forced solve failure'):model.step(state,.0005)
        for k,value in zip(('x','v','F'),before):np.testing.assert_array_equal(getattr(state,k),value)
        self.assertIsNone(model.last_grid)
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            SpatialGrid(np.array([[.1,.1,.1]]),np.ones(1),H)
        for dt in (-.1,0,np.nan):
            with self.assertRaises(ValueError):model.step(state,dt)

    @unittest.skipUnless(os.environ.get('ANISO_TEST_CUDA'),'real CUDA material audit')
    def test_gpu_cpu_multistep_grid_change_and_commit_match(self):
        import warp as wp
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        cpu,gpu=SpatialMPM(),SpatialMPM(device='cuda:0');a=b=initial()
        for step in range(8):
            origin=origin_at(step*.0005,True)
            a,ra=cpu.step(a,.0005,origin);b,rb=gpu.step(b,.0005,origin)
            np.testing.assert_allclose(a.x,b.x,atol=2e-10,rtol=0)
            np.testing.assert_allclose(a.F,b.F,atol=2e-9,rtol=0)
            np.testing.assert_allclose(a.v,b.v,atol=2e-8,rtol=0)
            self.assertLess(rb['prediction_commit_F_absolute'],1e-12)
            self.assertLess(abs(rb['energy_budget_residual']),1e-14)

if __name__=='__main__':unittest.main()
