import os
import unittest
import numpy as np
from benchmarks.aniso_spatial_mpm import initial,H,origin_at
from engine.aniso_phase1.spatial_mpm import SpatialGrid,SpatialMPM,ParticleHistory
from engine.aniso_phase1.spatial_basis import sample_basis,residual_modes


class SpatialBasisTests(unittest.TestCase):
    def test_quadratic_partition_affine_and_continuous_gradient(self):
        x=np.array([[.499,.493,.477],[.5,.5,.5],[.521,.478,.49]])
        for q in (x,x+1e-7):
            c,N,D=sample_basis(q,H,np.zeros(3),'quadratic');nodes=H*c
            np.testing.assert_allclose(np.asarray(N.sum(1)),1,atol=1e-14)
            np.testing.assert_allclose(N@nodes,q,atol=2e-15)
            np.testing.assert_allclose(np.stack([d@nodes for d in D],axis=2),np.broadcast_to(np.eye(3),(len(q),3,3)),atol=1e-13)
        # Cross a quadratic knot with a fixed nodal field, including support changes.
        def value(q):
            c,N,D=sample_basis(q,H,np.zeros(3),'quadratic');v=np.sin(c*.37)
            return N@v,np.stack([d@v for d in D],axis=2)
        q=np.array([[8.5*H,.492,.483]]);eps=1e-7
        _,gradient=value(q)
        for j in range(3):
            delta=np.eye(3)[j]*eps
            a,ga=value(q+delta);b,gb=value(q-delta)
            np.testing.assert_allclose((a-b)/(2*eps),gradient[:,:,j],atol=2e-7)
            self.assertLess(np.max(abs(ga-gb)),2e-5)

    def test_affine_extension_mass_and_support(self):
        s=initial();origin=np.array([.25,.125,-.125])*H
        raw=SpatialGrid(s.x,s.mass,H,origin)
        for kernel in ('q1','quadratic'):
            g=SpatialGrid(s.x,s.mass,H,origin,kernel,'affine_extension')
            self.assertGreater(g.support_diagnostics['constrained_nodes'],0)
            self.assertLess(g.mass_condition,raw.mass_condition)
            np.testing.assert_allclose(np.asarray(g.N.sum(1)),1,atol=2e-14)
            np.testing.assert_allclose(g.N@g.x,s.x,atol=2e-14)
            np.testing.assert_allclose(g.gradient(g.x),np.broadcast_to(np.eye(3),s.F.shape),atol=3e-13)
            v=g.x@np.array([[.1,.2,.3],[0,.3,.1],[.2,0,.1]])+[1,2,3]
            np.testing.assert_allclose(g.project(g.N@v),v,atol=1e-11)
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            SpatialGrid(s.x,s.mass,H,origin,'quadratic','raw')

    def test_residual_radial_derivative_at_frozen_sites(self):
        s=initial();r=np.random.default_rng(9).normal(size=s.v.shape)
        Q,D,_,radius=residual_modes(s.x,s.mass,r)
        def kernel(x):
            distance=np.linalg.norm(x[:,None]-s.x[None],axis=2)/radius
            return np.maximum(1-distance,0)**4*(4*distance+1)
        coef=np.linalg.solve(kernel(s.x),Q);eps=1e-7
        for j in range(3):
            delta=eps*np.eye(3)[j]
            fd=((kernel(s.x+delta)-kernel(s.x-delta))/(2*eps))@coef
            np.testing.assert_allclose(fd,D[j],rtol=2e-6,atol=1e-7)

    def test_residual_enrichment_reproduces_velocity_and_preserves_mass(self):
        s=initial();g=SpatialGrid(s.x,s.mass,H,np.zeros(3),'quadratic','affine_extension')
        v=s.v+.01*np.sin(s.x*31);before=np.linalg.norm(v-g.N@g.project(v))
        self.assertGreater(before,1e-4);g.enrich_velocity(s.x,v)
        self.assertGreater(g.enrichment_modes,0)
        np.testing.assert_allclose(g.N@g.project(v),v,atol=1e-11)
        np.testing.assert_allclose(np.asarray(g.N.sum(1)),1,atol=1e-13)
        np.testing.assert_allclose(g.N@g.x,s.x,atol=1e-13)
        np.testing.assert_allclose(g.gradient(g.x),np.broadcast_to(np.eye(3),s.F.shape),atol=1e-12)
        nodal=g.project(v)
        self.assertAlmostEqual(np.sum(nodal*(g.M@nodal)),np.sum(s.mass[:,None]*v*v),delta=1e-16)

    def test_enriched_potential_gradient_hessian_and_commit(self):
        old=initial();s=ParticleHistory(old.X,old.x,old.v+.005*np.sin(old.x*31),old.F,old.mass,old.volume,old.A)
        model=SpatialMPM(kernel='quadratic',boundary='affine_extension',velocity_enrichment=True)
        sys=model.frozen_step(s);g=sys.grid;v=g.project(s.v)
        a=np.random.default_rng(8).normal(size=v.shape)*.01;dt=.000125;eps=1e-4
        _,r=sys.potential(v,v,dt);ep,rp=sys.potential(v+eps*a,v,dt);em,rm=sys.potential(v-eps*a,v,dt)
        self.assertAlmostEqual((ep-em)/(2*eps),np.sum(r*a),delta=2e-12)
        action=sys.tangent(v,a,dt)
        np.testing.assert_allclose((rp-rm)/(2*eps),action,rtol=2e-6,atol=2e-12)
        np.testing.assert_allclose(sys.whitened_hessian(v,dt)@g.to_y(a).ravel(),g.force_to_y(action).ravel(),rtol=2e-9,atol=2e-12)
        result,stats=model.step(s,dt)
        self.assertLess(stats['prediction_commit_F_absolute'],1e-12)
        self.assertLess(stats['kinetic_relative'],1e-12)
        self.assertLess(abs(stats['projection_delta']),1e-15)
        self.assertLess(abs(stats['energy_budget_residual']),1e-14)

    @unittest.skipUnless(os.environ.get('ANISO_TEST_CUDA'),'real CUDA material audit')
    def test_enriched_gpu_cpu_grid_change(self):
        a=b=initial();options=dict(kernel='quadratic',boundary='affine_extension',velocity_enrichment=True)
        cpu,gpu=SpatialMPM(**options),SpatialMPM(device='cuda:0',**options)
        for step in range(10):
            origin=origin_at(step*.00025,True)
            a,ra=cpu.step(a,.00025,origin);b,rb=gpu.step(b,.00025,origin)
            np.testing.assert_allclose(a.F,b.F,atol=2e-8,rtol=0)
            np.testing.assert_allclose(a.v,b.v,atol=2e-7,rtol=0)
            self.assertLess(rb['affine_gradient_error'],1e-10)

if __name__=='__main__':unittest.main()
