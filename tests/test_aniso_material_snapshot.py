"""Independent material response, partition/cubature and frozen-state checks."""
import unittest
import numpy as np
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,center_support,moments,response,compare_snapshot,moment_points,particle_response
from engine.aniso_phase1.constitutive import energy,pk1
from engine.aniso_phase1 import AnisotropicMaterialParams
from benchmarks.aniso_material_snapshot import make_snapshot,analytic_bend_F
from engine.aniso_phase1.rotation_probe import axis_rotation


class MaterialSnapshotTests(unittest.TestCase):
    def setUp(self):self.params=AnisotropicMaterialParams(10,20,200)

    def test_batch_material_and_moments_match_individual_law_and_derivative(self):
        rng=np.random.default_rng(33);F=np.eye(3)+rng.normal(size=(7,3,3))*.03
        a=rng.normal(size=(7,3));a/=np.linalg.norm(a,axis=1)[:,None];A=np.einsum('pi,pj->pij',a,a)
        s=MaterialSnapshot(rng.random((7,3)),rng.random((7,3)),F,A,np.ones(7))
        E,P,_=particle_response(s,self.params)
        np.testing.assert_allclose(E,[energy(f,t,self.params) for f,t in zip(F,A)],atol=1e-12)
        np.testing.assert_allclose(P,[pk1(f,t,self.params) for f,t in zip(F,A)],atol=1e-11)
        w=rng.random(7);w/=w.sum();A2,A4=moments(A,w);e,p,_=response(F[:1],A2,A4,self.params)
        self.assertAlmostEqual(e[0],sum(wi*energy(F[0],t,self.params) for wi,t in zip(w,A)),places=10)
        np.testing.assert_allclose(p[0],sum(wi*pk1(F[0],t,self.params) for wi,t in zip(w,A)),atol=1e-11)
        d=rng.normal(size=(1,3,3));eps=1e-6
        de=(response(F[:1]+eps*d,A2,A4,self.params)[0]-response(F[:1]-eps*d,A2,A4,self.params)[0])/(2*eps)
        np.testing.assert_allclose(de,np.sum(p*d,axis=(1,2)),rtol=1e-6,atol=1e-7)

    def test_positive_cubature_matches_cloud_first_second_moments(self):
        rng=np.random.default_rng(41);x=rng.normal(size=(23,3));w=rng.random(23);w/=w.sum();q=moment_points(x,w)
        np.testing.assert_allclose(q.mean(axis=0),w@x,atol=1e-12)
        np.testing.assert_allclose(q.T@q/8,np.einsum('p,pi,pj->ij',w,x,x),atol=1e-12)

    def test_common_F_crossed_fibers_and_partition_are_exact(self):
        s=make_snapshot(kind='tension',field='crossed');original=[a.copy() for a in (s.x,s.F,s.A,s.X,s.volume)]
        r,_=compare_snapshot(s,1/16,self.params)
        for name in r['methods']:
            self.assertLess(abs(r['methods'][name]['energy_relative_error']),1e-8)
            self.assertLess(r['methods'][name]['center_P_rms_relative_error'],1e-8)
        self.assertLess(r['partition_volume_error'],1e-14)
        for a,b in zip(original,(s.x,s.F,s.A,s.X,s.volume)):np.testing.assert_array_equal(a,b)
        with self.assertRaises(ValueError):s.x[0,0]=0

    def test_bending_error_is_quadrature_not_just_reference_fit(self):
        s=make_snapshot();r,_=compare_snapshot(s,1/16,self.params,analytic_bend_F);m=r['methods']
        self.assertLess(m['center']['energy_relative_error'],-.1)
        self.assertGreater(m['grid8']['energy_relative_error'],.1)
        self.assertGreater(m['analytic_grid8']['energy_relative_error'],.1)
        self.assertLess(abs(m['moment8']['energy_relative_error']),.01)
        self.assertLess(abs(m['reconstructed_particles']['energy_relative_error']),.01)
        np.testing.assert_allclose(analytic_bend_F(s.x),s.F,atol=1e-12)

    def test_particle_reference_rotates_covariantly_and_rejects_bad_states(self):
        s=make_snapshot(field='smooth');Q=axis_rotation(.7)
        t=MaterialSnapshot((s.x-.5)@Q.T+.5,s.X,Q@s.F,s.A,s.volume)
        e,p,tau=particle_response(s,self.params);er,pr,tr=particle_response(t,self.params)
        np.testing.assert_allclose(er,e,atol=1e-11)
        np.testing.assert_allclose(pr,Q@p,atol=1e-10)
        np.testing.assert_allclose(tr,Q@tau@Q.T,atol=1e-10)
        bad=s.F.copy();bad[0]*=-1
        with self.assertRaisesRegex(ValueError,'det F'):MaterialSnapshot(s.x,s.X,bad,s.A,s.volume)


if __name__=='__main__':unittest.main()
