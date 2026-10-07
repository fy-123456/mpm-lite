"""Positive paired samples, conditional moments, invariance and frozen response."""
import unittest
import numpy as np
from engine.aniso_phase1 import AnisotropicMaterialParams
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,center_support,response,particle_response
from engine.aniso_phase1.joint_sampling import partition_material,build_rule,compare_joint,METHODS
from engine.aniso_phase1.rotation_probe import axis_rotation
from benchmarks.aniso_material_snapshot import make_snapshot


class JointSamplingTests(unittest.TestCase):
    def setUp(self):self.params=AnisotropicMaterialParams(10,20,200)

    def test_positive_volume_partition_paired_directions_and_unoriented_groups(self):
        s=make_snapshot(field='smooth');_,ids,w=max(center_support(s,1/16),key=lambda g:len(g[1]))
        for mode in METHODS:
            x,W,A2,A4=build_rule(s,ids,w,1/16,mode)
            self.assertTrue(np.all(W>0));self.assertAlmostEqual(W.sum(),w.sum(),places=14)
            if mode.startswith('paired'):
                self.assertLessEqual(len(W),int(mode[6:]))
                for p,a in zip(x,A2):
                    selected=np.flatnonzero(np.all(s.x==p,axis=1));self.assertEqual(len(selected),1)
                    np.testing.assert_array_equal(a,s.A[selected[0]])
            else:
                np.testing.assert_allclose(np.einsum('p,pij->ij',W,A2),np.einsum('p,pij->ij',w,s.A[ids]),atol=1e-14)
                a=s.A[ids].reshape(-1,9)
                np.testing.assert_allclose(np.einsum('p,pij->ij',W,A4),np.einsum('p,pi,pj->ij',w,a,a),atol=1e-14)

    def test_grouped_common_F_matches_arbitrary_direction_mixture_and_virtual_work(self):
        rng=np.random.default_rng(11);n=39;X=rng.uniform(.4,.6,(n,3));F=np.array([[1.04,.03,0],[0,.98,.01],[0,0,1.01]])
        x=(X-.5)@F.T+.5;a=rng.normal(size=(n,3));a/=np.linalg.norm(a,axis=1)[:,None]
        s=MaterialSnapshot(x,X,np.broadcast_to(F,(n,3,3)),np.einsum('pi,pj->pij',a,a),rng.uniform(.1,1,n))
        ep,pp,_=particle_response(s,self.params)
        for mode in ('group2x8','group4x8','group8x8'):
            _,W,A2,A4=build_rule(s,np.arange(n),s.volume,.1,mode);Fs=np.broadcast_to(F,(len(W),3,3));e,p,_=response(Fs,A2,A4,self.params)
            np.testing.assert_allclose(W@e,s.volume@ep,rtol=1e-10)
            np.testing.assert_allclose(np.einsum('p,pij->ij',W,p),np.einsum('p,pij->ij',s.volume,pp),rtol=1e-10,atol=1e-10)
            d=rng.normal(size=Fs.shape);eps=1e-6
            fd=(W@response(Fs+eps*d,A2,A4,self.params)[0]-W@response(Fs-eps*d,A2,A4,self.params)[0])/(2*eps)
            self.assertAlmostEqual(fd/np.einsum('p,pij,pij->',W,p,d),1,places=5)

    def test_rule_rigid_motion_keeps_pairing_and_positive_weights(self):
        s=make_snapshot(field='smooth');Q=axis_rotation(.63);shift=np.array([.01,.02,0.])
        t=MaterialSnapshot(s.x@Q.T+shift,s.X,Q@s.F,s.A,s.volume);_,ids,w=max(center_support(s,1/16),key=lambda g:len(g[1]))
        for mode in METHODS:
            x,W,A,M=build_rule(s,ids,w,1/16,mode);xr,Wr,Ar,Mr=build_rule(t,ids,w,1/16,mode)
            np.testing.assert_allclose(W,Wr,atol=1e-14);np.testing.assert_allclose(A,Ar,atol=1e-14)
            # Covariance square roots may permute samples in degenerate eigenspaces.
            np.testing.assert_allclose(W@xr,W@(x@Q.T+shift),atol=1e-12)
            np.testing.assert_allclose(np.einsum('p,pi,pj->ij',W,xr,xr),np.einsum('p,pi,pj->ij',W,x@Q.T+shift,x@Q.T+shift),atol=1e-12)

    def test_smooth_bending_improves_energy_and_stress_without_mutation(self):
        s=make_snapshot(field='smooth');before=[a.copy() for a in (s.x,s.X,s.F,s.A,s.volume)];r,_=compare_joint(s,1/16,self.params)
        for name in ('paired8','paired16','group4x8'):
            self.assertLess(abs(r['methods'][name]['energy_relative_error']),.025)
            self.assertLess(r['methods'][name]['center_P_rms_relative_error'],.025)
        for a,b in zip(before,(s.x,s.X,s.F,s.A,s.volume)):np.testing.assert_array_equal(a,b)
        self.assertLess(r['max_rule_volume_error'],1e-14)
        self.assertGreater(r['min_sample_weight'],0)


if __name__=='__main__':unittest.main()
