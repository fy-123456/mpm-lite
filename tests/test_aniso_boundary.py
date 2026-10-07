"""Boundary field reproduction, exact constraints, and actual frozen GPU maps."""
import os,unittest
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from engine.aniso_phase1.quadratic import node_patch,node_patch_basis,polynomial
from engine.aniso_phase1.trace_probe import (BlendedMLS,ConstrainedStatic,face_points,grouped_maps,stiffness,lite_values,CORNERS)
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,center_support
from benchmarks.aniso_material_snapshot import make_snapshot


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h=1/16;s=make_snapshot()
        cls.snapshot=MaterialSnapshot(s.X,s.X,np.tile(np.eye(3),(len(s.X),1,1)),s.A,s.volume)
        cls.nodes=np.unique(np.concatenate([key+CORNERS for key,_,_ in center_support(cls.snapshot,cls.h)]),axis=0)*cls.h
        cls.blend=BlendedMLS(cls.nodes,cls.h)

    def test_patch_values_and_derivatives_share_the_production_fit(self):
        x=self.nodes;center=np.array([.4,.5,.5]);q=np.array([[.41,.49,.507],[.393,.515,.501]])
        ids,N,G,_,_=node_patch_basis(x,center,self.h,cKDTree(x),queries=q)
        old_ids,oldG,_,_=node_patch(x,center,self.h,cKDTree(x),queries=q)
        np.testing.assert_array_equal(ids,old_ids);np.testing.assert_array_equal(G,oldG)
        P,_=polynomial((x-center)/self.h,True);expected,_=polynomial((q-center)/self.h,True)
        np.testing.assert_allclose(N@P[ids],expected,atol=1e-11)
        for d in range(3):
            eps=np.eye(3)[d]*1e-7
            plus=node_patch_basis(x,center,self.h,cKDTree(x),queries=q+eps)[1]
            minus=node_patch_basis(x,center,self.h,cKDTree(x),queries=q-eps)[1]
            np.testing.assert_allclose((plus-minus)/2e-7,G[:,:,d],atol=1e-8)

    def test_global_gradient_includes_partition_derivatives(self):
        q=np.array([[.251,.461,.549],[.417,.507,.499],[.746,.544,.463]])
        N,G=self.blend.evaluate(q);coef=np.random.default_rng(10).normal(size=len(self.nodes))
        np.testing.assert_allclose(N.sum(axis=1),1,atol=1e-11);np.testing.assert_allclose(N@self.nodes,q,atol=1e-11)
        for d in range(3):
            np.testing.assert_allclose(G[d].sum(axis=1),0,atol=1e-10)
            np.testing.assert_allclose(G[d]@self.nodes,np.tile(np.eye(3)[d],(len(q),1)),atol=1e-10)
            eps=np.eye(3)[d]*1e-7
            fd=(self.blend.evaluate(q+eps)[0]@coef-self.blend.evaluate(q-eps)[0]@coef)/2e-7
            np.testing.assert_allclose(fd,G[d]@coef,rtol=1e-7,atol=1e-7)

    def test_exact_material_face_clamp_reaction_and_work_balance(self):
        _,V,M,G=grouped_maps(self.snapshot,self.nodes,self.h);K=stiffness(G,V,M)
        C=self.blend.local_traces(face_points(17,.25,6,True));s=ConstrainedStatic(K,C)
        L=self.blend.evaluate(face_points(17,.75))[0]
        from benchmarks.aniso_boundary import load_vector
        f=load_vector(L);u,R=s.solve(f);uv=u.reshape(-1,3)
        check=self.blend.local_traces(face_points(17,.25,7,True))
        self.assertLess(np.max(abs(check@uv)),1e-12)
        np.testing.assert_allclose(R.reshape(-1,3).sum(0),[0,1e-4,0],atol=1e-12)
        self.assertLess(np.linalg.norm(s.Z.T@R)/np.linalg.norm(f),1e-9)
        self.assertAlmostEqual(u@K@u/(f@u),1,places=9)
        # Actual old node constraints do not fix the same material face.
        node=ConstrainedStatic(K,np.eye(len(self.nodes))[self.nodes[:,0]<=.25]);un,_=node.solve(f)
        self.assertGreater(np.max(abs(check@un.reshape(-1,3))),1e-7)

    def test_actual_sparse_maps_with_nonlocal_constraints(self):
        from benchmarks.aniso_boundary import gpu_audit
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        for result in gpu_audit(os.environ.get('ANISO_TEST_DEVICE','cpu')):
            self.assertLess(result['constraint_direction_error'],1e-10)
            self.assertLess(result['potential_fd'],1e-5)
            self.assertLess(result['tangent_fd'],1e-5)
            self.assertLess(result['static_matvec_relative'],1e-9)


if __name__=='__main__':unittest.main()
