import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_local_q3 import assemble,gradient
from benchmarks.aniso_v21_common import BASE,controlled_case,hessian
from engine.aniso_phase1.tensor_reference import interpolate,coordinates
from engine.aniso_phase1.high_order_space import BoxElastic,local_box,scalar_grams,reference_stress_load,HighOrderPotential
from engine.aniso_phase1.stress_local_space import reduced_solve,LocalPotential

class HighOrderSpaceTests(unittest.TestCase):
    def test_local_operator_matches_global_principal_submatrix_and_natural_faces(self):
        edges=[np.array([.125,.25,.375,.5,.625,.75,.875]),np.array([.375,.4375,.5,.5625,.625]),np.array([.375,.5,.625])];rng=np.random.default_rng(44)
        for p in (2,3,4):
            X,K=assemble(edges,p,hessian('F45'));n=len(X)
            for lo,hi in [([.25,.375,.375],[.5,.5,.625]),([.375,.4375,.375],[.625,.5625,.625])]:
                idx,op,_=local_box(edges,p,dict(lo=lo,hi=hi),hessian('F45'));ids=np.concatenate([idx+j*n for j in range(3)]);v=rng.normal(size=len(ids));A=K[ids][:,ids]
                np.testing.assert_allclose(op.free_apply(v),A@v,atol=2e-10,rtol=2e-10)
                rhs=rng.normal(size=len(ids));w,r=op.correction(rhs);self.assertLess(r['relative_residual'],1e-8);np.testing.assert_allclose(A@w,rhs,atol=2e-8,rtol=2e-8)
    def test_original_tensor_grams_reproduce_assembled_stiffness(self):
        edges=[np.array([.25,.5,.75]),np.array([.375,.625]),np.array([.375,.625])];rng=np.random.default_rng(7)
        for p in (3,4):
            X,K=assemble(edges,p,hessian('F45'));T=rng.normal(size=(len(X),8));G=scalar_grams(edges,p,T);H=hessian('F45').reshape(3,3,3,3)
            KK=np.block([[sum(H[a,i,b,j]*G[i,j] for i in range(3) for j in range(3)) for b in range(3)] for a in range(3)])
            J=la.block_diag(T,T,T);np.testing.assert_allclose(KK,J.T@(K@J),atol=2e-10,rtol=2e-12)
    def test_stress_load_matches_independent_same_mesh_squared_stress_operator(self):
        edges=[np.array([.25,.5,.75]),np.array([.375,.625]),np.array([.375,.625])];rng=np.random.default_rng(2)
        for p in (3,4):
            H=hessian('F45');X,K=assemble(edges,p,H.T@H);u=rng.normal(scale=.001,size=(len(X),3));f,norm=reference_stress_load(edges,p,(edges,p,u),H)
            np.testing.assert_allclose(f,K@u.T.ravel(),atol=2e-9,rtol=2e-11);self.assertAlmostEqual(norm,u.T.ravel()@(K@u.T.ravel()),places=10)

class HigherOrderPotentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s,cls.e,_,_,_=controlled_case()
        with np.load(BASE/'v19/space/reconstruction16.npz') as z:cls.edges=[z[f'axis{k}'] for k in range(3)];cls.oldA=z['A']
    def test_rigid_affine_and_bending_preserved_at_higher_orders(self):
        x=self.s.Y;poly=lambda z:np.column_stack((np.ones(len(z)),z,z*z,z[:,0]*z[:,1],z[:,0]*z[:,2],z[:,1]*z[:,2]));b=np.column_stack((-x[:,0]*x[:,1],.5*x[:,0]**2,np.zeros(len(x))))
        for p in (3,4):
            A=interpolate(self.edges,2,self.oldA,self.edges,p);X=np.array(np.meshgrid(*coordinates(self.edges,p),indexing='ij')).reshape(3,-1).T;np.testing.assert_allclose(A@poly(x),poly(X),atol=2e-11)
            P=np.array([[.283,.431,.48],[.482,.56,.533],[.721,.599,.51]]);L=gradient(P,self.edges,p,A@b);exact=np.zeros_like(L);exact[:,0,0]=-P[:,1];exact[:,0,1]=-P[:,0];exact[:,1,0]=P[:,0];np.testing.assert_allclose(L,exact,atol=3e-11)
    def test_nonlinear_same_energy_force_tangent_and_finite_rotation(self):
        p=3;A=interpolate(self.edges,2,self.oldA,self.edges,p);X=np.array(np.meshgrid(*coordinates(self.edges,p),indexing='ij')).reshape(3,-1).T;z=np.maximum(0,(X[:,0]-.25)*(.75-X[:,0]));Z=np.column_stack((z,z*(X[:,1]-.5),z*(X[:,2]-.5)));Z/=la.norm(Z,axis=0);pot=HighOrderPotential(self.edges,p,A,Z,self.e.Ks,self.e.params,self.e.A[0]);Y=np.vstack((self.s.Y,np.zeros((3,3))));rng=np.random.default_rng(70);Y+=rng.normal(scale=2e-5,size=Y.shape);d=rng.normal(size=Y.shape);d/=la.norm(d);eps=1e-6;a=pot.evaluate(Y,d);b=pot.evaluate(Y+eps*d);c=pot.evaluate(Y-eps*d)
        self.assertLess(abs((b['U']-c['U'])/(2*eps)-np.sum(a['force']*d)),1e-8);np.testing.assert_allclose((b['force']-c['force'])/(2*eps),a['tangent_action'],atol=3e-8,rtol=2e-5)
        R=la.expm(np.array([[0,-.3,.2],[.3,0,.1],[-.2,-.1,0]]));rot=Y@R.T;rot[:len(self.s.Y)]+=[.03,-.02,.01];b=pot.evaluate(rot);self.assertLess(abs(b['U']-a['U']),1e-11);np.testing.assert_allclose(b['force'],a['force']@R.T,atol=2e-10)
    def test_new_q2_quadrature_potential_matches_previous_native_implementation(self):
        A=self.oldA;Z=np.empty((len(A),0));Y=self.s.Y+np.random.default_rng(5).normal(scale=1e-5,size=self.s.Y.shape);d=np.random.default_rng(6).normal(size=Y.shape)
        a=HighOrderPotential(self.edges,2,A,Z,self.e.Ks,self.e.params,self.e.A[0]).evaluate(Y,d);b=LocalPotential(self.edges,A,Z,self.e.Ks,self.e.params,self.e.A[0]).evaluate(Y,d)
        self.assertAlmostEqual(a['U'],b['U'],places=12);np.testing.assert_allclose(a['force'],b['force'],atol=2e-11);np.testing.assert_allclose(a['tangent_action'],b['tangent_action'],atol=2e-10)
if __name__=='__main__':unittest.main()
