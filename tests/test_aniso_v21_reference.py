import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_local_q3 import assemble,solve as assembled_solve,gradient
from engine.aniso_phase1.tensor_reference import TensorElastic,solve,interpolate,coordinates

class TensorReferenceTests(unittest.TestCase):
    def setUp(self):self.e=[np.array([.25,.3125,.5,.75]),np.array([.375,.4375,.625]),np.array([.375,.5,.625])]
    def test_operator_matches_independent_assembled_all_directions(self):
        for p in (2,3,4):
            for label in ('ISO','F0','F45','F90'):
                H=hessian(label);x,K=assemble(self.e,p,H);op=TensorElastic(self.e,p,H);v=np.random.default_rng(10).normal(size=3*len(x));np.testing.assert_allclose(op.apply(v),K@v,atol=2e-11,rtol=3e-13)
    def test_nonuniform_q4_polynomial_and_prolongation(self):
        e=[np.union1d(x,(x[:-1]+x[1:])/2) for x in self.e];axes=coordinates(self.e,3);x=np.array(np.meshgrid(*axes,indexing='ij')).reshape(3,-1).T;u=np.column_stack((x[:,0]**3,x[:,0]*x[:,1]*x[:,2],x[:,1]**2*x[:,2]));v=interpolate(self.e,3,u,e,4);X=np.random.default_rng(11).uniform([.25,.375,.375],[.75,.625,.625],(50,3));np.testing.assert_allclose(gradient(X,e,4,v),gradient(X,self.e,3,u),atol=2e-11)
    def test_preconditioner_symmetric_positive_and_rigid_kernel(self):
        op=TensorElastic(self.e,3,hessian('F45'));n=3*np.prod(op.free_shape);rng=np.random.default_rng(12);v,w=rng.normal(size=(2,n));self.assertGreater(v@op.precondition(v),0);self.assertLess(abs(v@op.precondition(w)-w@op.precondition(v))/la.norm(v)/la.norm(w),1e-10)
        x=np.array(np.meshgrid(*coordinates(self.e,3),indexing='ij')).reshape(3,-1).T;W=np.array([[0,.2,-.1],[-.2,0,.3],[.1,-.3,0]]);self.assertLess(la.norm(op.apply((x@W.T+.03).T.ravel())),1e-10)
    def test_solution_work_identity_and_unshifted_positivity(self):
        e=[np.union1d(self.e[0],[.125,.875]),*self.e[1:]]
        for p in (2,3,4):
            x,u,r=assembled_solve(e,p,hessian('F45'));v,s=solve(e,p,hessian('F45'));self.assertTrue(s['passed']);self.assertLess(s['work_identity_relative'],1e-8);np.testing.assert_allclose(u,v,atol=1e-11,rtol=1e-8);self.assertAlmostEqual(r['reaction_N'],s['reaction_N'],places=10)
            xa,K=assemble(self.e,p,hessian('F45'));free=np.tile((xa[:,0]>.25)&(xa[:,0]<.75),3);self.assertGreater(la.eigvalsh(K[free][:,free].toarray(),subset_by_index=[0,0])[0],0)
if __name__=='__main__':unittest.main()
