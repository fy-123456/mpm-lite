import unittest
import numpy as np
from engine.aniso_phase1.tensor_reference import coordinates,axis,gradient

class QuarticReferenceTests(unittest.TestCase):
    def test_quartic_tensor_gradient_exact_and_gauss5_matches6(self):
        edges=[np.array([.125,.25,.265625,.5,.75,.875]),np.array([.375,.390625,.5,.625]),np.array([.375,.5,.609375,.625])];x=np.array(np.meshgrid(*coordinates(edges,4),indexing='ij')).reshape(3,-1).T;u=np.column_stack((x[:,0]**4,x[:,0]**2*x[:,1]**2,x[:,0]*x[:,1]*x[:,2]**4));X=np.random.default_rng(211).uniform([.125,.375,.375],[.875,.625,.625],(71,3));L=gradient(X,edges,4,u);exact=np.zeros_like(L);exact[:,0,0]=4*X[:,0]**3;exact[:,1,0]=2*X[:,0]*X[:,1]**2;exact[:,1,1]=2*X[:,0]**2*X[:,1];exact[:,2,0]=X[:,1]*X[:,2]**4;exact[:,2,1]=X[:,0]*X[:,2]**4;exact[:,2,2]=4*X[:,0]*X[:,1]*X[:,2]**3;np.testing.assert_allclose(L,exact,atol=3e-11,rtol=2e-11)
        for e in edges:
            for a,b in zip(axis(e,4,5)[1],axis(e,4,6)[1]):self.assertLess(np.linalg.norm((a-b).data),2e-10)
if __name__=='__main__':unittest.main()
