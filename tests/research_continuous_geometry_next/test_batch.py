import unittest
import numpy as np
import warp as wp
from engine.aniso_phase1.research_continuous_geometry_next.geometry import contract

class Batch(unittest.TestCase):
    def test_tail_batch_zero_support_and_owned_copy(self):
        wp.init();P=np.array([[1.,0.,2.],[0.,0.,-1.],[.3,0.,.2],[.7,0.,.8]]);grad=np.arange(4*4*3,dtype=float).reshape(4,4,3);out=wp.full(4*3*3,-123.,dtype=wp.float64,device='cpu');pd=wp.array(P.ravel(),dtype=wp.float64,device='cpu');gd=wp.array(grad.ravel(),dtype=wp.float64,device='cpu');wp.launch(contract,dim=(3,9),inputs=[pd,gd,out,4,3],device='cpu');v=out.numpy().reshape(4,3,3)
        np.testing.assert_allclose(v[:3],np.einsum('ij,bik->bjk',P,grad[:3]),rtol=1e-13,atol=1e-13);self.assertTrue(np.all(v[3]==-123));self.assertTrue(np.all(v[:3,1]==0));copy=v.copy();copy[:]=0;self.assertTrue(np.any(out.numpy()!=0))

if __name__=='__main__':unittest.main()
