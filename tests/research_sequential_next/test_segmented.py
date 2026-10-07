"""Independent sparse algebra checks, including empty and long rows."""
import os
import unittest
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedCSR


class SegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir=os.environ.get('MPM_NEXT_WARP_CACHE','/tmp/mpm-next-test-warp')

    def test_long_and_empty_rows_match_scipy_and_the_adjoint_identity(self):
        rng=np.random.default_rng(381);matrix=rng.normal(size=(7,1051));matrix[2]=0.;matrix[4,40:]=0.
        matrix=sp.csr_matrix(matrix);operator=SegmentedCSR(matrix,'cpu')
        x=rng.normal(size=(1051,3));u=rng.normal(size=(7,3))
        y,shape=operator.apply(wp.array(x.ravel(),dtype=wp.float64,device='cpu'),x.shape)
        actual=y.numpy().reshape(shape)
        np.testing.assert_allclose(actual,matrix@x,rtol=1e-11,atol=1e-11)
        self.assertAlmostEqual(float(np.sum(u*actual)),float(np.sum((matrix.T@u)*x)),places=10)
        with self.assertRaises(ValueError):operator.apply(wp.zeros(2,dtype=wp.float64,device='cpu'),(2,1))

    def test_all_zero_matrix_and_general_axis_fallback(self):
        zero=SegmentedCSR(sp.csr_matrix((3,9)),'cpu')
        y,shape=zero.apply(wp.ones(27,dtype=wp.float64,device='cpu'),(9,3))
        np.testing.assert_array_equal(y.numpy().reshape(shape),np.zeros((3,3)))
        matrix=np.array([[1.,-2.,0.],[0.,1.,3.]])
        x=np.arange(12,dtype=float).reshape(2,3,2);op=SegmentedCSR(matrix,'cpu',chunk=1)
        y,shape=op.apply(wp.array(x.ravel(),dtype=wp.float64,device='cpu'),x.shape,axis=1)
        np.testing.assert_allclose(y.numpy().reshape(shape),np.einsum('ij,ajc->aic',matrix,x))


if __name__=='__main__':unittest.main()
