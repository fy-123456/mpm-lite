"""Small GPU witnesses for buffer sharing, row segmentation and private outputs."""
import unittest
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.aniso_phase1.research_d.stage2.gpu_space import CSR
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedCSR
from engine.aniso_phase1.research_local_span_next.reuse import ReusedSegmentedCSR
from benchmarks.research_local_span_next import config

class ReuseTests(unittest.TestCase):
    def test_same_segment_sum_shared_inputs_private_outputs(self):
        m=sp.csr_matrix(np.arange(3*385,dtype=float).reshape(3,385)/1000)
        original=CSR(m,'cuda:0');borrowed=ReusedSegmentedCSR.from_csr(original,m);reference=SegmentedCSR(m,'cuda:0')
        self.assertIs(borrowed.val,original.val);self.assertIs(borrowed.ptr,original.ptr)
        values=original.val.numpy().copy();x=wp.array(np.arange(385*3,dtype=float)/13,dtype=wp.float64,device='cuda:0')
        a,shape=borrowed.apply(x,(385,3));b,_=reference.apply(x,(385,3));c,_=borrowed.apply(x,(385,3))
        np.testing.assert_array_equal(a.numpy(),b.numpy());np.testing.assert_array_equal(original.val.numpy(),values)
        self.assertNotEqual(a.ptr,c.ptr);self.assertEqual(shape,(3,3))
    def test_shape_nan_and_chunk_rejected(self):
        m=sp.eye(3,format='csr');original=CSR(m,'cuda:0')
        for source,chunk in [(sp.eye(4,format='csr'),128),(sp.csr_matrix(np.full((3,3),np.nan)),128),(m,0)]:
            with self.assertRaises(ValueError):ReusedSegmentedCSR.from_csr(original,source,chunk)
    def test_runtime_switch_is_explicit_boolean(self):
        cfg=config.make(1.,space={'path':'p','sha256':'s'},reuse_transpose_buffers=True)
        self.assertIs(cfg['implementation']['reuse_transpose_buffers'],True)
        with self.assertRaisesRegex(ValueError,'boolean'):config.make(1.,space={'path':'p','sha256':'s'},reuse_transpose_buffers='true')
if __name__=='__main__':unittest.main()
