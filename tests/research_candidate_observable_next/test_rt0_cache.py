import unittest
from benchmarks.research_candidate_observable_next import __doc__
import numpy as np
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology
from engine.aniso_phase1.research_candidate_observable_next.rt0_cache import StaticRT0Metadata
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from benchmarks.research_phase_reference_next.coupling_study import quadrature

class StaticMetadataTests(unittest.TestCase):
    def setUp(self):
        self.top=CartesianTopology([[.125,.5,.875],[.375,.625],[.375,.625]])
        self.X,self.w,self.ids=quadrature(self.top)
        self.cache=StaticRT0Metadata(self.top,self.X,self.w,self.ids,MOBILITY,max_bytes=256*2**20,owner_identity='test')
    def test_current_F_and_full_off_diagonal_tensor(self):
        for f in (np.eye(3),np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]])):
            F=np.broadcast_to(f,(len(self.X),3,3));a,j=self.top.assemble(self.X,self.w,self.ids,F,MOBILITY);b,k=self.cache.assemble(self.X,self.w,self.ids,F,MOBILITY)
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12);self.assertEqual(j,k)
    def test_changed_input_rejected(self):
        F=np.broadcast_to(np.eye(3),(len(self.X),3,3))
        for X,w,ids,mob in ((self.X+.0001,self.w,self.ids,MOBILITY),(self.X,self.w*2,self.ids,MOBILITY),(self.X,self.w,self.ids,MOBILITY*2)):
            with self.assertRaisesRegex(ValueError,'foreign or changed'):self.cache.assemble(X,w,ids,F,mob)
    def test_owned_and_bounded(self):
        self.assertLessEqual(self.cache.bytes,self.cache.max_bytes)
        self.assertEqual(self.cache.bytes,len(self.w)*32+72)
        for chunks in self.cache.parts:
            for arrays in chunks:
                for value in arrays:self.assertFalse(value.flags.writeable)
        with self.assertRaises(MemoryError):StaticRT0Metadata(self.top,self.X,self.w,self.ids,MOBILITY,max_bytes=1,owner_identity='test')
    def test_reject_invalid_current_F(self):
        F=np.broadcast_to(np.eye(3),(len(self.X),3,3)).copy();F[0,0,0]=-.1
        with self.assertRaises(ValueError):self.cache.assemble(self.X,self.w,self.ids,F,MOBILITY)

if __name__=='__main__':unittest.main()
