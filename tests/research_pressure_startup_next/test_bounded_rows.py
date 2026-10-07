import unittest
import numpy as np
import scipy.sparse as sp
from engine.aniso_phase1.research_pressure_startup_next.reduced_geometry import bounded_segments

class BoundedRows(unittest.TestCase):
    def test_exact_action_on_compact_supported_vector(self):
        rng=np.random.default_rng(18);m=sp.random(7,25,density=.5,random_state=rng,format='csr');m.sort_indices()
        x=rng.normal(size=(25,3));x[:6]=0;x[19:]=0
        (starts,stops,ptr),used=bounded_segments(m.indptr,m.indices,6,19,3)
        out=np.zeros((7,3))
        for i in range(7):
            for seg in range(ptr[i],ptr[i+1]):
                for k in range(starts[seg],stops[seg]):out[i]+=m.data[k]*x[m.indices[k]]
        np.testing.assert_allclose(out,m@x,rtol=1e-13,atol=1e-13);self.assertLess(used,m.nnz)
    def test_empty_rows_and_right_open_bound(self):
        ptr=np.array([0,0,3,3]);cols=np.array([0,4,7]);(starts,stops,rows),used=bounded_segments(ptr,cols,4,7)
        self.assertEqual(used,1);np.testing.assert_array_equal(rows,[0,0,1,1]);np.testing.assert_array_equal(starts,[1]);np.testing.assert_array_equal(stops,[2])
    def test_bad_bounds(self):
        for lo,hi,chunk in [(1,1,128),(-1,2,128),(0,2,0)]:
            with self.assertRaises(ValueError):bounded_segments(np.array([0,1]),np.array([0]),lo,hi,chunk)

if __name__=='__main__':unittest.main()
