"""Conservative nonuniform grids and unchanged mixed equations."""
import unittest
import numpy as np
from engine.aniso_phase1.research_phase_boundary_next.rt0 import CartesianTopology,CartesianAVF
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import ScaledTensorCellAVF
from benchmarks.research_observable_pressure_next.coupling_study import affine_check
from benchmarks.research_phase_reference_next.coupling_study import quadrature

class CartesianChecks(unittest.TestCase):
    def test_shared_faces_cancel_in_three_directions(self):
        top=CartesianTopology([[0,.2,1],[0,.3,1],[0,.7,1]])
        self.assertEqual(top.cells,8);self.assertEqual(top.nflux,36)
        np.testing.assert_array_equal(top.B[:,top.internal].sum(axis=0),0)
        self.assertAlmostEqual(sum(top.V0),1.)
    def test_source_integrates_overlap_not_array_half(self):
        top=CartesianTopology([[0,.3,.7,1],[0,.2,1],[0,1]])
        self.assertAlmostEqual(top.source().sum(),.0005)
        self.assertTrue(np.any((top.source()>0)&(top.source()<.001*top.V0)))
    def test_affine_anisotropic_flux_and_boundary_work(self):
        t=CartesianTopology([[0,.1,.5,1],[0,.4,1],[0,1]])
        for F in (np.eye(3),np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]])):
            self.assertLess(affine_check(t,F)['error'],1e-9)
    def test_inverse_deformation_rejected(self):
        t=CartesianTopology([[0,.2,1],[0,1],[0,1]]);X,w,ids=quadrature(t)
        with self.assertRaises(ValueError):t.assemble(X,w,ids,np.broadcast_to(np.diag([-1.,1.,1.]),(len(X),3,3)))
    def test_invalid_cuts_and_budget_rejected(self):
        for cuts in ([[0,.4,.4,1],[0,1],[0,1]],[[0,1],[0,float('nan')],[0,1]],[list(range(18)),[0,1],[0,1]]):
            with self.assertRaises(ValueError):CartesianTopology(cuts)
    def test_cuts_owned_and_immutable(self):
        c=np.array([0.,.4,1.]);t=CartesianTopology([c,[0,1],[0,1]]);c[1]=.8
        self.assertEqual(t.cuts[0][1],.4)
        with self.assertRaises(ValueError):t.cuts[0][1]=.8
    def test_general_mixed_equations_inherited(self):
        self.assertIs(CartesianAVF._compute,ScaledTensorCellAVF._compute)

class SegmentChecks(unittest.TestCase):
    def test_order_and_empty_rows_match_sequential_segments(self):
        from engine.aniso_phase1.research_phase_boundary_next.segments import segment_metadata
        rng=np.random.default_rng(902)
        for lengths in ([0,1,127,128,129,0,1000,0],rng.integers(0,800,500)):
            ptr=np.r_[0,np.cumsum(lengths)]
            for chunk in (1,128,257,2**65):
                starts=[];stops=[];rows=[0]
                for a,b in zip(ptr[:-1],ptr[1:]):
                    for start in range(int(a),int(b),chunk):starts.append(start);stops.append(min(start+chunk,int(b)))
                    rows.append(len(starts))
                for actual,expected in zip(segment_metadata(ptr,chunk),(starts,stops,rows)):
                    np.testing.assert_array_equal(actual,expected)
                    self.assertEqual(actual.dtype,np.int32)
        self.assertEqual(segment_metadata(np.array([0],dtype=np.int64))[2].tolist(),[0])
    def test_invalid_csr_and_chunk_are_rejected(self):
        from engine.aniso_phase1.research_phase_boundary_next.segments import segment_metadata
        for ptr in ([],[1,2],[0,3,2],[0,-1],[0,2**31],[0.,1.]):
            with self.assertRaises(ValueError):segment_metadata(np.asarray(ptr))
        for chunk in (0,-1,True,1.5):
            with self.assertRaises(ValueError):segment_metadata(np.array([0,1]),chunk)

if __name__=='__main__':unittest.main()
