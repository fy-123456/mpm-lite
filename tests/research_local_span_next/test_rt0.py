"""Independent analytic flow and assembly invariants; no long trajectories."""
import unittest
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_local_span_next.rt0 import TwoCellTopology,MOBILITY
from benchmarks.research_local_span_next.coupling_study import quadrature,affine_check

class TensorFlowTests(unittest.TestCase):
    def test_single_owned_internal_face_and_reference_volume(self):
        t=TwoCellTopology([[0,1],[-.1,.1],[-.05,.05]])
        self.assertEqual(t.internal.tolist(),[1]);self.assertEqual(t.B[0,1],-t.B[1,1]);np.testing.assert_allclose(t.V0,[.01,.01])
        with self.assertRaises(ValueError):t.B[0,1]=0
    def test_rectangular_affine_crossflow_and_boundary_power(self):
        t=TwoCellTopology([[0,1],[-.1,.1],[-.05,.05]])
        F=np.array([[1.03,.08,.01],[.01,.98,.03],[.02,0.,1.01]])
        r=affine_check(t,F)
        self.assertLess(r['flux_error'],1e-10);self.assertLess(abs(r['closure']),1e-10);self.assertGreater(r['min_H_eigenvalue'],0)
        z=np.asarray(r['flux']);self.assertTrue(all(np.any(abs(z[t.axes==axis])>1e-8) for axis in range(3)))
    def test_tensor_coupling_produces_crossflow_for_transverse_gradient(self):
        t=TwoCellTopology([[0,2],[0,1],[0,1]]);X,w,c=quadrature(t);H,_=t.assemble(X,w,c,np.broadcast_to(np.eye(3),(len(w),3,3)));B=t.B;grad=np.array([0.,.03,0.]);gb=t.boundary_term(.1+t.centres@grad)
        A=np.block([[H,-B.T],[B,np.zeros((2,2))]]);z=la.solve(A,np.r_[-gb,0.,0.])[:11]
        expected=t.areas*(-MOBILITY@grad)[t.axes];np.testing.assert_allclose(z,expected,atol=1e-10,rtol=1e-7)
        self.assertGreater(abs(z[1]),1e-5)
    def test_invalid_mobility_and_inverted_geometry_rejected(self):
        t=TwoCellTopology([[0,2],[0,1],[0,1]]);X,w,c=quadrature(t);F=np.broadcast_to(np.eye(3),(len(w),3,3)).copy()
        with self.assertRaisesRegex(ValueError,'SPD'):t.assemble(X,w,c,F,np.diag([.1,-.1,.1]))
        F[:,0,0]=-1
        with self.assertRaisesRegex(ValueError,'positive-J'):t.assemble(X,w,c,F)
if __name__=='__main__':unittest.main()
