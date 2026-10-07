import unittest,json
from types import SimpleNamespace
import numpy as np
import warp as wp
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology
from engine.aniso_phase1.research_restoring_rt0_next.device_rt0 import DeviceRT0

def geometry():
    top=CartesianTopology(([0.,.5,1.],[0.,1.],[0.,1.]));points=(np.array([.1,.4,.6,.9]),np.array([.25,.75]),np.array([.25,.75]));X=np.stack(np.meshgrid(*points,indexing='ij'),axis=-1).reshape(-1,3)
    return SimpleNamespace(op=SimpleNamespace(device='cuda:0'),topology=top,count=len(X),shape=tuple(map(len,points)),points=points,X=X,total_weights=np.full(len(X),1/len(X)),cell_ids=top.locate(X),mobility=np.array([[.1,.02,.01],[.02,.08,-.015],[.01,-.015,.06]]),identity=dict(test='two-cell-tensor'))

class DeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):wp.init()
    def test_nonlinear_tensor_and_current_state(self):
        g=geometry();a=DeviceRT0(g,cap_bytes=2**20,chunk=3)
        F=np.broadcast_to(np.array([[1.05,.12,0.],[.02,.98,.07],[0.,.03,1.02]]),(g.count,3,3)).copy()
        x,j=a.assemble(wp.array(F,dtype=wp.mat33d,device='cuda:0'));y,k=g.topology.assemble(g.X,g.total_weights,g.cell_ids,F,g.mobility)
        np.testing.assert_allclose(x,y,rtol=2e-5,atol=1e-8);self.assertAlmostEqual(j,k)
        F[:,0,0]+=.1;z,_=a.assemble(wp.array(F,dtype=wp.mat33d,device='cuda:0'));self.assertGreater(np.linalg.norm(z-x),.01)
        self.assertGreaterEqual(np.linalg.eigvalsh(z)[0],0.)
    def test_owned_static_arrays(self):
        g=geometry();a=DeviceRT0(g,cap_bytes=2**20);F=wp.array(np.broadcast_to(np.eye(3),(g.count,3,3)).copy(),dtype=wp.mat33d,device='cuda:0');old=a.assemble(F)[0]
        g.total_weights[:]*=2;g.points[0][:]+=.01;g.mobility[:]*=2
        np.testing.assert_array_equal(a.assemble(F)[0],old)
    def test_invalid_current_J(self):
        g=geometry();a=DeviceRT0(g,cap_bytes=2**20);F=np.broadcast_to(np.eye(3),(g.count,3,3)).copy();F[0,0,0]=-.1
        with self.assertRaisesRegex(ValueError,'invalid'):a.assemble(wp.array(F,dtype=wp.mat33d,device='cuda:0'))
    def test_memory_cap(self):
        with self.assertRaises(MemoryError):DeviceRT0(geometry(),cap_bytes=1)
    def test_wrong_tensor_layout(self):
        g=geometry();g.X=g.X[::-1].copy()
        with self.assertRaisesRegex(ValueError,'tensor grid'):DeviceRT0(g,cap_bytes=2**20)
    def test_wrong_F_shape(self):
        a=DeviceRT0(geometry(),cap_bytes=2**20)
        with self.assertRaisesRegex(ValueError,'layout'):a.assemble(wp.array(np.eye(3)[None],dtype=wp.mat33d,device='cuda:0'))
    def test_identity_json_roundtrip(self):
        a=DeviceRT0(geometry(),cap_bytes=2**20)
        self.assertEqual(a.identity,json.loads(json.dumps(a.identity)))

if __name__=='__main__':unittest.main()
