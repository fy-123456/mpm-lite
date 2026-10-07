import unittest,time
from types import SimpleNamespace
import numpy as np
from engine.aniso_phase1.research_startup_substeps_next.schedule import make_times,interval_rows
from engine.aniso_phase1.research_continuous_geometry_next.epoch import extension_times
from engine.aniso_phase1.research_continuous_geometry_next.reference import prolongation,integrate
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction

class Contracts(unittest.TestCase):
    def test_extension_keeps_history_and_physical_nodes(self):
        old=make_times('U4');a=extension_times(old,expected_prefix=old);b=extension_times(old,True,expected_prefix=old)
        self.assertTrue(np.array_equal(a[:29],old));self.assertTrue(np.array_equal(b[:29],old));self.assertEqual(len(a),37);self.assertEqual(len(b),45)
        np.testing.assert_allclose(a[28:],b[28::2],atol=1e-18,rtol=0);self.assertFalse(a.flags.writeable)
    def test_bad_prefix_and_nonzero_origin_rejected(self):
        old=make_times('U4');bad=old.copy();bad[1]*=.9
        with self.assertRaises(ValueError):extension_times(bad,expected_prefix=old)
        with self.assertRaises(ValueError):extension_times(old[4:])
        with self.assertRaises(ValueError):extension_times(np.r_[old[:-1],3e-4])
    def test_partial_engineering_interval_not_published_or_duplicated(self):
        t=make_times('U4');rows=[dict(time=t[i+1],dt=t[i+1]-t[i],z=i+1.) for i in range(5)];out=interval_rows(rows,np.arange(4)*1.25e-5,average=['z'])
        self.assertEqual(len(out),1);self.assertAlmostEqual(out[0]['z'],2.5)
        with self.assertRaises(ValueError):interval_rows(rows[4:],np.arange(4)*1.25e-5,average=['z'])
        with self.assertRaises(ValueError):interval_rows(rows+rows[-1:],np.arange(4)*1.25e-5,average=['z'])
    def test_rt0_injection_conserves_and_reproduces_affine_axis_field(self):
        a=ReferenceTopology(([0,.1,1],[0,1],[0,1]));b=ReferenceTopology(([0,.1,1],[0,.5,1],[0,.5,1]));P,M,Z=restriction(a,b);E=prolongation(a,b)
        np.testing.assert_allclose(Z@E,np.eye(a.nflux),atol=1e-13);np.testing.assert_allclose(a.B@Z,M@b.B,atol=1e-13)
        ca=(2+3*a.centres[np.arange(a.nflux),a.axes])*a.areas;cb=(2+3*b.centres[np.arange(b.nflux),b.axes])*b.areas
        np.testing.assert_allclose(E@ca,cb,atol=1e-13)
        self.assertGreater(np.linalg.norm(E-Z.T),.1)
    def test_streamed_gradient_affine_volume_and_pressure_work(self):
        axes=(np.array([0.,1.]),)*3;X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        parent=SimpleNamespace(edges=axes,p=1,shape=(2,2,2),nodes=lambda q:q,adjoint=lambda f:f)
        reduction=SimpleNamespace(parent=parent,P=np.eye(8),expand=lambda q:q);top=ReferenceTopology(axes)
        q0=X@np.array([[.1,.03,0],[0,-.05,.02],[.01,0,.04]]).T;q1=X@np.array([[.2,-.02,.04],[.03,.05,0],[0,.02,-.03]]).T
        vv=[integrate(reduction,q,top,np.eye(3),3,time.perf_counter()+30) for q in (q0,.5*(q0+q1),q1)]
        G=(vv[0]['gradient']+4*vv[1]['gradient']+vv[2]['gradient'])/6;dv=vv[2]['volume']-vv[0]['volume'];work=np.einsum('kij,ij->k',G,q1-q0)
        np.testing.assert_allclose(work,dv,rtol=1e-12,atol=1e-13);np.testing.assert_allclose(.8*.2*(work-dv),0,atol=1e-13)
        self.assertGreater(np.linalg.eigvalsh(vv[1]['H'])[0],0)

if __name__=='__main__':unittest.main()
