"""Cheap tests exercise new contracts, not the unchanged full GPU problem."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_transverse_next.initial import pressure_profile,InitialCoupling,TAG
from engine.aniso_phase1.research_transverse_next.recovery import SafePublication
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from engine.aniso_phase1.research_d.common_state import CommonState,StateTransaction
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_transverse_next.observables import modes,spatial_statistics,face_signals
from tests.research_pressure3d_next.test_multirange_recovery import RecoveryModel

def topology(nz=2):return ReferenceTopology([np.linspace(.125,.875,33),[.375,.5,.625],np.linspace(.375,.625,nz+1)])

class IdentityModel(RecoveryModel):
    identity={'equations':'cpu-test'}
    def __init__(self,p):
        s=CommonState(np.zeros((2,3)),np.zeros((2,3)));s.child_states['fluid']={'pressure_Pa':p.tolist()};super().__init__(s)

class Contracts(unittest.TestCase):
    def test_profiles_modes_and_restriction(self):
        for case,nz,az in [('Y64',1,None),('Y128',2,0.),('YZ128',2,.02)]:
            t=topology(nz);p,d=pressure_profile(case,t);a=modes(t,p)
            self.assertFalse(p.flags.writeable);self.assertAlmostEqual(a['mean_Pa'],.2)
            self.assertAlmostEqual(a['Ay']['value_Pa'],.04);self.assertEqual(a['Az']['resolved'],az is not None)
            if az is not None:self.assertAlmostEqual(a['Az']['value_Pa'],az)
            self.assertEqual(a['Az']['value_Pa'] is None,az is None)
            self.assertAlmostEqual(t.V0@(p-.2),0)
            C=.0002*t.V0;self.assertAlmostEqual(.5*C@(p*p)-.5*C@np.full(t.cells,.2**2),.5*C@((p-.2)**2))
            zero=modes(t,np.full(t.cells,.2));self.assertAlmostEqual(zero['Ay']['value_Pa'],0)
        p64,_=pressure_profile('Y64',topology(1));p128,_=pressure_profile('Y128',topology())
        np.testing.assert_allclose(p64,p128.reshape(32,2,2).mean(axis=2).ravel(),rtol=0,atol=1e-15)
        with self.assertRaises(ValueError):pressure_profile('Y64',topology())

    def test_initial_identity_cross_load_and_payload(self):
        t=topology();p,d=pressure_profile('Y128',t);q,e=pressure_profile('YZ128',t)
        a=InitialCoupling(IdentityModel(p),d);b=InitialCoupling(IdentityModel(q),e)
        with self.assertRaises(ValueError):b.validate(a.state)
        s=a.state;s.child_states['fluid']['pressure_Pa'][0]+=.001
        with self.assertRaises(ValueError):a.validate(s)
        s=a.state;s.child_states[TAG]='foreign'
        with self.assertRaises(ValueError):a.validate(s)
        with tempfile.TemporaryDirectory() as tmp:
            GenerationStore(tmp,a.identity).save(a.state,[])
            with self.assertRaises(ValueError):GenerationStore(tmp,b.identity).load()

    def test_all_spatial_axes_and_oriented_faces(self):
        shape=(2,3,4);X=np.zeros((*shape,3));v=np.arange(72).reshape(*shape,3);w=np.arange(24).reshape(shape)+1
        s=spatial_statistics(X,v,w);np.testing.assert_allclose(s['mean'],np.average(v.reshape(-1,3),axis=0,weights=w.ravel()))
        t=topology();z=np.arange(t.nflux)*1e-9;a=face_signals(t,z)
        self.assertAlmostEqual(sum(a['boundary_outward_m3_s'].values()),np.sum(t.B@z))
        for axis,name in enumerate('xyz'):self.assertTrue(all(t.axes[i]==axis for i in a['internal'][name]['indices']))

    def test_frame_failure_and_nonfinite_do_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=CommonState(np.zeros((2,3)),np.zeros((2,3)));c=RecoveryModel(s);store=GenerationStore(tmp,{'case':'frame'});store.save(s,[]);safe=SafePublication(c,store)
            def broken(state):raise MemoryError('frame device unavailable')
            with self.assertRaises(MemoryError):safe.advance(frame_builder=broken)
            self.assertEqual(store.load()['state'].digest(),s.digest());self.assertEqual(c.state.digest(),s.digest());self.assertTrue(safe.pending)
            with self.assertRaises(ValueError):safe.advance(frame_builder=lambda s:{'x':np.array([np.nan])})
            self.assertFalse(list(Path(tmp).rglob('frame.npz')))
            safe.advance(frame_builder=lambda s:{'step':np.array([s.step])});self.assertEqual(len(store.history()),2)

    def test_frame_pointer_failures_and_lost_ack_once(self):
        for where in ('before_pointer','after_pointer'):
            with tempfile.TemporaryDirectory() as tmp:
                s=CommonState(np.zeros((2,3)),np.zeros((2,3)));c=RecoveryModel(s);store=GenerationStore(tmp,{'case':'pointer'});store.save(s,[]);safe=SafePublication(c,store)
                def fault(stage):
                    if stage==where:raise OSError('injected pointer failure')
                if where=='before_pointer':
                    with self.assertRaises(OSError):safe.advance(frame_builder=lambda s:{'step':np.array([s.step])},inject_store=fault)
                    self.assertEqual(store.load()['state'].step,0);self.assertEqual(len(store.history()),1)
                    safe.advance(frame_builder=lambda s:{'step':np.array([s.step])})
                else:safe.advance(frame_builder=lambda s:{'step':np.array([s.step])},inject_store=fault)
                data=store.load();self.assertEqual(data['state'].step,1)
                with np.load(data['folder']/'frame.npz') as f:self.assertEqual(f['step'][0],1)
                safe.prepare();self.assertEqual(len(store.history()),2)

if __name__=='__main__':unittest.main()
