"""Noncontiguous sparse support, and CPU-only authenticated recovery contracts."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_pressure3d_next.multirange import transverse_segments
from engine.aniso_phase1.research_pressure3d_next.recovery import SafePublication
from engine.aniso_phase1.research_d.common_state import CommonState,StateTransaction
from benchmarks.research_sequential_next.checkpoint import GenerationStore

class RecoveryModel:
    def __init__(self,s):
        self.core=self;self.block=False;self.geometry=type('Geometry',(),{'cache':{}})();self._transaction=StateTransaction(s,validator=self.validate)
    @property
    def state(self):return self._transaction.snapshot()
    def validate(self,s):
        if self.block:raise MemoryError('unavailable device')
    def step(self,inject=None):
        t=self._transaction.begin_trial();t.state.step+=1;t.state.time+=1
        if inject:inject('before_commit',t.state)
        self._transaction.commit(t);return dict(step=self.state.step,time=self.state.time)

class Contracts(unittest.TestCase):
    def test_asymmetric_device_gather_and_adjoint_pairing(self):
        import warp as wp
        from engine.aniso_phase1.research_pressure3d_next.geometry import gather,indices
        wp.init()
        rng=np.random.default_rng(151);shape=(4,5,7);bounds=((1,4),(1,3),(2,6));ids=indices(bounds,shape)
        values=rng.normal(size=(140,3,3));weights=rng.uniform(.1,1,size=140);n=len(ids)
        source=wp.array(values,dtype=wp.mat33d,device='cpu');w=wp.array(weights,dtype=wp.float64,device='cpu')
        target=wp.zeros(n+3,dtype=wp.mat33d,device='cpu');tw=wp.zeros(n+3,dtype=wp.float64,device='cpu')
        wp.launch(gather,dim=n,inputs=[source,w,target,tw,1,1,2,2,4,5,7],device='cpu')
        np.testing.assert_array_equal(target.numpy()[:n],values[ids]);np.testing.assert_array_equal(tw.numpy()[:n],weights[ids])
        np.testing.assert_array_equal(target.numpy()[n:],0)
        local=rng.normal(size=(n,3,3));direct=np.zeros_like(values);direct[ids]=local*weights[ids,None,None]
        self.assertAlmostEqual(float(np.sum(target.numpy()[:n]*local*tw.numpy()[:n,None,None])),float(np.sum(values*direct)),places=12)

    def test_sparse_runs_and_empty_rows(self):
        cols=np.r_[np.arange(60),[2,5,10,14,23,44]];ptr=np.array([0,60,60,66]);shape=(3,4,5);bounds=((0,3),(1,3),(1,4))
        (starts,stops,rows),used=transverse_segments(ptr,cols,bounds,shape,chunk=3)
        for row in range(3):
            actual=np.concatenate([np.arange(starts[k],stops[k]) for k in range(rows[row],rows[row+1])]) if rows[row+1]>rows[row] else np.array([],dtype=int)
            ids=np.arange(ptr[row],ptr[row+1]);x=cols[ids]//20;y=cols[ids]//5%4;z=cols[ids]%5
            expected=ids[(x>=0)&(x<3)&(y>=1)&(y<3)&(z>=1)&(z<4)]
            np.testing.assert_array_equal(actual,expected)
        self.assertEqual(used,int(sum(stops-starts)))

    def test_pending_restore_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=CommonState(np.zeros((2,3)),np.zeros((2,3)));c=RecoveryModel(s);store=GenerationStore(tmp,{'model':'test'});store.save(s,[]);safe=SafePublication(c,store)
            def fault(where,state):c.block=True;raise MemoryError('injected')
            with self.assertRaises(MemoryError):safe.advance(inject_step=fault)
            self.assertEqual(c.state.digest(),s.digest());self.assertTrue(safe.pending)
            with self.assertRaises(MemoryError):safe.prepare()
            c.block=False;safe.advance();self.assertEqual(c.state.step,1);self.assertFalse(safe.pending)
            bad=GenerationStore(tmp,{'model':'foreign'})
            with self.assertRaises(ValueError):bad.load()
            path=store.load()['folder']/'state.json';path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):safe.prepare()

    def test_after_pointer_is_committed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=CommonState(np.zeros((2,3)),np.zeros((2,3)));c=RecoveryModel(s);store=GenerationStore(tmp,{'model':'test'});store.save(s,[]);safe=SafePublication(c,store)
            def fault(where):
                if where=='after_pointer':raise OSError('published but acknowledgement lost')
            row=safe.advance(inject_store=fault);self.assertEqual(row['step'],1);self.assertTrue(safe.pending)
            safe.advance();self.assertEqual(len(store.history()),3);self.assertEqual(c.state.step,2)

if __name__=='__main__':unittest.main()
