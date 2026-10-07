"""One authenticated generation owns state, ledger and optional display frame."""
import time
import numpy as np
from engine.aniso_phase1.research_pressure3d_next.recovery import SafePublication as Parent

class SafePublication(Parent):
    def __init__(self,coupling,store):
        super().__init__(coupling,store);self.profile={}

    def _record(self,name,tick):
        v=self.profile.setdefault(name,dict(calls=0,seconds=0.));v['calls']+=1;v['seconds']+=time.perf_counter()-tick

    def advance(self,*,frame_builder=None,inject_step=None,inject_store=None):
        tick=time.perf_counter();old=self.prepare();self._record('prepare',tick)
        candidate=None
        try:
            tick=time.perf_counter();row=self.c.step(inject=inject_step);candidate=self.c.state;self._record('physical_step',tick)
            self.authorized[candidate.digest()]=candidate.clone()
            tick=time.perf_counter();frame=frame_builder(candidate) if frame_builder else None
            if frame is not None and any(not np.isfinite(np.asarray(v)).all() for v in frame.values()):
                raise ValueError('nonfinite frame refused before publication')
            self._record('frame',tick);tick=time.perf_counter()
            self.store.save(candidate,[*old['rows'],row],frame=frame,inject=inject_store);self._record('store',tick)
        except Exception:
            record=self.store.load()
            if record is None:raise ValueError('lost authenticated persistent commit')
            self._restore_known(record)
            if candidate is not None and record['state'].digest()==candidate.digest() and candidate.step==old['state'].step+1:
                return record['rows'][-1]
            raise
        tick=time.perf_counter();self.last=self.store.load();self.authorized={candidate.digest():candidate.clone()};self._record('readback',tick)
        return row
