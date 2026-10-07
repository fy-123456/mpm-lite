"""Explicit-grid owner around unchanged solid AVF and pressure midpoint physics."""
import copy
import numpy as np
from engine.aniso_phase1.research_cost_phase_next.pressure import MultiCellAVF
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction


class ExplicitGridCoupling:
    def __init__(self,model,config,times,*,source_m3_s=0.,state=None,**kwargs):
        ts=np.asarray(times,dtype=np.float64)
        if ts.ndim!=1 or len(ts)<2 or not np.isfinite(ts).all() or ts[0]!=0 or np.any(np.diff(ts)<=0):
            raise ValueError('finite increasing explicit grid starting at zero required')
        self.times=ts.copy();self.times.setflags(write=False)
        self.core=MultiCellAVF(model,config,state=state,**kwargs)
        self.source=np.broadcast_to(source_m3_s,(self.core.cells,)).astype(float).copy()
        if not np.isfinite(self.source).all():raise ValueError('finite owned source required')
        self.source.setflags(write=False)
        self.identity=dict(schema='phase-stress-explicit-pressure-grid-v1',core=self.core.identity,
            times_s=ts.tolist(),source_m3_s=self.source.tolist(),source_semantics='constant rate, integrated with each actual interval')
        self.signature=digest(self.identity)
        if state is None:
            s=self.core.state;s.child_states['explicit_pressure_grid']=self.signature
            self.core._transaction=StateTransaction(s,validator=self.validate)
        self.validate(self.core.state)

    def __getattr__(self,name):return getattr(self.core,name)

    @property
    def state(self):return self.core.state

    def validate(self,state):
        self.core.validate(state)
        if state.child_states.get('explicit_pressure_grid')!=self.signature:raise ValueError('foreign pressure time grid/source history')
        if state.step>=len(self.times) or state.step<0 or abs(state.time-self.times[state.step])>1e-12:
            raise ValueError('pressure checkpoint time/index differs from actual grid')

    def step(self,dt=None,*,inject=None,external_force=None):
        self.validate(self.state);index=self.state.step
        if index>=len(self.times)-1:raise ValueError('pressure time grid exhausted')
        h=float(self.times[index+1]-self.times[index])
        if dt is not None and abs(dt-h)>1e-12:raise ValueError('requested pressure dt differs from frozen actual interval')
        def check(stage,state):
            if inject is not None:inject(stage,state)
            self.validate(state)
        return self.core.step(h,source_m3_s=self.source,external_force=external_force,inject=check)

    def restore(self,state):
        self.validate(state);self.core._transaction=StateTransaction(state,validator=self.validate);self.geometry.cache.clear()


def advance_publish(c,store,rows,*,frame_builder=None,inject_step=None,inject_store=None):
    """Recover exactly one published generation, otherwise restore all owned fields."""
    base=c.state
    try:
        row=c.step(inject=inject_step)
        store.save(c.state,[*rows,row],frame=frame_builder(c.state) if frame_builder else None,inject=inject_store)
        return row
    except Exception:
        data=store.load()
        if data is not None and data['state'].digest()==c.state.digest() and c.state.step==base.step+1:
            c.validate(data['state']);return copy.deepcopy(data['rows'][-1])
        c.restore(base);raise
