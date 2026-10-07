"""Bind explicit research pressure parameters to owned current-geometry AVF state."""
import copy
import numpy as np
from engine.aniso_phase1.research_pressure_window_next.coupled import WindowCoupling
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY


def parameters(protocol):
    p=copy.deepcopy(protocol['parameters'])
    for key in ('alpha','storage','pressure0_Pa','reservoir_Pa','mobility_scale','source_density_s_inv'):
        if key not in p or not np.isfinite(p[key]):raise ValueError('finite explicit fixture parameter required: '+key)
    if p['alpha']!=.8 or p['storage']<=0 or p['mobility_scale']<=0 or p['pressure0_Pa']<0 or p['reservoir_Pa']<0:raise ValueError('invalid positive research fixture')
    return p


class ObservableCoupling(WindowCoupling):
    def __init__(self,model,config,protocol,grid,*,fine=False,state=None):
        p=parameters(protocol);ts=np.asarray(protocol['times_s'],dtype=float)
        if fine:ts=np.sort(np.r_[ts,.5*(ts[:-1]+ts[1:])])
        from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology
        top=CartesianTopology(protocol['cuts'][grid]);source=top.source(p['source_density_s_inv'])
        # Build the parent in its own valid initial identity, then atomically bind
        # all explicit fixture inputs before restoring any descendant checkpoint.
        super().__init__(model,config,ts,cuts=protocol['cuts'][grid],alpha=p['alpha'],storage=p['storage'],pressure0=p['pressure0_Pa'],reservoir=p['reservoir_Pa'],mobility=p['mobility_scale']*MOBILITY,source_m3_s=source)
        self.identity=dict(schema='observable-fixture-v1',parent=self.identity,parameters=p,grid=grid,time_refinement=2 if fine else 1,protocol_sha256=digest(protocol))
        self.signature=digest(self.identity)
        initial=self.core.state;initial.child_states['explicit_pressure_grid']=self.signature
        self.core._transaction=StateTransaction(initial,validator=self.validate)
        if state is not None:self.restore(state)
