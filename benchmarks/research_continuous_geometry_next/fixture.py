"""Construct the unchanged parent16-cell fixture with the new result cache path."""
import copy
import numpy as np
from .provenance import *
from .physics import baseline_model
from engine.aniso_phase1.research_zero_source_next.coupled import ZeroSourceCoupling,explicit_zero
from engine.aniso_phase1.research_restoring_rt0_next.device_rt0 import DeviceGeometry
from engine.aniso_phase1.research_stabilization_boundary_next.local_geometry import install as local_install
from engine.aniso_phase1.research_pressure_startup_next.reduced_geometry import install as reduced_install
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload

def inherited_setup(run,state):
    p=read(ZERO/'S0/input-contract.json');params=p['parameters'];source=explicit_zero(params['source_density_s_inv']);m,cfg=baseline_model(run,pressure=True)
    c=ZeroSourceCoupling(m,cfg,p['times_s'],method='startup',source_m3_s=source,cuts=p['cuts']['coarse'],alpha=params['alpha'],storage=params['storage'],pressure0=params['pressure0_Pa'],reservoir=params['reservoir_Pa'],mobility=params['mobility_scale']*MOBILITY)
    cap=min(256*2**20,int(.02*m.operator.memory_budget.initial_free));owned=c.state;geo=DeviceGeometry.from_owned(c.geometry,cap_bytes=cap);core=c.core;core.geometry=geo
    core.identity=dict(core.identity,geometry=geo.identity,implementation='current-device-RT0');core.signature=digest(core.identity);c.identity=dict(c.identity,core=core.identity);c.signature=digest(c.identity);owned.child_states['fluid']['model']=core.signature;owned.child_states['explicit_pressure_grid']=c.signature;core._transaction=StateTransaction(owned,validator=c.validate);local_install(c)
    if c.identity!=read(ZERO/'cases/zero-startup-coarse-h/identity.json')['coupling']:raise ValueError('parent physical fixture differs')
    c.restore(state);before=physical_payload(c.state);bridge=reduced_install(c)
    if physical_payload(c.state)!=before:raise ValueError('bounded geometry changed state')
    return c,m,cfg,bridge

def new_setup(run,fine=False,state=None,shared=False):
    p=read(Path(run)/'S0/coupled-protocol.json');params=p['parameters'];times=p['times']['half' if fine else 'h'];m,cfg=baseline_model(run,pressure=True)
    c=ZeroSourceCoupling(m,cfg,times,method='startup',source_m3_s=explicit_zero(params['source_density_s_inv']),cuts=p['cuts'],alpha=params['alpha'],storage=params['storage'],pressure0=params['pressure0_Pa'],reservoir=params['reservoir_Pa'],mobility=params['mobility_scale']*MOBILITY)
    cap=min(256*2**20,int(.02*m.operator.memory_budget.initial_free));owned=c.state;before=physical_payload(owned);geo=DeviceGeometry.from_owned(c.geometry,cap_bytes=cap);core=c.core;core.geometry=geo
    core.identity=dict(core.identity,geometry=geo.identity,implementation='current-device-RT0');core.signature=digest(core.identity);c.identity=dict(c.identity,core=core.identity);c.signature=digest(c.identity);owned.child_states['fluid']['model']=core.signature;owned.child_states['explicit_pressure_grid']=c.signature;core._transaction=StateTransaction(owned,validator=c.validate)
    bridge=local_install(c);bridge=reduced_install(c,bridge)
    if physical_payload(c.state)!=before:raise ValueError('new-grid installation changed initial physics')
    if state is not None:c.restore(state)
    if shared:
        from engine.aniso_phase1.research_boundary_reference_next.geometry import install as shared_install
        bridge=shared_install(c,bridge)
    return c,m,cfg,bridge
