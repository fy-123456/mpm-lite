"""Construct an owned implementation branch after authenticating the old state."""
import copy
from engine.aniso_phase1.research_candidate_observable_next.fixture import ObservableCoupling
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_d.identity import digest
from .device_rt0 import DeviceGeometry

def physical_payload(state):
    data=state.to_dict();data['child_states'].pop('explicit_pressure_grid',None)
    data['child_states']['fluid'].pop('model',None)
    return data

def construct(model,config,protocol,grid,*,fine=False,state=None,backend='original',bridge=False):
    if backend not in ('original','device'):raise ValueError('unknown geometry backend')
    original=ObservableCoupling(model,config,protocol,grid,fine=fine)
    if state is not None and (backend=='original' or bridge):original.validate(state)
    if backend=='original':
        if state is not None:original.restore(state)
        return original,None
    fresh=original.state;core=original.core;physical_identity=copy.deepcopy(original.identity)
    cap=min(256*2**20,int(.02*model.operator.memory_budget.initial_free))
    geometry=DeviceGeometry.from_owned(core.geometry,cap_bytes=cap);core.geometry=geometry
    core.identity=dict(core.identity,geometry=geometry.identity,implementation='current-device-RT0-chunk-v1');core.signature=digest(core.identity)
    original.identity=dict(schema='restoring-RT0-fixture-v1',physical_parent=physical_identity,core=core.identity);original.signature=digest(original.identity)
    fresh.child_states['fluid']['model']=core.signature;fresh.child_states['explicit_pressure_grid']=original.signature
    core._transaction=StateTransaction(fresh,validator=original.validate)
    record=None
    if state is not None:
        target=state.clone()
        if bridge:
            before=physical_payload(state);target.child_states['fluid']['model']=core.signature;target.child_states['explicit_pressure_grid']=original.signature
            if physical_payload(target)!=before:raise ValueError('implementation bridge changed physical history')
            record=dict(old_digest=state.digest(),new_digest=target.digest(),physical_payload_sha256=digest(before),physical_state_exact=True,old_owner_validated=True)
        original.restore(target)
    return original,record
