"""Recertify an immutable zero-source prefix for a longer residual-budget window."""
import copy
from pathlib import Path
import numpy as np
from .provenance import read,sha,digest
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload


def extended_identity(value):
    if isinstance(value,dict):return {k:extended_identity(v) for k,v in value.items() if k not in ('times_s','theta_schedule','window_s')}
    if isinstance(value,list):return [extended_identity(v) for v in value]
    return value


def mass_budget(V0,capacity,p0,p1,h,window):
    V0=np.asarray(V0);return 1e-10*(V0/V0.sum())*(h/window)+64*np.finfo(float).eps*(.8*V0+np.asarray(capacity)*(abs(np.asarray(p0))+abs(np.asarray(p1))))


def bridge_state(c,state):
    owned=state.clone();before=physical_payload(owned)
    owned.child_states['fluid']['model']=c.core.signature
    owned.child_states['explicit_pressure_grid']=c.signature
    if physical_payload(owned)!=before:raise ValueError('continuation changed physical history')
    c.validate(owned);return owned


def prefix_audit(c,folder):
    folder=Path(folder);ident=read(folder/'identity.json');old=ident['coupling']
    if extended_identity(old)!=extended_identity(c.identity):raise ValueError('physical coupling identity differs beyond registered window extension')
    history=GenerationStore(folder,ident).history()
    if not history or history[0]['state'].step!=0:raise ValueError('incomplete prefix')
    if physical_payload(history[0]['state'])!=physical_payload(c.state):raise ValueError('prefix initial state differs')
    oldcore=old['core'];thetas=np.asarray(oldcore['theta_schedule'])
    if not np.array_equal(thetas,c.core.thetas[:len(thetas)]):raise ValueError('prefix theta changed')
    records=[]
    for i,item in enumerate(history):
        state=item['state'];f=state.child_states['fluid'];bridge_state(c,state)
        if abs(state.time-c.times[i])>8*abs(np.spacing(max(abs(state.time),abs(c.times[i])))):raise ValueError('prefix times differ')
        if np.any(np.asarray(f['cumulative_source_m3'])!=0):raise ValueError('prefix has a volume source')
        if i==0:continue
        row=item['rows'][-1];p0=history[i-1]['state'].child_states['fluid']['pressure_Pa']
        budget=mass_budget(c.geometry.V0,c.core.capacity,p0,f['pressure_Pa'],row['dt'],c.times[-1])
        fraction=float(np.max(abs(np.asarray(row['mass_defect_m3']))/budget))
        conservative=max(fraction,row['true_scaled_residual'])
        if conservative>1 or row['source_work_J']!=0:raise ValueError('prefix fails new residual/source gate')
        if sum(r['numerical_dissipation_J'] for r in item['rows'])!=f['cumulative_numerical_dissipation_J']:raise ValueError('prefix dissipation ledger differs')
        records.append(dict(step=i,new_mass_budget_fraction=fraction,conservative_full_residual_fraction=conservative,state_sha256=sha(item['folder']/'state.json')))
    return history,dict(status='passed_scoped',source=str(folder),identity_sha256=sha(folder/'identity.json'),inherited_steps=len(history)-1,new_steps=len(c.times)-len(history),physical_identity_except_extension_equal=True,only_changed_identity_fields=['times_s','theta_schedule','window_s'],all_physical_payloads_exact=True,source_zero=True,original_rows_preserved=True,records=records)
