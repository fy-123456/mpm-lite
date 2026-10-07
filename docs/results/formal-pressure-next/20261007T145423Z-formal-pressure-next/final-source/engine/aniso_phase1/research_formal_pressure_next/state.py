"""Finite original-material poro state and identity-bound checkpoint."""
from dataclasses import dataclass
import json
import numpy as np
from engine.aniso_phase1.research_unified_lite_poro.model import sha_arrays

@dataclass
class State:
    q:np.ndarray
    velocity:np.ndarray
    p:np.ndarray
    time:float=0.
    step:int=0
    boundary_volume:float=0.
    dissipation:float=0.
    external_work:float=0.
    rule:int=7
    rule_work:float=0.
    def digest(self):
        return sha_arrays(self.q,self.velocity,self.p,np.array([self.time,self.step,self.boundary_volume,self.dissipation,self.external_work,self.rule,self.rule_work]))

def validate(model,s):
    model.validate(s)
    if s.p.shape!=(2,) or not np.isfinite(s.p).all():raise ValueError('two finite material-volume pressures required')
    if not np.isfinite([s.boundary_volume,s.dissipation,s.external_work,s.rule_work]).all():raise ValueError('invalid cumulative ledger')
    if s.rule<1 or int(s.rule)!=s.rule:raise ValueError('invalid rule identity')

def save(s,path,identity):
    meta=dict(identity=identity,time=s.time,step=s.step,boundary_volume=s.boundary_volume,dissipation=s.dissipation,external_work=s.external_work,rule=s.rule,rule_work=s.rule_work,digest=s.digest())
    np.savez_compressed(path,q=s.q,velocity=s.velocity,p=s.p,metadata=np.array(json.dumps(meta,sort_keys=True)))

def load(path,model,identity):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']))
        if meta['identity']!=identity:raise ValueError('foreign formal coupled checkpoint identity')
        s=State(z['q'].copy(),z['velocity'].copy(),z['p'].copy(),**{k:meta[k] for k in ('time','step','boundary_volume','dissipation','external_work','rule','rule_work')})
    validate(model,s)
    if s.digest()!=meta['digest']:raise ValueError('corrupt formal coupled checkpoint')
    return s
