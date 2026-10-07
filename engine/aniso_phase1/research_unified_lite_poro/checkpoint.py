"""Atomic, identity-bound complete particle/fluid checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from .space import Space
from .model import Bridge,Particles,State


def config(bridge):
    return dict(space=bridge.space.identity(),groups=bridge.groups,order=bridge.order,rule=bridge.rule,
                physics='Hencky10-20-fiber200-alpha0.8-storage0.01-full-mobility-v1')


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def save(bridge,path):
    path=Path(path);s=bridge.state;conf=config(bridge)
    payload={f'particle_{k}':getattr(s.particles,k) for k in s.particles.__dataclass_fields__}
    payload.update(q=s.q,v=s.v,p=s.p,content=s.content)
    meta=dict(config=conf,signature=signature(conf),state_digest=s.digest(),time=s.time,step=s.step,
              boundary_volume=s.boundary_volume,dissipation=s.dissipation)
    payload['metadata']=np.array(json.dumps(meta));temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**payload);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def load(path,expected=None):
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']));conf=meta['config']
        if signature(conf)!=meta['signature'] or (expected is not None and signature(expected)!=meta['signature']):
            raise ValueError('foreign or corrupt checkpoint identity')
        spec=conf['space'];space=Space(tuple(spec['shape']),spec['bubbles'],spec['lengths'])
        b=Bridge.__new__(Bridge);b.space=space;b.groups=conf['groups'];b.order=conf['order'];b.rule=conf.get('rule','moments');b.attempts=0;b.accepted=0
        particles=Particles(**{k:d['particle_'+k].copy() for k in Particles.__dataclass_fields__})
        b.state=State(particles,*(d[k].copy() for k in ('q','v','p','content')),
                       *(meta[k] for k in ('time','step','boundary_volume','dissipation')))
    if b.state.digest()!=meta['state_digest']:raise ValueError('checkpoint payload digest differs')
    op,q,v,_=b.prepare();V=op.geometry(q,False)[0]
    expected_content=op.alpha*(V-op.top.V0)+op.capacity*b.state.p
    if not np.allclose(expected_content,b.state.content,rtol=1e-8,atol=1e-10):
        raise ValueError('checkpoint fluid content differs')
    return b
