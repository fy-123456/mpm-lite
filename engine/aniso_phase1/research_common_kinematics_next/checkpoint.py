"""Complete point histories and physical identity; no migration of old F."""
import json,hashlib
import numpy as np
from .mapping import SCHEMA
from .model import CommonBridge,State,Particles,Space,PARAMS,MOBILITY

def config(b):
    return dict(schema=SCHEMA,space=b.space.identity(),rule=b.rule,order=b.order,
        params=PARAMS,alpha=.8,storage=.01,rho=1.,mobility=MOBILITY.tolist(),time_layer='committed-step-end',
        pressure_control='fixed material',velocity='PIC/Hermite; no APIC state in modal solver')

def signature(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()

def save(b,path):
    s=b.state;conf=config(b)
    data={'particle_'+k:getattr(s.particles,k) for k in Particles.__dataclass_fields__}
    scalars={};arrays={}
    for k in State.__dataclass_fields__:
        if k=='particles':continue
        if isinstance(getattr(s,k),np.ndarray):arrays[k]=getattr(s,k)
        else:scalars[k]=getattr(s,k)
    np.savez_compressed(path,**data,**arrays,metadata=np.array(json.dumps(dict(config=conf,
        signature=signature(conf),state_digest=s.digest(),scalars=scalars),sort_keys=True)))

def load(path,expected=None):
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']));c=meta['config']
        if signature(c)!=meta['signature'] or (expected is not None and signature(expected)!=meta['signature']):
            raise ValueError('foreign or corrupt checkpoint identity')
        b=CommonBridge.__new__(CommonBridge);s=c['space']
        b.space=Space(tuple(s['shape']),s['bubbles'],s['lengths']);b.rule=c['rule'];b.order=c['order']
        b.attempts=0;b.accepted=0
        if config(b)!=c or c['schema']!=SCHEMA:raise ValueError('unsupported physical/time identity')
        p=Particles(**{k:d['particle_'+k].copy() for k in Particles.__dataclass_fields__})
        fields={k:d[k].copy() for k in ('qx','qF','qv','p','content','last_increment')}
        b.state=State(p,**fields,**meta['scalars'])
    if b.state.digest()!=meta['state_digest']:raise ValueError('checkpoint state digest differs')
    op,v,fit=b.prepare();V=op.geometry(np.zeros_like(v),False)[0]
    if not np.allclose(op.alpha*(V-op.top.V0)+op.capacity*b.state.p,b.state.content,rtol=1e-8,atol=1e-10):
        raise ValueError('checkpoint fluid content mismatch')
    return b
