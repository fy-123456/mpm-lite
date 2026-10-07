"""Complete bounded histories, selected rule and immutable physical identity."""
import json
import numpy as np
from .model import BoundedBridge,State,Particles,Space,SCHEMA,BUDGET
from engine.aniso_phase1.research_common_kinematics_next.checkpoint import config as old_config,signature

def config(b):
    c=old_config(b);c.update(schema=SCHEMA,primary_order=b.primary_order,reserve_order=6,
        geometry_order=4,mass_order=4,budget=BUDGET,history='continuously advanced from initialization')
    return c

def save(b,path):
    s=b.state;c=config(b);data={'particle_'+k:getattr(s.particles,k) for k in Particles.__dataclass_fields__}
    scalars={}
    for k in State.__dataclass_fields__:
        if k=='particles':continue
        value=getattr(s,k)
        if isinstance(value,np.ndarray):data[k]=value
        else:scalars[k]=value
    np.savez_compressed(path,**data,metadata=np.array(json.dumps(dict(config=c,signature=signature(c),
        state_digest=s.digest(),scalars=scalars),sort_keys=True)))

def load(path,expected=None):
    with np.load(path,allow_pickle=False) as d:
        m=json.loads(str(d['metadata']));c=m['config']
        if c.get('schema')!=SCHEMA or signature(c)!=m['signature'] or (expected is not None and signature(expected)!=m['signature']):
            raise ValueError('foreign or corrupt bounded-history identity')
        b=BoundedBridge.__new__(BoundedBridge);s=c['space']
        b.space=Space(tuple(s['shape']),s['bubbles'],s['lengths']);b.rule=c['rule'];b.order=4;b.primary_order=c['primary_order'];b.attempts=0;b.accepted=0
        if config(b)!=c:raise ValueError('unsupported physical identity')
        part=Particles(**{k:d['particle_'+k].copy() for k in Particles.__dataclass_fields__})
        fields={k:d[k].copy() for k in State.__dataclass_fields__ if k!='particles' and k not in m['scalars']}
        b.state=State(part,**fields,**m['scalars'])
    if b.state.digest()!=m['state_digest']:raise ValueError('state digest differs')
    if b.state.active_order not in (b.primary_order,6):raise ValueError('unsupported active rule')
    op,res,v,fit=b.prepare_pair();V=op.geometry(np.zeros_like(v),False)[0]
    if not np.allclose(op.alpha*(V-op.top.V0)+op.capacity*b.state.p,b.state.content,rtol=1e-8,atol=1e-10):
        raise ValueError('fluid content mismatch')
    return b
