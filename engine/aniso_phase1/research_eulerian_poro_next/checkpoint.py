"""Identity-bound descendant restore, retaining every inherited state field."""
import json
import numpy as np
from engine.aniso_phase1.research_unified_lite_poro.checkpoint import save,config,signature
from engine.aniso_phase1.research_unified_lite_poro.model import Particles,State
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from .bridge import ImprovedBridge


def load(path,expected=None):
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']));conf=meta['config']
        if signature(conf)!=meta['signature'] or (expected is not None and signature(expected)!=meta['signature']):
            raise ValueError('foreign or corrupt checkpoint identity')
        if conf['rule'] not in ('director-log','director-or-positive','fixed-positive'):
            raise ValueError('unknown descendant rule')
        spec=conf['space'];b=ImprovedBridge.__new__(ImprovedBridge)
        b.space=Space(tuple(spec['shape']),spec['bubbles'],spec['lengths'])
        b.rule=conf['rule'];b.order=conf['order'];b.groups=conf['groups'];b.attempts=0;b.accepted=0
        particles=Particles(**{k:d['particle_'+k].copy() for k in Particles.__dataclass_fields__})
        b.state=State(particles,*(d[k].copy() for k in ('q','v','p','content')),
                     *(meta[k] for k in ('time','step','boundary_volume','dissipation')))
    if b.state.digest()!=meta['state_digest']:raise ValueError('checkpoint state digest differs')
    if config(b)!=conf:raise ValueError('unsupported checkpoint physical configuration')
    op,q,v,_=b.prepare();V=op.geometry(q,False)[0]
    if not np.allclose(op.alpha*(V-op.top.V0)+op.capacity*b.state.p,b.state.content,rtol=1e-8,atol=1e-10):
        raise ValueError('checkpoint fluid content mismatch')
    return b
