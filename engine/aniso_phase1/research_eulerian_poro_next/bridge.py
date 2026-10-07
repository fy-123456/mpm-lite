"""Material-rule descendant; inherited transaction and inertia are unchanged."""
import time
from engine.aniso_phase1.research_unified_lite_poro.model import Bridge, unload
from .materials import make_operator


class ImprovedBridge(Bridge):
    def __init__(self,space,ppc=3,rule='director-log'):
        super().__init__(space,ppc,rule='fixed-positive')
        if rule not in ('director-log','director-or-positive','fixed-positive'):
            raise ValueError('unknown material rule')
        self.rule=rule

    def prepare(self):
        start=time.perf_counter();s=self.state
        q,v,fit=unload(self.space,s.particles)
        fit['q_recovery']=float(abs(q-s.q).max());fit['v_recovery']=float(abs(v-s.v).max())
        if max(fit['q_recovery'],fit['v_recovery'],fit['weighted_history_residual'])>1e-7:
            raise ValueError('incompatible particle history; no projection/reset')
        op,info=make_operator(self.space,s.particles,self.rule,self.order)
        fit.update(material_rule=info,seconds=time.perf_counter()-start)
        return op,q,v,fit
