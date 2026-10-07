"""Exact reduced-coordinate adapter, bounded response cache and call budget."""
import time
import numpy as np
from engine.aniso_phase1.research_b.tensor import TensorMaterialOperator,TensorRule

class Material:
    def __init__(self,model,order=7):
        self.model=model;self.order=order
        self.op=TensorMaterialOperator(model.space,TensorRule.uniform(model.space.edges,order))
        self.cache={};self.calls=0;self.seconds=0.;self.deadline=float('inf');self.limit=10**9
    def evaluate(self,q,direction=None):
        key=q.tobytes()
        if direction is None and key in self.cache:return self.cache[key]
        if time.perf_counter()>self.deadline or self.calls>=self.limit:raise RuntimeError('material evaluation budget exhausted')
        start=time.perf_counter();r=self.model.r
        raw=self.op.evaluate(r.expand(q),None if direction is None else r.velocity(direction))
        self.calls+=1;self.seconds+=time.perf_counter()-start
        raw['force']=r.P.T@raw['force']
        if direction is not None:raw['tangent_action']=r.P.T@raw['tangent_action']
        if direction is None:
            if len(self.cache)>=3:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=raw
        return raw
