import time
from collections import Counter

class BudgetExceeded(RuntimeError):pass

class Budget:
    """One owner for primary/reserve/old-rule, geometry, Jv and wall time."""
    def __init__(self,seconds=600,material_limit=24,clock=time.perf_counter):
        if seconds<=0 or material_limit<1:raise ValueError('positive shared budget required')
        self.clock=clock;self.started=clock();self.deadline=self.started+seconds;self.limit=material_limit
        self.counts=Counter();self.seconds=Counter();self.events=[]
    def check(self):
        if self.clock()>self.deadline:raise BudgetExceeded('shared wall budget exhausted')
    def begin(self,kind,label):
        self.check()
        if kind=='material' and self.counts[kind]>=self.limit:raise BudgetExceeded('shared material budget exhausted')
        self.counts[kind]+=1;self.events.append(dict(kind=kind,label=label,event='begin',at=self.clock()-self.started))
        return self.clock()
    def end(self,kind,label,start,success):
        elapsed=self.clock()-start;self.seconds[kind]+=elapsed
        self.events.append(dict(kind=kind,label=label,event='end',seconds=elapsed,success=success))
        if success:self.check()
    def report(self):return dict(seconds=self.clock()-self.started,counts=dict(self.counts),component_seconds=dict(self.seconds),events=self.events)
