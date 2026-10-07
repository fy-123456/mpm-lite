"""Single bounded material retry, with the whole trial state privately owned."""
import numpy as np
from .material import Material
from .solver import Solver

class MaterialBudgetFailure(RuntimeError):pass

class Transaction:
    def __init__(self,model,geometry,primary=7,reserve=7,*,drained=False,alpha=1.,budget=.03):
        self.model=model;self.geometry=geometry;self.primary=Material(model,primary)
        self.reserve=self.primary if primary==reserve else Material(model,reserve)
        self.options=dict(drained=drained,alpha=alpha);self.budget=budget
    def step(self,old,h,target,*,fault=None):
        digest=old.digest();attempts=[];before_rule=old.rule
        def solve(op):return Solver(self.model,self.geometry,op,**self.options).step(old,h,target)
        trial,row=solve(self.primary)
        if self.reserve is self.primary:
            comparison={};accepted=True
        else:
            low=self.primary.evaluate(trial.q);high=self.reserve.evaluate(trial.q)
            comparison={k:float(np.linalg.norm(low[k]-high[k])/max(np.linalg.norm(high[k]),1e-12)) for k in ('force','weak_moments')}
            accepted=all(e<=self.budget for e in comparison.values())
            attempts.append(dict(rule=self.primary.order,accepted=accepted,comparison=comparison))
        if not accepted:
            if old.digest()!=digest:raise RuntimeError('rejected attempt mutated committed state')
            trial,row=solve(self.reserve);attempts.append(dict(rule=self.reserve.order,accepted=True))
        chosen=self.primary if accepted else self.reserve
        delta=0.
        if before_rule!=chosen.order:
            previous=Material(self.model,before_rule)
            delta=chosen.evaluate(old.q)['U']-previous.evaluate(old.q)['U']
        trial.rule_work=old.rule_work+delta
        if fault:raise RuntimeError('injected after complete retry before commit')
        if old.digest()!=digest:raise RuntimeError('transaction mutated committed state')
        row.update(attempts=attempts,delta_U_rule=delta,cumulative_rule_work=trial.rule_work,rollback_digest=digest,
                   fallback=not accepted,geometry_rule_unchanged=True,mass_unchanged=True)
        return trial,row
