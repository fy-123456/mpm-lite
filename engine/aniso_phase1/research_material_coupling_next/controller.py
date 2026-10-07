"""One run entry owns all rules, attempts, budget and private trial states."""
from .operators import Material,compare
from .solver import Solver

class Controller:
    def __init__(self,model,geometry,budget,*,primary=7,reserve=7,options=None,journal=None):
        self.model=model;self.geometry=geometry;self.budget=budget;self.primary=primary;self.reserve=reserve
        self.options={} if options is None else dict(options);self.materials={};self.solvers={};self.attempts=[];self.journal=journal
        self._solver(primary);self._solver(reserve)
    def _material(self,order):
        if order not in self.materials:self.materials[order]=Material(self.model,order,self.budget)
        return self.materials[order]
    def _solver(self,order):
        if order not in self.solvers:self.solvers[order]=Solver(self.model,self.geometry,self._material(order),**self.options)
        return self.solvers[order]
    @property
    def identity(self):
        return dict(**self._solver(self.primary).identity,material_policy=dict(primary=self.primary,reserve=self.reserve,limit=.03,check='endpoint regional force/stress/energy; optional initial guard'))
    def _log(self,event):
        self.attempts.append(event)
        if self.journal:self.journal(dict(attempts=self.attempts,budget=self.budget.report()))
    def step(self,old,h,target,*,initial_guard=False,fault=None):
        digest=old.digest();self._log(dict(stage='begin',time=old.time,step=old.step,digest=digest,primary=self.primary))
        active=self.primary;rejected=False;report=None
        try:
            if self.primary!=self.reserve and initial_guard:
                report=compare(self.model,self._material(active).evaluate(old.q),self._material(self.reserve).evaluate(old.q))
                self._log(dict(stage='initial-material-check',rule=active,comparison=report))
                if not report['accepted']:active=self.reserve;rejected=True
            trial,row=self._solver(active).step(old,h,target)
            if active!=self.reserve:
                report=compare(self.model,self._material(active).evaluate(trial.q),self._material(self.reserve).evaluate(trial.q))
                self._log(dict(stage='endpoint-material-check',rule=active,comparison=report))
                if not report['accepted']:
                    if old.digest()!=digest:raise RuntimeError('trial mutated committed state')
                    rejected=True;active=self.reserve
                    trial,row=self._solver(active).step(old,h,target)
            delta=0.
            if old.rule!=active:
                delta=self._material(active).evaluate(old.q)['U']-self._material(old.rule).evaluate(old.q)['U']
            trial.rule_work=old.rule_work+delta
            if fault:raise RuntimeError('injected before controller commit')
            self.budget.check()
            if old.digest()!=digest:raise RuntimeError('controller changed old state')
            row.update(active_rule=active,fallback=rejected,material_comparison=report,delta_U_rule=delta,cumulative_rule_work=trial.rule_work,
                controller_identity=self.identity,attempt_count=len(self.attempts),shared_budget=self.budget.report())
            self._log(dict(stage='accept',active_rule=active,fallback=rejected,new_digest=trial.digest(),old_digest=digest))
            return trial,row
        except Exception as e:
            self._log(dict(stage='failure',type=type(e).__name__,reason=str(e),old_unchanged=old.digest()==digest))
            raise
