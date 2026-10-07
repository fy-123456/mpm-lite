"""A fixed material rule per AVF trial, with an owned full-rule retry proposal."""
from __future__ import annotations
import copy
from dataclasses import dataclass
import numpy as np
from ..research_d.common_state import StateTransaction,CommonState
from ..research_d.identity import digest
from ..research_d.stage2.contracts import array_digest
from .integrator import ValidatedAVF,StepRejected


@dataclass(frozen=True)
class RuleProposal:
    parent_state_sha256:str
    source_model_sha256:str
    target_model_sha256:str
    source_rule:str
    target_rule:str
    reason:str

    def value(self):return dict(self.__dict__)


class FixedRuleRetry:
    """One parent transaction owns both attempts; the successful one commits.

    Every state retains its actual model/rule identity. Portable checkpoints
    must bind this controller's whitelist identity, rather than pretending a
    changed material rule is compatible with a fixed-rule case.
    """
    def __init__(self,compressed,full,config,state=None):
        self.compressed=compressed;self.full=full;self.config=copy.deepcopy(config);self.failures=[]
        if compressed.identity['material']==full.identity['material']:raise ValueError('distinct compressed/full rules required')
        for key in ('model','boundary','device','dtype'):
            if compressed.identity[key]!=full.identity[key]:raise ValueError('rule switch changes physical model: '+key)
        if not np.array_equal(compressed.M,full.M):raise ValueError('rule switch changes mass')
        self.models={m.signature:m for m in [compressed,full]}
        self.identity=dict(schema='fixed-rule-retry-v1',allowed_models={m.signature:m.identity for m in [compressed,full]},
            mass_sha256=array_digest(full.M),config=digest(config),policy='fixed trial, full retry, full sticky after success')
        initial=compressed.rest() if state is None else state.clone()
        self._transaction=StateTransaction(initial,validator=self.validate)

    @property
    def state(self):return self._transaction.snapshot()

    def validate(self,state):
        matches=[m for m in self.models.values() if state.child_states.get('identity')==m.identity]
        if len(matches)!=1:raise ValueError('state model is outside the fixed-rule whitelist')
        matches[0].validate(state,material=True)
        return True

    def step(self,dt,*,prepare_children=None,validate_trial=None,inject=None):
        trial=self._transaction.begin_trial();base=trial.state.clone();parent_id=base.digest()
        active=next(m for m in self.models.values() if m.identity==base.child_states['identity'])
        candidates=[active] if active is self.full else [active,self.full]
        try:
            for index,model in enumerate(candidates):
                start=base.clone();proposal=None;energy_shift=0.
                if index:
                    proposal=RuleProposal(parent_id,active.signature,model.signature,active.rule.signature,model.rule.signature,
                        self.failures[-1]['reason'])
                    energy_shift=float(model.evaluate(base.q)['U']-active.evaluate(base.q)['U'])
                    cumulative=base.child_states.get('cumulative_abs_rule_switch_error_J',0.)+abs(energy_shift)
                    budget=self.config['acceptance']['energy_fraction']*self.config['acceptance']['energy_scale_J']
                    if not np.isfinite(energy_shift) or cumulative>budget:raise StepRejected('material-rule migration energy exceeds frozen budget')
                    start.child_states['identity']=copy.deepcopy(model.identity)
                    start.child_states['B_rule_proposal']=proposal.value()
                    start.child_states['cumulative_abs_rule_switch_error_J']=cumulative
                cfg=copy.deepcopy(self.config);cfg['material_order']=model.rule.orders[0]
                stepper=ValidatedAVF(model,cfg,start)
                try:
                    def hook(where,state):
                        if inject is not None:inject(index,where,state)
                    row=stepper.step(dt,prepare_children=prepare_children,validate_trial=validate_trial,inject=hook)
                except StepRejected as exc:
                    if stepper.state.digest()!=start.digest():raise RuntimeError('failed material trial mutated its initial state')
                    self.failures.append(dict(parent_state_sha256=parent_id,attempt=index,rule=model.rule.signature,
                        reason=str(exc),rolled_back=True))
                    if index==len(candidates)-1:raise
                    continue
                if row['material_rule']!=model.rule.signature:raise ValueError('rule changed within AVF trial')
                candidate=stepper.state
                row=dict(row,rule_switch_energy_J=energy_shift,rule_proposal=None if proposal is None else proposal.value(),
                         material_attempts=index+1)
                candidate.child_states['last_ledger']=copy.deepcopy(row)
                for name in CommonState.__dataclass_fields__:setattr(trial.state,name,copy.deepcopy(getattr(candidate,name)))
                self._transaction.commit(trial)
                return row
        except Exception:
            try:self._transaction.rollback(trial)
            except ValueError:pass
            raise
        raise RuntimeError('no material attempt made')
