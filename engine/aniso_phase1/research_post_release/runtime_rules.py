"""Daily fixed-rule retries: numerical rejection, owned histories and sentinels."""
import copy
import numpy as np
from engine.aniso_phase1.research_d.common_state import CommonState,StateTransaction
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
from engine.aniso_phase1.research_sequential_next.rules import FixedRuleRetry,RuleProposal


class RetryableRuleError(ValueError):
    """A registered material qualification/sentinel or deliberate test rejection."""


def retryable(exc):
    cause=exc.__cause__
    if isinstance(cause,RetryableRuleError):return True
    if not isinstance(cause,ValueError):return False
    return any(x in str(cause) for x in (
        'true AVF residual','residual line search','exact tangent GMRES','non-positive/non-finite detF',
        'non-finite material response','path leaves','endpoint below','precommit deformation/residual',
        'cumulative_abs_path_quadrature_error','cumulative_abs_solve_work_error','step energy ledger'))


class RuntimeRetry(FixedRuleRetry):
    def __init__(self,compressed,full,config,state=None):
        super().__init__(compressed,full,config,state)
        self.failures=copy.deepcopy(self.state.child_states.get('material_failure_history',[]))
        self.identity=dict(self.identity,schema='post-release-fixed-rule-retry-v1',
            sentinel_times=[.5,.6,1.1],retry_policy='classified numerical errors only; programming errors propagate')

    def _clear_trial_caches(self):
        for model in self.models.values():
            model._cached_q=None;model._cached_response=None
            if hasattr(model,'linearizations'):model.linearizations.clear()

    def step(self,dt,*,prepare_children=None,validate_trial=None,inject=None):
        trial=self._transaction.begin_trial();base=trial.state.clone();parent_id=base.digest()
        active=next(m for m in self.models.values() if m.identity==base.child_states['identity'])
        candidates=[active] if active is self.full else [active,self.full]
        try:
            for index,model in enumerate(candidates):
                start=base.clone();proposal=None;shift=0.;sentinel=None
                if index:
                    proposal=RuleProposal(parent_id,active.signature,model.signature,active.rule.signature,model.rule.signature,self.failures[-1]['reason'])
                    shift=float(model.evaluate(base.q)['U']-active.evaluate(base.q)['U'])
                    cumulative=base.child_states.get('cumulative_abs_rule_switch_error_J',0.)+abs(shift)
                    budget=self.config['acceptance']['energy_fraction']*self.config['acceptance']['energy_scale_J']
                    if not np.isfinite(shift) or cumulative>budget:raise StepRejected('material-rule migration energy exceeds frozen budget')
                    start.child_states['identity']=copy.deepcopy(model.identity)
                    start.child_states['B_rule_proposal']=proposal.value()
                    start.child_states['cumulative_abs_rule_switch_error_J']=cumulative
                cfg=copy.deepcopy(self.config);cfg['material_order']=model.rule.orders[0]
                stepper=ValidatedAVF(model,cfg,start)
                try:
                    def hook(where,state):
                        if inject is not None:inject(index,where,state)
                    def validation(candidate):
                        nonlocal sentinel
                        if model is self.compressed and any(abs(candidate.time-t)<1e-10 for t in [.5,.6,1.1]):
                            a=model.evaluate(candidate.q);b=self.full.evaluate(candidate.q)
                            e=abs(a['material_U']-b['material_U']);f=float(np.linalg.norm(a['material_force']-b['material_force']))
                            ep=1e-10+.02*abs(b['material_U']);fp=1e-8+.02*float(np.linalg.norm(b['material_force']))
                            sentinel=dict(time_s=candidate.time,energy_difference_J=e,energy_budget_J=ep,force_difference_N=f,force_budget_N=fp,passed=bool(e<=ep and f<=fp))
                            if not sentinel['passed']:raise RetryableRuleError('registered full-rule sentinel failed')
                        return True if validate_trial is None else validate_trial(candidate)
                    row=stepper.step(dt,prepare_children=prepare_children,validate_trial=validation,inject=hook)
                except StepRejected as exc:
                    if stepper.state.digest()!=start.digest():raise RuntimeError('failed material trial mutated base')
                    self.failures.append(dict(parent_state_sha256=parent_id,attempt=index,rule=model.rule.signature,
                        reason=str(exc),rolled_back=True,retryable=retryable(exc)))
                    self._clear_trial_caches()
                    if index==len(candidates)-1 or not retryable(exc):raise
                    continue
                if row['material_rule']!=model.rule.signature:raise ValueError('rule changed within AVF trial')
                row=dict(row,rule_switch_energy_J=shift,rule_proposal=None if proposal is None else proposal.value(),
                         material_attempts=index+1,material_sentinel=sentinel,
                         delta_total_including_rule_J=row['delta_total_J']+shift)
                candidate=stepper.state
                candidate.child_states['last_ledger']=copy.deepcopy(row)
                candidate.child_states['material_failure_history']=copy.deepcopy(self.failures)
                for name in CommonState.__dataclass_fields__:setattr(trial.state,name,copy.deepcopy(getattr(candidate,name)))
                self._transaction.commit(trial)
                return row
        except Exception:
            try:self._transaction.rollback(trial)
            except ValueError:pass
            self._clear_trial_caches()
            raise
        raise RuntimeError('no material attempt made')


def advance_publish(stepper,store,rows,dt,*,frame_builder=None,inject_step=None,inject_store=None):
    """Disk pointer is authoritative. Failed prepublication restores owned state."""
    base=stepper.state
    try:
        kwargs={} if inject_step is None else dict(inject=inject_step)
        row=stepper.step(dt,**kwargs);state=stepper.state
        frame=None if frame_builder is None else frame_builder(state)
        store.save(state,[*rows,row],frame=frame,inject=inject_store)
        return row
    except Exception:
        published=store.load()
        if published is not None and published['state'].digest()==stepper.state.digest() and stepper.state.step==base.step+1:
            # Publication already completed before an observational after-pointer
            # hook failed. Treat the persisted generation as accepted exactly once.
            return published['rows'][-1]
        validator=stepper.validate if isinstance(stepper,RuntimeRetry) else stepper.model.validate
        stepper._transaction=StateTransaction(base,validator=validator)
        if isinstance(stepper,RuntimeRetry):stepper._clear_trial_caches()
        raise
