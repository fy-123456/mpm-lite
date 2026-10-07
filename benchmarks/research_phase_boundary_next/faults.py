"""Actual current-space rule retries and publication faults."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import read,write,register,source_files,serial_lock
from benchmarks.research_sequential_next.checkpoint import GenerationStore
def faults(run):
    from .run import load_model,make_stepper
    from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError,advance_publish
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
    run=Path(run);cfg=read(run/'cases/daily-q5-retry/execution-protocol.json');m,_=load_model(run,cfg);base=make_stepper(run,cfg,m,m.rest());full=base.full
    register(run,'S5/fault-protocol.json',dict(final_source=source_files(),faults=['before_material','after_prepare','before_commit','double_failure','KeyError','before_pointer','after_pointer'],
        restart='S6/cache-retry-check.json',qualified_main=True,steps_per_trial=1))
    records=[]
    def children(state,values):values['owned_child']={'pressure':[.1,.2],'time':state.time};return values
    reference=ValidatedAVF(full,cfg);reference.step(.0125,prepare_children=children)
    for stage in ('before_material','after_prepare','before_commit'):
        c=RuntimeRetry(m,full,cfg)
        def fail(attempt,where,state):
            if attempt==0 and where==stage:raise RetryableRuleError('controlled '+stage)
        row=c.step(.0125,inject=fail,prepare_children=children)
        err=max(float(np.max(abs(getattr(c.state,k)-getattr(reference.state,k)))) for k in ('q','velocity','predictor'))
        if err>1e-8 or row['material_attempts']!=2 or c.state.child_states['owned_child']['time']!=.0125:raise ValueError('whole-step retry differs')
        records.append(dict(fault=stage,error=err,attempts=2,owned_history=True,energy_switch_J=row['rule_switch_energy_J']))
    for label,error in [('double_failure',RetryableRuleError),('programming_error',KeyError)]:
        c=RuntimeRetry(m,full,cfg);before=c.state.digest();calls=[]
        def fail(attempt,where,state):
            if where=='before_material':calls.append(attempt);raise error(label)
        try:c.step(.0125,inject=fail)
        except StepRejected:pass
        else:raise AssertionError('failure was swallowed')
        if c.state.digest()!=before or len(calls)!=(2 if label=='double_failure' else 1):raise ValueError('bad error classification or partial rollback')
        records.append(dict(fault=label,attempts=len(calls),unchanged_digest=True))
    c=RuntimeRetry(m,full,cfg);folder=run/'S5/publication-fixture';store=GenerationStore(folder,dict(model=m.identity,source=source_files()));store.save(c.state,[])
    before=c.state.digest();pointer=read(store.pointer)
    def fail_before(where):
        if where=='before_pointer':raise OSError('controlled disk error')
    try:advance_publish(c,store,[],.0125,inject_store=fail_before)
    except OSError:pass
    else:raise AssertionError('disk error swallowed')
    if c.state.digest()!=before or read(store.pointer)!=pointer:raise ValueError('publication rollback failed')
    def fail_after(where):
        if where=='after_pointer':raise OSError('controlled observer error')
    row=advance_publish(c,store,[],.0125,inject_store=fail_after)
    if c.state.step!=1 or len(store.history())!=2:raise ValueError('publication not accepted once')
    records.extend([dict(fault='before_pointer',unchanged=True),dict(fault='after_pointer',accepted_once=True)])
    write(run/'S5/fault-and-restart.json',dict(status='passed_scoped',records=records,numeric_sources=source_files(),actual_restart=read(run/'S6/cache-retry-check.json')))
    write(run/'S5/runtime-policy.json',dict(status='passed_scoped',requested_actual_recorded=True,classification='only numerical/retryable errors',
        whole_step_rollback=True,programming_errors_propagate=True,publication_authoritative=True,sensitive_full_cycle_q5=False))
    print('FAULTS_PASSED',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):faults(a.run)
