"""Controlled resource rejection at precommit, then authenticated normal retry."""
import argparse
from .provenance import *
from .fixture import setup
from .runtime import attempt,update
from .trajectory import case_name
from engine.aniso_phase1.research_pressure3d_next.recovery import SafePublication
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_continuous_geometry_next.continuous import state_checks

def main(run):
    run=Path(run);mutable(run);c,m,cfg,ident=setup(run,'yz');h=history(run/'cases'/case_name('yz'));c.restore(h[8]['state']);folder=run/'S5/resource-fault'
    for f in ('engine/aniso_phase1/research_pressure3d_next/recovery.py','benchmarks/research_pressure3d_next/recovery.py'):ident['numerical_sources'][f]=sha(ROOT/f)
    write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources']);store=GenerationStore(folder,ident);store.save(c.state,[]);safe=SafePublication(c,store);old=c.state.digest();pointer=sha(store.pointer)
    guard=c.model.operator.memory_budget if hasattr(c,'model') else m.operator.memory_budget;original=guard.observe;starved=[False]
    def observe(*args,**kwargs):
        if starved[0]:raise MemoryError('registered allocation refusal during device validation')
        return original(*args,**kwargs)
    guard.observe=observe
    def fault(where,state):
        if where=='before_commit':starved[0]=True;c.geometry.cache.clear()
    try:attempt(run,'S5','resource-rejection',lambda:safe.advance(inject_step=fault),fault=True)
    except MemoryError as e:
        if 'registered allocation refusal' not in str(e):raise
    else:raise ValueError('resource failure not triggered')
    exact=c.state.digest()==old and store.load()['state'].digest()==old and sha(store.pointer)==pointer and safe.pending
    if not exact:raise ValueError('resource recovery changed persistent state')
    try:safe.prepare()
    except MemoryError:blocked=True
    else:raise ValueError('pending state bypassed full device validation')
    starved[0]=False;row=attempt(run,'S5','resource-retry',safe.advance);checks=state_checks(c.state,h[9]['state'])
    if not all(v['passed'] for v in checks.values()) or safe.pending:raise ValueError('resource retry differs from normal')
    guard.observe=original;reopened=GenerationStore(folder,ident).load(validator=c.validate)
    write(run/'S5/resource-fault-check.json',dict(status='passed_scoped',rollback_exact=exact,device_validation_blocked_until_resources_return=blocked,retry_checks=checks,final_status=safe.status,final_reload_digest_equal=reopened['state'].digest()==c.state.digest(),original_guard_unchanged=True,no_other_process_affected=True,new_display_frames=0,scope='controlled allocation refusal in valid CUDA context; real driver/context loss requires fresh process'))
    write(run/'S5/recovery-scope-decision.json',dict(status='passed_scoped',entry='benchmarks.research_pressure3d_next.recovery',restoration='only authenticated state already physically validated by this owner',future_step_requires_full_validation=True,real_GPU_starvation_or_driver_loss_certified=False))
    update(run,'S5：提交前模拟资源拒绝后，以CPU认证恢复完整已提交状态；资源仍被拒绝时继续推进被阻止，解除后正常重算与原第9步等价。原显存保护未改，不宣称已覆盖真实驱动上下文损坏。')
    print('RESOURCE_RECOVERY passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
