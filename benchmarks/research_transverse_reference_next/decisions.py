"""Scope decisions remain explicit when expensive branches were not triggered."""
import argparse
from .provenance import *
from .runtime import update


def main(run):
    run=Path(run);mutable(run)
    parent=read(APP/'release.json');reference=read(APP/'S5/reference-applicability.json')
    write(run/'S4/space-entry-decision.json',dict(status='not_triggered',inherited_reference=reference,reference_source_sha256=sha(APP/'S5/reference-applicability.json'),reason='fixed-skeleton reference is valid but transverse increments unresolved; coupled pressure change mainly volume exchange; no matched solid dynamic reference',new_training=0,new_reference_equilibria=0,formal_local_functions=144,formal_space_changed=False))
    entries=read(APP/'S5/extension-entry-decision.json')
    entries['preferred_next']='matched coupled linearized or moving-skeleton reference accounting for alpha*DeltaV, before enlarging time or reallocating basis'
    entries['entries'].append(dict(direction='real GPU driver loss',input='new process and immutable checkpoint identity',missing=['actual driver/device loss test, device revalidation and side-effect recovery'],cost='separate bounded fixture; no destructive driver test in this delivery',stop='cannot restore authenticated physical state'))
    write(run/'S4/next-entry-decisions.json',dict(entries,status='future_work_only'))
    write(run/'S5/default-scene-decision.json',dict(status='inherited',default_case=parent['default_case'],default_case_source=parent['default_case_source'],formal_steps=252,new_formal_steps=0))
    from benchmarks.research_sequential_next.checkpoint import GenerationStore
    p=parent['default_case_source'];folder=Path(p['release_root'])/'cases'/p['case'];store=GenerationStore(folder,read(folder/'identity.json'));s=store.load()['state']
    if s.step!=252 or abs(s.time-1.6)>1e-12:raise ValueError('inherited solid scene changed')
    write(run/'S5/daily-load-check.json',dict(status='passed_scoped',source=p,step=s.step,time_s=s.time,zero_new_steps=True,physics_validator_not_rerun=True,authenticated_generation_load=True))
    update(run,'S4：正式144空间、M7/q7与原稳定化保持；没有同工况固体动态参考，不触发重分配。完整周期、生产C/E、耦合q5和真实驱动丢失恢复仍留作后续。继承纯固体252步终态只读认证加载通过。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
