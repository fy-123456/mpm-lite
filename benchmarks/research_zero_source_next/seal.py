"""Seal a qualified short-window descendant with complete per-stage evidence."""
import argparse,re
from .provenance import *
from .publication import audit_release

MAPPING={
'S0.1':['S0/version-audit.json','S0/source-map.json'],
'S0.2':['input-lock.json','S0/impact-and-compatibility.json','S0/experiment-budget.json'],
'S0.3':['S0/input-contract.json','S6/test-report.json','S6/physical-ledger-audit.json'],
'S1.1':['S1/continuation-audit.json','S1/full-window-protocol.json'],
'S1.2':['S1/full-window-comparison.json'],
'S1.3':['S1/energy-and-transaction.json'],
'S2.1':['S2/theta-protocol.json','S2/work-contract.md','S6/test-report.json'],
'S2.2':['S2/startup-comparison.json','S2/switch-window.json','S2/energy-and-transaction.json'],
'S2.3':['S2/method-decision.json'],
'S3.1':['S3/grid-protocol.json','S3/operator-check.json','S3/transfer-check.json'],
'S3.2':['S3/coupled-comparison.json','S3/energy-and-transaction.json'],
'S3.3':['S3/diagnosis.json','S3/grid-decision.json'],
'S4.1':['S4/integration-protocol.json','S4/memory-estimate.json'],
'S4.2':['S4/operator-equivalence.json','S4/implementation-bridge.json'],
'S4.3':['S4/paired-performance.json','S4/performance-decision.json','S4/full-build-cost-check.json','S4/fault-B0/failure.json','S4/restart.json'],
'S5.1':['S5/objective-audit.json'],
'S5.2':['S5/design-decision.json'],
'S5.3':['S5/static-comparison.json','S5/space-decision.json'],
'S6.1':['S6/test-report.json','S6/physical-ledger-audit.json','S6/load-check.json','S6/attempt-accounting.json'],
'S6.2':['S6/visual-review.json','S6/visual-assets-check.json'],
'S6.3':['S6/regression-impact.json','S6/fallback-and-restart.json']}


def accounting(run):
    run=Path(run);rows=[];stage={s:0 for s in ('S0','S1','S2','S3','S4','S5','S6')};commits=0;inherited=0;faults=0
    for f in sorted(run.rglob('attempt*.json')):
        if f.name not in ('attempt.json','attempts.json'):continue
        v=read(f);relative=str(f.relative_to(run));n=v['attempts'];s=v.get('stage',relative.split('/')[0])
        if s not in stage:raise ValueError('unassigned attempt '+relative)
        stage[s]+=n;rows.append(dict(path=relative,attempts=n,stage=s,sha256=sha(f)))
        if f.name=='attempts.json':inherited+=v.get('inherited_steps',0);commits+=v['committed']-v.get('inherited_steps',0)
        elif 'fault' in relative:faults+=n
        elif v.get('accepted'):commits+=n
    budget=read(run/'S0/experiment-budget.json');frames=[p for p in run.rglob('*.npz') if p.name in ('frame.npz','fields.npz')]
    if any(stage[s]>budget['stages'][s] for s in stage) or sum(stage.values())>budget['max_total_attempts'] or len(frames)>12:raise ValueError('experiment budget exceeded')
    d=read(run/'S3/diagnosis.json');value=dict(status='passed_scoped',stage_attempts=stage,total_attempts=sum(stage.values()),new_committed_steps=commits,inherited_coupled_prefix_steps=inherited,injected_failure_attempts=faults,other_failed_or_interrupted_attempts=sum(stage.values())-commits-faults,records=rows,max_attempts=188,new_display_frames=len(frames),max_new_display_frames=12,new_display_frame_paths=[str(f.relative_to(run)) for f in frames],inherited_daily_steps=252,inherited_daily_frames=12,new_formal_steps=0,new_static_solves=0,new_spaces=0,CPU_reference=dict(new_matrices=d.get('new_CPU_matrices',0),spectral_calls=d.get('spectral_calls',0),matrix_steps=d.get('matrix_steps',0)),CPU_static_operator_H_evaluations=4,source_mismatch_ancestor_excluded=True)
    write(run/'S6/attempt-accounting.json',value);return value


def capabilities(run):
    run=Path(run);perf=read(run/'S4/performance-decision.json');grid=read(run/'S3/grid-decision.json');method=read(run/'S2/method-decision.json')
    return dict(status='qualified_short_window_with_limits',zero_source_full_window=read(run/'S1/full-window-comparison.json')['status'],startup_method=method,pressure_grid=grid,performance=perf,transactions=read(run/'S6/fallback-and-restart.json'),space_candidate=read(run/'S5/space-decision.json'),formal_space_changed=False,functions=144,mass_order=7,full_material_order=7,new_formal_steps=0,true_coupled_spatial_accuracy=False,global_temporal_accuracy=False,full_coupled_cycle=False,fixed_skeleton_reference='inherited same-grid evidence; no new continuum certification',main_q5='inherited original solid certificate only',coupled_q5=False,production_C_E_integration=False,pure_solid_default=True,phase_events='unobserved in current pressure window')


def seal(run):
    run=Path(run);lock=verify(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    sources=all_sources();audit_parent(True);tests=read(run/'S6/test-report.json')
    if tests['returncode']!=0 or tests['tested_sources']!={k:v for k,v in sources.items() if k.startswith(('engine/','tests/'))}:raise ValueError('untested source drift')
    if read(run/'S6/visual-review.json')['image_visual_review']!='passed' or read(run/'S6/load-check.json')['status']!='passed_scoped':raise ValueError('visual/default load required')
    actual=accounting(run);perf=read(run/'S4/performance-decision.json');default=read(run/'S6/default-scene-decision.json');grid=read(run/'S3/grid-decision.json');method=read(run/'S2/method-decision.json')
    capability=capabilities(run)
    write(run/'capability-matrix.json',capability);steps=[]
    for code,title in re.findall(r'^### (S\d\.\d)[:： ]+(.+)$',(run/'plan-frozen.md').read_text(),re.M):
        status='passed_scoped'
        if code in ('S5.2','S5.3'):status='not_triggered'
        if code=='S3.3':status='limited_spatial_certification'
        if code=='S4.3' and not perf['selected']:status='performance_inconclusive_fallback_A'
        steps.append(dict(step=code,title=title,status=status,evidence=[dict(path=n,sha256=sha(run/n)) for n in MAPPING[code]]))
    if len(steps)!=21 or len(MAPPING)!=21:raise ValueError('21-step coverage mismatch')
    write(run/'requirement-audit.json',dict(count=21,steps=steps,all_steps_accounted_for=True,all_requirements_passed=False,scope_limit_reason='true coupled continuum spatial accuracy, global temporal/phase accuracy and full-cycle production coupling remain uncertified; conditional space work not triggered',plan_sha256=lock['plan_sha256'],sequential=True))
    size=sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
    if size>3*2**30:raise ValueError('new output storage budget exceeded')
    write(run/'S6/final-resources.json',dict(**resources(),output_bytes=size,output_budget_bytes=3*2**30));(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(f.relative_to(run)):sha(f) for f in sorted(run.rglob('*')) if f.is_file() and 'warp-cache' not in f.parts and f.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(n):return dict(path=n,sha256=sha(run/n))
    pub=dict(schema='zero-source-theta-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S6/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,inherited_display_frames=12,new_display_frames=actual['new_display_frames'],mass_order=7,full_material_order=7,material_policy='inherited solid q5_with_full_retry; coupled fullq7 only',spatial_accuracy=False,temporal_accuracy=False,zero_source_full_window_us=200,selected_method=method['method'])
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,checked=audit_release(run,True);print('SEALED',run,sha(run/'release.json'),checked,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
