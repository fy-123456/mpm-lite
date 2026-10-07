"""Seal all evidence, including rejected protocol runs, under a scoped release."""
from pathlib import Path
import argparse,re
from .provenance import *
from .publication import audit_release

MAPPING={
 'S0.1':['S0/version-audit.json','S0/source-map.json'],
 'S0.2':['input-lock.json','S0/resource-policy.json','S0/experiment-budget.json'],
 'S0.3':['S0/impact-and-compatibility.json'],
 'S1.1':['S1/fixed-inputs.json'],'S1.2':['S1/startup-screen.json'],
 'S1.3':['S1/startup-decision.json'],'S1.4':['S1/discrete-work-contract.md','S1/equation-protocol.json','S6/test-report.json'],
 'S2.1':['S2/reference-protocol.json','S2/prefix-reuse.json','S2/short-reference-comparison.json','S2/physical-check.json'],
 'S2.2':['S2/stage-window-decision.json'],'S2.3':['S2/reference-scope.json'],
 'S3.1':['S3/objective-audit.json','S3/design-decision.json'],'S3.2':['S3/space-decision.json'],'S3.3':['S3/space-decision.json'],
 'S4.1':['S4/profile.json'],'S4.2':['S4/optimization-protocol.json','S4/operator-equivalence.json'],
 'S4.3':['S4/performance-decision.json','S4/paired-performance.json','S4/fault-B0/failure.json','S4/restart.json'],
 'S5.1':['S5/selected-protocol.json','S5/source-mismatch/diagnosis.json','S5/repair-protocol.json','S5/operator-coarse.json'],
 'S5.2':['S5/coupled-comparison.json','S5/coupling-decision.json'],
 'S5.3':['S5/energy-and-transaction.json'],
 'S6.1':['S6/test-report.json','S6/load-check.json','S6/attempt-accounting.json'],
 'S6.2':['S6/visual-review.json'],'S6.3':['S6/regression-impact.json','S6/fallback-and-restart.json']}

def accounting(run):
    rows=[];stage={s:0 for s in ('S0','S1','S2','S3','S4','S5','S6')}
    for f in sorted(run.rglob('attempt*.json')):
        if f.name not in ('attempt.json','attempts.json'):continue
        relative=str(f.relative_to(run));n=read(f)['attempts']
        s='S2' if relative.startswith('cases/candidate-') else ('S4' if relative.startswith('S4/') else 'S5')
        rows.append(dict(path=relative,attempts=n,stage=s,sha256=sha(f)));stage[s]+=n
    budget=read(run/'S0/experiment-budget.json');frames=[f for f in run.rglob('*.npz') if f.name in ('frame.npz','fields.npz')]
    if any(stage[s]>budget['stages'][s] for s in stage) or sum(stage.values())>budget['max_total_attempts'] or len(frames)>12:raise ValueError('actual experiments exceeded budget')
    value=dict(status='passed_scoped',stage_attempts=stage,total_attempts=sum(stage.values()),records=rows,max_attempts=budget['max_total_attempts'],new_display_frames=len(frames),max_new_display_frames=12,new_display_frame_paths=[str(f.relative_to(run)) for f in frames],inherited_daily_steps=252,inherited_daily_frames=12,new_formal_steps=0,inherited_solid_coarse_steps=16,inherited_solid_fine_steps=16,new_solid_reference_steps=16,corrected_zero_source_steps=36,rejected_source_protocol_attempts=60,rejected_source_protocol_commits=59,interrupted_attempts=1,injected_failure_attempts=2,static_solves=0,new_spaces=0,CPU_pressure_screen=dict(algebra_assemblies=6,spectral_exact_calls=6,theta_matrix_steps=384),S2_attempt_log_is_precommit=True,S2_final_committed_step=read(run/'S2/physical-check.json')['total_steps'])
    write(run/'S6/attempt-accounting.json',value);return value

def seal(run):
    run=Path(run);lock=verify(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    sources=all_sources();audit_parent(True);tests=read(run/'S6/test-report.json')
    if tests['returncode']!=0 or tests['tested_sources']!={k:v for k,v in sources.items() if k.startswith(('engine/','tests/'))}:raise ValueError('untested source drift')
    if read(run/'S6/visual-review.json')['image_visual_review']!='passed':raise ValueError('actual image review required')
    if read(run/'S6/load-check.json')['status']!='passed_scoped':raise ValueError('default load failed')
    actual=accounting(run);perf=read(run/'S4/performance-decision.json');default=read(run/'S6/default-scene-decision.json')
    capabilities=dict(status='limited',scene_stability='inherited_passed_scoped',formal_space_changed=False,functions=144,mass_order=7,formal_steps=252,new_formal_steps=0,spatial_accuracy=False,temporal_accuracy=False,local_reference=read(run/'S2/reference-scope.json'),fixed_skeleton_pressure_time=read(run/'S1/startup-decision.json'),actual_coupling=read(run/'S5/coupling-decision.json'),space_candidate=read(run/'S3/space-decision.json'),performance=perf,raw_event_status='inherited_limited',main_q5='original solid certificate only',coupled_q5=False,production_C_E_integration=False,transactions=read(run/'S6/fallback-and-restart.json'),pure_solid_default=True)
    write(run/'capability-matrix.json',capabilities);steps=[]
    for code,title in re.findall(r'^### (S\d\.\d)[:： ]+(.+)$',(run/'plan-frozen.md').read_text(),re.M):
        names=MAPPING[code];status='passed_scoped'
        if code in ('S2.2','S3.2','S3.3'):status='not_triggered'
        if code in ('S1.3','S2.3','S5.2'):status='limited'
        if code=='S5.1':status='fixed_then_passed_scoped'
        steps.append(dict(step=code,title=title,status=status,evidence=[dict(path=n,sha256=sha(run/n)) for n in names]))
    if len(steps)!=22 or len(MAPPING)!=22:raise ValueError('22-step coverage mismatch')
    write(run/'requirement-audit.json',dict(count=22,steps=steps,all_steps_accounted_for=True,all_original_requirements_passed=False,plan_sha256=lock['plan_sha256'],sequential=True,source_repair_scope='12/24 zero-source prefix instead of complete16/32 by16/32 matrix'))
    size=sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
    if size>3*2**30:raise ValueError('new output storage budget exceeded')
    write(run/'S6/final-resources.json',dict(**resources(),output_bytes=size,output_budget_bytes=3*2**30));(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(f.relative_to(run)):sha(f) for f in sorted(run.rglob('*')) if f.is_file() and 'warp-cache' not in f.parts and f.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(n):return dict(path=n,sha256=sha(run/n))
    pub=dict(schema='pressure-startup-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S6/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,inherited_display_frames=12,new_display_frames=actual['new_display_frames'],mass_order=7,full_material_order=7,material_policy='inherited parent q5_with_full_retry only',spatial_accuracy=False,temporal_accuracy=False,local_prefix_time_passed=True)
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,checked=audit_release(run,True);print('SEALED',run,sha(run/'release.json'),checked,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
