"""Seal all 21 planned steps with scoped acceptance and preserved ancestry."""
import argparse
from .provenance import *
from .publication import audit_release

MAPPING={
'S0.1':['S0/version-audit.json','input-lock.json'],
'S0.2':['S0/physical-contract.json','selected-space.json'],
'S0.3':['S0/experiment-budget.json','S0/resource-policy.json','S0/storage-migration.json'],
'S1.1':['S1/time-protocol.json','S6/test-report.json'],
'S1.2':['S1/observation-contract.json','S6/test-report.json'],
'S1.3':['S1/transaction-contract.json','S3/transaction-check.json'],
'S2.1':['S2/screen-protocol.json','S2/reference-check.json','S2/reference-binding.json'],
'S2.2':['S2/candidate-U2.json','S2/candidate-U4.json','S2/baseline.json'],
'S2.3':['S2/method-decision.json','S2/screen-accounting.json'],
'S3.1':['S3/coupled-protocol.json','S3/operator-check.json','S3/memory-estimate.json'],
'S3.2':['S3/engineering-comparison.json','S3/raw-substep-diagnostics.json','cases/boundary32-h/summary.json','cases/boundary32-half/summary.json'],
'S3.3':['S3/transaction-check.json','S3/grid-decision.json'],
'S4.1':['S4/legacy/integration-protocol.json','S4/legacy/environment-check.json'],
'S4.2':['S4/legacy/paired-performance.json','S4/legacy/setup-cost.json'],
'S4.3':['S4/backend-decision.json','S6/selected-backend-load.json','S6/fallback-and-restart.json'],
'S5.1':['S5/bottleneck-review.md','S5/worst-errors.json'],
'S5.2':['S5/next-scope-decision.json'],
'S5.3':['implementation-report.md'],
'S6.1':['S6/test-report.json','S6/load-check.json','S6/attempt-accounting.json','S6/regression-impact.json'],
'S6.2':['S6/visual-assets-check.json','S6/visual-review.json','S6/visualization-origin.json'],
'S6.3':['S6/final-resources.json','capability-matrix.json','documentation.json']}


def accounting(run):
    run=Path(run);budget=read(run/'S0/experiment-budget.json');stage={s:0 for s in budget['stages']};commits=faults=0;records=[]
    for f in sorted((run/'cases').glob('*/attempts.json')):
        v=read(f);stage['S3']+=v['attempts'];commits+=v['committed'];records.append(dict(path=str(f.relative_to(run)),sha256=sha(f),attempts=v['attempts'],stage='S3'))
    for s in ('S3','S4'):
        for f in sorted((run/s).rglob('attempt.json')):
            v=read(f);n=v['attempts'];stage[s]+=n;commits+=int(v.get('accepted',False))*n;faults+=n if 'fault' in str(f.relative_to(run)) else 0;records.append(dict(path=str(f.relative_to(run)),sha256=sha(f),attempts=n,stage=s))
    frames=[f for f in run.rglob('frame.npz')];cpu=read(run/'S2/screen-accounting.json');peak=max(read(p)['peak_RSS_GiB'] for p in (run/'cases').glob('*/summary.json'));peak=max(peak,read(run/'S3/memory-estimate.json')['peak_RSS_GiB'])
    for p in (run/'S4').glob('*/*/measurement.json'):peak=max(peak,read(p)['peak_rss_GiB'])
    # First operator + two trajectories started before the common outer wall timer.
    # Their in-process times include model construction. Bound their import/audit/exit
    # overhead by the entire registration-to-latest-finish elapsed interval, a strict
    # upper bound including idle time and CPU work, then add separately timed later jobs.
    from datetime import datetime,timezone
    unwrapped_paths=[run/'S3/operator-check.json',run/'cases/boundary32-h/summary.json',run/'cases/boundary32-half/summary.json']
    measured_initial=sum(read(p)['seconds'] for p in unwrapped_paths)
    start=datetime.fromisoformat(read(run/'input-lock.json')['utc']).timestamp();end=max(p.stat().st_mtime for p in unwrapped_paths);initial_upper=end-start
    logs=[read(p) for p in (run/'execution-logs').glob('*.json')];later=sum(x['seconds'] for x in logs);wall_upper=initial_upper+later
    output=sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
    if any(stage[s]>budget['stages'][s] for s in stage) or sum(stage.values())>100 or len(frames)>6 or peak>16 or cpu['new_matrices']>6 or cpu['spectral_calls']>8 or cpu['matrix_steps']>512 or cpu['seconds']>600 or wall_upper>2400 or output>3*2**30:raise ValueError('registered budget exceeded')
    v=dict(status='passed_scoped',stage_attempts=stage,total_attempts=sum(stage.values()),new_committed_steps=commits,injected_failure_attempts=faults,other_failed_or_interrupted_attempts=sum(stage.values())-commits-faults,records=records,max_attempts=100,new_display_frames=len(frames),new_display_frame_paths=[str(f.relative_to(run)) for f in frames],max_new_display_frames=6,inherited_daily_steps=252,inherited_daily_frames=12,new_formal_steps=0,new_grid_coupled_trajectory_steps=84,new_spaces=0,CPU_reference=cpu,peak_RSS_GiB=peak,initial_in_process_seconds=measured_initial,initial_wall_strict_upper_seconds=initial_upper,later_complete_process_wall_seconds=later,all_GPU_related_wall_upper_seconds=wall_upper,wall_bound_notes='Initial bound spans registration to last of two trajectory summaries, thus also contains idle and CPU time; later logs include construction/import/fault/restart/review. No failed process excluded.',output_bytes_at_check=output)
    write(run/'S6/attempt-accounting.json',v);return v


def capabilities(run):
    run=Path(run);backend=read(run/'S4/backend-decision.json')
    return dict(status='engineering_window_and_scoped_geometry_delivery',engineering_window_accuracy=read(run/'S3/grid-decision.json')['engineering_window_accuracy'],fixed_skeleton_engineering_reference=read(run/'S2/reference-check.json')['status'],raw_substep_accuracy=False,actual_coupled_spatial_accuracy=False,geometry_correctness=True,exclusive_performance={k:v['selected'] for k,v in backend['scopes'].items()},research_backends=backend['scopes'],production_default='inherited daily-q5-retry pure solid; 252steps1.6s',formal_space_changed=False,local_functions=144,mass_order=7,full_material_order=7,new_formal_steps=0,global_temporal_accuracy=False,full_coupled_cycle=False,three_dimensional_reference_certified=False,coupled_q5=False,production_C_E_integration=False)


def seal(run):
    run=Path(run);lock=mutable(run);sources=all_sources();audit_parent(True);tests=read(run/'S6/test-report.json')
    if tests['returncode']!=0 or tests['tested_sources']!={k:v for k,v in sources.items() if k.startswith(('engine/','tests/'))}:raise ValueError('untested engine/test drift')
    if read(run/'S6/visual-review.json')['image_visual_review']!='passed' or read(run/'S6/load-check.json')['status']!='passed_scoped':raise ValueError('visual/default load required')
    actual=accounting(run);default=read(run/'S6/default-scene-decision.json');write(run/'capability-matrix.json',capabilities(run));(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    size=sum(f.stat().st_size for f in run.rglob('*') if f.is_file());write(run/'S6/final-resources.json',dict(**resources(),output_bytes=size,output_budget_bytes=3*2**30))
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    import re
    codes=re.findall(r'^### (S[0-6]\.[1-3]) ',(run/'plan-frozen.md').read_text(),re.M)
    if set(codes)!=set(MAPPING) or len(codes)!=21:raise ValueError('plan step map differs')
    steps=[]
    for code,evidence in MAPPING.items():
        status='passed_scoped';reason='required scoped evidence complete'
        if code=='S2.2':status='limited';reason='U2 fails engineering gate, U4 passes; raw microstep time accuracy remains limited and is retained'
        if code=='S3.2':reason='same new32 grid engineering interval accuracy and stability only; raw and spatial accuracy separately limited'
        if code=='S4.3' and not all(v['selected'] for v in read(run/'S4/backend-decision.json')['scopes'].values()):status='limited';reason='backend/performance eligibility remains scope dependent'
        if code=='S5.2':reason='future directions and missing entry conditions recorded; no expansion performed'
        steps.append(dict(step=code,status=status,reason=reason,evidence=[dict(path=n,sha256=sha(run/n)) for n in evidence]))
    write(run/'requirement-audit.json',dict(count=21,steps=steps,all_steps_accounted_for=True,all_accuracy_goals_passed=False,scope_limit_reason='raw startup interval bias, actual spatial convergence, full coupled cycle and full3D reference remain unqualified',plan_sha256=lock['plan_sha256'],sequential=True))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(f.relative_to(run)):sha(f) for f in sorted(run.rglob('*')) if f.is_file() and 'warp-cache' not in f.parts and f.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(n):return dict(path=n,sha256=sha(run/n))
    pub=dict(schema='startup-substeps-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S6/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,inherited_display_frames=12,new_display_frames=actual['new_display_frames'],mass_order=7,full_material_order=7,material_policy='inherited solid q5_with_full_retry; coupled fullq7 only',spatial_accuracy=False,temporal_accuracy=False,engineering_window_accuracy=True,selected_geometry_by_scope={k:v['backend'] for k,v in read(run/'S4/backend-decision.json')['scopes'].items()})
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,checked=audit_release(run,True);print('SEALED',run,sha(run/'release.json'),{k:v for k,v in checked.items() if k!='ancestors'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
