"""Seal completed and conditionally untriggered work without upgrading its scope."""
import argparse
from .provenance import *
from .publication import audit_release

MAPPING={
'R0':['S0/version-audit.json','input-lock.json','R0/input-contract.json','S0/experiment-budget.json'],
'R1':['R1/reference-protocol.json','R1/scalar-reference.json','S6/test-report.json'],
'R2':['R2/grid-screen-protocol.json','R2/grid-screen.json','R2/diagnosis.json'],
'R3':['R3/coupling-decision.json'],
'R4':['R4/integration-protocol.json','R4/optimization-protocol.json','R4/operator-equivalence.json','R4/implementation-bridge.json','R4/paired-performance.json','R4/performance-decision.json','R4/performance-environment-review.json','R4/fault-B0/failure.json','R4/restart.json'],
'R5':['S6/physical-ledger-audit.json','S6/frame-equivalence.json','S6/load-check.json','S6/attempt-accounting.json','S6/visual-assets-check.json','S6/visual-review.json','S6/regression-impact.json','S6/fallback-and-restart.json']}


def accounting(run):
    run=Path(run);budget=read(run/'S0/experiment-budget.json');stage={s:0 for s in budget['stages']};records=[];commits=0;faults=0;seconds=0.;peak=0.
    for f in sorted(run.rglob('attempt.json')):
        v=read(f);relative=str(f.relative_to(run));s=relative.split('/')[0];n=v['attempts'];stage[s]+=n;commits+=int(v.get('accepted',False))*n;faults+=n if 'fault-' in relative else 0;records.append(dict(path=relative,attempts=n,stage=s,sha256=sha(f)))
    for f in (run/'R4').glob('*/measurement.json'):
        v=read(f);seconds+=v['advance_s'];peak=max(peak,v['peak_rss_GiB'])
    frames=[f for f in run.rglob('*.npz') if f.name in ('frame.npz','fields.npz')];scalar=read(run/'R1/scalar-reference.json');grid=read(run/'R2/grid-screen.json')
    cpu={k:scalar[k]+grid[k] for k in ('new_matrices','spectral_calls','matrix_steps','seconds')}
    if any(stage[s]>budget['stages'][s] for s in stage) or sum(stage.values())>56 or len(frames)>4 or peak>16 or cpu['new_matrices']>12 or cpu['spectral_calls']>16 or cpu['matrix_steps']>512 or cpu['seconds']>600:raise ValueError('budget exceeded')
    value=dict(status='passed_scoped',stage_attempts=stage,total_attempts=sum(stage.values()),new_committed_steps=commits,injected_failure_attempts=faults,other_failed_or_interrupted_attempts=sum(stage.values())-commits-faults,records=records,max_attempts=56,new_display_frames=len(frames),new_display_frame_paths=[str(f.relative_to(run)) for f in frames],max_new_display_frames=4,inherited_daily_steps=252,inherited_daily_frames=12,new_formal_steps=0,new_grid_coupled_steps=0,new_spaces=0,CPU_reference=cpu,measured_AB_dynamic_seconds=seconds,peak_AB_RSS_GiB=peak,fault_runtime_separately_measured=False,fault_process_completed_within_timeout_s=1200,all_dynamic_seconds_upper_bound=1200+seconds)
    write(run/'S6/attempt-accounting.json',value);return value


def capabilities(run):
    run=Path(run)
    return dict(status='fixed_skeleton_reference_and_shared_geometry_correctness_passed_with_limits',independent_scalar_reference=read(run/'R1/scalar-reference.json')['status'],full_tensor_fixed_skeleton_reference='64/128 x-only refinement passes at uniform engineering intervals, quarter-budget gate',fixed_skeleton_32cell_spatial_comparison='passes against128 at uniform engineering intervals',raw_startup_temporal_accuracy=False,new_grid_actual_coupling=read(run/'R3/coupling-decision.json'),shared_geometry_correctness=read(run/'R4/operator-equivalence.json')['status'],performance=read(run/'R4/performance-decision.json'),transactions=read(run/'S6/fallback-and-restart.json'),formal_space_changed=False,functions=144,mass_order=7,full_material_order=7,new_formal_steps=0,true_coupled_spatial_accuracy=False,global_temporal_accuracy=False,full_coupled_cycle=False,three_dimensional_reference_certified=False,main_q5='inherited original solid certificate only',coupled_q5=False,production_C_E_integration=False,pure_solid_default=True)


def seal(run):
    run=Path(run);lock=mutable(run);sources=all_sources();audit_parent(True);tests=read(run/'S6/test-report.json')
    if tests['returncode']!=0 or tests['tested_sources']!={k:v for k,v in sources.items() if k.startswith(('engine/','tests/'))}:raise ValueError('untested engine/test drift')
    if read(run/'S6/visual-review.json')['image_visual_review']!='passed' or read(run/'S6/load-check.json')['status']!='passed_scoped':raise ValueError('visual/default load required')
    actual=accounting(run);perf=read(run/'R4/performance-decision.json');default=read(run/'S6/default-scene-decision.json');write(run/'capability-matrix.json',capabilities(run));steps=[]
    for code,evidence in MAPPING.items():
        status='passed_scoped'
        if code=='R2':status='spatial_passed_temporal_limited'
        if code=='R3':status='not_triggered'
        if code=='R4' and not perf['selected']:status='performance_inconclusive_fallback_A'
        steps.append(dict(step=code,status=status,evidence=[dict(path=n,sha256=sha(run/n)) for n in evidence]))
    write(run/'requirement-audit.json',dict(count=6,steps=steps,all_steps_accounted_for=True,all_requirements_passed=False,scope_limit_reason='startup raw interval time accuracy fails on new boundary grids; conditional actual grid branch not triggered; full coupled space/time/cycle accuracy not certified',plan_sha256=lock['plan_sha256'],sequential=True))
    size=sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
    if size>3*2**30:raise ValueError('storage budget exceeded')
    write(run/'S6/final-resources.json',dict(**resources(),output_bytes=size,output_budget_bytes=3*2**30));(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(f.relative_to(run)):sha(f) for f in sorted(run.rglob('*')) if f.is_file() and 'warp-cache' not in f.parts and f.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(n):return dict(path=n,sha256=sha(run/n))
    pub=dict(schema='boundary-reference-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S6/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,inherited_display_frames=12,new_display_frames=actual['new_display_frames'],mass_order=7,full_material_order=7,material_policy='inherited solid q5_with_full_retry; coupled fullq7 only',spatial_accuracy=False,temporal_accuracy=False,selected_geometry=perf['backend'])
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,checked=audit_release(run,True);print('SEALED',run,sha(run/'release.json'),{k:v for k,v in checked.items() if k!='ancestors'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
