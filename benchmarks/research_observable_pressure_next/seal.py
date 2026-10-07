"""Seal only verified scoped outcomes, including explicit non-adoption decisions."""
from pathlib import Path
import argparse,re
from .provenance import ROOT,APP,APP_SHA,PARENT,PLAN,PROGRESS,read,write,sha,all_sources,source_files,snapshot,verify,audit_parent,check,utc,resources,serial_lock
from .publication import audit_release

def seal(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    lock=verify(run);protocol=read(run/'S6/final-protocol.json');default=protocol['default_case']
    required=['S0/compatibility.json','S3/two-cell-coupled.json','S3/rollback-restart.json','S6/final-scene.json','S6/raw-physical-check.json','S6/physical-review.json','S6/final-tests-result.json','S4/runtime-check.json']
    if protocol['q5_qualified']:required+=['S5/window-runtime-final.json','S5/fault-and-restart.json','S6/cache-retry-check.json']
    for name in required:
        if read(run/name)['status'] not in ('passed','passed_scoped'):raise ValueError('unmet publication gate '+name)
    for name in ('S6/final-model-lock.json','S6/final-tests-result.json'):
        if read(run/name)['numerical_source_sha256']!=source_files():raise ValueError('final numerical source drift')
    decisions={'S0':('passed_scoped',['S0/version-audit.json','S0/compatibility.json','S0/protocol.json','S0/storage-migration.json']),
        'S1':('passed_scoped_local_outputs',['S1/output-baseline.json','S1/observable-importance.json','S1/local-reference-check.json','S1/time-decision.json']),
        'S2':('limited_F60_formal_stress',['S2/reference-input-audit.json','S2/search-correction.json','S2/direction-check.json','S2/space-scope-decision.json']),
        'S3':('passed_scoped_pressure',['S3/pressure-time-protocol.json','S3/fixed-2.json','S3/two-cell-coupled.json','S3/rollback-restart.json','S3/pressure-scope-decision.json']),
        'S4':('adopt_two_cell_pressure_only',['S4/current-profile.json','S4/operator-equivalence.json','S4/paired-performance.json','S4/performance-decision.json','S4/runtime-check.json']),
        'S5':('passed_scoped_current_certificates',['S5/continuous-window-qualification.json','S5/window-runtime-final.json','S5/scope-boundaries.json','S5/fault-and-restart.json']),
        'S6':('passed_scoped_final_scene',['S6/final-tests-result.json','S6/qualification-final.json','S6/final-scene.json','S6/raw-physical-check.json','S6/physical-review.json','S6/parent-scene-review.json','S6/load-check.json','S6/time-transition-review.json'])}
    steps=[]
    for code,title in re.findall(r'^### (S\d\.\d) (.+)$',(run/'plan-frozen.md').read_text(),re.M):
        status,evidence=decisions[code.split('.')[0]]
        if code=='S3.5':
            status='grid_sensitive_not_qualified';evidence=[*evidence,'S3/common-time-grid-review.json','S3/grid-trigger-correction.json']
        if code=='S2.4':status='formal_space_retained_F60_not_qualified'
        for name in evidence:
            if not (run/name).is_file():raise ValueError('missing evidence '+name)
        scope=status
        status='limited' if code in ('S2.3','S3.5') else 'passed_scoped'
        if code=='S2.1':scope='authenticated_R5_reload_without_new_main_solve'
        if code=='S2.2':scope='preregistered_F60_true_material_and_direction_read_back'
        if code=='S2.4':scope='formal_space_retained_no_F60_qualification'
        steps.append(dict(step=code,title=title,status=status,scope=scope,evidence=evidence))
    if len(steps)!=27:raise ValueError('27-step audit incomplete')
    write(run/'requirement-audit.json',dict(count=27,steps=steps,all_steps_accounted_for=True,original_plan_sha256=lock['plan_sha256'],
        bounded_limits=['F60 formal stress exceeds engineering budget despite resolved R4/R5 reference; formal space retained',
        'local 252-step schedule improvement does not certify full-cycle temporal accuracy',
        'four-cell cumulative boundary volume comparison exceeds grid budget; individually stable diagnostic, no grid qualification',
        'pressure time scales derived from fixed-solid rest operator; limited aligned 2/4-cell scope, no production coupling',
        'search-matrix repair changes static solver direction only, never physical model or residual budgets']))
    space=read(run/'selected-space.json');scene=read(run/'S6/final-scene.json');time=read(run/'S1/time-decision.json');pressure=read(run/'S3/pressure-scope-decision.json');perf=read(run/'S4/performance-decision.json')
    caps=dict(scene_stability='passed_scoped',selected_space=space['selected'],functions=144,mass_order=7,steps=scene['steps'],time_scheme=time['status'],spatial_accuracy=False,temporal_accuracy=False,
        local_time_outputs='two paired windows with continuous candidate history; actual final-cycle initial states checked',F60_static_stress_qualified=False,
        formal_space_unchanged=True,material_scope='main .005 exact final grid; .0075 continuous .6-.7 eight-step authenticated window',sensitive_full_cycle_q5=False,
        pressure=pressure,performance=perf,pure_solid_default=True,production_C_E_integration=False,coupled_q5=False)
    write(run/'capability-matrix.json',caps);write(run/'S6/final-resources.json',resources())
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes());write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    sources=all_sources();snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(path):return dict(path=path,sha256=sha(run/path))
    pub=dict(schema='observable-pressure-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default,
        sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),qualification=entry('S6/qualification-final.json'),window_qualification=entry('S5/continuous-window-qualification.json'),scene=entry('S6/final-scene.json'),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),
        case_identity_sha256=sha(run/'cases'/default/'identity.json'),case_protocol_sha256=sha(run/'cases'/default/'execution-protocol.json'),steps=scene['steps'],display_frames=scene['frames'],mass_order=7,full_material_order=7,material_policy='q5_with_full_retry' if protocol['q5_qualified'] else 'full_only',spatial_accuracy=False,temporal_accuracy=False,coupling_scope=pressure)
    check(ROOT,sources);check(run/'final-source',sources);check(run,artifacts);audit_parent(True);write(run/'release.json',pub);_,counts=audit_release(run)
    print('SEALED',run,sha(run/'release.json'),counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
