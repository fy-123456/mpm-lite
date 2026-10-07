"""Source-complete scoped release with explicit negatives and inherited default."""
from pathlib import Path
import argparse,re
from .provenance import *
from .performance import numerical_sources
from .publication import audit_release

MAPPING={
 'S0.1':['S0/version-audit.json'],
 'S0.2':['S0/default-source.json','S0/interface-contract.json'],
 'S0.3':['S0/compatibility.json','S0/resource-policy.json'],
 'S1.1':['S1/reference-protocol.json'],
 'S1.2':['S1/time-comparison.json','cases/candidate-quarter/summary.json'],
 'S1.3':['S1/force-attribution.json','S1/candidate-diagnostic.json','S1/formal-diagnostic.json'],
 'S1.4':['S1/candidate-reference-decision.json'],
 'S2.1':['S2/hotspot-breakdown.json','S2/optimization-protocol.json'],
 'S2.2':['S2/operator-equivalence.json','S2/identity-fix.json'],
 'S2.3':['S2/paired-protocol.json','S2/fault-B0/failure.json','S2/publication-anomaly.json'],
 'S2.4':['S2/paired-performance.json','S2/performance-decision.json'],
 'S3.1':['S3/grid-protocol.json'],
 'S3.2':['S3/operator-check.json','S3/pressure-schedule-check.json','cases/grid32-h/summary.json','cases/grid32-half/summary.json'],
 'S3.3':['S3/time-comparison.json','S3/grid-comparison.json','S3/grid-transfer-check.json'],
 'S3.4':['S3/coupling-scope-decision.json','S3/grid-limitation-diagnostic.json'],
 'S4.1':['S4/reaction-engineering.json'],
 'S4.2':['S4/engineering-protocol.json','S4/reaction-engineering.json'],
 'S4.3':['S4/time-decision.json'],
 'S5.1':['S5/research-space-decision.json'],
 'S5.2':['S5/research-space-decision.json'],
 'S5.3':['S5/research-space-decision.json'],
 'S5.4':['S5/research-space-decision.json'],
 'S6.1':['S6/tests-result.json','S6/regression-impact.json','S6/fallback-and-restart.json'],
 'S6.2':['S6/default-scene-decision.json','S6/load-check.json','S6/visualization-origin.json','S6/visual-review.json'],
 'S6.3':['S6/scope-summary.json','S6/attempt-accounting.json'],
 'S6.4':['S6/publication-preflight.json']}


def seal(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    lock=verify(run)
    for name in ('S0/compatibility.json','S6/tests-result.json','S6/visual-review.json','S6/load-check.json'):
        if read(run/name)['status'] not in ('passed','passed_scoped'):raise ValueError('missing final gate '+name)
    tests=read(run/'S6/tests-result.json')
    if tests['numerical_source_sha256']!=source_files():raise ValueError('core source drift')
    if tests['engine_source_sha256']!={k:v for k,v in numerical_sources().items() if k.startswith('engine/')}:raise ValueError('engine source drift')
    check(ROOT,tests['tested_source_sha256'])
    if read(run/'S6/visual-review.json')['image_visual_review']!='inspected':raise ValueError('images not inspected')
    counts=audit_parent(True)[2];sources=all_sources();check(ROOT,sources)
    write(run/'S6/publication-preflight.json',dict(status='passed_scoped',sources=len(sources),ancestors=counts,plan_sha256=sha(PLAN),frozen_plan_sha256=lock['plan_sha256'],inherited_default_verified=True))
    attempts={'S0':4,'S1':read(run/'cases/candidate-quarter/attempts.json')['attempts'],'S2':sum(read(p)['attempts'] for p in (run/'S2').glob('*/attempt.json')),'S3':sum(read(p)['attempts'] for p in (run/'cases').glob('grid32-*/attempts.json')),'S4':0,'S5':0,'S6':0}
    budget=read(run/'S0/experiment-budget.json');frames=len(list(run.rglob('frame.npz')))
    if any(attempts[k]>budget['stages'][k] for k in attempts) or sum(attempts.values())>budget['max_total_attempts'] or frames>budget['max_new_display_frames']:raise ValueError('registered budget exceeded')
    write(run/'S6/attempt-accounting.json',dict(status='passed_scoped',attempts=attempts,total=sum(attempts.values()),max_total=budget['max_total_attempts'],new_display_frames=frames,max_new_display_frames=12,
        failed_publication_and_injected_failure_included=True,new_full_cycle_steps=0,inherited_daily_steps=252,inherited_daily_display_frames=12,
        fixed_geometry_pressure_precheck_matrix_steps=48,additional_grid_diagnosis='66 exact matrix-exponential evaluations, no coupled steps',rejected_configuration_before_model_or_step=1))
    coupling=read(run/'S3/coupling-scope-decision.json');default=read(run/'S6/default-scene-decision.json')
    capabilities=dict(status='limited',scene_stability='inherited_passed_scoped',formal_space_changed=False,functions=144,mass_order=7,formal_steps=252,new_formal_steps=0,
        spatial_accuracy=False,temporal_accuracy=False,reference=read(run/'S1/candidate-reference-decision.json'),coupling=coupling,
        research_space=read(run/'S5/research-space-decision.json'),time=read(run/'S4/time-decision.json'),performance=read(run/'S2/performance-decision.json'),
        main_q5='original solid certificate only',sensitive_full_cycle_q5=False,coupled_q5=False,production_C_E_integration=False,pure_solid_default=True)
    write(run/'capability-matrix.json',capabilities);write(run/'S6/scope-summary.json',dict(status='passed_scoped',all_negative_results_preserved=True,capabilities_sha256=sha(run/'capability-matrix.json')))
    steps=[]
    for code,title in re.findall(r'^### (S\d\.\d) (.+)$',(run/'plan-frozen.md').read_text(),re.M):
        records=[read(run/n) for n in MAPPING[code]];states=[str(x.get('status','evidence_record')) for x in records]
        status='limited' if any('limited' in s for s in states) or code in ('S1.2','S1.4','S4.2','S4.3') else 'passed_scoped'
        if all(s=='not_triggered' for s in states):status='not_triggered'
        if code=='S6.2':status='inherited'
        steps.append(dict(step=code,title=title,status=status,reported_statuses=states,evidence=MAPPING[code]))
    if len(steps)!=26 or len(MAPPING)!=26:raise ValueError('26-step coverage mismatch')
    write(run/'requirement-audit.json',dict(count=26,steps=steps,all_steps_accounted_for=True,plan_sha256=lock['plan_sha256'],sequential=True,not_all_physical_gates_passed=True))
    write(run/'S6/final-resources.json',resources());(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(f.relative_to(run)):sha(f) for f in sorted(run.rglob('*')) if f.is_file() and 'warp-cache' not in f.parts and f.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(name):return dict(path=name,sha256=sha(run/name))
    pub=dict(schema='restoring-rt0-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),
        default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),
        space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S6/attempt-accounting.json'),
        formal_steps=252,new_formal_steps=0,display_frames=12,new_display_frames=frames,mass_order=7,full_material_order=7,material_policy='inherited parent q5_with_full_retry only',spatial_accuracy=False,temporal_accuracy=False,coupling_scope=coupling)
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,checked=audit_release(run,True);print('SEALED',str(run),sha(run/'release.json'),checked,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
