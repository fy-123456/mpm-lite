"""Seal all 28 plan steps while separating inherited and new physical evidence."""
from pathlib import Path
import argparse,re
from .provenance import *
from .publication import audit_release

MAPPING={
'S0.1':['S0/version-audit.json'],
'S0.2':['S0/interface-contract.json','S0/compatibility.json'],
'S0.3':['S0/resource-policy.json','S0/experiment-budget.json'],
'S1.1':['S1/memory-observer-check.json'],
'S1.2':['S1/checkpoint-bridge.json','S1/prefix-witness.json'],
'S1.3':['S1/candidate-coarse-check.json'],
'S1.4':['S1/candidate-half-protocol.json','cases/candidate-half/summary.json'],
'S1.5':['S1/time-vs-space-comparison.json','S1/candidate-reference-decision.json'],
'S2.1':['S2/projection-energy-audit.json'],
'S2.2':['S2/physical-force-acceleration-decomposition.json','S2/directional-energy-check.json'],
'S2.3':['S2/modal-output-diagnostic.json'],
'S2.4':['S2/research-space-decision.json'],
'S3.1':['S3/physical-scale-screening.json','S3/new-scene-protocol.json'],
'S3.2':['S3/model-parameter-binding.json','S3/initial-state-check.json'],
'S3.3':['S3/operator-check.json','S3/pressure-schedule-check.json','S3/quasi-newton-scope.json'],
'S3.4':['S3/observable-coupled-check.json','S3/coupled-time-comparison.json'],
'S3.5':['S3/grid-and-transaction-review.json','S3/coupling-scope-decision.json'],
'S4.1':['S4/output-phase-attribution.json','S4/window-selection.json'],
'S4.2':['S4/local-phase-check.json'],
'S4.3':['S4/time-decision.json'],
'S5.1':['S5/profile.json'],
'S5.2':['S5/performance-candidate.json','S5/compact-performance-candidate.json','S5/operator-equivalence.json'],
'S5.3':['S5/paired-performance.json'],
'S5.4':['S5/performance-decision.json','S5/final-numeric-lock.json'],
'S6.1':['S6/tests-result.json','S6/regression-impact.json'],
'S6.2':['S6/inherited-solid-evidence.json','S6/default-scene-decision.json','S6/fallback-and-restart.json'],
'S6.3':['S6/visual-review.json','S6/load-check.json'],
'S6.4':['S6/publication-preflight.json']}


def seal(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    lock=verify(run)
    for name in ('S0/compatibility.json','S6/tests-result.json','S6/visual-review.json','S6/load-check.json'):
        if read(run/name)['status'] not in ('passed','passed_scoped'):raise ValueError('missing release gate '+name)
    if read(run/'S6/tests-result.json')['numerical_source_sha256']!=source_files():raise ValueError('numeric source drift')
    check(ROOT,read(run/'S6/tests-result.json').get('tested_source_sha256',{}))
    counts=audit_parent(True)[2];sources=all_sources();check(ROOT,sources)
    write(run/'S6/publication-preflight.json',dict(status='passed_scoped',sources=len(sources),ancestors=counts,plan_sha256=sha(PLAN),frozen_plan_sha256=lock['plan_sha256']))
    steps=[]
    for code,title in re.findall(r'^#### (S\d\.\d) (.+)$',(run/'plan-frozen.md').read_text(),re.M):
        records=[read(run/n) for n in MAPPING[code]];states=[str(x.get('status','evidence_record')) for x in records]
        status='failed' if 'failed' in states else 'limited' if any('limited' in x or x=='retain_252_scoped' for x in states) else 'passed_scoped'
        if all(x=='not_triggered' for x in states):status='not_triggered'
        if any('inherited' in x for x in states):status='inherited_passed_scoped'
        steps.append(dict(step=code,title=title,status=status,reported_statuses=states,evidence=MAPPING[code]))
    if len(steps)!=28 or len(MAPPING)!=28:raise ValueError('28-step coverage mismatch')
    attempts={'S0':4,'S1':read(run/'S1/prefix-witness.json')['attempts']+sum(read(run/'cases'/c/'attempts.json')['attempts'] for c in ('candidate-tail','candidate-half')),'S2':0,'S3':sum(read(p)['attempts'] for p in (run/'cases').glob('observable-*/attempts.json')),'S4':len(read(run/'cases/phase-quarter/ledger.json')),'S5':sum(read(p)['attempts'] for p in (run/'S5').glob('*/attempt.json')),'S6':0}
    budget=read(run/'S0/experiment-budget.json')
    if any(attempts[k]>budget['stages'][k] for k in attempts) or sum(attempts.values())>budget['max_total_attempts']:raise ValueError('experiment budget exceeded')
    write(run/'S6/attempt-accounting.json',dict(status='passed_scoped',attempts=attempts,total=sum(attempts.values()),max_total=1026,includes_failed_and_witness_steps=True,fixed_geometry_pressure_matrix_probes_separate=64,inherited_formal_steps_not_new=True))
    write(run/'requirement-audit.json',dict(count=28,steps=steps,all_steps_accounted_for=True,plan_sha256=lock['plan_sha256'],sequential=True,not_all_physical_gates_passed=True))
    coupling=read(run/'S3/coupling-scope-decision.json');default=read(run/'S6/default-scene-decision.json')['default_case']
    write(run/'capability-matrix.json',dict(scene_stability='inherited_passed_scoped',selected_space=read(run/'selected-space.json')['selected'],formal_space_changed=False,functions=144,mass_order=7,formal_steps=252,new_formal_steps=0,spatial_accuracy=False,temporal_accuracy=False,coupling=coupling,research_candidate=read(run/'S2/research-space-decision.json'),time=read(run/'S4/time-decision.json'),performance=read(run/'S5/performance-decision.json'),main_q5='parent scope inherited only',sensitive_full_cycle_q5=False,coupled_q5=False,production_C_E_integration=False,pure_solid_default=True))
    write(run/'S6/final-resources.json',resources());(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(f.relative_to(run)):sha(f) for f in sorted(run.rglob('*')) if f.is_file() and 'warp-cache' not in f.parts and f.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(name):return dict(path=name,sha256=sha(run/name))
    pub=dict(schema='candidate-observable-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default,default_case_source=dict(release_root=str(APP),release_sha256=APP_SHA,case=default,identity_sha256=sha(APP/'cases'/default/'identity.json'),execution_protocol_sha256=sha(APP/'cases'/default/'execution-protocol.json')),sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S6/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,display_frames=12,mass_order=7,full_material_order=7,material_policy='inherited parent q5_with_full_retry only',spatial_accuracy=False,temporal_accuracy=False,coupling_scope=coupling)
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,checked=audit_release(run,True);print('SEALED',str(run),sha(run/'release.json'),checked,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
