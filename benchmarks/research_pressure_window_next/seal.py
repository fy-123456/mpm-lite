"""Seal the 26-step child and preserve every bounded research limitation."""
from pathlib import Path
import argparse,re
from .provenance import *
from .publication import audit_release

MAPPING={
 'S0.1':['S0/version-audit.json','S0/frozen-inputs.json'],
 'S0.2':['S0/interface-contract.json','S0/scope-map.json'],
 'S0.3':['S0/compatibility.json'],
 'S1.1':['S1/full-window-protocol.json'],
 'S1.2':['S1/exact-discrete-reference.json','S1/operator-check.json'],
 'S1.3':['S1/full-window-check.json','S1/grid-comparison.json'],
 'S1.4':['S1/pressure-scope-decision.json'],
 'S2.1':['S2/common-model-lock.json','S2/coupled-protocol.json'],
 'S2.2':['S2/residual-and-time-scale-audit.json'],
 'S2.3':['S2/volume-gradient-check.json','S2/mixed-operator-check.json'],
 'S2.4':['S2/coupled-scene-check.json','S2/coupled-time-reference.json'],
 'S2.5':['S2/transaction-and-restart.json'],
 'S2.6':['S2/coupling-scope-decision.json'],
 'S3.1':['S3/projection-audit.json','S3/operator-check.json'],
 'S3.2':['S3/force-acceleration-modal-diagnostic.json','S3/mass-modal-check.json'],
 'S3.3':['S3/candidate-time-check.json'],
 'S3.4':['S3/research-space-decision.json'],
 'S4.1':['S4/observable-phase-review.json'],
 'S4.2':['S4/time-decision.json'],
 'S5.1':['S5/profile.json'],
 'S5.2':['S5/performance-candidate.json'],
 'S5.3':['S5/performance-decision.json'],
 'S6.1':['S6/final-tests-result.json','S6/raw-physical-check.json','S6/parent-trajectory-comparison.json'],
 'S6.2':['S6/qualification-final.json','S6/final-scene.json','S6/fallback-and-restart.json'],
 'S6.3':['S6/physical-review.json','S6/load-check.json'],
 'S6.4':['S6/publication-preflight.json']}


def seal(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    lock=verify(run);p=read(run/'S6/final-protocol.json');default=p['default_case']
    for name in ('S0/compatibility.json','S6/final-scene.json','S6/raw-physical-check.json','S6/physical-review.json','S6/final-tests-result.json','S6/load-check.json','S2/transaction-and-restart.json'):
        if read(run/name)['status'] not in ('passed','passed_scoped'):raise ValueError('missing publication gate '+name)
    for name in ('S6/numeric-lock.json','S6/final-model-lock.json','S6/final-tests-result.json'):
        if read(run/name)['numerical_source_sha256']!=source_files():raise ValueError('numeric source drift')
    if p['q5_qualified']:
        for name in ('S5/fault-and-restart.json','S6/cache-retry-check.json','S5/scope-boundaries.json'):
            if read(run/name)['status']!='passed_scoped':raise ValueError('missing q5 recovery gate')
    counts=audit_parent(True)[2];sources=all_sources();check(ROOT,sources)
    write(run/'S6/publication-preflight.json',dict(status='passed_scoped',sources=len(sources),ancestors=counts,plan_sha256=sha(PLAN),frozen_plan_sha256=lock['plan_sha256']))
    steps=[]
    for code,title in re.findall(r'^#### (S\d\.\d) (.+)$',(run/'plan-frozen.md').read_text(),re.M):
        records=[read(run/n) for n in MAPPING[code]];states=[str(x.get('status','evidence_record')) for x in records]
        status='failed' if 'failed' in states else 'limited' if any('limited' in x or x=='retain_252_scoped' for x in states) else 'passed_scoped'
        if code=='S2.4' and read(run/'S2/coupled-scene-check.json')['mechanical_response_unresolved']:status='limited'
        if all(x=='not_triggered' for x in states):status='not_triggered'
        steps.append(dict(step=code,title=title,status=status,reported_statuses=states,evidence=MAPPING[code]))
    if len(steps)!=26 or len(MAPPING)!=26:raise ValueError('26-step coverage mismatch')
    write(run/'requirement-audit.json',dict(count=26,steps=steps,all_steps_accounted_for=True,plan_sha256=lock['plan_sha256'],sequential=True,not_all_physical_gates_passed=True))
    scene=read(run/'S6/final-scene.json');pressure=read(run/'S1/pressure-scope-decision.json');coupling=read(run/'S2/coupling-scope-decision.json')
    write(run/'capability-matrix.json',dict(scene_stability='passed_scoped',selected_space=read(run/'selected-space.json')['selected'],formal_space_changed=False,functions=144,mass_order=7,steps=252,spatial_accuracy=False,temporal_accuracy=False,pressure=pressure,coupling=coupling,research_candidate=read(run/'S3/research-space-decision.json'),time=read(run/'S4/time-decision.json'),performance=read(run/'S5/performance-decision.json'),main_q5=p['q5_qualified'],sensitive_full_cycle_q5=False,coupled_q5=False,production_C_E_integration=False,pure_solid_default=True))
    write(run/'S6/final-resources.json',resources());(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(f.relative_to(run)):sha(f) for f in sorted(run.rglob('*')) if f.is_file() and 'warp-cache' not in f.parts and f.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(name):return dict(path=name,sha256=sha(run/name))
    pub=dict(schema='pressure-window-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default,sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),qualification=entry('S6/qualification-final.json'),scene=entry('S6/final-scene.json'),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),case_identity_sha256=sha(run/'cases'/default/'identity.json'),case_protocol_sha256=sha(run/'cases'/default/'execution-protocol.json'),steps=252,display_frames=scene['frames'],mass_order=7,full_material_order=7,material_policy='q5_with_full_retry' if p['q5_qualified'] else 'full_only',spatial_accuracy=False,temporal_accuracy=False,coupling_scope=coupling)
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,checked=audit_release(run,True);print('SEALED',str(run),sha(run/'release.json'),checked,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
