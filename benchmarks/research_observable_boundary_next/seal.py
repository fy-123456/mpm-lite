"""Seal accounted work and explicit limited/not-triggered research branches."""
from pathlib import Path
import argparse,re
from .provenance import *
from .publication import audit_release

MAPPING={
'S0.1':['S0/version-audit.json','S0/environment.json'],
'S0.2':['S0/interface-contract.json','S0/dependency-invalidation.json'],
'S0.3':['S0/protocol.json','S0/frozen-inputs.json','S0/storage-migration.json'],
'S0.4':['S0/compatibility.json'],
'S1.1':['S1/origin-audit.json','S1/modal-observable-map.json'],
'S1.2':['S1/time-reference-protocol.json'],
'S1.3':['S1/time-raw-check.json'],
'S1.4':['S1/time-comparison.json','S1/time-decision.json'],
'S1.5':['S1/time-decision.json','S1/time-scope.json'],
'S2.1':['S2/observation-times.json','S2/early-flow-diagnostic.json'],
'S2.2':['S2/grid-protocol.json'],
'S2.3':['S2/topology-operator-check.json'],
'S2.4':['S2/exact-time-reference.json','S2/fixed-grid-comparison.json','S2/time-error-diagnostic.json','S2/corrected-start-protocol.json','S2/corrected-start-check.json'],
'S2.5':['S2/pressure-scope-decision.json'],
'S3.1':['S3/candidate-protocol.json','S3/operator-check.json'],
'S3.2':['S3/mass-modal-check.json','S3/projection-check.json'],
'S3.3':['S3/dynamic-check.json','S3/dynamic-limit-review.json'],
'S3.4':['S3/research-space-decision.json'],
'S4.1':['S4/profile.json'],
'S4.2':['S4/candidate-decision.json','S4/equivalence.json'],
'S4.3':['S4/paired-performance.json','S4/performance-decision.json'],
'S5.1':['S5/common-model-lock.json'],
'S5.2':['S5/volume-gradient-check.json','S5/mixed-operator-check.json'],
'S5.3':['S5/coupled-scene-check.json'],
'S5.4':['S5/transaction-and-restart.json'],
'S6.1':['S6/numeric-lock.json','S6/final-tests-result.json'],
'S6.2':['S6/final-scene.json','S6/raw-physical-check.json','S6/parent-trajectory-comparison.json'],
'S6.3':['S6/qualification-final.json','S6/material-decision.json','S6/fallback-and-restart.json'],
'S6.4':['S6/physical-review.json','S6/load-check.json']}

def requirement_status(code,records):
    states=[str(r.get('status','evidence_record')) for r in records]
    if any(x=='failed' for x in states):return 'failed'
    if any(x=='limited' or 'limited' in x or x=='retain_252_scoped' for x in states):return 'limited'
    if any(x=='not_triggered' for x in states):return 'not_triggered'
    return 'passed_scoped'

def seal(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    lock=verify(run);protocol=read(run/'S6/final-protocol.json');default=protocol['default_case']
    for name in ('S0/compatibility.json','S6/final-scene.json','S6/raw-physical-check.json','S6/physical-review.json','S6/final-tests-result.json','S6/load-check.json'):
        if read(run/name)['status'] not in ('passed','passed_scoped'):raise ValueError('unmet publication gate '+name)
    for name in ('S6/numeric-lock.json','S6/final-model-lock.json','S6/final-tests-result.json'):
        if read(run/name)['numerical_source_sha256']!=source_files():raise ValueError('final numeric source drift')
    if protocol['q5_qualified']:
        for name in ('S5/fault-and-restart.json','S6/cache-retry-check.json','S5/scope-boundaries.json'):
            if read(run/name)['status']!='passed_scoped':raise ValueError('missing q5 recovery gate')
    steps=[]
    for code,title in re.findall(r'^### (S\d\.\d) (.+)$',(run/'plan-frozen.md').read_text(),re.M):
        evidence=MAPPING[code];records=[read(run/n) for n in evidence];steps.append(dict(step=code,title=title,status=requirement_status(code,records),reported_statuses=[x.get('status','evidence_record') for x in records],evidence=evidence))
    if len(steps)!=29 or len(MAPPING)!=29:raise ValueError('29-step coverage mismatch')
    write(run/'requirement-audit.json',dict(count=29,steps=steps,all_steps_accounted_for=True,plan_sha256=lock['plan_sha256'],sequential=True,meaning='accounted for does not mean every numerical research gate passed'))
    scene=read(run/'S6/final-scene.json');space=read(run/'selected-space.json');pressure=read(run/'S2/pressure-scope-decision.json');perf=read(run/'S4/performance-decision.json')
    caps=dict(scene_stability='passed_scoped',selected_space=space['selected'],formal_space_changed=False,functions=144,mass_order=7,steps=scene['steps'],spatial_accuracy=False,temporal_accuracy=False,research_candidate=read(run/'S3/research-space-decision.json'),time=read(run/'S1/time-decision.json'),pressure=pressure,performance=perf,main_q5=protocol['q5_qualified'],sensitive_full_cycle_q5=False,coupled_q5=False,production_C_E_integration=False,pure_solid_default=True)
    write(run/'capability-matrix.json',caps);write(run/'S6/final-resources.json',resources());(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    sources=all_sources();snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(path):return dict(path=path,sha256=sha(run/path))
    pub=dict(schema='observable-boundary-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default,sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),qualification=entry('S6/qualification-final.json'),scene=entry('S6/final-scene.json'),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),case_identity_sha256=sha(run/'cases'/default/'identity.json'),case_protocol_sha256=sha(run/'cases'/default/'execution-protocol.json'),steps=scene['steps'],display_frames=scene['frames'],mass_order=7,full_material_order=7,material_policy='q5_with_full_retry' if protocol['q5_qualified'] else 'full_only',spatial_accuracy=False,temporal_accuracy=False,coupling_scope=pressure)
    check(ROOT,sources);check(run/'final-source',sources);check(run,artifacts);audit_parent(True);write(run/'release.json',pub);_,counts=audit_release(run);print('SEALED',run,sha(run/'release.json'),counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
