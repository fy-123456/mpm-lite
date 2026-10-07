"""Seal 29 accounted steps; limited and untriggered branches remain explicit."""
from pathlib import Path
import argparse,re
from .provenance import *
from .publication import audit_release

def seal(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    lock=verify(run);protocol=read(run/'S6/final-protocol.json');default=protocol['default_case']
    required=['S0/compatibility.json','S6/final-scene.json','S6/raw-physical-check.json','S6/physical-review.json','S6/final-tests-result.json','S6/load-check.json']
    if protocol['q5_qualified']:required+=['S5/fault-and-restart.json','S6/cache-retry-check.json']
    for name in required:
        if read(run/name)['status'] not in ('passed','passed_scoped'):raise ValueError('unmet publication gate '+name)
    for name in ('S5/numeric-lock.json','S6/final-model-lock.json','S6/final-tests-result.json'):
        if read(run/name)['numerical_source_sha256']!=source_files():raise ValueError('final numerical source drift')
    mapping={
        'S0.1':['S0/version-audit.json'],'S0.2':['S0/compatibility.json'],'S0.3':['S0/protocol.json','S0/scope-map.json','S0/storage-migration.json'],
        'S1.1':['S1/reference-audit.json'],'S1.2':['S1/design-protocol.json','S1/ambient-projection-review.json'],
        'S1.3':['S1/novelty-audit.json','S1/conditioning-review.json','S1/second-candidate-decision.json'],
        'S1.4':['S1/candidates/cross-direction-snapshot6/operator-audit.json'],
        'S1.5':['S1/training-comparison.json'],'S1.6':['S1/reserved-direction-check.json'],'S1.7':['S1/dynamic-smoke.json','S1/space-decision.json'],
        'S2.1':['S2/baseline-output-review.json'],'S2.2':['S2/extension-protocol.json'],'S2.3':['S2/propagation-comparison.json'],'S2.4':['S2/time-scope-decision.json'],
        'S3.1':['S3/grid-error-audit.json'],'S3.2':['S3/topology-operator-check.json'],'S3.3':['S3/fixed-grid-comparison.json'],'S3.4':['S3/coupled-grid-check.json','S3/transaction-check.json'],'S3.5':['S3/pressure-scope-decision.json'],
        'S4.1':['S4/current-profile.json'],'S4.2':['S4/candidate-decision.json'],'S4.3':['S4/paired-performance.json','S4/performance-decision.json'],
        'S5.1':['S5/numeric-lock.json'],'S5.2':['S5/main-qualification-protocol.json','S6/qualification-final.json'],'S5.3':['S5/window-runtime-final.json','S5/scope-boundaries.json'],
        'S6.1':['S6/final-tests-result.json'],'S6.2':['S6/final-scene.json','S6/raw-physical-check.json'],'S6.3':['S6/material-decision.json','S5/fault-and-restart.json'],'S6.4':['S6/physical-review.json','S6/load-check.json']}
    steps=[]
    for code,title in re.findall(r'^### (S\d\.\d) (.+)$',(run/'plan-frozen.md').read_text(),re.M):
        evidence=mapping[code];records=[read(run/n) for n in evidence];raw=records[0].get('status','passed_scoped')
        status='not_triggered' if raw in ('not_triggered','condition_not_triggered') else 'limited' if raw in ('space_limited','reference_limited','grid_sensitive','limited_research_scope','limited') else 'passed_scoped'
        steps.append(dict(step=code,title=title,status=status,reported_status=raw,evidence=evidence))
    if len(steps)!=29:raise ValueError('29-step audit incomplete')
    write(run/'requirement-audit.json',dict(count=29,steps=steps,all_steps_accounted_for=True,plan_sha256=lock['plan_sha256'],sequential=True))
    space=read(run/'selected-space.json');scene=read(run/'S6/final-scene.json');pressure=read(run/'S3/pressure-scope-decision.json');perf=read(run/'S4/performance-decision.json')
    same=space['package']['sha256']==read(run/'baseline-space.json')['package']['sha256']
    caps=dict(scene_stability='passed_scoped',selected_space=space['selected'],formal_space_changed=not same,functions=144,mass_order=7,steps=scene['steps'],spatial_accuracy=False,temporal_accuracy=False,
        space_scope='F45/F60 training plus reserved52.5 static and eight-step smoke; full F45 cycle separately checked' if not same else 'inherited BASELINE',
        time_scope='BASELINE local 1.1->1.125 continuation only; unchanged formal252 grid',pressure=pressure,performance=perf,
        main_q5=protocol['q5_qualified'],sensitive_window=read(run/'S5/window-runtime-final.json')['status'],sensitive_full_cycle_q5=False,coupled_q5=False,production_C_E_integration=False,pure_solid_default=True)
    write(run/'capability-matrix.json',caps);write(run/'S6/final-resources.json',resources())
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes());write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    sources=all_sources();snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(path):return dict(path=path,sha256=sha(run/path))
    pub=dict(schema='cross-direction-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default,
        sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),qualification=entry('S6/qualification-final.json'),scene=entry('S6/final-scene.json'),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),
        case_identity_sha256=sha(run/'cases'/default/'identity.json'),case_protocol_sha256=sha(run/'cases'/default/'execution-protocol.json'),steps=scene['steps'],display_frames=scene['frames'],mass_order=7,full_material_order=7,material_policy='q5_with_full_retry' if protocol['q5_qualified'] else 'full_only',spatial_accuracy=False,temporal_accuracy=False,coupling_scope=pressure)
    if (run/'S5/continuous-window-qualification.json').exists():pub['window_qualification']=entry('S5/continuous-window-qualification.json')
    check(ROOT,sources);check(run/'final-source',sources);check(run,artifacts);audit_parent(True);write(run/'release.json',pub);_,counts=audit_release(run)
    print('SEALED',run,sha(run/'release.json'),counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
