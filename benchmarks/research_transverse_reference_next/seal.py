"""Twenty-step scoped release with complete source and artifact provenance."""
import argparse,re
from .provenance import *
from .publication import audit_release
from .finalize import accounting

MAPPING={
'S0.1':['S0/version-audit.json'],
'S0.2':['S0/physical-contract.json','S0/checkpoint-sources.json','S0/observation-contract.json'],
'S0.3':['S0/experiment-budget.json','S0/resource-policy.json','S0/storage-migration.json'],
'S1.1':['S1/reference-input.json','S1/operator-check.json'],
'S1.2':['S1/reference-self-check.json'],
'S1.3':['S1/fixed-skeleton-time-review.json'],
'S1.4':['S1/window-and-time-decision.json','S1/timescales.json','S1/coupled-model-difference.json','S1/geometric-volume-crosscheck.json'],
'S2.1':['S2/backend-binding.json'],
'S2.2':['S2/bridge-check.json','S2/recovery-scope.json','S5/load-check.json'],
'S2.3':['S2/extension-review.json'],
'S2.4':['S2/time-not-triggered.json'],
'S3.1':['S3/DV-profile.json','S3/DV-adjoint-profile.json','S3/profile-interpretation.json'],
'S3.2':['S3/candidate-protocol.json'],
'S3.3':['S3/operator-check.json','S3/buffer-lifecycle.json'],
'S3.4':['S3/paired-performance.json','S3/continuous-check.json','S3/backend-decision.json','S5/selected-load-check.json'],
'S4.1':['S4/space-entry-decision.json'],
'S4.2':['S4/next-entry-decisions.json'],
'S5.1':['S5/test-report.json','S5/load-check.json','S5/daily-load-check.json','S5/physical-ledger-audit.json','S5/attempt-accounting.json'],
'S5.2':['S5/visual-assets-check.json','S5/visual-review.json','S5/cli-http-check.json','S5/visualization-origin.json'],
'S5.3':['capability-matrix.json','documentation.json','S5/numerical-source-audit.json','S5/final-resources.json']}


def main(run):
    run=Path(run).absolute();lock=mutable(run);audit_parent(True);sources=all_sources();ac=accounting(run)
    for p in ('S1/reference-self-check.json','S2/bridge-check.json','S3/operator-check.json','S3/continuous-check.json','S5/test-report.json','S5/load-check.json','S5/selected-load-check.json','S5/physical-ledger-audit.json','S5/numerical-source-audit.json','S5/visual-review.json','S5/cli-http-check.json'):
        if read(run/p)['status']!='passed_scoped':raise ValueError('required qualification missing '+p)
    test=read(run/'S5/test-report.json');check(ROOT,test['tested_sources'])
    write(run/'S5/attempt-accounting.json',ac);res=resources()
    if res['system_free_GiB']<5:raise ValueError('system storage migration needed')
    write(run/'S5/final-resources.json',dict(**res,output_bytes=ac['output_bytes_at_check'],new_frames=0,material_and_mass_order=7))
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    codes=re.findall(r'^\*\*(S\d\.\d+)｜',(run/'plan-frozen.md').read_text(),re.M)
    if set(codes)!=set(MAPPING) or len(codes)!=20:raise ValueError('plan coverage differs')
    skipped={'S2.3':'fixed-skeleton extra signal below floor; coupled extension deferred until matched reference','S2.4':'engineering time passed; model difference explained by geometric volume exchange','S4.1':'no matched dynamic reference, retain144'}
    limited={'S1.3':'engineering checks pass; raw startup interval flow and small-signal relative accuracy remain limited','S1.4':'small fixed-skeleton signal; moving-solid difference dominated by volume exchange','S3.1':'local segment remains partly combined; construction and diagnostic counters kept separate','S4.2':'entry contracts only, production extensions not executed'}
    steps=[dict(step=k,status='not_triggered' if k in skipped else 'limited' if k in limited else 'passed_scoped',reason=skipped.get(k,limited.get(k,'registered scope verified')),evidence=[dict(path=n,sha256=sha(run/n)) for n in paths]) for k,paths in MAPPING.items()]
    write(run/'requirement-audit.json',dict(count=20,steps=steps,all_steps_accounted_for=True,all_accuracy_goals_passed=False,sequential=True))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(name):return dict(path=name,sha256=sha(run/name))
    default=read(run/'S5/default-scene-decision.json');backend=read(run/'S3/backend-decision.json')
    pub=dict(schema='transverse-reference-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S5/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,new_display_frames=0,mass_order=7,full_material_order=7,material_policy='inherited pure-solid q5; coupled fullq7',spatial_accuracy=False,temporal_accuracy=False,fixed_skeleton_engineering_time_checks=True,selected_geometry=backend['backend'])
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,counts=audit_release(run,True)
    print('SEALED',run,sha(run/'release.json'),{k:v for k,v in counts.items() if k!='ancestors'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
