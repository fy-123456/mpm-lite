"""Seal a scoped delivery; conditional branches are accounted for, not invented."""
import argparse,re
from .provenance import *
from .finalize import accounting
from .publication import audit_release

MAPPING={
'S0.1':['S0/version-audit.json'],
'S0.2':['S0/physical-contract.json','S0/checkpoint-sources.json','S2/backend-binding.json'],
'S0.3':['S0/resource-policy.json','S0/storage-migration.json','S0/experiment-budget.json'],
'S1.1':['S1/equation-map.md','S1/reference-protocol.json'],
'S1.2':['S1/derivative-check.json','S1/reference-cost-decision.json'],
'S1.3':['S1/assembly.json','S1/linear-blocks.npz'],
'S1.4':['S1/reference-self-check.json','S1/local-validity.json','S1/holdout-review.json'],
'S1.5':['S1/window-decision.json','S1/engineering-review.json'],
'S2.1':['S2/bridge-check.json','S2/recovery-scope.json'],
'S2.2':['S2/time-review.json'],
'S2.3':['S2/window-review.json'],
'S3.1':['S3/BD-profile.json'],
'S3.2':['S3/candidate-protocol.json'],
'S3.3':['S3/operator-check.json','S3/buffer-lifecycle.json','S3/paired-performance.json'],
'S3.4':['S3/backend-decision.json','S3/continuous-check.json'],
'S4.1':['S4/spatial-entry-decision.json'],
'S4.2':['S4/capability-matrix.json'],
'S5.1':['S5/test-report.json','S5/selected-load-check.json','S5/daily-load-check.json','S5/physical-ledger-audit.json','S5/attempt-accounting.json'],
'S5.2':['S5/visual-assets-check.json','S5/visual-review.json','S5/cli-http-check.json','S5/visualization-origin.json'],
'S5.3':['capability-matrix.json','documentation.json','S5/numerical-source-audit.json','S5/final-resources.json']}

def main(run):
    run=Path(run).absolute();mutable(run);audit_parent(True);sources=all_sources()
    for path in ('S1/derivative-check.json','S1/reference-self-check.json','S1/holdout-review.json','S2/bridge-check.json','S3/operator-check.json','S3/buffer-lifecycle.json','S3/paired-performance.json','S5/test-report.json','S5/selected-load-check.json','S5/physical-ledger-audit.json','S5/numerical-source-audit.json','S5/visual-review.json','S5/cli-http-check.json'):
        if read(run/path)['status']!='passed_scoped':raise ValueError('qualification missing '+path)
    check(ROOT,read(run/'S5/test-report.json')['tested_sources']);ac=accounting(run);write(run/'S5/attempt-accounting.json',ac)
    res=resources()
    if res['system_free_GiB']<5:raise ValueError('storage migration required')
    write(run/'S5/final-resources.json',dict(**res,output_bytes=ac['output_bytes'],new_frames=0))
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN)))
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    codes=re.findall(r'^\*\*(S\d\.\d+)｜',(run/'plan-frozen.md').read_text(),re.M)
    if len(codes)!=20 or set(codes)!=set(MAPPING):raise ValueError('plan coverage mismatch')
    skip={'S2.2':'no isolated time error requiring new trajectory','S2.3':'no informative extension required','S3.4':'candidate speed/setup gate failed; keep BD, no continuity','S4.1':'no independent spatial reference; keep144'}
    limited={'S1.2':'bounded projected route selected','S1.3':'r8/r12 diagnostic only','S1.4':'independent BDF and one unused state pass, full-space bound unavailable','S1.5':'retain reference scope and current window','S4.2':'future production capability gates only'}
    steps=[dict(step=k,status='not_triggered' if k in skip else 'limited' if k in limited else 'passed_scoped',reason=skip.get(k,limited.get(k,'registered scoped checks passed')),evidence=[dict(path=n,sha256=sha(run/n)) for n in paths]) for k,paths in MAPPING.items()]
    write(run/'requirement-audit.json',dict(count=20,steps=steps,all_steps_accounted_for=True,all_accuracy_goals_passed=False,sequential=True))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(name):return dict(path=name,sha256=sha(run/name))
    parent=read(APP/'release.json');d=read(run/'S3/backend-decision.json')
    pub=dict(schema='coupled-reference-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,input_lock_sha256=sha(run/'input-lock.json'),sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S5/attempt-accounting.json'),default_case=parent['default_case'],default_case_source=parent['default_case_source'],formal_steps=252,new_formal_steps=0,new_display_frames=0,mass_order=7,full_material_order=7,material_policy=parent['material_policy'],spatial_accuracy=False,temporal_accuracy=False,coupled_affine_diagnostic=True,selected_geometry=d['backend'])
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,counts=audit_release(run,True)
    print('SEALED',run,sha(run/'release.json'),{k:v for k,v in counts.items() if k!='ancestors'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
