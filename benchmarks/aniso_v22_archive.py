"""Verify provenance and seal a complete v22 delivery, including failed gates."""
import ast,json,re,shutil,zipfile,hashlib
from benchmarks.aniso_v22_common import *

def tests():
    log=Path('/tmp/mpm-v22-regression.log').read_text();m=re.search(r'^Ran (\d+) tests? in ',log,re.MULTILINE);assert m and int(m[1])==40 and log.rstrip().endswith('OK')
    files=list((ROOT/'engine/aniso_phase1').glob('*.py'))+list((ROOT/'tests').glob('test_aniso_v2*.py'))+[ROOT/'tests/test_aniso_compatible_carrier.py',ROOT/'benchmarks/aniso_local_q3.py',ROOT/'benchmarks/aniso_local_reference.py']
    write(OUT/'tests.json',dict(passed=True,tests=40,new_tests=8,existing_related_tests=32,full_repository_suite=False,cuda_tested=False,log='execution-logs/mpm-v22-regression.log',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))

def main():
    assert not (OUT/'completion-check.json').exists();assert load(OUT/'postprocess-status.json')['visual_review_completed'];t=load(OUT/'tests.json');assert t['passed']
    for n,d in t['source_sha256'].items():assert sha(ROOT/n)==d,n
    for path in [OUT/'protocol.json',OUT/'space-protocol.json',OUT/'multiscale/space-protocol.json',OUT/'metrics-protocol.json',OUT/'reference-resume-protocol.json']:
        for n,d in load(path)['source_sha256'].items():assert sha(ROOT/n)==d,n
    prior=load(BASE/'v21/artifact-sha256.json')
    for n,d in prior.items():assert sha(ROOT/n)==d,n
    oldpy={n:d for n,d in load(BASE/'v21/source-delivered-sha256.json').items() if n.endswith('.py')}
    for n,d in oldpy.items():assert sha(ROOT/n)==d,n
    r=load(OUT/'reference-summary.json');f=load(OUT/'final-spatial-acceptance.json');s=load(OUT/'static-audits.json');m=load(OUT/'multiscale-audit.json');a=load(OUT/'nonlinear-audits.json')
    assert r['completed'] and f['completed'] and s['completed'] and m['completed'] and a['completed'];assert s['candidates']==72 and m['static_candidates']==24 and s['all_passed'] and m['all_static_passed'];assert len(a['records'])==3 and all(v['passed'] for v in a['records']) and m['nonlinear']['passed']
    assert sha(ROOT/f['reference'])==f['reference_sha256'];plans=load(OUT/'metrics-protocol.json')['cases']
    for name,row in f['cases'].items():assert sha(ROOT/plans[name])==row['source_sha256'],name
    for p in (2,3,4):
        folder=OUT/f'space/q{p}';d=load(folder/'summary.json');audit=a['records'][p-2];assert d['completed'] and d['basis_reconstruction_relative_error']<1e-9;assert sha(folder/'round6.npz')==audit['source_sha256'];assert sha(folder/'basis-raw.npz')==audit['basis_raw_sha256'];assert sha(folder/'basis-transform.npz')==audit['basis_transform_sha256']
    folder=OUT/'multiscale/space/q4';audit=m['nonlinear'];assert sha(folder/'round6.npz')==audit['source_sha256'];assert sha(folder/'basis-raw.npz')==audit['basis_raw_sha256'];assert sha(folder/'basis-transform.npz')==audit['basis_transform_sha256']
    resume=load(OUT/'reference-resume-protocol.json');assert sha(Path(resume['failed_log']))==resume['failed_log_sha256']
    for name,v in r['cases'].items():
        assert v['passed'] and v['relative_residual']<1e-8 and v['work_identity_relative']<1e-8 and not v['mass_included'] and v['stiffness_shift']==0.
        assert load(OUT/'reference'/(name+'.json'))==v
    logdir=OUT/'execution-logs';logdir.mkdir(exist_ok=False)
    for p in Path('/tmp').glob('mpm-v22-*.log'):shutil.copy2(p,logdir/p.name)
    files=[]
    for directory in ('engine','demos','benchmarks','tests','utils'):files+=list((ROOT/directory).rglob('*.py'))
    files+=list((ROOT/'docs').glob('*.md'));files+=[ROOT/n for n in ('README.md','RUNNING_RESTORED.md','pyproject.toml','uv.lock','LICENSE')];manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()}
    for n in manifest:
        if n.endswith('.py') and ('v22' in n or n.startswith('engine/aniso_phase1/')):ast.parse((ROOT/n).read_text())
    with zipfile.ZipFile(OUT/'source-delivered.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in manifest:z.write(ROOT/n,n)
        z.writestr('SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'source-delivered.zip') as z:
        for n,d in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==d,n
    write(OUT/'source-delivered-sha256.json',manifest)
    report=ROOT/'docs/ANISO_LITE_GRIP_HIGH_ORDER_ZH.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if target.endswith(('completion-check.json','artifact-sha256.json')):continue
        assert (report.parent/target).exists(),target
    c=f['cases']['q4-multiscale144'];completion=dict(requested_implementation_and_testing_completed=True,version='v22',related_tests=40,new_tests=8,static_candidates=96,all_static_candidates_positive=True,real_nonlinear_candidate_audits=4,accepted_high_order_reference_fields=4,original_reference_attempt_failed=True,tighter_reference_attempts=len(list((OUT/'reference/attempts').glob('*.json'))),full_space_limit_solves=2,reference_self_checks_passed=r['all_stress_and_fiber_passed'],reference_regions=r['regions'],spatial_stress_accuracy_passed=c['stress_passed'],fiber_strain_accuracy_passed=c['fiber_strain_passed'],final_candidate=c,new_local_space_in_full_moving_dynamics=False,production_default_changed=False,full_repository_suite=False,cuda_tested=False,old_v21_artifacts_unchanged=len(prior),old_python_sources_unchanged=len(oldpy),source_files=len(manifest),source_zip_sha256=sha(OUT/'source-delivered.zip'),report=str(report.relative_to(ROOT)))
    write(OUT/'completion-check.json',completion);artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'artifact-sha256.json'};write(OUT/'artifact-sha256.json',artifacts)
    for n,d in artifacts.items():assert sha(ROOT/n)==d,n
    print('SEALED',len(artifacts),'artifacts',len(manifest),'sources',json.dumps(completion),flush=True)
if __name__=='__main__':
    import sys
    tests() if len(sys.argv)>1 and sys.argv[1]=='tests' else main()
