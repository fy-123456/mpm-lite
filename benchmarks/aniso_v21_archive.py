"""Seal v21 evidence without modifying earlier delivered code or results."""
import ast,json,re,shutil,zipfile,hashlib
from pathlib import Path
from benchmarks.aniso_v21_common import *

def tests():
    log=Path('/tmp/mpm-v21-regression.log').read_text();match=re.search(r'^Ran (\d+) tests? in ',log,re.MULTILINE);assert match and log.rstrip().endswith('OK');assert int(match[1])==31
    supplemental=Path('/tmp/mpm-v21-quartic-tests.log').read_text();assert re.search(r'^Ran 1 test in ',supplemental,re.MULTILINE) and supplemental.rstrip().endswith('OK')
    files=list((ROOT/'engine/aniso_phase1').glob('*.py'))+list((ROOT/'benchmarks').glob('aniso_v21*.py'))+list((ROOT/'tests').glob('test_aniso_v21*.py'))
    files+=[ROOT/n for n in ['tests/test_aniso_compatible_carrier.py','tests/test_aniso_v20_variational.py','benchmarks/aniso_local_q3.py','benchmarks/aniso_local_reference.py']]
    write(OUT/'tests.json',dict(passed=True,tests=32,new_tests=14,existing_related_regressions=18,full_repository_suite=False,cuda_tested=False,logs=['execution-logs/mpm-v21-regression.log','execution-logs/mpm-v21-quartic-tests.log'],source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files}))

def main():
    assert not (OUT/'completion-check.json').exists();assert load(OUT/'finish-status.json')['completed'];assert load(OUT/'postprocess-status.json')['visual_review_completed'];t=load(OUT/'tests.json');assert t['passed'];s=load(OUT/'static-audits.json');assert s['completed'] and s['candidates']==176 and s['all_passed'];f=load(OUT/'final-spatial-acceptance.json');r=load(OUT/'reference-self-checks.json');assert f['completed'] and r['completed'];assert f['reference'].endswith('level1-q4.npz') and sha(ROOT/f['reference'])==f['reference_sha256']
    for n,d in t['source_sha256'].items():assert sha(ROOT/n)==d,n
    protocols=[OUT/'protocol.json',OUT/'space-protocol.json',OUT/'adaptive-protocol.json',OUT/'adaptive/gain/protocol.json',OUT/'reference-extension/protocol.json',OUT/'reference-extension/completion-protocol.json']
    for p in protocols:
        for n,d in load(p)['source_sha256'].items():assert sha(ROOT/n)==d,n
    old=load(BASE/'v20/artifact-sha256.json')
    for n,d in old.items():assert sha(ROOT/n)==d,n
    oldpy={n:d for n,d in load(BASE/'v20/source-delivered-sha256.json').items() if n.endswith('.py')}
    for n,d in oldpy.items():assert sha(ROOT/n)==d,n
    for n,row in load(OUT/'final-candidate-selection.json')['cases'].items():assert sha(ROOT/row)==f['cases'][n]['source_sha256'],n
    audits=load(OUT/'candidate-nonlinear-audits.json')['records']+load(OUT/'candidate-gain-nonlinear-audit.json')['records'];assert len(audits)==3 and all(a['passed'] for a in audits)
    for a in audits:
        folder=OUT/'adaptive'/a['name'];assert sha(folder/'round6.npz')==a['source_sha256'];assert sha(folder/'round6-basis.npz')==a['basis_sha256']
    for mesh in ('coarse','graded'):assert load(OUT/'space'/mesh/'summary.json')['completed']
    for ordering in ('stress','geometric','gain'):assert load(OUT/'adaptive'/ordering/'summary.json')['completed']
    logdir=OUT/'execution-logs';logdir.mkdir(exist_ok=False)
    for p in Path('/tmp').glob('mpm-v21-*.log'):shutil.copy2(p,logdir/p.name)
    files=[]
    for directory in ('engine','demos','benchmarks','tests','utils'):files+=list((ROOT/directory).rglob('*.py'))
    files+=list((ROOT/'docs').glob('*.md'));files+=[ROOT/n for n in ('README.md','RUNNING_RESTORED.md','pyproject.toml','uv.lock','LICENSE')];manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()}
    for n in manifest:
        if n.endswith('.py') and ('v21' in n or n.startswith('engine/aniso_phase1/')):ast.parse((ROOT/n).read_text())
    with zipfile.ZipFile(OUT/'source-delivered.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in manifest:z.write(ROOT/n,n)
        z.writestr('SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'source-delivered.zip') as z:
        for n,d in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==d
    write(OUT/'source-delivered-sha256.json',manifest)
    report=ROOT/'docs/ANISO_LITE_FIBER_SPACE_ADAPTIVITY_ZH.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if target.endswith(('completion-check.json','artifact-sha256.json')):continue
        assert (report.parent/target).exists(),target
    case=f['cases']['adaptive/gain/round6'];completion=dict(requested_implementation_and_testing_completed=True,version='v21 static fiber strain / higher-order references / stress-selected local spaces',related_tests=32,new_tests=14,static_candidates=176,all_static_candidates_positive=True,real_nonlinear_candidate_audits=3,reference_high_order_new_solves=7,old_reference_operator_replay=1,material_only_Q2_limit_solves=1,final_reference_nodes=load(OUT/'reference-completed-summary.json')['cases']['level1-q4']['nodes'],reference_self_checks_passed=r['all_stress_and_fiber_passed'],spatial_stress_accuracy_passed=case['stress_passed'],fiber_strain_accuracy_passed=case['fiber_passed'],new_local_space_in_full_moving_dynamics=False,production_default_changed=False,full_repository_suite=False,cuda_tested=False,old_v20_artifacts_unchanged=len(old),old_python_sources_unchanged=len(oldpy),source_files=len(manifest),source_zip_sha256=sha(OUT/'source-delivered.zip'),final_gain_case=case,report=str(report.relative_to(ROOT)),limitations=['Linear static F45 precision study; other directions check static consistency, not certified accuracy gains.','Reference h/p differences remain explicit; no rigorous continuum error bound.','The corrected-error ranking is not superior at every basis budget.','Material integration mesh error remains even in the full Q2 limit.','New spatial variables are not integrated into the moving dynamic algorithm.'])
    write(OUT/'completion-check.json',completion);artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'artifact-sha256.json'};write(OUT/'artifact-sha256.json',artifacts)
    for n,d in artifacts.items():assert sha(ROOT/n)==d,n
    print('SEALED',len(artifacts),'artifacts',len(manifest),'sources',json.dumps(completion),flush=True)
if __name__=='__main__':
    import sys
    tests() if len(sys.argv)>1 and sys.argv[1]=='tests' else main()
