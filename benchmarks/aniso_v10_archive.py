"""Seal/check v10 implementation evidence, preserving failed accuracy gates."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'docs/results/lite-aniso-mainline'
VERSIONS=('v10','v10-reference','v10-reference-fine','v10-reference-q2')


def read(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def verify(mapping):
    bad=[name for name,value in mapping.items() if digest(ROOT/name)!=value]
    if bad:raise RuntimeError('changed files: '+str(bad))
    return len(mapping)


def evidence():
    old={name:verify(read(BASE/name/'artifact-sha256.json'))
         for name in [f'v{i}' for i in range(1,10)]+['v9-space','v9-dynamic-space','v9-dynamic-refined']}
    source={name:verify(read(BASE/name/'protocol.json')['source_sha256']) for name in VERSIONS}
    verify(read(BASE/'v10-reference-fine/protocol.json')['reference_sha256'])
    verify(read(BASE/'v10/static-inputs.json'))
    with zipfile.ZipFile(BASE/'v9/source-delivered.zip') as archive:
        for name,h in read(BASE/'v9/protocol.json')['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest()==h
    t=read(BASE/'v10/tests.json');s=read(BASE/'v10/static-summary.json');d=read(BASE/'v10/summary.json')
    a=read(BASE/'v10/artifact-check.json');r=read(BASE/'v10-reference/summary.json');q=read(BASE/'v10-reference-q2/summary.json')
    rt=read(BASE/'v10-reference/tests.json');qt=read(BASE/'v10-reference-q2/tests.json')
    assert t['tests_run']==83 and t['required_checks_passed']
    assert rt['tests']==3 and rt['passed'] and qt['tests']==2 and qt['passed']
    assert s['completed'] and len(s['records'])==36 and len(s['beams'])==24
    assert s['all_compatible_tensile_solves_passed'] and s['candidate_static_gate']['residual_corotated']
    assert d['completed'] and d['physical_checks_passed'] and a['passed'] and len(a['snapshots'])==32
    for name in ('v10-reference','v10-reference-fine','v10-reference-q2'):
        runs=read(BASE/name/'runs.json');assert runs['completed'] and all(x['passed'] for x in runs['records'])
    return dict(work_complete=True,implementation_checks_passed=True,overall_accuracy_accepted=False,
        regression_checks=83,independent_Q1_checks=3,independent_Q2_checks=2,skipped=0,
        reference_solutions=20,tensile_static_cases=36,beam_static_cases=24,
        candidate_static_gate=s['candidate_static_gate'],dynamic_trajectories=8,dynamic_steps=1200,
        independent_dynamic_snapshots=32,history_closure_max=d['history_closure_max'],
        independent_state_error_max=a['max_state_error'],independent_grip_force_error_N=a['max_force_error_N'],
        dynamic_physical_checks_passed=True,dynamic_time_sensitivity_passed=d['time_sensitivity_passed'],
        reference_full_stress_certified=False,long_loading_validated=False,default_changed=False,
        old_artifact_files_verified=old,frozen_source_files_verified=source,v9_source_snapshot_verified=True,
        report='docs/ANISO_LITE_LOCAL_HISTORY_ZH.md',device='cpu',precision='float64',cuda='not_run')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('seal','check'));args=parser.parse_args()
    result=evidence()
    if args.action=='seal':
        if any((BASE/v/'artifact-sha256.json').exists() for v in VERSIONS):raise RuntimeError('preserve existing manifests')
        files=set()
        for folder in ('engine','utils','demos','benchmarks','tests'):files.update((ROOT/folder).rglob('*.py'))
        files.update(ROOT.glob('*.md'));files.update((ROOT/'docs').glob('*.md'))
        files.update(ROOT/name for name in ('pyproject.toml','uv.lock','.python-version'))
        archive=BASE/'v10/source-delivered.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):z.write(f,f.relative_to(ROOT))
        result.update(sealed_at=datetime.now(timezone.utc).isoformat(),source_archive_files=len(files),source_archive_sha256=digest(archive))
        write(BASE/'v10/completion-check.json',result)
        for name in VERSIONS:
            mapping={str(f.relative_to(ROOT)):digest(f) for f in sorted((BASE/name).rglob('*'))
                     if f.is_file() and f.name!='artifact-sha256.json'}
            write(BASE/name/'artifact-sha256.json',mapping)
    manifests={name:verify(read(BASE/name/'artifact-sha256.json')) for name in VERSIONS}
    completion=read(BASE/'v10/completion-check.json')
    assert completion['source_archive_sha256']==digest(BASE/'v10/source-delivered.zip')
    with zipfile.ZipFile(BASE/'v10/source-delivered.zip') as z:
        for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==digest(ROOT/name),name
    for name in ('README.md','RUNNING_RESTORED.md','docs/ANISO_LITE_LOCAL_HISTORY_ZH.md'):
        p=ROOT/name
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in link and not link.startswith('#'):assert (p.parent/link.split('#')[0]).exists(),link
    print(json.dumps(dict(work_complete=True,implementation_checks_passed=True,overall_accuracy_accepted=False,
        new_manifest_files_verified=manifests,candidate_static_gate=result['candidate_static_gate'],
        dynamic_time_sensitivity_passed=result['dynamic_time_sensitivity_passed']),indent=2))


if __name__=='__main__':main()
