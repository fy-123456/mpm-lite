"""Seal v12 provenance and report work completion separately from full accuracy."""
import argparse,hashlib,json,re,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/results/lite-aniso-mainline'
VERSIONS=('v12','v12-exploration','v12-transfer-controls')


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def verify(mapping):
    bad=[f for f,h in mapping.items() if digest(ROOT/f)!=h]
    if bad:raise RuntimeError('changed files: '+str(bad))
    return len(mapping)


def evidence():
    old={p.parent.name:verify(read(p)) for p in sorted(BASE.glob('*/artifact-sha256.json')) if p.parent.name not in VERSIONS}
    for v,archive in [('v11','source-delivered.zip'),('v12-exploration','source-frozen.zip'),('v12-transfer-controls','source-frozen.zip')]:
        with zipfile.ZipFile(BASE/v/archive) as z:
            for f,h in read(BASE/v/'protocol.json')['source_sha256'].items():assert hashlib.sha256(z.read(f)).hexdigest()==h,(v,f)
    source=verify(read(BASE/'v12/protocol.json')['source_sha256'])
    t=read(BASE/'v12/tests.json');s=read(BASE/'v12/static-preservation.json');d=read(BASE/'v12/summary.json');a=read(BASE/'v12/artifact-check.json');f=read(BASE/'v12/factorial-attribution.json');e=read(BASE/'v12/public-api-equivalence.json')
    assert t['required_checks_passed'] and t['tests_run']==95 and s['passed']
    assert d['completed'] and d['physical_checks_passed'] and d['trajectories']==16 and d['steps']==30000
    assert a['passed'] and len(a['snapshots'])==64 and e['passed']
    assert all(r['completed'] and r['physical_passed'] for r in f['modes'].values())
    assert f['independent_snapshots']==48 and len(read(BASE/'v12-transfer-controls/same-input-attribution.json')['records'])==16
    assert d['F45_stress_reduction_target_passed']
    hold=read(BASE/'v12/hold/summary.json');assert hold['completed'] and all(x['completed'] and x['steps']==400 and x['restart_replay_max']<1e-11 and x['max_force_error']<1e-7 for x in hold['records'].values())
    assert hold['records']['candidate']['affine_kinetic_ratio']>1 and hold['records']['candidate']['stress_drift_relative']>hold['records']['baseline']['stress_drift_relative']
    assert d['history_closure_max']<1e-12 and d['particle_commit_max']<1e-12
    assert f['modes']['velocity_only']['pairs'][-1]['P_terminal_relative']>f['modes']['baseline']['pairs'][-1]['P_terminal_relative']
    assert all(x['exit_code']==0 for v in VERSIONS for x in read(BASE/v/'batch.json'))
    for p in [ROOT/'README.md',ROOT/'RUNNING_RESTORED.md',ROOT/'docs/ANISO_LITE_AFFINE_HISTORY_TIME_ZH.md']:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in link and not link.startswith('#'):assert (p.parent/link.split('#')[0]).exists(),(p,link)
    return dict(work_complete=True,requested_stress_reduction_achieved=True,static_improvement_preserved=True,
        regression_tests=95,new_regression_tests=5,official_trajectories=16,official_steps=30000,
        additional_control_trajectories=12,additional_control_steps=22500,total_full_loading_steps=52500,hold_steps=800,restart_replay_steps=2,total_new_steps=53302,
        hold_stress_drift_relative={k:x['stress_drift_relative'] for k,x in hold['records'].items()},
        hold_stability_improved=False,affine_energy_growth_remains=True,
        official_independent_snapshots=64,additional_control_transfer_snapshots=48,same_input_attribution_snapshots=16,
        history_closure_max=d['history_closure_max'],particle_commit_max=d['particle_commit_max'],
        independent_force_error_max_N=a['max_force_error_N'],independent_state_error_max=a['max_state_error'],
        public_api_control_equivalence_max=e['max_difference'],F45_improvement_fraction=d['F45_improvement_fraction'],
        time_threshold_passed=d['time_threshold_passed'],reliable_time_trend=d['reliable_time_trend'],overall_accuracy_accepted=False,
        attribution='remove only affine PIC relaxation; retain v11 velocity damping, individual material history, patch energy and reconstructed reference rules',
        old_artifact_files_verified=old,frozen_source_files_verified=source,default_changed=False,device='cpu',precision='float64',cuda='not_run',
        report='docs/ANISO_LITE_AFFINE_HISTORY_TIME_ZH.md')


def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('action',choices=('seal','check'));args=a.parse_args();r=evidence()
    if args.action=='seal':
        if any((BASE/v/'artifact-sha256.json').exists() for v in VERSIONS):raise RuntimeError('preserve manifests')
        files=set()
        for folder in ('engine','utils','demos','benchmarks','tests'):files.update((ROOT/folder).rglob('*.py'))
        files.update(ROOT.glob('*.md'));files.update((ROOT/'docs').glob('*.md'));files.update(ROOT/f for f in ('pyproject.toml','uv.lock','.python-version'))
        archive=BASE/'v12/source-delivered.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):z.write(f,f.relative_to(ROOT))
        r.update(sealed_at=datetime.now(timezone.utc).isoformat(),source_archive_sha256=digest(archive),source_archive_files=len(files));write(BASE/'v12/completion-check.json',r)
        for v in VERSIONS:write(BASE/v/'artifact-sha256.json',{str(f.relative_to(ROOT)):digest(f) for f in sorted((BASE/v).rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'})
    counts={v:verify(read(BASE/v/'artifact-sha256.json')) for v in VERSIONS}
    with zipfile.ZipFile(BASE/'v12/source-delivered.zip') as z:
        for f in z.namelist():assert hashlib.sha256(z.read(f)).hexdigest()==digest(ROOT/f),f
    print(json.dumps(dict(work_complete=True,stress_reduction_achieved=r['requested_stress_reduction_achieved'],static_preserved=True,manifest_files=counts,time_threshold_passed=r['time_threshold_passed'],overall_accuracy_accepted=False),indent=2))

if __name__=='__main__':main()
