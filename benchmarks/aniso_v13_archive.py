"""Seal v13 results, preserving failed diagnostics and separating accuracy gates."""
import argparse,hashlib,json,re,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/results/lite-aniso-mainline'
VERSIONS=('v13','v13-screen','v13-diagnostic-attempt','v13-screen-diagnostic-attempt')

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def verify(mapping):
    bad=[f for f,h in mapping.items() if digest(ROOT/f)!=h]
    if bad:raise RuntimeError('changed files: '+str(bad))
    return len(mapping)


def evidence():
    out=BASE/'v13';old={p.parent.name:verify(read(p)) for p in sorted(BASE.glob('*/artifact-sha256.json')) if p.parent.name not in VERSIONS}
    for name in ('v11','v12'):
        with zipfile.ZipFile(BASE/name/'source-delivered.zip') as z:
            for f,h in read(BASE/name/'protocol.json')['source_sha256'].items():assert hashlib.sha256(z.read(f)).hexdigest()==h,(name,f)
    source=verify(read(out/'protocol.json')['source_sha256'])
    t=read(out/'tests.json');static=read(out/'static-preservation.json');s=read(out/'summary.json');a=read(out/'artifact-check.json')
    assert t['required_checks_passed'] and t['tests_run']==103 and static['passed']
    assert s['completed'] and s['physical_checks_passed'] and s['trajectories']==12 and s['steps']==54000
    assert s['history_closure_max']<1e-12 and s['particle_commit_max']<1e-12 and s['snapshot_count']==96
    assert a['passed'] and len(a['snapshots'])==96 and a['max_state_error']<1e-10 and a['max_force_error']<1e-8
    screen={m:read(BASE/'v13-screen'/m/'summary.json') for m in ('none','null','weak')}
    assert all(r['completed'] and r['steps']==400 and r['stage_budget_max']<1e-12 and r['momentum_error_max']<1e-7 for r in screen.values())
    assert read(BASE/'v13-screen/independent-filter-check.json')['passed']
    crossing=read(out/'crossing-filter-check.json');assert crossing['passed'] and crossing['cases']==74
    assert crossing['source_sha256']==digest(ROOT/'benchmarks/aniso_unresolved_crossing.py')
    cli=json.loads([line for line in (out/'cli-smoke.log').read_text().splitlines() if line.startswith('{')][-1]);assert cli['converged'] and cli['steps']==20 and cli['velocity_dissipation']=='weak'
    assert all(r['exit_code']==0 for v in ('v13','v13-screen') for r in read(BASE/v/'batch.json'))
    assert len(read(BASE/'v13-screen/same-input-spectral.json')['records'])==32
    for f in (ROOT/'README.md',ROOT/'RUNNING_RESTORED.md',ROOT/'docs/ANISO_LITE_UNRESOLVED_VELOCITY_ZH.md'):
        for link in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            if '://' not in link and not link.startswith('#'):assert (f.parent/link.split('#')[0]).exists(),(f,link)
    return dict(work_complete=True,regression_tests=103,new_regression_tests=8,static_improvement_preserved=True,
        official_trajectories=12,official_steps=54000,same_input_hold_trajectories=3,same_input_hold_steps=1200,cli_steps=20,total_accepted_steps=55220,
        official_independent_snapshots=96,same_input_hold_independent_snapshots=9,frozen_same_input_interventions=32,isolated_support_crossing_cases=74,
        baseline_v12_loading_equivalent=True,history_closure_max=s['history_closure_max'],particle_commit_max=s['particle_commit_max'],
        independent_state_error_max=a['max_state_error'],independent_force_error_max_N=a['max_force_error'],
        stage_budget_max_J=max(r['max_stage_budget_error'] for r in s['checks'].values()),
        filter_energy_increase_max_J=max(r['max_filter_energy_increase'] for r in s['checks'].values()),
        loading_stress_time_difference={m:s['refinement'][m]['ramp']['pairs'][-1]['P_relative'] for m in ('baseline','null','weak')},
        weak_all_phase_time_2pct_passed=s['weak_all_phase_time_2pct_passed'],weak_all_phase_orders_passed=s['weak_all_phase_orders_passed'],
        weak_all_holds_net_energy_nonincrease=s['weak_all_holds_nonincrease'],
        reference_rebuild_energy_growth_eliminated=False,spatial_stress_accuracy_accepted=False,overall_accuracy_accepted=False,default_changed=False,
        old_artifact_files_verified=old,frozen_source_files_verified=source,device='cpu',precision='float64',cuda='not_run',
        report='docs/ANISO_LITE_UNRESOLVED_VELOCITY_ZH.md',scope='small CPU joint v/C dissipation; stage attribution and full cyclic dt study; preserve all failed gates')


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('seal','check'));args=parser.parse_args()
    out=BASE/'v13'
    # Evidence link validation includes completion-check: create only an explicit pending stub.
    if args.action=='seal' and not (out/'completion-check.json').exists():write(out/'completion-check.json',dict(work_complete=False,pending='sealing'))
    r=evidence()
    if args.action=='seal':
        if any((BASE/v/'artifact-sha256.json').exists() for v in VERSIONS):raise RuntimeError('preserve manifests')
        files=set()
        for folder in ('engine','utils','demos','benchmarks','tests'):files.update((ROOT/folder).rglob('*.py'))
        files.update(ROOT.glob('*.md'));files.update((ROOT/'docs').glob('*.md'));files.update(ROOT/f for f in ('pyproject.toml','uv.lock','.python-version'))
        archive=out/'source-delivered.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):z.write(f,f.relative_to(ROOT))
        r.update(sealed_at=datetime.now(timezone.utc).isoformat(),source_archive_sha256=digest(archive),source_archive_files=len(files));write(out/'completion-check.json',r)
        for v in VERSIONS:write(BASE/v/'artifact-sha256.json',{str(f.relative_to(ROOT)):digest(f) for f in sorted((BASE/v).rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'})
    counts={v:verify(read(BASE/v/'artifact-sha256.json')) for v in VERSIONS}
    with zipfile.ZipFile(out/'source-delivered.zip') as z:
        for f in z.namelist():assert hashlib.sha256(z.read(f)).hexdigest()==digest(ROOT/f),f
    print(json.dumps(dict(work_complete=True,manifest_files=counts,overall_accuracy_accepted=False,weak_all_phase_time_2pct_passed=r['weak_all_phase_time_2pct_passed'],weak_all_holds_net_energy_nonincrease=r['weak_all_holds_net_energy_nonincrease']),indent=2))

if __name__=='__main__':main()
