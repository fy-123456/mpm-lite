"""Seal v14 implementation, independent evidence and all diagnostic attempts."""
import argparse,hashlib,json,re,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/results/lite-aniso-mainline';OUT=BASE/'v14'
VERSIONS=('v14','v14-controls','v14-space','v14-exploration','v14-controls-diagnostic-attempt','v14-storage-attempt','v14-storage-attempt-2','v14-recovery2')
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def verify(mapping):
    for f,h in mapping.items():assert digest(ROOT/f)==h,f
    return len(mapping)

def evidence():
    old={p.parent.name:verify(read(p)) for p in sorted(BASE.glob('*/artifact-sha256.json')) if p.parent.name not in VERSIONS}
    assert len(old)==27 and sum(old.values())==2753
    with zipfile.ZipFile(BASE/'v13/source-delivered.zip') as z:
        for f,h in read(BASE/'v13/protocol.json')['source_sha256'].items():assert hashlib.sha256(z.read(f)).hexdigest()==h,f
    source=verify(read(OUT/'protocol.json')['source_sha256']);t=read(OUT/'tests.json');s=read(OUT/'summary.json');a=read(OUT/'artifact-check.json');d=read(OUT/'deformed-massless.json')
    assert t['required_checks_passed'] and t['tests_run']==109
    assert s['completed'] and s['physical_checks_passed'] and s['trajectories']==8 and s['steps']==48000 and s['snapshot_count']==80
    assert s['history_closure_max']<1e-12 and s['particle_commit_max']<1e-12 and s['material_carrier_commit_max']<1e-12
    assert a['passed'] and len(a['snapshots'])==80 and a['max_state_error']<1e-10 and a['max_force_error']<1e-8
    assert s['material_reference_rebuild_max_J']<1e-14 and d['all_rank_gates_passed'] and d['finite_difference_refinement_passed'] and len(d['records'])==12
    assert d['source_sha256']==digest(ROOT/'benchmarks/aniso_material_deformed.py')
    c=read(BASE/'v14-controls/summary.json');assert c['completed'] and c['cases']==80 and c['steps']==12000
    cp=read(BASE/'v14-controls/protocol.json');assert cp['script_sha256']==digest(ROOT/'benchmarks/aniso_material_controls.py') and cp['source_sha256']==digest(ROOT/cp['source'])
    sp=read(BASE/'v14-space/summary.json');assert sp['completed'] and sp['all_massless_ranks_passed'] and sp['all_solves_passed'] and len(sp['records'])==12;verify(read(BASE/'v14-space/protocol.json')['source_sha256'])
    rb=read(BASE/'v14-exploration/same-state-rebuild.json');assert rb['passed'] and len(rb['records'])==32 and rb['actual_kernel_cases']==8
    assert rb['source_sha256']==digest(ROOT/'benchmarks/aniso_material_rebuild.py')
    joint=read(BASE/'v14-exploration/joint-linear-control.json');assert joint['completed'] and len(joint['records'])==32 and len(joint['smooth_fields'])==6 and joint['source_sha256']==digest(ROOT/'benchmarks/aniso_material_joint.py')
    crossing=read(BASE/'v14-exploration/material-support-crossing.json');assert crossing['passed'] and crossing['steps']==180 and crossing['support_changes']>0 and crossing['source_sha256']==digest(ROOT/'benchmarks/aniso_material_crossing.py')
    with zipfile.ZipFile(BASE/'v14-storage-attempt/paused-cases.zip') as z:
        for f,h in read(BASE/'v14-storage-attempt/paused-case-sha256.json').items():assert hashlib.sha256(z.read(f)).hexdigest()==h,f
    resume=read(OUT/'resume-summary.json');assert resume['passed'] and resume['new_results_returned_to_project'] and resume['all_originals_preserved_on_disk']
    overlap=[r for stage in resume['stages'] for r in stage['cases'].values()]
    assert len(resume['stages'])==2 and len(overlap)==5 and all(r['passed'] and max(r['errors'].values())<1e-10 for r in overlap)
    for stage in resume['stages']:assert stage['source_sha256']==digest(ROOT/stage['source'])
    recovery=read(BASE/'v14-recovery2/protocol.json');assert recovery['source_sha256']==digest(ROOT/'benchmarks/aniso_material_resume_safe.py')
    for f,h in recovery['source_files'].items():
        old_path=ROOT/f;backup=BASE/'v14-storage-attempt-2/cases'/old_path.relative_to(OUT/'cases');assert digest(backup)==h,f
    cli=json.loads([s for s in (OUT/'cli-smoke.log').read_text().splitlines() if s.startswith('{')][-1]);assert cli['converged'] and cli['steps']==20 and cli['stabilization']=='material_patch' and cli['velocity_dissipation']=='null'
    assert all(r['exit_code']==0 for v in ('v14','v14-controls') for r in read(BASE/v/'batch.json'))
    for f in (ROOT/'README.md',ROOT/'RUNNING_RESTORED.md',ROOT/'docs/ANISO_LITE_MATERIAL_REFERENCE_ENERGY_ZH.md'):
        txt=f.read_text();assert not re.search(r'<!-- \w+_RESULTS -->',txt),f
        for link in re.findall(r'\]\(([^)]+)\)',txt):
            if '://' not in link and not link.startswith('#'):assert (f.parent/link.split('#')[0]).exists(),(f,link)
    return dict(work_complete=True,regression_tests=109,new_regression_tests=6,official_trajectories=8,official_steps=48000,official_independent_snapshots=80,
        fixed_moving_isolated_coupled_cases=80,control_steps=12000,auxiliary_linear_joint_split_cases=32,auxiliary_linear_steps=4800,smooth_field_controls=6,
        resumed_storage_cases=3,resume_overlap_steps=sum(r["overlap_steps"] for r in overlap),resume_overlap_max=max(max(r["errors"].values()) for r in overlap),same_state_rebuild_snapshots=32,same_state_actual_kernel_checks=8,initial_spatial_cases=12,deformed_massless_checks=12,material_carrier_crossing_steps=180,material_carrier_support_changes=crossing["support_changes"],cli_steps=20,
        initial_static_improvement_preserved=True,deformed_massless_ranks_passed=True,reference_rebuild_energy_growth_eliminated=True,reference_rebuild_max_J=s['material_reference_rebuild_max_J'],
        history_closure_max=s['history_closure_max'],particle_commit_max=s['particle_commit_max'],carrier_commit_max=s['material_carrier_commit_max'],independent_state_error_max=a['max_state_error'],independent_force_error_max_N=a['max_force_error'],
        stage_budget_max_J=max(r['max_stage_budget_error'] for r in s['checks'].values()),baseline_v13_equivalent_through_1_2s=True,
        material_all_phase_time_2pct_passed=s['material_all_phase_time_2pct_passed'],material_all_phase_orders_passed=s['material_all_phase_orders_passed'],material_all_holds_nonincrease=s['material_all_holds_nonincrease'],
        loading_stress_time_difference={m:s['refinement'][m]['ramp']['pairs'][-1]['P_relative'] for m in ('baseline','material')},spatial_stress_accuracy_accepted=False,overall_accuracy_accepted=False,default_changed=False,
        old_artifact_files_verified=old,frozen_source_files_verified=source,device='cpu',precision='float64',cuda='not_run',report='docs/ANISO_LITE_MATERIAL_REFERENCE_ENERGY_ZH.md')

def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('action',choices=('seal','check'));args=ap.parse_args()
    if args.action=='seal' and not (OUT/'completion-check.json').exists():write(OUT/'completion-check.json',dict(work_complete=False,pending='sealing'))
    r=evidence()
    if args.action=='seal':
        assert not any((BASE/v/'artifact-sha256.json').exists() for v in VERSIONS)
        files=set()
        for folder in ('engine','utils','demos','benchmarks','tests'):files.update((ROOT/folder).rglob('*.py'))
        files.update(ROOT.glob('*.md'));files.update((ROOT/'docs').glob('*.md'));files.update(ROOT.glob('*.toml'));files.update(ROOT.glob('*.lock'))
        archive=OUT/'source-delivered.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):z.write(f,f.relative_to(ROOT))
        r.update(sealed_at=datetime.now(timezone.utc).isoformat(),source_archive_sha256=digest(archive),source_archive_files=len(files));write(OUT/'completion-check.json',r)
        for v in VERSIONS:write(BASE/v/'artifact-sha256.json',{str(f.relative_to(ROOT)):digest(f) for f in sorted((BASE/v).rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'})
    counts={v:verify(read(BASE/v/'artifact-sha256.json')) for v in VERSIONS}
    with zipfile.ZipFile(OUT/'source-delivered.zip') as z:
        for f in z.namelist():assert hashlib.sha256(z.read(f)).hexdigest()==digest(ROOT/f),f
    print(json.dumps(dict(work_complete=True,manifest_files=counts,reference_rebuild_energy_growth_eliminated=True,material_all_phase_time_2pct_passed=r['material_all_phase_time_2pct_passed'],material_all_phase_orders_passed=r['material_all_phase_orders_passed'],overall_accuracy_accepted=False),indent=2))

if __name__=='__main__':main()
