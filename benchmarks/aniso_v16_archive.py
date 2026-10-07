"""Seal completed v16 work without promoting unaccepted global accuracy."""
import ast,json,zipfile,shutil,re
from pathlib import Path
from benchmarks.aniso_v16_experiments import OUT,ROOT,load,write,sha
from benchmarks.aniso_v16_avf import AVF,avf_sources


def pack_sources(path,files):
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(path) as z:
        import hashlib
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    return manifest


def main():
    assert not (OUT/'completion-check.json').exists(),'preserve sealed result'
    a=load(OUT/'summary.json');b=load(AVF/'summary.json');p=load(OUT/'protocol.json');p2=load(AVF/'protocol.json')
    assert a['completed'] and b['completed'] and a['formal_steps']==9000 and b['formal_steps']==3000
    assert len(a['cases'])==48 and len(b['cases'])==16
    assert load(OUT/'batch.json')['completed'] and load(AVF/'batch.json')['completed']
    tests=load(OUT/'tests.json');tests2=load(AVF/'tests.json')
    assert tests['required_checks_passed'] and tests['tests_run']==124
    assert tests2['required_checks_passed'] and tests2['tests_run']==4
    assert len(set(r['test'] for t in (tests,tests2) for r in t['records']))==128
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    assert avf_sources()==p2['source_sha256']
    support=load(OUT/'support-gates/summary.json');assert support['passed']
    assert len(support['translation_support_gates'])==24 and len(support['same_snapshot_gates'])==3
    for root in (OUT,AVF):
        c=load(root/'moving-support.json');assert c['passed'] and c['steps']==180
        assert all(r['gate']['passed'] for r in c['actual_massless_gates'])
    audits=a['independent_snapshots']+b['independent_snapshots'];assert len(audits)==64
    assert all(r['massless']['passed'] and max(r['saved_frame_errors'].values())==0 and r['fd_exact_relative']<1e-6 for r in audits)
    assert load(OUT/'modal-reference-check.json')['passed']
    prior=load(OUT/'prior-archives-check.json');assert prior['passed'] and len(prior['archives'])==36 and prior['total_files']==3931
    # Every archived input is still the one specified at freeze.
    for r in p['source_snapshots'].values():assert sha(ROOT/r['path'])==r['sha256']
    logs=OUT/'execution-logs';logs.mkdir(exist_ok=False)
    for path in Path('/tmp').glob('mpm-v16-*.log'):shutil.copyfile(path,logs/path.name)
    files=[]
    for directory in ('engine','demos','benchmarks','tests','utils'):
        files.extend((ROOT/directory).rglob('*.py'))
    files.extend((ROOT/'docs').glob('*.md'))
    files.extend(ROOT/n for n in ('README.md','RUNNING_RESTORED.md','pyproject.toml','uv.lock','LICENSE'))
    # Phase-one reproducible checkout has the exact algorithm file set. Final
    # docs/auxiliary analysis are included, explicitly distinguished from
    # the executed fingerprint archive. Original v14 inputs are separate.
    phase1=[f for f in files if f.name not in ('carrier_avf.py','test_aniso_carrier_avf.py') and not f.name.startswith('aniso_v16_avf')]
    first=pack_sources(OUT/'phase1-reproduction-source.zip',phase1)
    assert all(first[n]==h for n,h in p['source_sha256'].items())
    with zipfile.ZipFile(OUT/'phase1-executed-source.zip') as z:
        assert json.loads(z.read('SOURCE_SHA256.json'))==p['source_sha256']
    manifest=pack_sources(OUT/'source-delivered.zip',files);write(OUT/'source-delivered-sha256.json',manifest)
    assert all(manifest[n]==h for n,h in p2['source_sha256'].items())
    for f in files:
        if f.suffix=='.py' and ('v16' in f.name or 'carrier_joint' in f.name or 'carrier_avf' in f.name):ast.parse(f.read_text())
    controls=load(OUT/'small-step-controls.json')
    failed=[dict(start=r['start'],mode=r['mode'],dt=r['dt']) for r in controls['records'] if not r['completed']]
    assert all(r['mode']=='adjoint_split' for r in failed)
    completion=dict(requested_steps_completed=True,production_default_changed=False,production_ready=False,overall_accuracy_accepted=False,
        scope='Bounded material-coordinate support prototype; same-input frozen/moving joint and AVF comparisons. Not arbitrary grid-space stiffness, full load-cycle, or spatial certification.',
        formal_trajectories=64,formal_steps=12000,independent_snapshots=64,terminal_massless_gates=64,
        support_direction_gates=24,original_snapshot_massless_gates=3,moving_rigid_steps_per_scheme=180,
        moving_rigid_massless_gates=10,full_regression_tests=124,avf_supplemental_tests=4,distinct_tests=128,
        all_saved_frames_match_immediate_snapshots=True,phase1_source_files_unchanged_after_execution=True,
        shared_elastic_energy_unchanged=True,no_mass_shift_or_extra_stiffness=True,no_extra_dissipation=True,
        backward_euler_intrinsic_dissipation_accounted=True,moving_APIC_metric_change_accounted=True,
        small_step_controls=len(controls['records']),failed_counterfactual_controls=failed,
        preserved_initial_attempt_steps=9000,preserved_initial_attempt_tests=124,
        max_independent_F_error=max(r['errors']['F'] for r in audits),
        max_fd_tangent_relative=max(r['fd_exact_relative'] for r in audits),
        max_avf_kinetic_force_work_defect_J=max(r['max_work_defect_J'] for r in b['cases'].values()),
        max_avf_path_quadrature_error_J=max(r['max_quadrature_error_J'] for r in b['cases'].values()),
        avf_moving_last_dt_pair={label:b['refinement'][f'{label}-moving-avf'][-1] for label in ('early_hold','late_hold')},
        full_load_cycle_completed=False,spatial_accuracy_revalidated=False,cuda_tested=False,
        source_files=len(manifest),source_delivered_sha256=sha(OUT/'source-delivered.zip'),
        phase1_reproduction_sha256=sha(OUT/'phase1-reproduction-source.zip'),prior_archives=prior,
        report='docs/ANISO_LITE_CARRIER_JOINT_ZH.md')
    write(OUT/'completion-check.json',completion)
    report=ROOT/completion['report'];missing=[]
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if not (report.parent/target).exists():missing.append(target)
    assert not missing,missing
    artifacts={str(f.relative_to(ROOT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'}
    write(OUT/'artifact-sha256.json',artifacts)
    for name,h in artifacts.items():assert sha(ROOT/name)==h
    print('SEALED',len(artifacts),'artifacts;',len(manifest),'source files; 64 trajectories / 12000 steps / 128 distinct tests.',flush=True)

if __name__=='__main__':main()
