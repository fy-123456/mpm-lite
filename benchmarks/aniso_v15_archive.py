"""Seal v15 attribution and bounded prototype, retaining failed general gates."""
import hashlib,json,shutil,zipfile
from pathlib import Path
from benchmarks.aniso_compatible_history import OUT,ROOT,load,write,sources


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT/'completion-check.json').exists(),'preserve sealed output'
    summary=load(OUT/'summary.json');protocol=load(OUT/'protocol.json');tests=load(OUT/'tests.json')
    assert summary['completed'] and summary['trajectories']==26 and summary['total_steps']==9300
    assert tests['required_checks_passed'] and tests['tests_run']==115
    proof=load(OUT/'support-guard-provenance.json')
    assert proof['passed'] and proof['guard_inactive_on_all_formal_steps']
    guard_tests=load(OUT/'support-guard-tests.json');assert guard_tests['passed'] and guard_tests['tests_run']==7
    executed=load(OUT/'executed-source-sha256.json')
    with zipfile.ZipFile(OUT/'source-executed-before-support-guard.zip') as z:
        for name,h in executed.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
        for name,h in protocol['source_sha256'].items():assert executed[name]==h
        name='engine/aniso_phase1/compatible_patch.py';new=(ROOT/name).read_text()
        a=new.index('    def prepare(self):\n',new.index('class CompatiblePatchEnhancements'))
        b=new.index('    def commit(self):\n',a)
        assert new[:a]+new[b:]==z.read(name).decode()
    changed={n for n,h in protocol['source_sha256'].items() if sources()[n]!=h}
    assert changed==set(proof['source_changes_since_formal_runs'])
    assert len(summary['independent_snapshots'])==36
    assert all(max(r['saved_frame_errors'].values())==0 for r in summary['independent_snapshots'])
    controls=load(OUT/'controls/summary.json');assert controls['all_massless_passed'] and controls['all_fd_refinement_passed']
    crossing=load(OUT/'rigid-support-crossing.json');assert crossing['passed']
    rank=load(OUT/'support-rank-diagnostic.json');assert not rank['passed'] and rank['records'][-1]['extra_modes']==48
    assert rank['records'][-1]['v14_material_map_extra_modes']==33
    cli=load(OUT/'cli-check.json');assert cli['passed'] and cli['delivered_support_guard_enabled']
    prior=load(OUT/'prior-archives-check.json');assert prior['passed']
    deformed=[r['massless'] for r in summary['independent_snapshots'] if 'massless' in r]
    assert len(deformed)==12 and all(r['passed'] for r in deformed)
    report=ROOT/'docs/ANISO_LITE_COMPATIBLE_HISTORY_ZH.md';assert '## 最终结果' in report.read_text()
    assert '48 个额外零刚度模式' in report.read_text()
    # Retain the unmodified pre-followup summaries beside explicitly scoped ones.
    shutil.copyfile(OUT/'summary.json',OUT/'summary-before-support-followup.json')
    summary.update(support_followup=dict(unrestricted_candidate_massless_passed=False,
        support_expansion_extra_modes=48,v14_map_same_state_extra_modes=33,
        delivered_guard_rejects_expansion=True,guard_is_general_rank_certificate=False,
        formal_trajectory_guard_inactive=True,post_guard_candidate_tests=7,
        support_report='support-rank-diagnostic.json',provenance='support-guard-provenance.json'))
    write(OUT/'summary.json',summary)
    shutil.copyfile(OUT/'rigid-support-crossing.json',OUT/'rigid-support-crossing-before-rank-followup.json')
    crossing.update(passed_scope='kinematic invariants only, executed before support guard',
        unrestricted_massless_stiffness_passed=False,delivered_solver_completes_this_crossing=False,
        followup='support-rank-diagnostic.json: 48 extra modes; guarded code rejects expansion')
    write(OUT/'rigid-support-crossing.json',crossing)
    logs=OUT/'execution-logs';logs.mkdir(exist_ok=False)
    for name in ('tests-initial','tests-candidate','diagnosis','controls','controls-stable','equilibrium','visibility','crossing','regression','regression-safe','regression-frames','run','run-safe','run-frames','analysis','support-tests','support-rank'):
        p=Path('/tmp')/('mpm-v15-'+name+'.log')
        if p.is_file():shutil.copyfile(p,logs/p.name)
    files=[]
    for directory in ('engine','demos','benchmarks','tests','utils'):
        files.extend((ROOT/directory).rglob('*.py'))
    files.extend((ROOT/'docs').glob('*.md'))
    files.extend(ROOT/n for n in ('README.md','RUNNING_RESTORED.md','pyproject.toml','uv.lock','LICENSE'))
    manifest={str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files)) if p.is_file()}
    zip_path=OUT/'source-delivered.zip'
    with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(zip_path) as z:
        for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    write(OUT/'source-delivered-sha256.json',manifest)
    checks=[c['checks'] for c in summary['cases'].values()]
    maxima={k:max(c[k] for c in checks) for k in checks[0]}
    completion=dict(requested_diagnosis_and_candidate_testing_complete=True,
        implementation_gates_within_tested_f45_support_passed=True,
        unrestricted_candidate_massless_passed=False,support_expansion_supported=False,
        support_expansion_rejected_before_commit=True,guard_is_general_rank_certificate=False,
        overall_accuracy_accepted=False,default_changed=False,
        pre_guard_full_regression_tests=115,post_guard_candidate_tests=7,distinct_tests=116,
        formal_trajectories=26,formal_steps=9300,formal_runs_executed_before_guard=True,
        guard_proven_inactive_on_formal_states=True,pre_guard_rigid_kinematic_steps=180,
        unrestricted_support_extra_modes=48,v14_map_same_state_extra_modes=33,
        independent_snapshots=36,all_saved_frames_match_immediate_snapshots=True,common_input_branch_groups=3,
        same_state_massless_gates=48,fresh_cycle_massless_gates=12,global_four_dt_completed=False,
        spatial_accuracy_revalidated=False,diagnostic_snapshots=24,frozen_nonlinear_relaxations=8,
        max_errors=maxima,source_files=len(manifest),delivered_source_zip_sha256=digest(zip_path),
        executed_source_zip_sha256=digest(OUT/'source-executed-before-support-guard.zip'),
        prior_archives=prior,report=str(report.relative_to(ROOT)),
        scope='Attribution and tested common F/Y prototype; no promotion, no general support or accuracy certification.',
        failed_attempts_preserved=['tests-storage-attempt.json','controls-cancellation-attempt','frame-alias-attempt'])
    write(OUT/'completion-check.json',completion)
    artifacts={str(p.relative_to(ROOT)):digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'}
    write(OUT/'artifact-sha256.json',artifacts)
    for name,h in artifacts.items():assert digest(ROOT/name)==h
    print('SEALED',len(artifacts),'artifacts;',len(manifest),'source files;',maxima,flush=True)

if __name__=='__main__':main()
