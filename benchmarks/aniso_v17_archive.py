"""Seal v17 diagnostics and tests without promoting full physical accuracy."""
import ast,json,hashlib,re,shutil,zipfile
from pathlib import Path
from benchmarks.aniso_v17_time import ROOT,BASE,OUT,load,write,sha,sources


def main():
    assert not (OUT/'completion-check.json').exists(),'preserve sealed v17'
    protocol=load(OUT/'protocol.json');assert sources()==protocol['source_sha256'];assert load(OUT/'batch.json')['completed']
    result=load(OUT/'time-summary.json');assert result['completed'] and result['formal_trajectories']==12 and result['formal_steps']==25200
    assert result['stress_outputs']==25212 and len(result['independent_audits'])==12
    physical=load(OUT/'independent-physical-audit.json');assert physical['passed'] and len(physical['records'])==12
    assert all(a['massless']['passed'] and a['fd_tangent_relative']<1e-6 for a in result['independent_audits'])
    assert load(OUT/'tests.json')['passed'] and load(OUT/'tests.json')['tests_run']==16
    log=(OUT/'time-metrics-tests.log').read_text();assert 'Ran 2 tests' in log and '\nOK\n' in log
    modal=load(OUT/'modal-snapshots.json');scan=load(OUT/'spatial-mode-scan.json');generic=load(OUT/'generic-geometry-scan.json');match=load(OUT/'stress-mode-matching.json')
    assert all(v['completed'] for v in (modal,scan,generic,match))
    assert len(scan['records'])==18 and len(scan['amplitude_ladder'])==4 and len(generic['records'])==12 and len(match['records'])==9
    assert all(r['gate']['passed'] for r in scan['records']+scan['amplitude_ladder']+modal['records'])
    old=load(BASE/'v16/artifact-sha256.json')
    for name,digest in old.items():assert sha(ROOT/name)==digest,name
    engine_old=load(BASE/'v16/avf/protocol.json')['source_sha256']
    for name,digest in engine_old.items():assert sha(ROOT/name)==digest,name
    for inp in protocol['inputs'].values():assert sha(ROOT/inp['path'])==inp['sha256']
    for name in ('generic-geometry-protocol-v2.json','stress-mode-matching-protocol.json'):
        for source,digest in load(OUT/name)['source_sha256'].items():assert sha(ROOT/source)==digest
    historical_attempt=(OUT/'generic-geometry-protocol.json').exists()
    if historical_attempt:
        first=load(OUT/'generic-geometry-protocol.json')['source_sha256']['benchmarks/aniso_v17_geometry.py']
        assert sha(OUT/'diagnostic-attempts/generic-eigen-attempt.py')==first
    for path in Path('/tmp').glob('mpm-v17-*.log'):
        dest=OUT/'execution-logs'/path.name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest)
    files=[]
    for directory in ('engine','demos','benchmarks','tests','utils'):files+=list((ROOT/directory).rglob('*.py'))
    files+=list((ROOT/'docs').glob('*.md'));files += [ROOT/n for n in ('README.md','RUNNING_RESTORED.md','pyproject.toml','uv.lock','LICENSE')]
    manifest={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(files)) if f.is_file()}
    for f in files:
        if f.suffix=='.py' and 'v17' in f.name:ast.parse(f.read_text())
    with zipfile.ZipFile(OUT/'source-delivered.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'source-delivered.zip') as z:
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    write(OUT/'source-delivered-sha256.json',manifest)
    report=ROOT/'docs/ANISO_LITE_STRESS_MODES_TIME_ZH.md'
    # Completion and manifest links are created below; all other references must exist now.
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if target.endswith(('completion-check.json','artifact-sha256.json')):continue
        assert (report.parent/target).exists(),target
    completion=dict(requested_work_completed=True,algorithm_version='v16 unchanged; v17 validation and diagnostics',formal_trajectories=12,formal_steps=25200,stress_outputs=25212,
        independent_terminal_audits=12,independent_constitutive_patch_kinetic_audits=12,related_tests=18,related_tests_execution=[16,2],full_repository_suite_rerun=False,
        snapshot_modal_modes=450,controlled_modal_scans=22,generic_inertia_scans=12,stress_shape_matching_scans=9,
        fixed_geometry_last_pair_accepted=result['fixed_geometry_last_pair_accepted'],
        final_dt_pairs={k:v[-1] for k,v in result['refinement'].items()},
        maximum_history_error=max(c['max_history_error'] for c in result['cases'].values()),
        maximum_work_defect_J=max(c['max_work_defect_J'] for c in result['cases'].values()),
        maximum_quadrature_error_J=max(c['max_quadrature_error_J'] for c in result['cases'].values()),
        maximum_fd_tangent_relative=max(a['fd_tangent_relative'] for a in result['independent_audits']),
        original_v16_artifacts_unchanged=len(old),v16_executed_sources_unchanged=True,
        rejected_generic_eigen_attempt_retained=historical_attempt,source_files=len(manifest),
        source_zip_sha256=sha(OUT/'source-delivered.zip'),report=str(report.relative_to(ROOT)),
        full_load_cycle_completed=False,moving_particle_refinement_completed=False,spatial_accuracy_accepted=False,
        nonlinear_continuum_reference_available=False,production_default_changed=False,cuda_tested=False)
    write(OUT/'completion-check.json',completion)
    artifacts={str(f.relative_to(ROOT)):sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'}
    write(OUT/'artifact-sha256.json',artifacts)
    for name,digest in artifacts.items():assert sha(ROOT/name)==digest
    print('SEALED',len(artifacts),'artifacts;',len(manifest),'source files; 12 trajectories / 25200 steps / 18 related tests.',flush=True)
if __name__=='__main__':main()
