"""Seal completed v18 validation without promoting spatial accuracy."""
import ast,json,hashlib,re,shutil,zipfile
from pathlib import Path
from benchmarks.aniso_v18_runs import ROOT,OUT,BASE,load,write,sha,sources

def main():
    assert not (OUT/'completion-check.json').exists();protocol=load(OUT/'protocol.json');assert sources()==protocol['source_sha256']
    moving,cycle=[load(OUT/f'{g}-acceptance.json') for g in ('moving','cycle')]
    for g,result in [('moving',moving),('cycle',cycle)]:
        assert load(OUT/f'batch-{g}.json')['completed'] and result['completed']
        for case in load(OUT/f'batch-{g}.json')['records']:
            for name,digest in case['files'].items():assert sha(OUT/'cases'/case['case']/name)==digest
    assert moving['steps']==12000 and cycle['steps']==96000
    audits=moving['independent_audits']+cycle['independent_audits'];assert len(audits)==56 and all(r['current_massless']['passed'] for r in audits)
    tests=[load(OUT/f'{n}.json') for n in ('tests','space-tests')];assert all(r['passed'] for r in tests) and sum(r['tests_run'] for r in tests)==21
    for name in ('space-quadrature','compatible-reference','fixed-mode-quadrature','input-sensitivity','reaction-diagnosis','boundary-reflection'):assert load(OUT/f'{name}.json')['completed']
    for p in ('space-protocol.json','compatible-reference-protocol.json','cycle-analysis-protocol.json'):
        d=load(OUT/p)
        for name,digest in d.get('sources',d.get('source_sha256',{})).items():assert sha(ROOT/name)==digest,name
    for name,digest in load(OUT/'compatible-reference-protocol.json')['reused_solves'].items():assert sha(OUT/name)==digest
    for result,source in [('fixed-mode-quadrature.json','benchmarks/aniso_v18_fixed_mode.py'),('input-sensitivity-protocol.json','benchmarks/aniso_v18_sensitivity.py'),('reaction-diagnosis-protocol.json','benchmarks/aniso_v18_reaction.py'),('boundary-reflection.json','benchmarks/aniso_v18_boundary_reflection.py')]:
        assert load(OUT/result)['source_sha256']==sha(ROOT/source)
    first=load(OUT/'reference-comparison-quadrature-attempt/compatible-reference-protocol.json')['source_sha256']['benchmarks/aniso_v18_compatible_reference.py']
    assert sha(OUT/'reference-comparison-quadrature-attempt/executed-source.py')==first
    old=load(BASE/'v17/artifact-sha256.json')
    for name,digest in old.items():assert sha(ROOT/name)==digest,name
    for name,digest in load(BASE/'v17/protocol.json')['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for x in protocol['input_snapshots'].values():assert sha(ROOT/x['path'])==x['sha256']
    logdir=OUT/'execution-logs';logdir.mkdir(exist_ok=False)
    for p in Path('/tmp').glob('mpm-v18-*.log'):shutil.copy2(p,logdir/p.name)
    files=[]
    for directory in ('engine','demos','benchmarks','tests','utils'):files+=list((ROOT/directory).rglob('*.py'))
    files+=list((ROOT/'docs').glob('*.md'));files+=[ROOT/n for n in ('README.md','RUNNING_RESTORED.md','pyproject.toml','uv.lock','LICENSE')]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()}
    for name in manifest:
        if name.endswith('.py') and ('v18' in name or 'carrier_driven' in name):ast.parse((ROOT/name).read_text())
    with zipfile.ZipFile(OUT/'source-delivered.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in manifest:z.write(ROOT/name,name)
        z.writestr('SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'source-delivered.zip') as z:
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    write(OUT/'source-delivered-sha256.json',manifest)
    report=ROOT/'docs/ANISO_LITE_MOVING_CYCLE_SPACE_ZH.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if target.endswith(('completion-check.json','artifact-sha256.json')):continue
        assert (report.parent/target).exists(),target
    allcases=list(moving['cases'].values())+list(cycle['cases'].values())
    result=dict(requested_validation_work_completed=True,algorithm_version='v18 bounded driven AVF research implementation; canonical v16 moving snapshot controls',formal_trajectories=12,formal_steps=108000,stress_outputs=108012,
        related_tests=21,related_tests_execution=[16,5],full_repository_suite_rerun=False,independent_snapshot_audits=56,all_audited_massless_stiffness_passed=True,
        moving_time_passed=moving['passed'],full_cycle_completed=True,full_cycle_time_passed=cycle['passed'],spatial_accuracy_passed=False,
        full_cycle_stress_time_passed=max([cycle['refinement']['cycle'][-1][k] for k in ('relative','terminal_relative')]+[r[k] for r in cycle['refinement']['cycle'][-1]['stages'].values() for k in ('relative','terminal_relative')])<.02,
        raw_reaction_time_passed=cycle['refinement']['cycle'][-1]['reaction_RMS_relative']<.02,
        reaction_diagnosis=load(OUT/'reaction-diagnosis.json')['pairs'][-1],independent_reaction_component_audits=40,independent_boundary_reflection_audits=40,
        finite_element_reference_all_targets_passed=load(OUT/'compatible-reference.json')['finite_element_refinement_passed'],
        moving_finest_pair_seconds=[.00003125,.000015625],cycle_finest_pair_seconds=[.0000625,.00003125],
        final_cycle_pair=cycle['refinement']['cycle'][-1],maximum_history_error=max(r['max_history'] for r in allcases),maximum_solve_work_defect_J=max(r['max_work_defect_J'] for r in allcases),
        quadrature_cases=9,compatible_projection_meshes=7,projection_targets_per_mesh=4,input_sensitivity_trajectories=2,input_sensitivity_steps=800,
        rejected_fast_hold_attempt_retained=True,rejected_reference_comparison_quadrature_retained=True,
        old_v17_artifacts_unchanged=len(old),old_executed_sources_unchanged=True,source_files=len(manifest),source_zip_sha256=sha(OUT/'source-delivered.zip'),
        report=str(report.relative_to(ROOT)),production_default_changed=False,cuda_tested=False,nonlinear_continuum_reference_available=False)
    write(OUT/'completion-check.json',result)
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'artifact-sha256.json'}
    write(OUT/'artifact-sha256.json',artifacts)
    for name,digest in artifacts.items():assert sha(ROOT/name)==digest
    print('SEALED',len(artifacts),'artifacts;',len(manifest),'source files;',result,flush=True)
if __name__=='__main__':main()
