"""Seal v19 implementation and evidence, without promoting failed candidates."""
import ast,hashlib,json,re,shutil,zipfile
from pathlib import Path
from benchmarks.aniso_v19_runs import ROOT,BASE,OUT,load,write,sha,sources

def main():
    assert not (OUT/'completion-check.json').exists();c=load(OUT/'cycle-acceptance.json');s=load(OUT/'space-summary.json');short=load(OUT/'short-moving-acceptance.json');tests=load(OUT/'tests.json')
    assert c['completed'] and s['completed'] and short['completed'] and tests['passed'];assert sources()==load(OUT/'cycle-protocol.json')['source_sha256']
    for n,d in tests['source_sha256'].items():assert sha(ROOT/n)==d
    old=load(BASE/'v18/artifact-sha256.json')
    for n,d in old.items():assert sha(ROOT/n)==d,n
    for n,d in load(BASE/'v18/protocol.json')['source_sha256'].items():assert sha(ROOT/n)==d,n
    for row in load(OUT/'cycle-batch.json')['records']:
        for n,d in row['sha256'].items():assert sha(OUT/'cases'/row['case']/n)==d
    assert len(c['audits'])==28 and all(a['static_gate']['passed'] for a in c['audits'])
    logdir=OUT/'execution-logs';logdir.mkdir()
    for p in Path('/tmp').glob('mpm-v19-*.log'):shutil.copy2(p,logdir/p.name)
    files=[]
    for directory in ('engine','demos','benchmarks','tests','utils'):files+=list((ROOT/directory).rglob('*.py'))
    files+=list((ROOT/'docs').glob('*.md'));files+=[ROOT/n for n in ('README.md','RUNNING_RESTORED.md','pyproject.toml','uv.lock','LICENSE')]
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()}
    for n in manifest:
        if n.endswith('.py') and any(k in n for k in ('v19','endpoint_boundary','compatible_carrier','compatible_avf')):ast.parse((ROOT/n).read_text())
    with zipfile.ZipFile(OUT/'source-delivered.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in manifest:z.write(ROOT/n,n)
        z.writestr('SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'source-delivered.zip') as z:
        for n,d in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==d
    write(OUT/'source-delivered-sha256.json',manifest)
    report=ROOT/'docs/ANISO_LITE_ENDPOINT_INERTIA_COMPATIBILITY_ZH.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if target.endswith(('completion-check.json','artifact-sha256.json')):continue
        assert (report.parent/target).exists(),target
    completion=dict(requested_implementation_and_validation_work_completed=True,version='v19 bounded endpoint impulse and compatible carrier experiments',
        full_cycles=4,full_cycle_steps=sum(v['steps'] for v in c['cases'].values()),short_moving_cases=9,short_moving_steps=sum(v['steps'] for v in load(OUT/'short-moving-summary.json')['records']),related_tests=tests['tests'],full_repository_suite=False,
        independent_snapshot_audits=len(c['audits']),all_audited_static_stiffness_positive=True,
        boundary_stress_time_passed=c['stress_time_passed'],boundary_raw_reaction_time_passed=c['raw_reaction_time_passed'],boundary_constraint_loss_decreases=c['constraint_loss_decreases'],boundary_conservative_acceptance_passed=c['passed'],
        all_short_moving_time_passed=all(v['passed'] for v in short['cases'].values()),initial_quadrature_crosscheck_passed=True,deformed_quadrature_fully_converged=False,spatial_accuracy_passed=s['spatial_accuracy_passed'],compatible_candidate_promoted=False,
        final_cycle_pair=c['pairs'][-1],constraint_losses_J=[v['energy_sums_J']['constraint_kinetic_loss_J'] for v in c['cases'].values()],
        old_v18_artifacts_unchanged=len(old),old_executed_sources_unchanged=True,source_files=len(manifest),source_zip_sha256=sha(OUT/'source-delivered.zip'),report=str(report.relative_to(ROOT)),production_default_changed=False,cuda_tested=False,
        limitations=['Constraint loss is tiny but not shown to vanish with dt.','F45 compatible candidate is too stiff against the local Q3 reference.','Local continuum stress reference is not certified at 2%.','Gauss/compatible changes have short moving tests, not combined full-cycle acceptance.'])
    write(OUT/'completion-check.json',completion)
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'artifact-sha256.json'};write(OUT/'artifact-sha256.json',artifacts)
    for n,d in artifacts.items():assert sha(ROOT/n)==d
    print('SEALED',len(artifacts),'artifacts;',len(manifest),'sources;',json.dumps(completion),flush=True)
if __name__=='__main__':main()
