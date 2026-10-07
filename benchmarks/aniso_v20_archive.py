"""Seal completed v20 evidence; failed/superseded attempts remain explicit."""
import ast,hashlib,json,re,shutil,zipfile
from pathlib import Path
from benchmarks.aniso_v20_common import *

def tests():
    logs={'all-tests':35,'roundoff-tests':2,'adaptive-tests':1}
    for name,count in logs.items():
        text=Path(f'/tmp/mpm-v20-{name}.log').read_text();assert re.search(rf'^Ran {count} tests? in ', text, re.MULTILINE) and text.rstrip().endswith('OK'),name
    names=[p for p in (ROOT/'tests').glob('test_aniso_*.py') if any(x in p.name for x in ['v20','swept_adaptive','integrated_inertia','endpoint_boundary','carrier_driven','carrier_avf','compatible_carrier'])]
    names+=list((ROOT/'engine/aniso_phase1').glob('*.py'));names+=list((ROOT/'benchmarks').glob('aniso_v20*.py'))
    write(OUT/'tests.json',dict(passed=True,tests=sum(logs.values()),logs=logs,full_repository_suite=False,cuda_tested=False,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in names}))

def main():
    assert not (OUT/'completion-check.json').exists();c=load(OUT/'cycle-acceptance.json');a=load(OUT/'independent-audits.json');q=load(OUT/'runtime-quadrature.json');s=load(OUT/'local-space.json');r=load(OUT/'local-relaxation.json');d=load(OUT/'condensation-diagnosis.json');t=load(OUT/'tests.json');assert all(x['completed'] for x in [c,a,q,s,r]);assert a['total_snapshots']==42 and t['passed'];assert len(d['candidate_modes'])==8
    assert load(OUT/'ghost-inertia-scaling.json')['completed'];assert load(OUT/'local-stiffness-checks.json')['all_passed'];assert load(OUT/'full-cycle-ablation.json')['completed'];assert load(OUT/'snapshot-batch.json')['completed']
    for n,digest in t['source_sha256'].items():assert sha(ROOT/n)==digest,n
    for p in [OUT/'cycle-protocol.json',OUT/'fast-cycle/cycle-protocol.json']:
        for n,digest in load(p)['source_sha256'].items():assert sha(ROOT/n)==digest,n
    old=load(BASE/'v19/artifact-sha256.json')
    for n,digest in old.items():assert sha(ROOT/n)==digest,n
    oldsrc=load(BASE/'v19/source-delivered-sha256.json');oldsrc_count=0
    for n,digest in oldsrc.items():
        if n.endswith('.py'):assert sha(ROOT/n)==digest,n;oldsrc_count+=1
    for prefix in (OUT,OUT/'fast-cycle'):
        batch=load(prefix/'cycle-batch.json')
        for row in batch['records']:
            for name,digest in row['sha256'].items():assert sha(prefix/'cases'/row['case']/name)==digest
    paths=load(OUT/'completed-case-paths.json');assert paths['completed'];total=0;recoveries=[]
    for name,p in paths['cases'].items():
        folder=ROOT/p['path'];status=load(folder/'status.json');assert status['completed'];total+=status['steps'];assert status['steps']==status['requested_steps'];assert len(list(folder.glob('audit-*.npz')))==7
        if p['recovered']:
            for f,digest in status['source_sha256'].items():assert sha(ROOT/f)==digest
            origin=ROOT/status['resumed_from'];assert sha(origin)==status['resumed_from_sha256'];sourcecase=origin.parent;k=status['prefix_steps'];oldlines=(sourcecase/'steps.jsonl').read_text().splitlines();newlines=(folder/'steps.jsonl').read_text().splitlines();assert oldlines[:k]==newlines[:k]
            import numpy as np
            with np.load(sourcecase/'stress.npz') as x,np.load(folder/'stress.npz') as y:np.testing.assert_array_equal(x['P'][:k+1],y['P'][:k+1])
            recoveries.append(dict(case=name,prefix_steps=k,previous_failure_steps=status['previous_failure_steps'],final_steps=status['steps']))
    assert total==73600
    finest=ROOT/paths['cases']['gauss3-condensed-L3']['path']
    for v in q['records']:
        source=finest/f"audit-{round(v['time']/.0000625):06d}.npz";assert sha(source)==v['source_sha256'];v['canonical_source']=str(source.relative_to(ROOT))
    write(OUT/'runtime-quadrature.json',q)
    logdir=OUT/'execution-logs';logdir.mkdir(exist_ok=False)
    for p in Path('/tmp').glob('mpm-v20-*.log'):shutil.copy2(p,logdir/p.name)
    files=[]
    for directory in ('engine','demos','benchmarks','tests','utils'):files+=list((ROOT/directory).rglob('*.py'))
    files+=list((ROOT/'docs').glob('*.md'));files+=[ROOT/n for n in ('README.md','RUNNING_RESTORED.md','pyproject.toml','uv.lock','LICENSE')];manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(files)) if p.is_file()}
    for n in manifest:
        if n.endswith('.py') and ('v20' in n or n.startswith('engine/aniso_phase1/')):ast.parse((ROOT/n).read_text())
    with zipfile.ZipFile(OUT/'source-delivered.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in manifest:z.write(ROOT/n,n)
        z.writestr('SOURCE_SHA256.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(OUT/'source-delivered.zip') as z:
        for n,digest in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==digest
    write(OUT/'source-delivered-sha256.json',manifest)
    report=ROOT/'docs/ANISO_LITE_REACTION_MODES_LOCAL_DOF_ZH.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if target.endswith(('completion-check.json','artifact-sha256.json')):continue
        assert (report.parent/target).exists(),target
    completion=dict(requested_implementation_and_validation_work_completed=True,version='v20 bounded reaction modes / separated moving inertia / local deformation experiments',primary_full_cycles=4,full_cycles_with_ablations=6,canonical_full_cycle_steps=total,original_implementation_full_cycle_steps=3200,short_same_snapshot_cases=24,short_same_snapshot_steps=5600,related_tests=t['tests'],independent_snapshot_audits=42,static_space_checks=38,all_static_space_checks_positive=True,
        raw_reaction_time_passed=c['raw_reaction_time_passed'],stress_time_passed=c['stress_time_passed'],constraint_loss_decreases=c['constraint_loss_decreases'],dynamic_candidate_acceptance_passed=c['passed'],old_snapshot_moving_integration_passed=load(OUT/'moving-quadrature.json')['all_reference_checks_passed'],actual_moving_integration_reference_passed=all(x['cut_reference_passed'] for x in q['records']),actual_moving_reaction_mode_inertia_passed=all(x['native_reaction_mode_inertia_passed'] for x in q['records']),spatial_stress_accuracy_passed=False,continuum_reference_stress_certified=False,local_enrichment_in_full_moving_dynamics=False,production_default_changed=False,cuda_tested=False,full_repository_suite=False,
        recovery_provenance=recoveries,old_v19_artifacts_unchanged=len(old),old_python_sources_unchanged=oldsrc_count,source_files=len(manifest),source_zip_sha256=sha(OUT/'source-delivered.zip'),report=str(report.relative_to(ROOT)),final_time_pair=c['pairs'][-1],limitations=['Local Q3 stress reference is not fully certified at 2%.','Kinetic integration validates the chosen APIC functional, not continuum inertia accuracy.','Material-null condensation is exact statically; deformed-state dynamic equivalence to the original model is not claimed.','Local FE enrichment is a spatial prototype, not yet combined with the full moving dynamic candidate.','Frozen dense support and F45 cyclic benchmark do not certify arbitrary support expansion.'])
    write(OUT/'completion-check.json',completion);artifacts={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=OUT/'artifact-sha256.json'};write(OUT/'artifact-sha256.json',artifacts)
    for n,digest in artifacts.items():assert sha(ROOT/n)==digest
    print('SEALED',len(artifacts),'artifacts',len(manifest),'sources',json.dumps(completion),flush=True)
if __name__=='__main__':
    import sys
    tests() if len(sys.argv)>1 and sys.argv[1]=='tests' else main()
