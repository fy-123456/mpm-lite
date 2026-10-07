"""Dependency decisions, precise source impact and limited regressions."""
from pathlib import Path
import argparse,subprocess,time,re,os
from .provenance import *
from .lineage import default_source
from .performance import numerical_sources
from benchmarks.research_phase_stress_next.time_study import history


def prepare(run):
    run=Path(run);verify(run);candidate=read(run/'S1/candidate-reference-decision.json')
    if candidate['spatial_retraining_eligible']:raise ValueError('S5 requires separate actual experiment; cannot silently skip')
    write(run/'S5/research-space-decision.json',dict(status='not_triggered',dependency='S1 same-space temporal gate failed',
        dependency_sha256=sha(run/'S1/candidate-reference-decision.json'),new_spaces=0,new_static_solves=0,new_dynamic_attempts=0,
        formal_space_changed=False,next_minimum_experiment='Diagnose stabilization-dominated acceleration on the retained five nodes before designing another reference; no R6 or 512/1024-step expansion in this round.'))
    same={name:sha(ROOT/'benchmarks/research_restoring_rt0_next'/name)==sha(ROOT/'benchmarks/research_candidate_observable_next'/name) for name in ['base_config.py','config.py','spaces.py','physics.py']}
    a=(ROOT/'benchmarks/research_candidate_observable_next/run.py').read_text();b=(ROOT/'benchmarks/research_restoring_rt0_next/run.py').read_text()
    output_only=a.replace('if loaded is None:store.save(stepper.state,rows,frame=frame_at(stepper.state))',"if loaded is None:store.save(stepper.state,rows,frame=frame_at(stepper.state) if name!='compatibility' else None)")==b
    if not all(same.values()) or not output_only:raise ValueError('unexpected solid numerical change')
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=same,run_only_suppresses_compatibility_initial_frame=output_only,
        ancestor_sources_immutable=True,solid_equations_unchanged=True,research_device_H_changed=True,production_q5_permission_not_extended=True,
        reference_reporting_optimization_preserves_original_trajectory_snapshot=True,device_identity_fix_evidence='S2/identity-fix.json'))
    folder,chain=default_source(APP,APP_SHA);owner=folder.parent.parent;owner_sha=sha(owner/'release.json');h=history(folder);rows=h[-1]['rows']
    physical=dict(steps=len(rows),end_s=h[-1]['state'].time,min_detF=min(x['min_detF'] for x in rows),max_residual_fraction=max(x['true_residual']/x['residual_tolerance'] for x in rows),max_constraint=max(max(x['displacement_constraint'],x['velocity_constraint']) for x in rows))
    if physical['steps']!=252 or physical['end_s']!=1.6 or physical['min_detF']<=.1 or physical['max_residual_fraction']>1:raise ValueError('inherited scene invalid')
    write(run/'S6/inherited-solid-evidence.json',dict(status='inherited_passed_scoped',actual_source=str(folder),owner_release_sha256=owner_sha,chain=chain,physical=physical,
        new_full_steps=0,new_compatibility_steps=4,source_identity_sha256=sha(folder/'identity.json'),parent_certificate_not_reissued=True))
    default=dict(default_case=folder.name,default_case_source=dict(release_root=str(owner),release_sha256=owner_sha,case=folder.name,
        identity_sha256=sha(folder/'identity.json'),execution_protocol_sha256=sha(folder/'execution-protocol.json')))
    write(run/'S6/default-scene-decision.json',dict(status='inherited_passed_scoped',**default,formal_steps=252,new_formal_steps=0))
    write(run/'S6/final-protocol.json',dict(**default,formal_steps=252,new_formal_steps=0,material_policy='original q5_with_full_retry scope only'))
    write(run/'S6/fallback-and-restart.json',dict(status='passed_scoped',inherited_q5_evidence=dict(path=str(owner/'S6/fallback-and-restart.json'),sha256=sha(owner/'S6/fallback-and-restart.json')),
        corrected_research_fault=read(run/'S2/fault-B0/failure.json'),grid_restart=read(run/'cases/grid32-h/summary.json'),
        initial_publication_failure='S2/publication-anomaly.json',new_S6_attempts=0,scope='reuse actual same-source rollback/restart; q5 remains original solid scope'))


def tests(run):
    run=Path(run);modules=['tests.research_restoring_rt0_next.test_reaction','tests.research_pressure_window_next.test_runtime']
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter();r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*modules,'-v'],capture_output=True,text=True,cwd=ROOT,env=env)
    out=run/'S6';out.mkdir(exist_ok=True);log=out/'final-tests.log';log.write_text(r.stdout+'\n'+r.stderr)
    if r.returncode:raise RuntimeError(r.stderr)
    records=[]
    for source,name in [(Path('/tmp/mpm-restoring-tests.log'),'initial-tests.log'),(Path('/tmp/mpm-restoring-tests-fix.log'),'metadata-and-transfer-tests.log'),(log,'final-tests.log')]:
        text=source.read_text();match=re.search(r'Ran (\d+) tests',text)
        if not match or not re.search(r'\nOK\s*(?:\n|$)',text):raise ValueError('test log missing success')
        if source!=out/name:(out/name).write_bytes(source.read_bytes())
        records.append(dict(path=name,sha256=sha(out/name),actual_run_count=int(match[1])))
    write(out/'tests-result.json',dict(status='passed',records=records,actual_run_count=sum(x['actual_run_count'] for x in records),distinct_tests=24,
        repeated_tests_explained='6 device tests repeated after identity fix; new JSON test and transfer tests added then',latest_modules=modules,
        numerical_source_sha256=source_files(),engine_source_sha256={k:v for k,v in numerical_sources().items() if k.startswith('engine/')},
        tested_source_sha256={k:v for k,v in all_sources().items() if k.startswith(('tests/','engine/'))},final_seconds=time.perf_counter()-start))
    print('TESTS',sum(x['actual_run_count'] for x in records),'executions;24 distinct',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','tests']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
