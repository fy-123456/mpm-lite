"""Scoped source-impact review and inherited-solid regression evidence."""
from pathlib import Path
import argparse,subprocess,time
from .provenance import *
from benchmarks.research_phase_stress_next.time_study import history


def prepare(run):
    run=Path(run);verify(run);old=read(APP/'S6/final-protocol.json');default=old['default_case'];files=['base_config.py','config.py','spaces.py','physics.py','run.py'];same={name:sha(ROOT/'benchmarks/research_candidate_observable_next'/name)==sha(ROOT/'benchmarks/research_pressure_window_next'/name) for name in files}
    if not all(same.values()):raise ValueError('solid driver impact requires re-review')
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=same,ancestor_sources_immutable=True,new_solid_numerical_change=False,new_formal_default_entry=False,new_full_cycles_required=False,research_coupling_parameter_adapter_only=True,production_q5_permission_not_extended=True))
    h=history(APP/'cases'/default);cfg=read(APP/'cases'/default/'execution-protocol.json');rows=h[-1]['rows'];physical=dict(steps=len(rows),end_s=h[-1]['state'].time,min_detF=min(x['min_detF'] for x in rows),max_residual_fraction=max(x['true_residual']/x['residual_tolerance'] for x in rows),max_constraint=max(max(x['displacement_constraint'],x['velocity_constraint']) for x in rows))
    if physical['steps']!=252 or physical['end_s']!=1.6 or physical['min_detF']<=.1 or physical['max_residual_fraction']>1:raise ValueError('inherited scene failed physical inspection')
    evidence={name:dict(path=str(APP/name),sha256=sha(APP/name)) for name in ['S6/final-scene.json','S6/raw-physical-check.json','S6/qualification-final.json','S6/fallback-and-restart.json','S2/transaction-and-restart.json','S6/final-tests-result.json']}
    write(run/'S6/inherited-solid-evidence.json',dict(status='inherited_passed_scoped',source_release=APP_SHA,physical=physical,evidence=evidence,new_full_steps=0,compatibility_steps=4,parent_certificate_not_reissued=True))
    write(run/'S6/default-scene-decision.json',dict(status='retain_inherited_default',default_case=default,default_case_source=str(APP),source_release_sha256=APP_SHA,new_full_cycles=0,reason='all solid equations, spatial package, mass, material and original entry remain immutable; only child research drivers/diagnostics changed',formal_steps=252))
    write(run/'S6/fallback-and-restart.json',dict(status='inherited_passed_scoped',parent_evidence=evidence['S6/fallback-and-restart.json'],new_compatibility=read(run/'S0/compatibility.json'),candidate_bridge=read(run/'S1/checkpoint-bridge.json'),new_fault_attempts=0,scope='parent q5 only; coupled q5 unauthorized; new coupling restart assessed separately'))
    write(run/'S6/final-protocol.json',dict(default_case=default,default_case_source='parent',parent_root=str(APP),formal_steps=252,new_full_steps=0,material_policy='inherited q5_with_full_retry'))


def tests(run):
    run=Path(run);modules=['tests.research_candidate_observable_next.test_observed','tests.research_candidate_observable_next.test_fixture','tests.research_candidate_observable_next.test_rt0_cache']
    # Focused inherited mixed residual/ownership/transaction tests cover the actual adapter.
    modules+=['tests.research_pressure_window_next.test_runtime']
    started=time.perf_counter();result=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*modules,'-v'],capture_output=True,text=True,cwd=ROOT)
    log=run/'S6/tests.log';log.parent.mkdir(exist_ok=True);log.write_text(result.stdout+'\n'+result.stderr)
    import re
    match=re.search(r'Ran (\d+) tests',result.stderr)
    write(run/'S6/tests-result.json',dict(status='passed' if result.returncode==0 else 'failed',actual_run_count=int(match[1]) if match else 0,new_test_count=14,inherited_prior_full_suite_count=50,inherited_suite_not_rerun=True,modules=modules,log_sha256=sha(log),seconds=time.perf_counter()-started,numerical_source_sha256=source_files()))
    if result.returncode:raise RuntimeError(result.stderr)
    print('FOCUSED_TESTS',match[1],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','tests']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
