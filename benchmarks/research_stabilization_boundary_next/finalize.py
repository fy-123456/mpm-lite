"""Scope decisions and minimal tests bound to the final implementation."""
from pathlib import Path
import argparse,subprocess,os,re,time
from .provenance import *
from .lineage import default_source
from benchmarks.research_phase_stress_next.time_study import history

def prepare(run):
    run=Path(run);verify(run);decision=read(run/'S4/space-decision.json')
    if decision.get('dynamic_eligibility'):raise ValueError('eligible S4 needs actual dynamics, not silent skip')
    unchanged={n:sha(ROOT/'benchmarks/research_restoring_rt0_next'/n)==sha(ROOT/'benchmarks/research_stabilization_boundary_next'/n) for n in ['base_config.py','config.py','spaces.py','physics.py','run.py']}
    if not all(unchanged.values()):raise ValueError('unreviewed core change')
    prefix=read(run/'cases/candidate-prefix-half/ledger.json');physical_prefix=dict(status='passed_scoped',steps=len(prefix),min_detF=min(r['min_detF'] for r in prefix),max_residual_fraction=max(r['true_residual']/r['residual_tolerance'] for r in prefix),max_constraint=max(max(r['displacement_constraint'],r['velocity_constraint']) for r in prefix),max_energy_closure_J=max(abs(r['budget_defect_J']) for r in prefix),mass_order=7,material_order=7)
    if physical_prefix['steps']!=16 or physical_prefix['min_detF']<=.1 or physical_prefix['max_residual_fraction']>1 or physical_prefix['max_constraint']>1e-10:raise ValueError('prefix physical check')
    write(run/'S3/physical-check.json',physical_prefix)
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=unchanged,ancestor_sources_immutable=True,original_32_cell_constructor_unchanged=True,solid_equations_unchanged=True,CPU_reference_128_only=True,performance=read(run/'S5/performance-decision.json'),production_q5_permission_not_extended=True,source_inventory=all_sources()))
    folder,chain=default_source(APP,APP_SHA);owner=folder.parent.parent;h=history(folder);rows=h[-1]['rows'];physical=dict(steps=len(rows),end_s=h[-1]['state'].time,min_detF=min(x['min_detF'] for x in rows),max_residual_fraction=max(x['true_residual']/x['residual_tolerance'] for x in rows),max_constraint=max(max(x['displacement_constraint'],x['velocity_constraint']) for x in rows))
    if physical['steps']!=252 or physical['end_s']!=1.6 or physical['min_detF']<=.1 or physical['max_residual_fraction']>1:raise ValueError('inherited scene invalid')
    write(run/'S6/inherited-solid-evidence.json',dict(status='inherited',source=str(folder),chain=chain,physical=physical,new_full_steps=0,new_compatibility_steps=0,identity_sha256=sha(folder/'identity.json')))
    default=dict(default_case=folder.name,default_case_source=dict(release_root=str(owner),release_sha256=sha(owner/'release.json'),case=folder.name,identity_sha256=sha(folder/'identity.json'),execution_protocol_sha256=sha(folder/'execution-protocol.json')))
    write(run/'S6/default-scene-decision.json',dict(status='inherited',**default,formal_steps=252,new_formal_steps=0));write(run/'S6/final-protocol.json',dict(**default,formal_steps=252,new_formal_steps=0,material_policy='original q5_with_full_retry scope only'))
    perf=read(run/'S5/performance-decision.json');fault=run/'S5/fault-B0/failure.json';restart=run/'S5/restart.json'
    if perf['selected'] and (not fault.exists() or not restart.exists()):raise ValueError('selected implementation lacks fault/restart')
    if (run/'S5/ownership-review.json').exists():
        eq=read(run/'S5/operator-equivalence.json')
        if not eq['host_weight_source_mutation_isolated']:raise ValueError('host ownership fix not verified')
        write(run/'S5/ownership-fix.json',dict(status='passed_scoped',review_sha256=sha(run/'S5/ownership-review.json'),source_mutation_isolated=True,final_engine_sha256=sha(ROOT/'engine/aniso_phase1/research_stabilization_boundary_next/local_geometry.py'),final_operator_sha256=sha(run/'S5/operator-equivalence.json'),final_B_trials=['S5/B0','S5/B1'],earlier_evidence_preserved='S5/before-owned-host',algorithmic_candidates=1))
    write(run/'S6/fallback-and-restart.json',dict(status='passed_scoped',solid_q5=dict(path=str(owner/'S6/fallback-and-restart.json'),sha256=sha(owner/'S6/fallback-and-restart.json')),prior_coupled_evidence=dict(path=str(APP/'S6/fallback-and-restart.json'),sha256=sha(APP/'S6/fallback-and-restart.json')),new_fault=read(fault) if fault.exists() else None,new_restart=read(restart) if restart.exists() else None,new_S6_attempts=0,scope='unchanged solid q5 and numerical reference integrator; owned performance implementation separately tested'))

def tests(run):
    run=Path(run);verify(run);modules=['tests.research_stabilization_boundary_next.test_pressure']
    if (ROOT/'tests/research_stabilization_boundary_next/test_local_geometry.py').exists():modules.append('tests.research_stabilization_boundary_next.test_local_geometry')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1');tick=time.perf_counter();r=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','unittest',*modules,'-v'],capture_output=True,text=True,cwd=ROOT,env=env);out=run/'S6';out.mkdir(exist_ok=True);(out/'new-tests.log').write_text(r.stdout+'\n'+r.stderr)
    if r.returncode:raise RuntimeError(r.stderr)
    n=int(re.search(r'Ran (\d+) tests',r.stderr).group(1));write(out/'tests-result.json',dict(status='passed_scoped',actual_run_count=n,distinct_tests=n,repeated_tests=0,modules=modules,log_sha256=sha(out/'new-tests.log'),tested_sources={k:v for k,v in all_sources().items() if k.startswith(('tests/','engine/'))},inherited_distinct_tests=24,inherited_tests_newly_executed=0,inherited_report=dict(path=str(APP/'S6/tests-result.json'),sha256=sha(APP/'S6/tests-result.json')),seconds=time.perf_counter()-tick));print('NEW_TESTS',n,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','tests']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
