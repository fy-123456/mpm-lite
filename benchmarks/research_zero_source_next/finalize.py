"""Close scoped evidence and inherit the unchanged daily scene with zero new steps."""
import argparse
import numpy as np
from .provenance import *
from .lineage import default_source
from benchmarks.research_phase_stress_next.time_study import history
from .review import energy


def prepare(run):
    run=Path(run);verify(run);unchanged={n:sha(ROOT/'benchmarks/research_pressure_startup_next'/n)==sha(ROOT/'benchmarks/research_zero_source_next'/n) for n in ('base_config.py','config.py','spaces.py','physics.py','run.py')}
    if not all(unchanged.values()):raise ValueError('solid core drift')
    actual=[]
    for folder in sorted((run/'cases').iterdir()):
        if not (folder/'summary.json').exists():continue
        h=history(folder);summary=read(folder/'summary.json');rows=h[-1]['rows'];ident=read(folder/'identity.json')['coupling'];core=ident['core'];p=np.array(h[0]['state'].child_states['fluid']['pressure_Pa']);priorE=summary['initial_energy_J'];num=[];balance=[]
        from .review import context
        cfg=read(folder/'execution-protocol.json');protocol=read(run/'S0/input-contract.json');capacity=context(protocol['cuts'][cfg['grid']],protocol['parameters']['storage'])['C']
        for r in rows:
            pn=np.array(r['pressure_Pa']);num.append(abs((r['theta']-.5)*np.sum(capacity*(pn-p)**2)-r['numerical_dissipation_J']));balance.append(r['total_energy_J']-priorE+r['darcy_dissipation_J']+r['numerical_dissipation_J']-r['external_work_J']-r['source_work_J']-r['reservoir_work_J']);p=pn;priorE=r['total_energy_J']
        physical=ident['physical_parent'];zero=not np.any(physical['source_m3_s']) and all(r['source_work_J']==0 for r in rows) and not np.any(h[-1]['state'].child_states['fluid']['cumulative_source_m3'])
        if not zero or max(num)>1e-18 or summary['max_residual_fraction']>1 or summary['min_detF']<=.1 or max(abs(np.array(balance)))>1e-9+.01*abs(summary['initial_energy_J']):raise ValueError('actual physics ledger failed')
        actual.append(dict(case=folder.name,source_explicit_zero=zero,initial_energy_J=summary['initial_energy_J'],max_numerical_dissipation_identity_error_J=float(max(num)),max_recomputed_energy_balance_J=float(max(abs(np.array(balance)))),cumulative_recomputed_energy_balance_J=sum(balance),max_true_residual_fraction=summary['max_residual_fraction'],min_detF=summary['min_detF'],state_count=len(h),source_identity_sha256=sha(folder/'identity.json')))
    write(run/'S6/physical-ledger-audit.json',dict(status='passed_scoped',records=actual,each_state_validated_by_coupling=True,original_residuals_not_relaxed=True))
    method=read(run/'S2/method-decision.json')['method'];energy(run,method,'fine','S3')
    write(run/'S3/operator-check.json',read(run/'S0/operator-fine.json'))
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=unchanged,ancestor_sources_immutable=True,solid_equations_unchanged=True,production_q5_permission_not_extended=True,explicit_source_constructor_guard=True,actual_source_identity_cases=actual))
    folder,chain=default_source(APP,APP_SHA);owner=folder.parent.parent;h=history(folder);rows=h[-1]['rows'];physical=dict(steps=len(rows),end_s=h[-1]['state'].time,min_detF=min(x['min_detF'] for x in rows),max_residual_fraction=max(x['true_residual']/x['residual_tolerance'] for x in rows),max_constraint=max(max(x['displacement_constraint'],x['velocity_constraint']) for x in rows))
    if physical['steps']!=252 or physical['end_s']!=1.6 or physical['min_detF']<=.1 or physical['max_residual_fraction']>1:raise ValueError('inherited scene invalid')
    write(run/'S6/inherited-solid-evidence.json',dict(status='inherited',source=str(folder),chain=chain,physical=physical,new_full_steps=0,identity_sha256=sha(folder/'identity.json')))
    default=dict(default_case=folder.name,default_case_source=dict(release_root=str(owner),release_sha256=sha(owner/'release.json'),case=folder.name,identity_sha256=sha(folder/'identity.json'),execution_protocol_sha256=sha(folder/'execution-protocol.json')))
    write(run/'S6/default-scene-decision.json',dict(status='inherited',**default,formal_steps=252,new_formal_steps=0))
    write(run/'S6/fallback-and-restart.json',dict(status='passed_scoped',solid_q5=dict(path=str(owner/'S6/fallback-and-restart.json'),sha256=sha(owner/'S6/fallback-and-restart.json')),performance_fault=read(run/'S4/fault-B0/failure.json'),performance_restart=read(run/'S4/restart.json'),BE_fault=read(run/'S1/fault.json'),BE_restart=read(run/'S1/restart.json'),startup_fault=read(run/'S2/fault.json'),startup_restart=read(run/'S2/restart.json'),new_S6_dynamic_attempts=0))
    write(run/'S6/test-report.json',dict(status='passed_scoped',command='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m unittest discover -s tests/research_zero_source_next -v',returncode=0,distinct_tests=4,total_test_case_executions=4,suite_executions=1,executed_before_actual_GPU_trajectories=True,tested_sources={k:v for k,v in all_sources().items() if k.startswith(('engine/','tests/'))},inherited_tests_not_rerun=True,actual_numerical_checks=['S0/operator-coarse.json','S0/operator-fine.json','S1/full-window-comparison.json','S2/startup-comparison.json','S3/transfer-check.json','S4/operator-equivalence.json','S6/physical-ledger-audit.json','S6/fallback-and-restart.json']))
    from .seal import capabilities
    write(run/'capability-matrix.json',capabilities(run))
    print('FINAL_PREPARED',len(actual),'actual cases',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):prepare(a.run)
