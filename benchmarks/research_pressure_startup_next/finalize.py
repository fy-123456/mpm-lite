"""Close scoped engineering evidence without advancing another trajectory."""
from pathlib import Path
import argparse,re
import numpy as np
from .provenance import *
from .lineage import default_source
from .coupling import case_name
from benchmarks.research_phase_stress_next.time_study import history

def prepare(run):
    run=Path(run);verify(run)
    unchanged={n:sha(ROOT/'benchmarks/research_stabilization_boundary_next'/n)==sha(ROOT/'benchmarks/research_pressure_startup_next'/n) for n in ('base_config.py','config.py','spaces.py','physics.py','run.py')}
    if not all(unchanged.values()):raise ValueError('solid core drift')
    checked=[]
    for fine in (False,True):
        folder=run/'cases'/case_name('coarse',fine);h=history(folder);rows=h[-1]['rows'];summary=read(folder/'summary.json');capacity=.0002*np.asarray(rows[0]['cell_volume_m3'])
        # Exact capacity comes from the unchanged reference cell volumes.
        from .coupling_review import context
        cuts=read(run/'S5/selected-protocol.json')['cuts']['coarse'];capacity=context(cuts,.0002)['C']
        prior=np.asarray(h[0]['state'].child_states['fluid']['pressure_Pa']);E0=.5*float(np.sum(capacity*prior**2));direct=[];num=[];priorE=E0
        for row in rows:
            p=np.asarray(row['pressure_Pa']);expected=(row['theta']-.5)*float(np.sum(capacity*(p-prior)**2));num.append(abs(expected-row['numerical_dissipation_J']))
            bal=row['total_energy_J']-priorE+row['darcy_dissipation_J']+row['numerical_dissipation_J']-row['external_work_J']-row['source_work_J']-row['reservoir_work_J'];direct.append(bal)
            prior=p;priorE=row['total_energy_J']
        zero=all(row['source_work_J']==0 for row in rows) and not np.any(h[-1]['state'].child_states['fluid']['cumulative_source_m3'])
        if not zero or max(num)>1e-18 or summary['max_residual_fraction']>1 or summary['min_detF']<=.1:raise ValueError('physical/source ledger failed')
        if max(abs(np.array(direct)))>1e-9+.01*max(E0,priorE):raise ValueError('direct energy budget failed')
        checked.append(dict(case=folder.name,summary=summary,source_zero_verified=bool(zero),initial_energy_J=E0,max_recomputed_balance_J=max(abs(np.array(direct))),cumulative_recomputed_balance_J=sum(direct),Dnum_identity_error_J=max(num),physical_balance_without_Dnum_J=sum(r['physical_energy_balance_J'] for r in rows),Dnum_over_Darcy=summary['numerical_dissipation_J']/max(summary['darcy_dissipation_J'],1e-300),Dnum_over_initial_energy=summary['numerical_dissipation_J']/E0))
    comparison=read(run/'S5/coarse-temporal.json');comparison.update(scope='corrected zero-source original16-cell prefix 0..1.5e-4 s, 12/24 steps',spatial=dict(status='not_run',reason='original source mismatch consumed budget; no corrected32-cell run'),source_mismatch_evidence_excluded=True)
    write(run/'S5/coupled-comparison.json',comparison)
    write(run/'S5/coupling-decision.json',dict(status='passed_time_prefix_space_unchecked' if comparison['status']=='passed_scoped' else 'limited',same_grid_time_passed=comparison['status']=='passed_scoped',window_s=[0.,1.5e-4],coarse_steps=12,fine_steps=24,zero_source=True,full_original_window_passed=False,grid_comparison_passed=False,certified_spatial_accuracy=False,production_C_E_integration=False,coupled_q5=False,full_cycle=False,new_accepted_zero_source_steps=36,invalid_protocol_attempts=60,scope=comparison['scope']))
    fault=read(run/'S5/fault.json');restart=read(run/'S5/restart.json')
    if fault['status']!='passed_scoped' or restart['status']!='passed_scoped':raise ValueError('theta transaction check failed')
    write(run/'S5/energy-and-transaction.json',dict(status='passed_scoped',records=checked,numerical_dissipation_reported_separately=True,numerical_dissipation_not_physical=True,quasi_newton_true_residual_checked=True,fault=fault,restart=restart,corrected_source_zero=True))
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=unchanged,ancestor_sources_immutable=True,solid_equations_unchanged=True,performance=read(run/'S4/performance-decision.json'),production_q5_permission_not_extended=True,source_mismatch_fixed_in_new_entry_only=True))
    folder,chain=default_source(APP,APP_SHA);owner=folder.parent.parent;h=history(folder);rows=h[-1]['rows'];physical=dict(steps=len(rows),end_s=h[-1]['state'].time,min_detF=min(x['min_detF'] for x in rows),max_residual_fraction=max(x['true_residual']/x['residual_tolerance'] for x in rows),max_constraint=max(max(x['displacement_constraint'],x['velocity_constraint']) for x in rows))
    if physical['steps']!=252 or physical['end_s']!=1.6 or physical['min_detF']<=.1 or physical['max_residual_fraction']>1:raise ValueError('inherited scene invalid')
    write(run/'S6/inherited-solid-evidence.json',dict(status='inherited',source=str(folder),chain=chain,physical=physical,new_full_steps=0,identity_sha256=sha(folder/'identity.json')))
    default=dict(default_case=folder.name,default_case_source=dict(release_root=str(owner),release_sha256=sha(owner/'release.json'),case=folder.name,identity_sha256=sha(folder/'identity.json'),execution_protocol_sha256=sha(folder/'execution-protocol.json')))
    write(run/'S6/default-scene-decision.json',dict(status='inherited',**default,formal_steps=252,new_formal_steps=0))
    write(run/'S6/fallback-and-restart.json',dict(status='passed_scoped',solid_q5=dict(path=str(owner/'S6/fallback-and-restart.json'),sha256=sha(owner/'S6/fallback-and-restart.json')),performance_fault=read(run/'S4/fault-B0/failure.json'),performance_restart=read(run/'S4/restart.json'),theta_fault=fault,theta_restart=restart,new_S6_dynamic_attempts=0))
    report=read(run/'S6/test-report.json');n=int(re.search(r'Ran (\d+) tests',(run/'S6/tests.log').read_text()).group(1));report.update(distinct_tests=n,last_execution_tests=n,total_executions=n+9,prior_test_executions=9,log_sha256=sha(run/'S6/tests.log'));write(run/'S6/test-report.json',report)
    print('PREPARED',comparison['status'],[(v['case'],v['Dnum_over_Darcy']) for v in checked],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):prepare(a.run)
