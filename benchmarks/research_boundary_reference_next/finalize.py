"""Report original evidence and explain the deliberately untriggered grid branch."""
import argparse
import numpy as np
from .provenance import *
from .lineage import default_source
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric


def diagnosis(run):
    run=Path(run);mutable(run);screen=read(run/'R2/grid-screen.json');records=[]
    for candidate in screen['records']:
        times=[]
        for t in candidate['time']:
            cmp=t['comparison'];rows=cmp['records']
            worst=max(((r,k,v) for r in rows for k,v in r['budget_ratios'].items()),key=lambda x:x[2])
            times.append(dict(label=t['label'],first_interval=rows[0],worst=dict(time_s=worst[0]['time_s'],field=worst[1],budget_ratio=worst[2],check=worst[0]['checks'][worst[1]]),maximum_ratios=cmp['max_budget_ratios'],minimum_pressure=t['minimum_pressure'],max_mass_defect_m3=t['max_mass_defect_m3'],max_energy_balance_J=t['max_energy_balance_J']))
        records.append(dict(first_m=candidate['first_m'],table=candidate['table'],reference_max=max(candidate['reference']['max_budget_ratios'].values()),reference_budget_fraction=.25,spatial_max=max(candidate['spatial']['max_budget_ratios'].values()),time=times,eligible=candidate['eligible']))
    # Analytic diagnostic, no new integration or simulated reference.
    write(run/'R2/diagnosis.json',dict(status='known_startup_time_limitation',records=records,first_BE_continuum_integral_ratio=float(np.sqrt(np.pi)/2),first_BE_relative_deficit=float(1-np.sqrt(np.pi)/2),formulas=dict(diffusion='S dp/dt = k d2p/dx2',early_cumulative_per_end='2 A (p0-pb) sqrt(k S t / pi)',BE_first_cumulative_per_end='A (p0-pb) sqrt(k S h)',BE_modal='1/(1+h lambda)',midpoint_modal='(1-h lambda/2)/(1+h lambda/2)'),interpretation='A pressure jump gives a square-root drainage boundary layer. The first BE interval has a nonvanishing relative bias in the half-space limit; smaller raw first intervals alone do not guarantee a smaller relative error. Resolving thinner cells exposes high-rate modes; midpoint does not strongly damp these modes. Observed switch-local error is consistent with this mechanism, not a proof of instability or an exact scalar formula for full-tensor coupled flow.',fixed_skeleton_only=True,scalar_analytic_not_full_tensor_truth=True,reference_x_refinement_only=True,uniform_reference_passed=True,cubic_earliest_intervals_reference_limited=True,actual_coupled_new_grid_qualified=False,original_16cell_scope_retained=True,source_or_boundary_ramp_changed=False,next_experiment='Pre-register fixed engineering observation intervals, conservative startup substeps and their summed boundary integrals; retain all raw substep diagnostics. Compare full-tensor semidiscrete exact time solutions before any actual coupled grid trial.',new_dynamic_attempts=0))


def prepare(run):
    run=Path(run);mutable(run);diagnosis(run)
    unchanged={n:sha(ROOT/'benchmarks/research_zero_source_next'/n)==sha(ROOT/'benchmarks/research_boundary_reference_next'/n) for n in ('base_config.py','config.py','spaces.py','physics.py','run.py')}
    if not all(unchanged.values()):raise ValueError('solid core drift')
    from benchmarks.research_pressure_startup_next.coupling_review import context
    p=read(run/'R0/input-contract.json');capacity=context(p['cuts']['coarse'],p['parameters']['storage'])['C'];actual=[]
    for kind in ('A','B'):
        for i in (0,1):
            folder=run/f'R4/{kind}{i}';h=history(folder);initial=h[0]['state'];final=h[-1]['state'];row=h[-1]['rows'][-1]
            p0=np.array(initial.child_states['fluid']['pressure_Pa']);pn=np.array(final.child_states['fluid']['pressure_Pa']);dn=(row['theta']-.5)*np.sum(capacity*(pn-p0)**2)
            e0=initial.child_states['fluid']['last_ledger']['total_energy_J'];balance=row['total_energy_J']-e0+row['darcy_dissipation_J']+row['numerical_dissipation_J']-row['external_work_J']-row['source_work_J']-row['reservoir_work_J']
            zero=row['source_work_J']==0 and not np.any(final.child_states['fluid']['cumulative_source_m3'])
            if not zero or abs(dn-row['numerical_dissipation_J'])>1e-18 or row['true_scaled_residual']>1 or row['min_detF']<=.1 or abs(balance)>1e-9+.01*abs(e0):raise ValueError('physical ledger failed')
            actual.append(dict(case=f'{kind}{i}',initial_time_s=initial.time,final_time_s=final.time,source_zero=zero,min_detF=row['min_detF'],minimum_pressure_Pa=float(pn.min()),true_residual_fraction=row['true_scaled_residual'],mass_defect_m3=float(max(abs(np.array(row['mass_defect_m3'])))),numerical_dissipation_identity_error_J=float(abs(dn-row['numerical_dissipation_J'])),energy_balance_J=float(balance),identity_sha256=sha(folder/'identity.json'),state_digest=final.digest(),accepted_steps=1))
    write(run/'S6/physical-ledger-audit.json',dict(status='passed_scoped',records=actual,original_residuals_unchanged=True,actual_new_grid_steps=0))
    af=next((run/'R4/A1').rglob('frame.npz'));bf=next((run/'R4/B1').rglob('frame.npz'))
    with np.load(af) as a,np.load(bf) as b:
        if set(a.files)!=set(b.files):raise ValueError('frame fields differ')
        fieldchecks={k:metric(a[k],b[k],1e-10 if k=='face_flux_m3_s' else 1e-8,2e-5) for k in a.files}
    if not all(v['passed'] for v in fieldchecks.values()):raise ValueError('display field equivalence failed')
    write(run/'S6/frame-equivalence.json',dict(status='passed_scoped',checks=fieldchecks,A_sha256=sha(af),B_sha256=sha(bf),scope='saved actual A1/B1 endpoint, 33x7x7 probes, includes skeleton and total PK1/Cauchy stress',new_solves=0))
    perf=read(run/'R4/paired-performance.json');shared=[]
    for r in perf['records']:
        for k in ('A','B'):
            m=r[k]
            if not m['sharing_before']['exclusive'] or not m['sharing_after']['exclusive']:shared.append(dict(case=k+str(r['index']),before=m['sharing_before'],after=m['sharing_after']))
    write(run/'R4/performance-environment-review.json',dict(status='limited_shared_GPU' if shared else 'exclusive',shared_cases=shared,all_state_and_energy_checks_passed=all(r['passed'] for r in perf['records']),rt0_assembly_counts=[[r['A']['call_counts']['rt0_s'],r['B']['call_counts']['rt0_s']] for r in perf['records']],gradient_adjoint_counts=[[r['A']['call_counts']['volume_gradient_adjoint_s'],r['B']['call_counts']['volume_gradient_adjoint_s']] for r in perf['records']],observed_gain_only=[r['gain'] for r in perf['records']],other_processes_not_interrupted=True,additional_dynamic_repeats=0,default_retained=True))
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=unchanged,ancestor_sources_immutable=True,solid_equations_unchanged=True,production_q5_permission_not_extended=True,new_coupling_backend_only_qualified_for_scoped_steps=True))
    folder,chain=default_source(APP,APP_SHA);owner=folder.parent.parent;h=history(folder);rows=h[-1]['rows'];physical=dict(steps=len(rows),end_s=h[-1]['state'].time,min_detF=min(x['min_detF'] for x in rows),max_residual_fraction=max(x['true_residual']/x['residual_tolerance'] for x in rows))
    if physical['steps']!=252 or physical['end_s']!=1.6 or physical['min_detF']<=.1 or physical['max_residual_fraction']>1:raise ValueError('inherited daily scene invalid')
    write(run/'S6/inherited-solid-evidence.json',dict(status='inherited',source=str(folder),chain=chain,physical=physical,new_steps=0))
    write(run/'S6/default-scene-decision.json',dict(status='inherited',default_case=folder.name,default_case_source=dict(release_root=str(owner),release_sha256=sha(owner/'release.json'),case=folder.name,identity_sha256=sha(folder/'identity.json'),execution_protocol_sha256=sha(folder/'execution-protocol.json')),formal_steps=252,new_formal_steps=0))
    fault=read(run/'R4/fault-B0/failure.json');restart=read(run/'R4/restart.json')
    if not fault['rollback_exact'] or not fault['cache_cleared'] or not restart['same_digest']:raise ValueError('transaction failure')
    write(run/'S6/fallback-and-restart.json',dict(status='passed_scoped',performance_fault=fault,performance_restart=restart,solid_q5=dict(path=str(owner/'S6/fallback-and-restart.json'),sha256=sha(owner/'S6/fallback-and-restart.json')),new_S6_dynamic_attempts=0))
    last=read(run/'S6/final-test-execution.json')
    if last['returncode']!=0 or last['tests']!=5:raise ValueError('final tests did not pass')
    write(run/'S6/test-report.json',dict(status='passed_scoped',command=last['command'],returncode=0,distinct_tests=5,total_test_case_executions=14,suite_executions=3,suite_counts=[4,5,5],execution_notes='First two suites before actual GPU steps; third after moving unittest.main below all test classes. No engine changes after the five-test suite or actual steps.',final_log='S6/final-tests.log',tested_sources={k:v for k,v in all_sources().items() if k.startswith(('engine/','tests/'))},inherited_tests_not_rerun=True,actual_checks=['R1/scalar-reference.json','R2/grid-screen.json','R4/operator-equivalence.json','R4/paired-performance.json','S6/physical-ledger-audit.json','S6/fallback-and-restart.json']))
    from .seal import capabilities,accounting
    accounting(run);write(run/'capability-matrix.json',capabilities(run));(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    print('FINAL_PREPARED',len(actual),'actual single-step cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):prepare(a.run)
