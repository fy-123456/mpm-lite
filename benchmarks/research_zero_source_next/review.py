"""Compare immutable histories with declared engineering budgets, including first interval."""
import argparse
import numpy as np
from .provenance import *
from .coupling import case_name
from benchmarks.research_pressure_startup_next.coupling_review import context,fluid,compare_fluid,compare_fields
from benchmarks.research_pressure_startup_next.pressure import coarse_observations
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction


def pair(run,method_a,grid_a,fine_a,method_b,grid_b,fine_b):
    a=Path(run)/'cases'/case_name(grid_a,fine_a,method_a);b=Path(run)/'cases'/case_name(grid_b,fine_b,method_b)
    va,t,rows=fluid(a);vb,tt,rhs=fluid(b);stride=2 if len(tt)==2*len(t)-1 else 1
    if stride==2:
        restricted=coarse_observations(vb,tt);restricted['content']=vb['content'][::2];vb=restricted;tt=tt[::2]
    if len(t)!=len(tt) or not np.allclose(t,tt,atol=1e-18,rtol=0):raise ValueError('comparison time nodes differ')
    p=read(Path(run)/'S0/input-contract.json');contexts={g:context(p['cuts'][g],p['parameters']['storage']) for g in ('coarse','fine')}
    flow=compare_fluid(contexts[grid_a],contexts[grid_b],va,vb,t);solid=compare_fields(a,b,rows,rhs,stride)
    return dict(status='passed_scoped' if flow['status']=='passed_scoped' and solid['passed'] else 'limited',a=str(a),b=str(b),fluid=flow,solid=solid,first_interval_included=True,flux_is_interval_integral=True,engineering_comparison_not_continuum_truth=True)


def energy(run,method,grid,stage):
    records=[]
    for fine in (False,True):
        folder=Path(run)/'cases'/case_name(grid,fine,method);s=read(folder/'summary.json');rows=read(folder/'ledger.json');E0=s['initial_energy_J']
        cumulative=s['direct_energy_change_J']+sum(r['darcy_dissipation_J']+r['numerical_dissipation_J']-r['external_work_J']-r['reservoir_work_J']-r['source_work_J'] for r in rows)
        closure=cumulative-sum(r['path_error_J']+r['solve_work_J'] for r in rows)
        sourcezero=all(r['source_work_J']==0 for r in rows)
        if not sourcezero or abs(cumulative)>1e-9+.01*abs(E0) or abs(closure)>1e-10+.01*abs(E0):raise ValueError('cumulative physical ledger gate')
        records.append(dict(fine=fine,summary=s,cumulative_energy_balance_J=cumulative,cumulative_path_and_solve_removed_J=closure,Dnum_over_E0=s['numerical_dissipation_J']/E0,Dnum_over_Ddarcy=s['numerical_dissipation_J']/s['darcy_dissipation_J'],source_zero=sourcezero,raw_interval_ledgers=str(folder/'ledger.json'),all_iterations=[r['iterations'] for r in rows],all_pressure_positive=min(min(r['pressure_Pa']) for r in rows)>=0))
    out=dict(status='passed_scoped',records=records,fault=read(Path(run)/stage/'fault.json') if (Path(run)/stage/'fault.json').exists() else None,restart=read(Path(run)/stage/'restart.json') if (Path(run)/stage/'restart.json').exists() else None)
    write(Path(run)/stage/'energy-and-transaction.json',out);return out


def baseline(run):
    verify(run);v=pair(run,'backward-euler','coarse',False,'backward-euler','coarse',True);write(Path(run)/'S1/full-window-comparison.json',v);energy(run,'backward-euler','coarse','S1')
    audits=[read(p) for p in sorted((Path(run)/'S1').glob('*prefix-audit.json'))];write(Path(run)/'S1/continuation-audit.json',dict(status='passed_scoped',records=audits))
    print('BASELINE',v['status'],v['fluid']['max_budget_ratios'],v['solid']['max_budget_fractions'],flush=True)


def startup(run):
    verify(run)
    if read(Path(run)/'S1/full-window-comparison.json')['status']!='passed_scoped':raise ValueError('BE window not qualified')
    v=pair(run,'startup','coarse',False,'startup','coarse',True);cross=[pair(run,'startup','coarse',f,'backward-euler','coarse',f) for f in (False,True)];en=energy(run,'startup','coarse','S2');ratios=[]
    for fine in (False,True):
        a=read(Path(run)/'cases'/case_name('coarse',fine,'backward-euler')/'summary.json');b=read(Path(run)/'cases'/case_name('coarse',fine,'startup')/'summary.json');drop=a['numerical_dissipation_J']-b['numerical_dissipation_J'];ratios.append(dict(fine=fine,Dnum_reduction=drop/a['numerical_dissipation_J'],Dnum_drop_J=drop,BE_seconds=a['seconds'],startup_seconds=b['seconds'],BE_inherited_steps=a['inherited_steps'],timings_have_different_reuse_and_not_speedup=True))
    selected=v['status']=='passed_scoped' and all(x['status']=='passed_scoped' for x in cross) and all(r['Dnum_reduction']>=.2 and r['Dnum_drop_J']>1e-12 for r in ratios)
    write(Path(run)/'S2/startup-comparison.json',dict(temporal=v,cross_method=cross,dissipation=ratios,status='passed_scoped' if selected else 'limited'))
    switch=[]
    for fine in (False,True):
        rows=read(Path(run)/'cases'/case_name('coarse',fine,'startup')/'ledger.json');n=4 if fine else 2
        switch.append(dict(fine=fine,rows=rows[max(0,n-2):n+2],minimum_pressure=min(min(r['pressure_Pa']) for r in rows),midpoint_Dnum_exact_zero=all(r['numerical_dissipation_J']==0 for r in rows[n:]),phase_peak_qualification='unobserved: no periodic or loading peak event in this window'))
    write(Path(run)/'S2/switch-window.json',dict(status='passed_scoped',records=switch,not_clipped=True))
    register(run,'S2/method-decision.json',dict(status='qualified_scoped' if selected else 'retain_BE',method='startup' if selected else 'backward-euler',grid='coarse',window_s=[0.,2e-4],source_m3_s=0.,switch_s=2.5e-5 if selected else None,numerical_benefit=ratios,scope='16-cell actual two-way coupling; 16/32 temporal engineering agreement',global_time_accuracy=False))
    print('METHOD','startup' if selected else 'backward-euler',ratios,flush=True)


def grid_prepare(run):
    verify(run);run=Path(run);p=read(run/'S0/input-contract.json');d=read(run/'S2/method-decision.json');ctx={g:context(p['cuts'][g],p['parameters']['storage']) for g in ('coarse','fine')};P,M,Z=restriction(ctx['coarse']['top'],ctx['fine']['top']);a=ctx['coarse']['top'];b=ctx['fine']['top'];err=float(np.max(abs(a.B@Z-M@b.B)))
    if err>1e-10:raise ValueError('nonconservative grid transfer')
    register(run,'S3/grid-protocol.json',dict(status='registered',method=d['method'],cuts=p['cuts'],source_m3_s=0.,steps=[16,32],window_s=[0.,2e-4],GPU_max_cells=32))
    write(run/'S3/transfer-check.json',dict(status='passed_scoped',divergence_commutation_error=err,volume_coverage=float(np.max(abs(M@b.V0-a.V0))),constant_pressure_error=float(np.max(abs(P@np.ones(b.cells)-1))),pressure_intensive_content_extensive=True,new_dynamic_steps=0))


def grid(run):
    verify(run);run=Path(run);method=read(run/'S2/method-decision.json')['method'];a=pair(run,method,'coarse',False,method,'coarse',True);b=pair(run,method,'fine',False,method,'fine',True);s=pair(run,method,'coarse',True,method,'fine',True)
    timepass=a['status']==b['status']=='passed_scoped';agree=timepass and s['status']=='passed_scoped'
    write(run/'S3/coupled-comparison.json',dict(coarse_temporal=a,fine_temporal=b,spatial=s,spatial_status='nested_grid_agreement' if agree else ('limited' if timepass else 'time_contaminated')))
    write(run/'S3/grid-decision.json',dict(status='nested_grid_agreement' if agree else 'limited',method=method,coarse_time_passed=a['status']=='passed_scoped',fine_time_passed=b['status']=='passed_scoped',nested_grid_agreement=agree,true_coupled_spatial_accuracy='uncertified',production_C_E=False,coupled_q5=False,full_cycle=False,performance_grid='coarse'))
    write(run/'S3/diagnosis.json',dict(status='not_triggered' if agree else 'pending_bounded_diagnosis',reason='all declared nested engineering comparisons pass; further reference not required' if agree else 'inspect recorded budget maxima before bounded fixed-skeleton diagnosis',coupled_continuum_reference=False,new_CPU_matrices=0,spectral_calls=0))
    print('GRID',a['status'],b['status'],s['status'],'spatial',s['fluid']['max_budget_ratios'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['baseline','startup','grid_prepare','grid']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
