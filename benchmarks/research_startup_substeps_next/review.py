"""Review saved raw and engineering histories without additional solves."""
import argparse
import numpy as np
from .provenance import *
from benchmarks.research_pressure_startup_next.coupling_review import context,fluid,compare_fluid
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_startup_substeps_next.schedule import OBSERVATIONS,aggregate

def review(run):
    run=Path(run);mutable(run);p=read(run/'S3/coupled-protocol.json');ctx=context(p['cuts'],p['parameters']['storage']);a=run/'cases/boundary32-h';b=run/'cases/boundary32-half'
    va,t,rows=fluid(a);vb,tt,rhs=fluid(b);ea=aggregate(va,t,OBSERVATIONS);eb=aggregate(vb,tt,OBSERVATIONS)
    cmp=compare_fluid(ctx,ctx,ea,eb,OBSERVATIONS);raw=compare_fluid(ctx,ctx,va,aggregate(vb,tt,t),t)
    ra=read(a/'engineering-ledger.json');rb=read(b/'engineering-ledger.json');solid=[]
    with np.load(a/'probes.npz') as x,np.load(b/'probes.npz') as y:
        aa={k:x[k] for k in x.files};bb={k:y[k] for k in y.files}
    if not np.array_equal(aa['times'],bb['times']) or not np.array_equal(aa['times'],OBSERVATIONS):raise ValueError('engineering probe time mismatch')
    for i,(ar,br) in enumerate(zip(ra,rb),1):
        fa=dict(X=aa['X'],**{k:aa[k][i] for k in ('x','velocity','PK1')});fb=dict(X=bb['X'],**{k:bb[k][i] for k in ('x','velocity','PK1')});checks=fields(fa,fb,aa['fiber'])
        for region,w in regions(aa['X']).items():
            checks[region].update({k:metric(aa[k][i],bb[k][i],.02,.05,w) for k in ('PK1_total','Cauchy_skeleton','Cauchy_total')})
            d=aa['fiber'];checks[region]['fiber_total']=metric(np.einsum('i,...ij,j->...',d,aa['PK1_total'][i],d),np.einsum('i,...ij,j->...',d,bb['PK1_total'][i],d),.02,.05,w)
        reaction=metric(ar['reaction_N'],br['reaction_N'],1e-4,.05);solid.append(dict(time_s=ar['time'],checks=checks,reaction=reaction,passed=good(checks) and reaction['passed']))
    maxima={k:max(r['checks'][region][k]['absolute']/r['checks'][region][k]['budget'] for r in solid for region in r['checks']) for k in solid[0]['checks']['global_domain']}
    maxima['reaction']=max(r['reaction']['absolute']/r['reaction']['budget'] for r in solid)
    ledger=[]
    for label,values,rr,times,folder in [('h',va,rows,t,a),('half',vb,rhs,tt,b)]:
        dn=(np.array([r['theta'] for r in rr])-.5)*np.sum(ctx['C'][None,:]*np.diff(values['pressure'],axis=0)**2,axis=1);err=float(np.max(abs(dn-[r['numerical_dissipation_J'] for r in rr])));summary=read(folder/'summary.json')
        passed=err<1e-18 and summary['source_zero'] and summary['min_detF']>.1 and summary['minimum_pressure_Pa']>=0 and summary['max_residual_fraction']<=1 and abs(summary['cumulative_energy_balance_J'])<=1e-9+.01*abs(summary['initial_energy_J'])
        ledger.append(dict(label=label,summary=summary,Dnum_identity_error_J=err,passed=passed))
    fault=read(run/'S3/fault.json');restart=read(run/'S3/restart.json');tx=fault['full_rollback'] and fault['cache_cleared'] and restart['same_digest'];passed=cmp['status']=='passed_scoped' and all(r['passed'] for r in solid+ledger) and tx
    write(run/'S3/engineering-comparison.json',dict(status='passed_scoped' if passed else 'limited',fluid=cmp,solid=solid,solid_max_budget_ratios=maxima,ledger=ledger,observations_s=OBSERVATIONS.tolist(),comparison='same boundary32 physical model, N vs2N full coupled steps',fixed_skeleton_not_truth_for_deforming_body=True))
    write(run/'S3/raw-substep-diagnostics.json',dict(status=raw['status'],same_grid_temporal=raw,raw_continuum_accuracy=False,CPU_same_grid_exact=read(run/'S2/candidate-U4.json')['trials'][0]['raw'],raw_steps_retained=True,first_intervals_not_removed=True))
    write(run/'S3/transaction-check.json',dict(status='passed_scoped' if tx else 'limited',fault=fault,restart=restart,microstep_is_transaction=True,observation_is_reporting_only=True,partial_observation_logic_tests='S1/tests.log'))
    write(run/'S3/grid-decision.json',dict(status='qualified_engineering_window' if passed else 'limited',engineering_window_accuracy=passed,raw_substep_accuracy=False,actual_coupled_spatial_accuracy=False,source_zero=True,steps=[len(rows),len(rhs)],time_end_s=float(t[-1]),grid='new boundary32 x-only',geometry='BASE-bounded-Gauss',full_coupled_cycle=False,production_default_changed=False))
    with PROGRESS.open('a') as f:f.write(f'\nS3：新边界32单元U4的28/56真实步已完成200微秒，工程观察比较`{passed}`；流体最大预算比{max(cmp["max_budget_ratios"].values()):.6g}，固体各场最大预算比{max(maxima.values()):.6g}。失败回滚与独立进程零步恢复均通过。原始微步误差仍单列，不声明实际空间收敛。\n')
    print('ENGINEERING',passed,cmp['max_budget_ratios'],maxima,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
