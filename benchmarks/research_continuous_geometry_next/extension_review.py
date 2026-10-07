"""Engineering comparison of the new epoch, retaining old cumulative accounts."""
import argparse
import numpy as np
from .provenance import *
from .runtime import update
from .review import balances
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_startup_next.coupling_review import fluid,context,compare_fluid
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_startup_substeps_next.schedule import aggregate

def main(run):
    run=Path(run);mutable(run);a=run/'cases/extension-h';b=run/'cases/extension-half';va,t,rows=fluid(a);vb,tt,rhs=fluid(b);p=read(run/'S0/coupled-protocol.json');ctx=context(p['cuts'],p['parameters']['storage']);aa=aggregate(va,t-t[0],t-t[0]);bb=aggregate(vb,tt-tt[0],t-t[0]);flow=compare_fluid(ctx,ctx,aa,bb,t)
    with np.load(a/'probes.npz') as x,np.load(b/'probes.npz') as y:pa={k:x[k] for k in x.files};pb={k:y[k] for k in y.files}
    if not np.allclose(pa['times'],pb['times'],rtol=0,atol=1e-18):raise ValueError('extension physical observation times differ')
    ra=read(a/'engineering-ledger.json');rb=read(b/'engineering-ledger.json');solid=[]
    for i,(ar,br) in enumerate(zip(ra,rb),1):
        fa=dict(X=pa['X'],**{k:pa[k][i] for k in ('x','velocity','PK1')});fb=dict(X=pb['X'],**{k:pb[k][i] for k in ('x','velocity','PK1')});checks=fields(fa,fb,pa['fiber'])
        for region,w in regions(pa['X']).items():
            checks[region].update({k:metric(pa[k][i],pb[k][i],.02,.05,w) for k in ('PK1_total','Cauchy_skeleton','Cauchy_total')});d=pa['fiber'];checks[region]['fiber_total']=metric(np.einsum('i,...ij,j->...',d,pa['PK1_total'][i],d),np.einsum('i,...ij,j->...',d,pb['PK1_total'][i],d),.02,.05,w)
        reaction=metric(ar['reaction_N'],br['reaction_N'],1e-4,.05);solid.append(dict(time_s=ar['time'],checks=checks,reaction=reaction,passed=good(checks) and reaction['passed']))
    hh,fh=history(a),history(b);bal=[balances(hh),balances(fh)];source_equal=np.array_equal(hh[0]['state'].q,fh[0]['state'].q) and hh[0]['state'].child_states['fluid']['cumulative_numerical_dissipation_J']==fh[0]['state'].child_states['fluid']['cumulative_numerical_dissipation_J'];tx=read(run/'S4/transaction-check.json');restart=read(a/'restart-36.json');passed=source_equal and flow['status']=='passed_scoped' and all(x['passed'] for x in solid+bal) and tx['rollback_exact'] and restart['same_digest'];out=dict(status='passed_scoped' if passed else 'limited',fluid=flow,solid=solid,balances=bal,source_equal=source_equal,residual_normalization_window_s=2e-4,time_range_s=[2e-4,3e-4],new_steps=[8,16],new_Dnum_J=[sum(r['numerical_dissipation_J'] for r in rr) for rr in (rows,rhs)],old_Dnum_retained=True,full_coupled_cycle=False,actual_coupled_spatial_accuracy=False)
    write(run/'S4/extension-comparison.json',out);write(run/'S4/extension-decision.json',dict(status='extended_window_qualified' if passed else 'limited',backend=read(run/'S4/extension-protocol.json')['backend'],scope='same BASE coarse 200us initial state, 200..300us engineering window only',original_residual_budget_window_s=2e-4,global_0_to_300_time_accuracy=False,actual_spatial_accuracy=False));update(run,f'S4：同一200微秒初态追加8/16步至300微秒，结果 `{out["status"]}`；流体最坏预算比{max(flow["max_budget_ratios"].values()):.4g}。新段Dnum为零，早期累计Dnum保留；不宣称0到300微秒全程时间/空间收敛。');print('EXTENSION_REVIEW',out['status'],flow['max_budget_ratios'],bal,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
