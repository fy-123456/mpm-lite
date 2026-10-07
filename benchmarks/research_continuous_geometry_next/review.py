"""Compare every saved continuation state and its engineering observation fields."""
import argparse
import numpy as np
from .provenance import *
from .continuous import state_checks
from .runtime import update
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_pressure_startup_next.coupling_review import context

def balances(h):
    a,b=h[0]['state'],h[-1]['state'];rows=h[-1]['rows'];f=a.child_states['fluid'];g=b.child_states['fluid']
    mass=float(np.sum(np.asarray(g['content_m3'])-f['content_m3'])+g['cumulative_boundary_m3']-f['cumulative_boundary_m3']-np.sum(np.asarray(g['cumulative_source_m3'])-f['cumulative_source_m3']))
    E0=f['last_ledger']['total_energy_J'];energy=rows[-1]['total_energy_J']-E0+sum(r['darcy_dissipation_J']+r['numerical_dissipation_J']-r['external_work_J']-r['source_work_J']-r['reservoir_work_J'] for r in rows)
    dn=g['cumulative_numerical_dissipation_J']-f['cumulative_numerical_dissipation_J']-sum(r['numerical_dissipation_J'] for r in rows)
    out=dict(mass_defect_m3=mass,incremental_energy_balance_J=energy,initial_energy_J=E0,Dnum_history_defect_J=dn,min_detF=min(r['min_detF'] for r in rows),max_residual_fraction=max(r['true_scaled_residual'] for r in rows),minimum_pressure_Pa=min(min(v['state'].child_states['fluid']['pressure_Pa']) for v in h),source_zero=all(r['source_work_J']==0 for r in rows) and not np.any(g['cumulative_source_m3']),nonnegative_dissipation=all(r['darcy_dissipation_J']>=-1e-18 and r['numerical_dissipation_J']>=-1e-18 for r in rows))
    out['passed']=abs(mass)<1e-10 and abs(energy)<1e-9+.01*abs(E0) and abs(dn)<1e-16 and out['min_detF']>.1 and out['max_residual_fraction']<=1 and out['minimum_pressure_Pa']>=0 and out['source_zero'] and out['nonnegative_dissipation'];return out

def review(run,kind='B'):
    run=Path(run);mutable(run);folder=run/'cases'/('continuous-'+kind);base=APP/'cases/boundary32-h' if kind=='B' else run/'cases/continuous-B';a=history(folder);ref={x['state'].step:x for x in history(base)};records=[];p=read(run/'S0/coupled-protocol.json');ctx=context(p['cuts'],p['parameters']['storage'])
    for x in a[1:]:
        y=ref[x['state'].step];checks=state_checks(x['state'],y['state']);row=x['rows'][-1];rr=y['rows'][-1]
        for k in ('reaction_N','total_energy_J','darcy_dissipation_J','numerical_dissipation_J','source_work_J','reservoir_work_J','external_work_J','pressure_solid_work_J','pressure_fluid_work_J','energy_balance_J'):
            if k in row:checks[k]=metric(row[k],rr[k],1e-8 if k=='reaction_N' else 1e-12,2e-5)
        checks['six_sides']=metric(ctx['groups']@np.asarray(row['flux_interval_m3_s']),ctx['groups']@np.asarray(rr['flux_interval_m3_s']),1e-10,2e-5)
        records.append(dict(step=x['state'].step,time_s=x['state'].time,checks=checks,passed=all(v['passed'] for v in checks.values())))
    def probes(path):
        files=sorted(path.glob('probes-*.npz')) or [path/'probes.npz'];out={}
        for file in files:
            with np.load(file) as z:
                for i,t in enumerate(z['times']):out[round(float(t),14)]={k:z[k][i] for k in ('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')}
        return out
    pa,pb=probes(folder),probes(base);fields=[]
    with np.load(next(folder.glob('probes-*.npz'))) as z:X=z['X'];fiber=z['fiber']
    for t,v in sorted(pa.items()):
        if t<=1.25e-5 or t not in pb:continue
        checks={k:metric(w-X if k=='x' else w,pb[t][k]-X if k=='x' else pb[t][k],1e-8,2e-5) for k,w in v.items()}
        regional={region:{k:metric(v[k],pb[t][k],1e-8,2e-5,w) for k in ('PK1','PK1_total','Cauchy_skeleton','Cauchy_total')} for region,w in regions(X).items()}
        for region,w in regions(X).items():regional[region]['fiber_total']=metric(np.einsum('i,...ij,j->...',fiber,v['PK1_total'],fiber),np.einsum('i,...ij,j->...',fiber,pb[t]['PK1_total'],fiber),1e-8,2e-5,w)
        fields.append(dict(time_s=t,checks=checks,regional=regional,passed=all(x['passed'] for x in checks.values()) and all(x['passed'] for rr in regional.values() for x in rr.values())))
    stage='S1' if kind=='B' else 'S2';bal=balances(a);tx=read(run/stage/'transaction-check.json');recovery=read(run/stage/'recovery-comparison.json');restart=read(folder/'restart-8.json');passed=all(x['passed'] for x in records+fields) and bal['passed'] and tx['rollback_exact'] and recovery['status']=='passed_scoped' and restart['same_digest']
    result=dict(status='passed_scoped' if passed else 'limited',backend=kind,comparison_source=str(base),states=records,fields=fields,balances=bal,restart=restart,transaction=tx,recovery=recovery,global_indices=[4,18],new_steps=14,full_coupled_cycle=False,spatial_accuracy=False)
    write(run/stage/'continuous-comparison.json',result);update(run,f'{stage} 连续后端{kind}：14个真实步、25微秒独立进程恢复、50微秒失败回滚后重算，比较结果 `{result["status"]}`；质量缺口{bal["mass_defect_m3"]:.3g}m³，增量能量收支{bal["incremental_energy_balance_J"]:.3g}J。');print('REVIEW',kind,result['status'],bal,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--kind',default='B');a=p.parse_args()
    with serial_lock(a.run):review(a.run,a.kind)
