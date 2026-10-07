"""Bounded independent scalar reference and full-tensor candidate screening."""
import argparse,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from engine.aniso_phase1.research_boundary_reference_next.reference import slab_reference,boundary_grid
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from benchmarks.research_phase_reference_next.coupling_study import quadrature
from benchmarks.research_stabilization_boundary_next.pressure import algebra,exact,comparison,bisect
from engine.aniso_phase1.research_pressure_startup_next.theta import integrate
from benchmarks.research_sequential_next.compare import metric


def scalar_model(cuts,params):
    top=ReferenceTopology(cuts);X,w,ids=quadrature(top);K=params['mobility_scale']*np.diag(np.diag(MOBILITY));H,_=top.assemble(X,w,ids,np.broadcast_to(np.eye(3),(len(X),3,3)),K)
    active=np.flatnonzero(top.axes==0);Hx=H[np.ix_(active,active)];B=top.B[:,active];factor=la.cho_factor(Hx);Z=la.cho_solve(factor,B.T);z0=-la.cho_solve(factor,top.boundary_term(params['reservoir_Pa'])[active]);C=params['storage']*top.V0;L=B@Z;rhs=-B@z0;D=np.sqrt(C);A=L/D[:,None]/D[None,:];lam,V=la.eigh(.5*(A+A.T));pe=la.solve(A,rhs/D,assume_a='pos')/D
    return dict(top=top,H=Hx,Z=Z,z0=z0,C=C,L=L,rhs=rhs,A=A,D=D,lam=lam,V=V,pe=pe,active=active)


def prepare():
    run=freeze();p=read(APP/'S0/input-contract.json');register(run,'R0/input-contract.json',dict(parameters=p['parameters'],cuts=p['cuts'],times_s=p['times_s'],source_m3_s=0.,parent_protocol_sha256=sha(APP/'S0/input-contract.json')))
    register(run,'R1/reference-protocol.json',dict(status='registered',scalar_first_m=4e-6,scalar_cells=[32,128],early_time_window_s=[0.,2e-4],cell_average_pressure=True,integral_interval_flux=True,source_zero=True,scalar_is_not_full_tensor_truth=True))
    register(run,'R2/grid-screen-protocol.json',dict(status='registered',mother_first_widths_m=[4e-6,8e-6],growth=2.2,cells=[32,64,128],method='startup',switch_s=2.5e-5,time_tables=['uniform16','cubic_start16'],reference_fraction=.25,field_rtol=.05,max_CPU_matrices=12,max_spectral_calls=16,max_matrix_steps=512,max_seconds=600,all_first_intervals_included=True))
    PROGRESS.write_text(f'# 边界排水参考与共享几何进展\n\n执行[{PLAN.name}]({PLAN.name})。最新父发布 `{APP.name}`，SHA `{APP_SHA}`；开工25源码/25快照/960产物及祖先链通过。旧文档完整封存，本轮承接未完成的边界参考与热点优化。系统盘约8GiB，无需迁移。\n\n结果：`{run.relative_to(ROOT)}`。R0已冻结。\n')
    print('RUN',run,flush=True)


def scalar(run):
    run=Path(run);mutable(run);begun=time.perf_counter();p=read(run/'R0/input-contract.json');params=p['parameters'];ts=np.linspace(0,2e-4,33);cuts=bisect(boundary_grid(p['cuts']['coarse'],4e-6));records=[]
    for n in (32,128):
        if n==128:cuts=bisect(bisect(cuts))
        a=scalar_model(cuts,params);v=exact(a,params,ts);area=np.prod(np.diff(a['top'].bounds[1:],axis=1));ref=slab_reference(cuts[0],ts,mobility=params['mobility_scale']*MOBILITY[0,0],storage=params['storage'],p0=params['pressure0_Pa'],reservoir=params['reservoir_Pa'],area=float(area));sign=a['top'].boundary_sign[a['active']];side=a['top'].centres[a['active'],0]<a['top'].bounds[0].mean();end=np.where(side,sign,0)
        Q=v['cumulative']@end;q=v['flux']@end;rows=[]
        for i in range(1,len(ts)):
            checks=dict(pressure=metric(v['pressure'][i],ref['pressure'][i],.001,.05,a['top'].V0),interval_flux=metric(q[i-1],ref['boundary_interval_per_end_m3_s'][i-1],1e-10,.05),cumulative=metric(Q[i],ref['boundary_cumulative_per_end_m3'][i],1e-10,.05))
            rows.append(dict(time_s=ts[i],checks=checks,passed=all(x['passed'] for x in checks.values())))
        conservation=np.max(abs((params['pressure0_Pa']-ref['pressure'])@ref['cell_capacity']-2*ref['boundary_cumulative_per_end_m3']))
        if conservation>1e-14:raise ValueError('analytic cell integral and drainage mismatch')
        np.savez_compressed(run/'R1'/f'scalar-{n}.npz',times=ts,pressure=v['pressure'],reference_pressure=ref['pressure'],flux=q,reference_flux=ref['boundary_interval_per_end_m3_s'],cumulative=Q,reference_cumulative=ref['boundary_cumulative_per_end_m3'],cuts=cuts[0])
        records.append(dict(cells=n,status='passed_scoped' if all(r['passed'] for r in rows) else 'limited',records=rows,analytic_mass_identity_error_m3=float(conservation),maximum_ratios={k:max(r['checks'][k]['absolute']/r['checks'][k]['budget'] for r in rows) for k in rows[0]['checks']}))
    write(run/'R1/scalar-reference.json',dict(status='passed_scoped' if records[-1]['status']=='passed_scoped' else 'limited',records=records,new_matrices=2,spectral_calls=2,matrix_steps=0,seconds=time.perf_counter()-begun,scope=ref['scope'],full_tensor_coupled_truth=False));print('SCALAR',[(r['cells'],r['maximum_ratios']) for r in records],flush=True)


def screen(run):
    run=Path(run);mutable(run);begun=time.perf_counter();p=read(run/'R0/input-contract.json');params=p['parameters'];protocol=read(run/'R2/grid-screen-protocol.json');records=[];matrices=0;calls=0;steps=0;eligible=[]
    tables=dict(uniform16=np.array(p['times_s']),cubic_start16=np.r_[2.5e-5*(np.arange(9)/8)**3,np.linspace(2.5e-5,2e-4,9)[1:]])
    for first in protocol['mother_first_widths_m']:
        cuts=bisect(boundary_grid(p['cuts']['coarse'],first,protocol['growth']));models={}
        for n in (32,64,128):models[n]=algebra(cuts,params);matrices+=1;cuts=bisect(cuts)
        for name,ts in tables.items():
            if time.perf_counter()-begun>600:raise TimeoutError('CPU screen time budget')
            fine=np.sort(np.r_[ts,.5*(ts[:-1]+ts[1:])]);ex={n:exact(a,params,fine) for n,a in models.items()};calls+=3
            ref=comparison(models[64],models[128],ex[64],ex[128],fine,.25);spatial=comparison(models[32],models[128],ex[32],ex[128],fine);checks=[]
            for label,t in [('h',ts),('half',fine)]:
                v=integrate(models[32],params,t,'startup');steps+=len(t)-1;ev=ex[32] if label=='half' else dict(pressure=ex[32]['pressure'][::2],cumulative=ex[32]['cumulative'][::2],flux=np.diff(ex[32]['cumulative'][::2],axis=0)/np.diff(ts)[:,None]);cmp=comparison(models[32],models[32],v,ev,t)
                checks.append(dict(label=label,comparison=cmp,minimum_pressure=float(v['pressure'].min()),max_mass_defect_m3=max(x['mass_defect_m3'] for x in v['ledger']),max_energy_balance_J=max(abs(x['energy_balance_J']) for x in v['ledger'])))
                np.savez_compressed(run/'R2'/f'grid-{first:g}-{name}-{label}.npz',times=t,pressure=v['pressure'],flux=v['flux'],cumulative=v['cumulative'],exact_pressure=ev['pressure'],exact_flux=ev['flux'],exact_cumulative=ev['cumulative'])
            ok=ref['status']==spatial['status']=='passed_scoped' and all(x['comparison']['status']=='passed_scoped' and x['minimum_pressure']>=0 for x in checks)
            row=dict(first_m=first,table=name,reference=ref,spatial=spatial,time=checks,eligible=ok);records.append(row)
            if ok:eligible.append(dict(cuts=[x.tolist() for x in models[32]['top'].cuts],times_s=ts.tolist(),parameters=params,method='startup',source_m3_s=0.,grid='boundary32',first_m=first,table=name))
            print('SCREEN',first,name,'reference',max(ref['max_budget_ratios'].values()),'space',max(spatial['max_budget_ratios'].values()),'time',[max(x['comparison']['max_budget_ratios'].values()) for x in checks],flush=True)
    write(run/'R2/grid-screen.json',dict(status='passed_scoped' if eligible else 'limited',records=records,new_matrices=matrices,spectral_calls=calls,matrix_steps=steps,seconds=time.perf_counter()-begun))
    if eligible:register(run,'R3/coupled-protocol.json',dict(status='registered',**eligible[0],max_attempts=49))
    else:write(run/'R3/coupling-decision.json',dict(status='not_triggered',reason='neither registered boundary grid/time pair jointly passes reference, spatial and raw first-interval temporal gates',new_attempts=0,retained_original16_scope=True,production=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','scalar','screen']);p.add_argument('--run',type=Path);a=p.parse_args()
    with serial_lock(a.run):prepare() if a.phase=='prepare' else globals()[a.phase](a.run)
