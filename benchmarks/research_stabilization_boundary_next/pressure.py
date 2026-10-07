"""Full-tensor CPU pressure references and bounded pre-registered mesh selection."""
from pathlib import Path
import argparse,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from benchmarks.research_phase_reference_next.coupling_study import quadrature
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction
from benchmarks.research_sequential_next.compare import metric

def bisect(cuts):
    return [np.sort(np.r_[cuts[0],.5*(np.asarray(cuts[0][:-1])+cuts[0][1:])]).tolist(),*cuts[1:]]

def graded(cuts,ratio):
    lo,hi=cuts[0][0],cuts[0][-1];weights=ratio**np.arange(8,dtype=float);w=(hi-lo)*.5*weights/weights.sum();left=lo+np.r_[0.,np.cumsum(w)];x=np.r_[left,hi-(left[:-1]-lo)[::-1]];x[0]=lo;x[-1]=hi
    return [x.tolist(),*cuts[1:]]

def algebra(cuts,params):
    top=ReferenceTopology(cuts);X,w,ids=quadrature(top);H,J=top.assemble(X,w,ids,np.broadcast_to(np.eye(3),(len(X),3,3)),params['mobility_scale']*MOBILITY);factor=la.cho_factor(H)
    Z=la.cho_solve(factor,top.B.T);z0=-la.cho_solve(factor,top.boundary_term(params['reservoir_Pa']));C=params['storage']*top.V0;L=top.B@Z;rhs=top.source(params['source_density_s_inv'])-top.B@z0
    D=np.sqrt(C);A=L/D[:,None]/D[None,:];lam,V=la.eigh(.5*(A+A.T));pe=la.solve(A,rhs/D,assume_a='pos')/D
    if lam[0]<=0:raise ValueError('fixed pressure generator is not positive')
    groups=np.array([top.boundary_sign*(top.axes==axis)*(top.boundary_sign==sign) for axis in range(3) for sign in (-1,1)])
    return dict(top=top,H=H,Z=Z,z0=z0,C=C,L=L,rhs=rhs,A=A,D=D,lam=lam,V=V,pe=pe,groups=groups)

def exact(a,params,times):
    t=np.array(times);c=a['V'].T@(a['D']*(params['pressure0_Pa']-a['pe']));decay=np.exp(-t[:,None]*a['lam']);p=a['pe']+(decay*c)@a['V'].T/a['D'];integ=t[:,None]*a['pe']+(-np.expm1(-t[:,None]*a['lam'])/a['lam']*c)@a['V'].T/a['D'];Q=integ@a['Z'].T+t[:,None]*a['z0']
    return dict(pressure=p,cumulative=Q,flux=np.diff(Q,axis=0)/np.diff(t)[:,None])

def midpoint(a,params,times):
    p=np.full(a['top'].cells,params['pressure0_Pa']);Q=np.zeros(a['top'].nflux);ps=[p.copy()];qs=[Q.copy()];flux=[]
    for h in np.diff(times):
        b=(np.eye(len(p))-.5*h*a['A'])@(a['D']*p)+h*a['rhs']/a['D'];new=la.solve(np.eye(len(p))+.5*h*a['A'],b,assume_a='pos')/a['D'];z=a['Z']@(.5*(p+new))+a['z0'];Q+=h*z;p=new;ps.append(p.copy());qs.append(Q.copy());flux.append(z)
    return dict(pressure=np.array(ps),cumulative=np.array(qs),flux=np.array(flux))

def comparison(a,b,va,vb,times,fraction=1.):
    P,C,Z=restriction(a['top'],b['top']);rows=[]
    for i,t in enumerate(times[1:],1):
        checks={'pressure':metric(va['pressure'][i],P@vb['pressure'][i],.001,.05,a['top'].V0),
            'content':metric(a['C']*va['pressure'][i],C@(b['C']*vb['pressure'][i]),1e-10,.05),
            'face_flux':metric(va['flux'][i-1],Z@vb['flux'][i-1],1e-10,.05),
            'boundary_by_side':metric(a['groups']@va['flux'][i-1],b['groups']@vb['flux'][i-1],1e-10,.05),
            'boundary_cumulative_by_side':metric(a['groups']@va['cumulative'][i],b['groups']@vb['cumulative'][i],1e-10,.05),
            'boundary_total':metric(a['top'].boundary_sign@va['cumulative'][i],b['top'].boundary_sign@vb['cumulative'][i],1e-10,.05)}
        ratios={k:v['absolute']/(fraction*v['budget']) for k,v in checks.items()};rows.append(dict(time_s=float(t),checks=checks,budget_ratios=ratios,passed=max(ratios.values())<=1))
    return dict(status='passed_scoped' if all(r['passed'] for r in rows) else 'limited',records=rows,max_budget_ratios={k:max(r['budget_ratios'][k] for r in rows) for k in rows[0]['budget_ratios']},reference_fraction=fraction)

def screen(run):
    run=Path(run);verify(run);started=time.perf_counter();protocol=read(OBS/'S3/new-scene-protocol.json');p=protocol['parameters'];basecuts=protocol['cuts']['coarse'];designs={str(r):graded(basecuts,r) for r in (4,5)}
    register(run,'S2/grid-design.json',dict(status='registered',ratios=[4,5],coarse_cuts=designs,nested_actual_cells=[16,32],CPU_reference_cells=[64,128],times={'uniform':np.linspace(0,2e-4,17).tolist(),'cubic':(2e-4*(np.arange(17)/16)**3).tolist()},parameters=p,new_coupled_attempts_max=100,original_gpu_constructor_unchanged=True,hard_seconds=600))
    # Audit new CPU topology/condensation against original <=32 implementation.
    from benchmarks.research_candidate_observable_next.coupling import algebra as original
    reproduction=[]
    for key in ('coarse','fine'):
        a=algebra(protocol['cuts'][key],p);b=original(protocol['cuts'][key],p);errors={k:float(np.max(abs(a[k]-b[k]))) for k in ('H','Z','z0','C','L','rhs')}
        if not all(np.allclose(a[k],b[k],rtol=2e-10,atol=1e-12) for k in errors):raise ValueError('CPU reference does not reproduce original')
        ts=np.array([0.,6.25e-6,2e-4]);e=exact(a,p,ts);n=a['top'].cells;aug=np.zeros((n+1,n+1));aug[:n,:n]=-a['L']/a['C'][:,None];aug[:n,-1]=a['rhs']/a['C'];ex=np.array([la.expm(t*aug)@np.r_[np.full(n,p['pressure0_Pa']),1.] for t in ts]);err=float(np.max(abs(e['pressure']-ex[:,:n])))
        if err>1e-8:raise ValueError('spectral matrix exponential differs')
        reproduction.append(dict(grid=key,errors=errors,matrix_exponential_error=err))
    write(run/'S2/topology-equivalence.json',dict(status='passed_scoped',records=reproduction,exact_algorithm='symmetric capacity scaling and spectral evaluation of the same matrix exponential; expm1 integral for every face',old_32_cell_guard_preserved=True))
    choices=[];selected=None
    for ratio,cuts in designs.items():
        models={16:algebra(cuts,p)}
        for n in (32,64,128):cuts=bisect(cuts);models[n]=algebra(cuts,p)
        records=[]
        for schedule in ('uniform','cubic'):
            coarse=np.asarray(read(run/'S2/grid-design.json')['times'][schedule]);fine=np.sort(np.r_[coarse,.5*(coarse[:-1]+coarse[1:])]);ex={n:exact(a,p,fine) for n,a in models.items()};ref=comparison(models[64],models[128],ex[64],ex[128],fine,.25);space=comparison(models[32],models[128],ex[32],ex[128],fine);timechecks={}
            for n in (16,32):
                for label,ts in [('h',coarse),('half',fine)]:
                    actual=midpoint(models[n],p,ts);cmp=comparison(models[n],models[n],actual,exact(models[n],p,ts),ts,.25);cmp['minimum_pressure_Pa']=float(actual['pressure'].min());cmp['nonnegative_pressure']=cmp['minimum_pressure_Pa']>=-1e-10;timechecks[str(n)+'-'+label]=cmp
            eligible=ref['status']==space['status']=='passed_scoped' and all(c['status']=='passed_scoped' and c['nonnegative_pressure'] for c in timechecks.values())
            item=dict(ratio=int(ratio),schedule=schedule,reference=ref,space=space,time=timechecks,eligible=eligible);records.append(item)
            print('PRESSURE_SCREEN',ratio,schedule,'eligible',eligible,'reference',ref['max_budget_ratios'],'space',space['max_budget_ratios'],'time',{k:max(v['max_budget_ratios'].values()) for k,v in timechecks.items()},flush=True)
            if selected is None and eligible:selected=dict(ratio=int(ratio),schedule=schedule,cuts={'coarse':[v.tolist() for v in models[16]['top'].cuts],'fine':[v.tolist() for v in models[32]['top'].cuts]},times_s=coarse.tolist(),parameters=p)
            if time.perf_counter()-started>600:raise TimeoutError('pressure screening time budget')
        choices.extend(records)
    write(run/'S2/fixed-skeleton-reference.json',dict(status='passed_scoped' if selected else 'limited',records=choices,selected=selected,seconds=time.perf_counter()-started,new_coupled_steps=0,exact_references_are_semidiscrete_not_continuum=True))
    # Preserve inherited actual directional flow, including transverse faces.
    from benchmarks.research_phase_stress_next.time_study import history
    attribution=[]
    for folder,cuts in [(OBS/'cases/observable-coarse-half',protocol['cuts']['coarse']),(APP/'cases/grid32-half',protocol['cuts']['fine'])]:
        a=algebra(cuts,p);h=history(folder);rows=h[-1]['rows'];groups=[]
        for row in rows:groups.append((a['groups']@np.array(row['flux_interval_m3_s'])).tolist())
        cumulative=np.sum(np.array(groups)*np.array([r['dt'] for r in rows])[:,None],axis=0);declared=h[-1]['state'].child_states['fluid']['cumulative_boundary_m3']
        if abs(cumulative.sum()-declared)>1e-10:raise ValueError('boundary attribution does not close')
        attribution.append(dict(source=str(folder),side_order=['-x','+x','-y','+y','-z','+z'],first_interval_m3_s=groups[0],cumulative_by_side_m3=cumulative.tolist(),total_m3=declared))
    write(run/'S2/boundary-attribution.json',dict(status='passed_scoped',records=attribution,inherited_pressure_jump_diagnostic=dict(path=str(APP/'S3/grid-limitation-diagnostic.json'),sha256=sha(APP/'S3/grid-limitation-diagnostic.json'))))
    if selected:write(run/'S2/selected-protocol.json',dict(status='registered',**selected,observations_s=[selected['times_s'][i] for i in [0,4,8,12,16]],max_attempts=100))
    else:write(run/'S2/coupling-decision.json',dict(status='not_triggered',reason='neither pre-registered grid/time pair meets separated reference, grid and time gates',new_dynamic_steps=0,old_actual_grid_accuracy=False,new_actual_grid_accuracy=False,production_C_E_integration=False,coupled_q5=False))
    print('PRESSURE_DECISION',selected is not None,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):screen(a.run)
