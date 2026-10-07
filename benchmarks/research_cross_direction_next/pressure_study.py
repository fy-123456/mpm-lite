"""Full-tensor 2/4/8 spatial diagnostics with exact cumulative boundary ODE."""
from pathlib import Path
import argparse,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .physics import baseline_model
from benchmarks.research_observable_pressure_next.coupling_study import affine_check,source
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_cross_direction_next.rt0 import BoundedTopology,BoundedGeometry,BoundedGridCoupling
from engine.aniso_phase1.research_observable_pressure_next.fast_rt0 import FastGeometry

def history(folder):return GenerationStore(folder,read(folder/'identity.json')).history()
def audit(run):
    run=Path(run);verify(run);hs={n:history(APP/f'cases/pressure-{n}') for n in (2,4)};records=[]
    for i in range(1,5):
        aa,bb=hs[2][i]['state'],hs[4][i]['state']
        if abs(aa.time-bb.time)>1e-14:raise ValueError('common physical endpoint mismatch')
        a,b=aa.child_states['fluid'],bb.child_states['fluid'];mass={}
        for n,f in ((2,a),(4,b)):
            initial=hs[n][0]['state'].child_states['fluid'];mass[n]=float(np.sum(np.array(f['content_m3'])-initial['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
            if abs(mass[n])>1e-10:raise ValueError('parent mass balance not reconstructed')
        records.append(dict(time=aa.time,mass_defect_m3=mass,pressure=metric(a['pressure_Pa'],np.array(b['pressure_Pa']).reshape(2,2).mean(axis=1),.001,.05),content=metric(sum(a['content_m3']),sum(b['content_m3']),1e-10,.05),boundary=metric(a['cumulative_boundary_m3'],b['cumulative_boundary_m3'],1e-10,.05)))
    old=read(APP/'S3/common-time-grid-review.json');expected=old['coupled']['fluid']['cumulative_boundary_m3']['absolute']
    if abs(records[-1]['boundary']['absolute']-expected)>1e-12:raise ValueError('parent boundary comparison differs')
    write(run/'S3/grid-error-audit.json',dict(status='grid_sensitive',records=records,old_review_sha256=sha(APP/'S3/common-time-grid-review.json'),old_fixed=old['fixed_common_time'],do_not_erase_old_failure=True))
    print('PRESSURE_PARENT',records[-1]['boundary'],flush=True)

def fixed(run):
    run=Path(run);verify(run);start=time.perf_counter();m,cfg=baseline_model(run,pressure=True);models={};checks=[];equiv={}
    for cells in (2,4,8):
        top=BoundedTopology([[e[0],e[-1]] for e in m.parent.edges],cells)
        if not np.all(top.B[:,top.internal].sum(axis=0)==0):raise ValueError('internal orientation incorrect')
        for F in (np.eye(3),np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]])):checks.append(affine_check(top,F))
        geo=BoundedGeometry(m,cells);v=geo.evaluate(m.rest().q)
        if cells in (2,4):
            old=FastGeometry(m,cells).evaluate(m.rest().q);err=max(float(np.max(abs(old[k]-v[k]))) for k in ('H','volume','gradient'))
            if err>1e-10:raise ValueError('2/4 geometry changed')
            equiv[cells]=err
        H=v['H'];B=geo.B;C=.0002*geo.V0;L=B@la.solve(H,B.T,assume_a='pos');rhs=source(geo)+B@la.solve(H,top.boundary_term(.002),assume_a='pos')
        rates=la.eigvalsh(L/np.sqrt(C[:,None]*C[None,:]));gterm=B@la.solve(H,top.boundary_term(.002),assume_a='pos')
        if rates[0]<=0:raise ValueError('nonpositive relaxation')
        aug=np.zeros((cells+2,cells+2));aug[:cells,:cells]=-L/C[:,None];aug[:cells,-1]=rhs/C;aug[cells,:cells]=np.ones(cells)@L;aug[cells,-1]=-sum(gterm)
        models[cells]=dict(geo=geo,L=L,C=C,rhs=rhs,rates=rates,aug=aug,source=source(geo),gterm=gterm)
    maxrate=max(x['rates'][-1] for x in models.values());target=7.594308444221303e-6;T=min(target,32*.5/maxrate);N=min(32,max(1,int(np.ceil(T*maxrate/.5-1e-12))));times=np.linspace(0,T,N+1)
    register(run,'S3/fixed-time-protocol.json',dict(target_s=target,actual_end_s=T,steps=N,times_s=times.tolist(),shortened=bool(T<target),decision_before_output_comparison=True,max_h_lambda_max=.5,max_steps=32,storage=.0002,alpha=.8,initial_pressure=.01,boundary_pressure=.002,source='0.001/s times reference volume in physical left half',BASELINE=read(run/'baseline-space.json')))
    exact={};reports={}
    for c,x in models.items():
        y0=np.r_[np.full(c,.01),0.,1.];Y=np.array([la.expm(x['aug']*t)@y0 for t in times]);exact[c]=Y;p=y0[:c].copy();b=0.;errors=[]
        for i,h in enumerate(np.diff(times),1):
            new=la.solve(np.diag(x['C'])+.5*h*x['L'],(np.diag(x['C'])-.5*h*x['L'])@p+h*x['rhs'],assume_a='pos');b+=h*(sum(x['L']@(.5*(p+new)))-sum(x['gterm']));p=new
            errors.append(dict(time=times[i],pressure_error_Pa=float(np.max(abs(p-Y[i,:c]))),boundary_volume=metric(b,Y[i,c],1e-10,.05)))
        mass=float(sum(x['C']*(Y[-1,:c]-.01))+Y[-1,c]-sum(x['source'])*T)
        reports[c]=dict(exact_pressure=Y[:,:c].tolist(),exact_boundary=Y[:,c].tolist(),content=(Y[:,:c]@x['C']).tolist(),time_errors=errors,mass_defect=mass,rates=x['rates'].tolist(),T_over_tau=(T*x['rates']).tolist(),source_m3_s=x['source'].tolist(),V0=x['geo'].V0.tolist(),time_resolved=max(e['pressure_error_Pa'] for e in errors)<=.001 and all(e['boundary_volume']['passed'] for e in errors))
        if abs(mass)>1e-10:raise ValueError('augmented exact mass balance failed')
    comparisons={}
    for a,b in ((2,4),(4,8)):
        comparisons[f'{a}-{b}']=[dict(time=t,pressure=metric(exact[a][i,:a],exact[b][i,:b].reshape(a,2).mean(axis=1),.001,.05),content=metric(reports[a]['content'][i],reports[b]['content'][i],1e-10,.05),boundary_volume=metric(exact[a][i,a],exact[b][i,b],1e-10,.05)) for i,t in enumerate(times[1:],1)]
    good=all(v['passed'] for row in comparisons['4-8'] for k,v in row.items() if k!='time') and all(reports[c]['time_resolved'] for c in (4,8))
    write(run/'S3/topology-operator-check.json',dict(status='passed_scoped',manufactured=checks,geometry_2_4_equivalence=equiv,eight_cells=8,eight_faces=41,volume=float(sum(models[8]['geo'].V0)),all_internal_faces_opposite=True))
    write(run/'S3/fixed-grid-comparison.json',dict(status='passed_scoped' if good else 'grid_sensitive',records=reports,comparisons=comparisons,eligible_coupled=good,matrix_exponential_includes_boundary=True,seconds=time.perf_counter()-start))
    write(run/'S3/pressure-scope-decision.json',dict(status='fixed_scoped_pending_coupled' if good else 'limited_research_scope',four_eight_fixed=good,four_eight_coupled=False,pressure_spatial_accuracy=False,production_C_E_integration=False,coupled_q5=False,pure_solid_default=True,solid='BASELINE',actual_end_s=T,old_2_4_failure_retained=True))
    if not good:
        for p in ('coupled-grid-check','transaction-check'):write(run/f'S3/{p}.json',dict(status='not_triggered',reason='complete fixed-grid gate failed'))
    print('PRESSURE_FIXED_4_8',good,'steps',N,'T',T,'end',comparisons['4-8'][-1],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','fixed']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
