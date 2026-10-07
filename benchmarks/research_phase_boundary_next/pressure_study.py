"""Early boundary flow with exact time references on registered explicit grids."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .physics import baseline_model
from benchmarks.research_phase_reference_next.coupling_study import quadrature
from benchmarks.research_observable_pressure_next.coupling_study import affine_check
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_cross_direction_next.rt0 import BoundedTopology,BoundedGeometry
from engine.aniso_phase1.research_phase_boundary_next.rt0 import CartesianTopology,CartesianGeometry
T=7.594308444221303e-6

def algebra(top,H=None):
    if H is None:
        X,w,ids=quadrature(top);H,_=top.assemble(X,w,ids,np.broadcast_to(np.eye(3),(len(X),3,3)))
    B=top.B;C=.0002*top.V0;Z=la.solve(H,B.T,assume_a='pos');z0=-la.solve(H,top.boundary_term(.002),assume_a='pos');L=B@Z
    src=top.source() if hasattr(top,'source') else .001*top.V0*(np.arange(top.cells)<top.cells//2)
    rhs=src-B@z0;rates=la.eigvalsh(L/np.sqrt(C[:,None]*C[None,:]));D=np.array([top.boundary_sign*(top.axes==a) for a in range(3)])
    c=top.cells;aug=np.zeros((c+4,c+4));aug[:c,:c]=-L/C[:,None];aug[:c,-1]=rhs/C;aug[c:c+3,:c]=D@Z;aug[c:c+3,-1]=D@z0
    return dict(top=top,H=H,B=B,C=C,Z=Z,z0=z0,L=L,source=src,rhs=rhs,rates=rates,D=D,aug=aug)

def exact(m,times):
    c=m['top'].cells;y0=np.r_[np.full(c,.01),np.zeros(3),1.];Y=np.array([la.expm(m['aug']*t)@y0 for t in times]);p=Y[:,:c];Q=Y[:,c:c+3]
    defect=(p-.01)@m['C']+Q.sum(axis=1)-np.asarray(times)*m['source'].sum()
    if np.max(abs(defect))>1e-10:raise ValueError('exact augmented mass imbalance')
    return dict(p=p,Q=Q,content=p@m['C'],mass=defect)

def restrict(values,coarse,fine):
    out=[]
    for b in coarse.cell_bounds:
        volumes=np.array([np.prod(np.maximum(0,np.minimum(b[:,1],c[:,1])-np.maximum(b[:,0],c[:,0]))) for c in fine.cell_bounds])
        if abs(volumes.sum()-np.prod(np.diff(b,axis=1)))>1e-12:raise ValueError('comparison cells do not cover same physical volume')
        out.append(values@volumes/volumes.sum())
    return np.stack(out,axis=-1)

def audit(run):
    run=Path(run);m,cfg=baseline_model(run,pressure=True);bounds=np.array([[e[0],e[-1]] for e in m.parent.edges]);times=read(APP/'S3/fixed-time-protocol.json')['times_s'];old=read(APP/'S3/fixed-grid-comparison.json');models={n:algebra(BoundedTopology(bounds,n)) for n in (4,8)};ex={n:exact(v,times) for n,v in models.items()}
    errors={n:float(np.max(abs(ex[n]['Q'].sum(axis=1)-old['records'][str(n)]['exact_boundary']))) for n in (4,8)}
    if max(errors.values())>1e-12:raise ValueError('old fixed RT0 report does not reproduce from physical rest geometry')
    signed=ex[8]['Q'][1:6]-ex[4]['Q'][1:6];axis=int(np.argmax(np.sum(abs(signed),axis=0)))
    actual=BoundedGeometry(m,4);v=actual.evaluate(m.rest().q);Herror=float(np.max(abs(v['H']-models[4]['H'])));Verror=float(np.max(abs(v['volume']-actual.V0)))
    if Herror>1e-8 or Verror>1e-11:raise ValueError('current formal rest geometry differs from exact constant-F fixture')
    write(run/'S3/current-solid-fixed-check.json',dict(status='passed_scoped',solid=read(run/'selected-space.json'),H_max=Herror,volume_max=Verror,scope='rest only; deformed geometry separately checked'))
    write(run/'S3/early-flow-diagnostic.json',dict(status='passed_scoped',reproduction_max=errors,first_five_axis_flow_difference_m3=signed.tolist(),axis_score=np.sum(abs(signed),axis=0).tolist(),selected_axis=axis,boundary_flow_8=ex[8]['Q'].tolist(),old_failure_retained=True))
    if axis==0:
        fractions=np.array([0,.02,.07,.2,.5,.8,.93,.98,1.]);base=[np.array(b) for b in bounds];base[0]=bounds[0,0]+fractions*np.ptp(bounds[0]);fine=[v.copy() for v in base];fine[0]=np.sort(np.r_[base[0],.5*(base[0][:-1]+base[0][1:])])
    else:
        base=[np.array(b) for b in bounds];base[0]=np.linspace(*bounds[0],3);base[axis]=bounds[axis,0]+np.array([0,.1,.5,.9,1.])*np.ptp(bounds[axis]);fine=[v.copy() for v in base];fine[axis]=np.sort(np.r_[base[axis],.5*(base[axis][:-1]+base[axis][1:])])
    register(run,'S3/grid-protocol.json',dict(cuts={'coarse':[v.tolist() for v in base],'fine':[v.tolist() for v in fine]},selected_axis=axis,max_cells=16,choice_before_new_results=True,source='physical left-half overlap volume times .001/s',target_end_s=T,comparison_times=times,only_one_refinement_direction=True))
    print('PRESSURE_AXIS',axis,'reproduction',errors,'geometry',Herror,flush=True)

def fixed(run):
    run=Path(run);settings=read(run/'S3/grid-protocol.json');models={};checks=[]
    for name,cuts in settings['cuts'].items():
        top=CartesianTopology(cuts)
        for F in (np.eye(3),np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]])):checks.append(affine_check(top,F))
        models[name]=algebra(top)
    rate=max(x['rates'][-1] for x in models.values());geometric=[0.];dt=.5/rate
    while geometric[-1]+dt<T:
        geometric.append(geometric[-1]+dt);dt*=1.7
    times=np.unique(np.r_[settings['comparison_times'],geometric,T]);within=len(times)-1<=32
    register(run,'S3/fixed-time-protocol.json',dict(times_s=times.tolist(),steps=len(times)-1,target_s=T,actual_end_s=T,first_h_lambda=.5,growth=1.7,includes_all_parent_nodes=True,within_step_budget=within,max_h_lambda=float(max(np.diff(times))*rate),storage=.0002,alpha=.8,initial_pressure=.01,boundary_pressure=.002,parameters_unchanged=True))
    exacts={n:exact(x,times) for n,x in models.items()};reports={};timegood=within
    for name,m in models.items():
        c=m['top'].cells;e=exacts[name];rows=[];p=np.full(c,.01);Q=np.zeros(3);mass=[]
        if within:
            for i,h in enumerate(np.diff(times),1):
                old=p.copy();p=la.solve(np.diag(m['C'])+.5*h*m['L'],(np.diag(m['C'])-.5*h*m['L'])@p+h*m['rhs'],assume_a='pos');z=m['Z']@(.5*(old+p))+m['z0'];Q+=h*m['D']@z
                pe=metric(p,e['p'][i],.001,.05);qe=metric(Q.sum(),e['Q'][i].sum(),1e-10,.05);defect=float((p-.01)@m['C']+sum(Q)-times[i]*sum(m['source']));D=h*float(z@m['H']@z)
                good=pe['absolute']<=.25*pe['budget'] and qe['absolute']<=.25*qe['budget'] and abs(defect)<=1e-10 and D>=0;timegood &= good
                rows.append(dict(time_s=times[i],pressure=pe,boundary=qe,mass_defect_m3=defect,dissipation_J=D,passed=bool(good)))
        reports[name]=dict(cells=c,faces=m['top'].nflux,V0=m['top'].V0.tolist(),source_m3_s=m['source'].tolist(),rates=m['rates'].tolist(),exact_pressure=e['p'].tolist(),exact_boundary_by_axis=e['Q'].tolist(),exact_content=e['content'].tolist(),exact_mass_defect=e['mass'].tolist(),time_rows=rows)
    coarse=models['coarse']['top'];fine=models['fine']['top'];finep=restrict(exacts['fine']['p'],coarse,fine);comparisons=[]
    for i,t in enumerate(times[1:],1):
        a,b=exacts['coarse'],exacts['fine'];comparisons.append(dict(time_s=t,pressure=metric(a['p'][i],finep[i],.001,.05,coarse.V0),content=metric(a['content'][i],b['content'][i],1e-10,.05),boundary=metric(a['Q'][i].sum(),b['Q'][i].sum(),1e-10,.05)))
    spacegood=all(v['passed'] for row in comparisons for key,v in row.items() if key!='time_s');eligible=bool(spacegood and timegood)
    write(run/'S3/topology-operator-check.json',dict(status='passed_scoped',manufactured=checks,shared_face_cancellation=True,physical_source_overlap=True,source_volume=float(models['fine']['source'].sum()/.001)))
    write(run/'S3/exact-time-reference.json',dict(status='passed_scoped',times_s=times.tolist(),records=reports,augmented_boundary_integrals=True))
    write(run/'S3/grid-comparison.json',dict(status='passed_scoped' if spacegood else 'grid_sensitive',records=comparisons,whole_window=True,only_one_axis_refined=True))
    write(run/'S3/fixed-grid-comparison.json',dict(status='passed_scoped' if eligible else 'limited',eligible_coupled=eligible,space_passed=spacegood,time_passed=bool(timegood),within_step_budget=within,records=reports,comparisons=comparisons))
    write(run/'S3/pressure-scope-decision.json',dict(status='fixed_scoped_pending_coupled' if eligible else 'limited_research_scope',eligible_coupled=eligible,fixed_grid_passed=eligible,actual_coupled=False,pressure_spatial_accuracy=False,production_C_E_integration=False,coupled_q5=False,pure_solid_default=True,solid=read(run/'selected-space.json')['package']['sha256'],end_s=T,old_failure_retained=True))
    if not eligible:
        for name in ('common-model-lock','volume-gradient-check','mixed-operator-check','coupled-scene-check','transaction-and-restart'):
            write(run/f'S4/{name}.json',dict(status='not_triggered',reason='fixed-pressure complete time/space gate failed',space_passed=spacegood,time_passed=bool(timegood)))
    print('PRESSURE_GRID',eligible,spacegood,timegood,'steps',len(times)-1,'last',comparisons[-1],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','fixed']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
