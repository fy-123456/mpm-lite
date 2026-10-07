"""Fixed physical observation times, one nested boundary-graded pair, RT0 tensor flow."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .physics import baseline_model
from benchmarks.research_phase_boundary_next.pressure_study import algebra,exact,restrict,T
from benchmarks.research_observable_pressure_next.coupling_study import affine_check
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology,CartesianGeometry
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY

def audit(run):
    run=Path(run);verify(run);oldgrid=read(APP/'S3/grid-protocol.json');oldref=read(APP/'S3/exact-time-reference.json');times=np.array(read(APP/'S3/fixed-time-protocol.json')['times_s']);models={k:algebra(CartesianTopology(c)) for k,c in oldgrid['cuts'].items()};e={k:exact(m,times) for k,m in models.items()}
    errors={k:float(np.max(abs(e[k]['Q']-oldref['records'][k]['exact_boundary_by_axis']))) for k in models}
    if max(errors.values())>1e-12:raise ValueError('baseline 8/16 early flow does not reproduce')
    signed=e['fine']['Q'][1:4]-e['coarse']['Q'][1:4];axis=int(np.argmax(np.sum(abs(signed),axis=0)))
    if axis!=0:raise ValueError('registered x-boundary design no longer justified')
    write(run/'S2/observation-times.json',dict(status='passed_scoped',times_s=times.tolist(),parent_path=str(APP/'S3/fixed-time-protocol.json'),parent_sha256=sha(APP/'S3/fixed-time-protocol.json'),old_failed_times_s=read(APP/'S3/grid-error-interpretation.json')['failed_nodes'],fixed_before_new_grids=True))
    write(run/'S2/early-flow-diagnostic.json',dict(status='passed_scoped',reproduction_max=errors,first_three_signed_axis_difference_m3=signed.tolist(),axis_score=np.sum(abs(signed),axis=0).tolist(),selected_axis=axis,old_failures_retained=True))
    bounds=models['coarse']['top'].bounds;fractions=np.array([0,.001,.003,.007,.015,.03,.07,.2,.5,.8,.93,.97,.985,.993,.997,.999,1.]);base=[np.array(b) for b in bounds];base[0]=bounds[0,0]+fractions*np.ptp(bounds[0]);fine=[x.copy() for x in base];fine[0]=np.sort(np.r_[base[0],.5*(base[0][:-1]+base[0][1:])])
    internal=np.sort(np.r_[times,.5*(times[1:]+times[:-1])])
    register(run,'S2/grid-protocol.json',dict(status='passed_scoped',cuts=dict(coarse=[x.tolist() for x in base],fine=[x.tolist() for x in fine]),cells=[16,32],selection='one x-boundary graded nested pair; first three frozen failures dominated by x flux',boundary_layer_lengths_m={str(t):np.sqrt(np.diag(MOBILITY)/.0002*t).tolist() for t in times[1:4]},no_new_grid_search=True))
    register(run,'S2/fixed-time-protocol.json',dict(status='passed_scoped',times_s=times.tolist(),internal_times_s=internal.tolist(),steps=len(internal)-1,max_steps=64,method='exactly two midpoint substeps per fixed observation interval; no grid-rate-dependent observation shift',first_four_observation_prefix_steps=8,storage=.0002,alpha=.8,mobility=MOBILITY.tolist(),mobility_units='m^2/(Pa s)',initial_pressure_Pa=.01,boundary_pressure_Pa=.002,source_density_per_s=.001,source_region='physical left half',unchanged_parameters=True))
    print('PRESSURE_REGISTERED',len(times),len(internal)-1,'cells16/32',flush=True)

def fixed(run):
    run=Path(run);protocol=read(run/'S2/fixed-time-protocol.json');times=np.array(protocol['times_s']);internal=np.array(protocol['internal_times_s']);models={};checks=[];geocheck=[];m,cfg=baseline_model(run,pressure=True)
    for label,cuts in read(run/'S2/grid-protocol.json')['cuts'].items():
        top=CartesianTopology(cuts)
        for F in (np.eye(3),np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]])):checks.append(affine_check(top,F))
        models[label]=algebra(top);g=CartesianGeometry(m,cuts);actual=g.evaluate(m.rest().q);errH=float(np.max(abs(actual['H']-models[label]['H'])));errV=float(np.max(abs(actual['volume']-top.V0)))
        if errH>1e-7 or errV>1e-11:raise ValueError('actual formal rest H/V differs')
        geocheck.append(dict(grid=label,H_max=errH,V_max=errV,geometry=g.identity));del g
    exacts={k:exact(v,times) for k,v in models.items()};reports={};timegood=True;within=len(internal)-1<=64
    for name,m in models.items():
        c=m['top'].cells;e=exacts[name];p=np.full(c,.01);Q=np.zeros(3);rows=[];steps=[];Dsum=0.;Cp=np.diag(m['C']);observation={float(t):i for i,t in enumerate(times)}
        for a,b in zip(internal[:-1],internal[1:]):
            dt=b-a;old=p.copy();p=la.solve(Cp+.5*dt*m['L'],(Cp-.5*dt*m['L'])@p+dt*m['rhs'],assume_a='pos');z=m['Z']@(.5*(p+old))+m['z0'];Q+=dt*m['D']@z;D=dt*float(z@m['H']@z);Dsum+=D;defect=float((p-.01)@m['C']+Q.sum()-b*m['source'].sum())
            if not np.isfinite(p).all() or abs(defect)>1e-10 or D<0:raise ValueError('fixed time physical gate')
            steps.append(dict(time_s=b,dt_s=dt,mass_defect_m3=defect,dissipation_J=D,min_pressure_Pa=float(p.min()),max_pressure_Pa=float(p.max())))
            if float(b) in observation:
                i=observation[float(b)];metrics=dict(pressure=metric(p,e['p'][i],.001,.05,m['top'].V0),content=metric(p@m['C'],e['content'][i],1e-10,.05),boundary=metric(Q.sum(),e['Q'][i].sum(),1e-10,.05),**{f'Q{axis}':metric(Q[axis],e['Q'][i,axis],1e-10,.05) for axis in range(3)})
                good=all(v['absolute']<=.25*v['budget'] for v in metrics.values());timegood &=good;rows.append(dict(time_s=b,metrics=metrics,passed=bool(good),pressure_Pa=p.tolist(),boundary_by_axis_m3=Q.tolist()))
        reports[name]=dict(cells=c,faces=m['top'].nflux,V0=m['top'].V0.tolist(),source_m3_s=m['source'].tolist(),rates=m['rates'].tolist(),exact_pressure=e['p'].tolist(),exact_boundary_by_axis=e['Q'].tolist(),exact_content=e['content'].tolist(),exact_mass_defect=e['mass'].tolist(),time_rows=rows,internal_steps=steps,dissipation_J=Dsum)
    coarse=models['coarse']['top'];fine=models['fine']['top'];finep=restrict(exacts['fine']['p'],coarse,fine);comparisons=[]
    for i,t in enumerate(times[1:],1):
        a,b=exacts['coarse'],exacts['fine'];comparisons.append(dict(time_s=t,pressure=metric(a['p'][i],finep[i],.001,.05,coarse.V0),content=metric(a['content'][i],b['content'][i],1e-10,.05),boundary=metric(a['Q'][i].sum(),b['Q'][i].sum(),1e-10,.05),**{f'Q{axis}':metric(a['Q'][i,axis],b['Q'][i,axis],1e-10,.05) for axis in range(3)}))
    spacegood=all(v['passed'] for row in comparisons for k,v in row.items() if k!='time_s');eligible=bool(spacegood and timegood and within)
    write(run/'S2/topology-operator-check.json',dict(status='passed_scoped',manufactured=checks,actual_formal_rest=geocheck,shared_faces_cancel=True,source_overlap_volume=float(models['fine']['source'].sum()/.001),mass_rule=7))
    write(run/'S2/exact-time-reference.json',dict(status='passed_scoped',times_s=times.tolist(),records=reports,exact_discrete_reference_only=True))
    write(run/'S2/grid-comparison.json',dict(status='passed_scoped' if spacegood else 'limited',records=comparisons,volume_overlap_restriction=True,all_frozen_nonzero_observations=len(times)-1,early_failure_times_preserved=True))
    write(run/'S2/fixed-grid-comparison.json',dict(status='passed_scoped' if eligible else 'limited',eligible_coupled=eligible,space_passed=spacegood,time_passed=bool(timegood),actual_steps_per_grid=len(internal)-1,comparisons=comparisons))
    write(run/'S2/pressure-scope-decision.json',dict(status='fixed_scoped_pending_coupled' if eligible else 'limited',eligible_coupled=eligible,fixed_grid_passed=eligible,actual_coupled=False,pressure_spatial_accuracy=False,production_C_E_integration=False,coupled_q5=False,pure_solid_default=True,old_failure_retained=True))
    print('PRESSURE_DECISION',eligible,spacegood,timegood,'steps',len(internal)-1,'failed',[(r['time_s'],[k for k,v in r.items() if k!='time_s' and not v['passed']]) for r in comparisons if any(not v['passed'] for k,v in r.items() if k!='time_s')],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','fixed']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
