"""One targeted stiff-start repair within the remaining fixed-pressure budget."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .pressure_study import algebra,exact,CartesianTopology
from benchmarks.research_sequential_next.compare import metric

def internal_schedule(observation_times,lambda_max,*,growth=1.7,initial_h_lambda=.5):
    obs=np.asarray(observation_times,dtype=float)
    if obs.ndim!=1 or len(obs)<2 or obs[0]!=0 or np.any(np.diff(obs)<=0) or not np.isfinite(obs).all():raise ValueError('fixed positive observation sequence required')
    if not np.isfinite(lambda_max) or lambda_max<=0 or not np.isfinite(growth) or growth<=1 or not 0<initial_h_lambda<=1:raise ValueError('positive stiffness and bounded resolved initial step required')
    ts=[0.];dt=initial_h_lambda/lambda_max
    if dt<=0 or ts[-1]+dt==ts[-1]:raise ValueError('unrepresentable internal time step')
    while ts[-1]+dt<obs[-1]:
        ts.append(ts[-1]+dt);dt*=growth
        if len(ts)>64:raise ValueError('registered internal step budget exceeded')
    out=np.unique(np.r_[obs,ts])
    if len(out)-1>64:raise ValueError('observation union exceeds fixed time budget')
    return out

def correction(run):
    run=Path(run);verify(run);obs=np.array(read(run/'S2/fixed-time-protocol.json')['times_s']);grids=read(run/'S2/grid-protocol.json')['cuts'];models={k:algebra(CartesianTopology(c)) for k,c in grids.items()};rate=max(m['rates'][-1] for m in models.values());full=internal_schedule(obs,rate);prefix=full[full<=obs[2]];prior=read(run/'S2/fixed-grid-comparison.json')['actual_steps_per_grid']
    if prior+len(prefix)-1>64:raise ValueError('targeted repair exceeds cumulative fixed-pressure budget')
    register(run,'S2/corrected-start-protocol.json',dict(status='passed_scoped',cause='resolved stiff initial pressure layer needs small initial h*lambda; physical observation times unchanged',observation_times_s=obs.tolist(),new_internal_times_s=full.tolist(),first_h_lambda_max=.5,growth=1.7,full_schedule_steps=len(full)-1,actual_test_times_s=prefix.tolist(),actual_extra_steps_per_grid=len(prefix)-1,total_fixed_steps_per_grid=prior+len(prefix)-1,full_corrected_schedule_executed=False,first_four_observation_steps=int(sum(full<=obs[4])-1),no_new_grids=True,solid_equations_unchanged=True))
    records={};passed=True
    for name,m in models.items():
        e=exact(m,prefix);p=np.full(m['top'].cells,.01);Q=np.zeros(3);rows=[];C=np.diag(m['C']);minimum=.01
        for i,(a,b) in enumerate(zip(prefix[:-1],prefix[1:]),1):
            dt=b-a;old=p.copy();p=la.solve(C+.5*dt*m['L'],(C-.5*dt*m['L'])@p+dt*m['rhs'],assume_a='pos');z=m['Z']@(.5*(old+p))+m['z0'];Q+=dt*m['D']@z;D=dt*float(z@m['H']@z);defect=float((p-.01)@m['C']+sum(Q)-b*sum(m['source']));minimum=min(minimum,float(p.min()))
            metrics=dict(pressure=metric(p,e['p'][i],.001,.05,m['top'].V0),boundary=metric(Q.sum(),e['Q'][i].sum(),1e-10,.05),**{f'Q{k}':metric(Q[k],e['Q'][i,k],1e-10,.05) for k in range(3)})
            isobs=bool(np.any(obs==b));ok=abs(defect)<=1e-10 and D>=0 and np.isfinite(p).all() and p.min()>=0 and (not isobs or all(v['absolute']<=.25*v['budget'] for v in metrics.values()));passed &=ok
            rows.append(dict(time_s=b,dt_s=dt,physical_observation=isobs,metrics=metrics,min_pressure_Pa=float(p.min()),mass_defect_m3=defect,dissipation_J=D,passed=bool(ok)))
        records[name]=dict(rows=rows,min_pressure_Pa=minimum,extra_steps=len(prefix)-1,total_fixed_steps=prior+len(prefix)-1)
    write(run/'S2/corrected-start-check.json',dict(status='passed_scoped' if passed else 'limited',records=records,full_corrected_time_window_qualified=False,old_failures_retained=True,scope='only first two frozen nonzero observation times; one targeted rerun, unchanged grids and equations'))
    write(run/'S2/pressure-scope-decision.json',dict(status='limited',eligible_coupled=False,original_fixed_time_gate=False,grid_comparison_passed=True,corrected_start_passed=bool(passed),full_corrected_time_window_qualified=False,actual_coupled=False,pressure_spatial_accuracy=False,production_C_E_integration=False,coupled_q5=False,pure_solid_default=True,steps_per_grid=prior+len(prefix)-1,reason='corrected initial layer tested within remaining budget; full corrected time trajectory untested and required four-observation prefix is19steps beyond8 coupling cap'))
    for name in ('common-model-lock','volume-gradient-check','mixed-operator-check','coupled-scene-check','transaction-and-restart'):
        write(run/f'S5/{name}.json',dict(status='not_triggered',reason='corrected start passed only scoped; complete corrected pressure time gate not run, required common coupling prefix19 exceeds8',evidence=['S2/corrected-start-check.json','S2/corrected-start-protocol.json'],coupled_steps=0))
    print('CORRECTED_PRESSURE_START',bool(passed),{k:v['min_pressure_Pa'] for k,v in records.items()},'totalsteps',prior+len(prefix)-1,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):correction(a.run)
