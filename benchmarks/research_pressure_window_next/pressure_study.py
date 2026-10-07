"""Complete the frozen repaired pressure schedule, retaining every physical node."""
from pathlib import Path
import argparse
import time
import numpy as np
import scipy.linalg as la
from .provenance import APP,read,write,sha,register,verify,serial_lock
from .physics import baseline_model
from benchmarks.research_observable_boundary_next.pressure_correction import internal_schedule
from benchmarks.research_phase_boundary_next.pressure_study import algebra,exact,restrict
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_observable_pressure_next.coupling_study import affine_check
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology,CartesianGeometry


def study(run):
    run=Path(run);verify(run);started=time.perf_counter()
    old=read(APP/'S2/corrected-start-protocol.json');obs=np.array(old['observation_times_s']);times=np.array(old['new_internal_times_s'])
    cuts=read(APP/'S2/grid-protocol.json')['cuts'];models={k:algebra(CartesianTopology(v)) for k,v in cuts.items()}
    regenerated=internal_schedule(obs,max(m['rates'][-1] for m in models.values()))
    if not np.array_equal(times,regenerated) or len(times)!=48 or not all(t in times for t in obs):raise ValueError('frozen repaired schedule differs')
    if sum(times<=obs[2])-1!=14 or sum(times<=obs[4])-1!=19:raise ValueError('physical prefix differs')
    register(run,'S1/full-window-protocol.json',dict(status='passed_scoped',times_s=times.tolist(),observations_s=obs.tolist(),steps_per_grid=47,max_steps_per_grid=64,cuts=cuts,parent_protocol_sha256=sha(APP/'S2/corrected-start-protocol.json'),time_fraction=.25,no_changed_observations=True))
    m,cfg=baseline_model(run,pressure=True);checks=[]
    for label,a in models.items():
        g=CartesianGeometry(m,cuts[label]);actual=g.evaluate(m.rest().q)
        rec=dict(grid=label,H_max=float(np.max(abs(actual['H']-a['H']))),volume_max=float(np.max(abs(actual['volume']-a['top'].V0))),geometry=g.identity,patches=[affine_check(a['top'],F) for F in (np.eye(3),np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]]))])
        if rec['H_max']>1e-7 or rec['volume_max']>1e-11:raise ValueError('actual rest geometry mismatch')
        checks.append(rec);del g
    write(run/'S1/operator-check.json',dict(status='passed_scoped',records=checks,full_tensor=True,source_overlap_volume_m3=float(models['fine']['source'].sum()/.001)))
    records={};allpass=True;exacts={k:exact(a,times) for k,a in models.items()};oldprefix=read(APP/'S2/corrected-start-check.json')['records']
    for label,a in models.items():
        e=exacts[label];p=np.full(a['top'].cells,.01);Q=np.zeros(3);C=np.diag(a['C']);rows=[];Dsum=0.;maxprefix=0.
        for i,(start,end) in enumerate(zip(times[:-1],times[1:]),1):
            h=end-start;before=p.copy();p=la.solve(C+.5*h*a['L'],(C-.5*h*a['L'])@p+h*a['rhs'],assume_a='pos')
            z=a['Z']@(.5*(before+p))+a['z0'];Q+=h*a['D']@z;D=h*float(z@a['H']@z);Dsum+=D
            residual=a['C']*(p-before)+h*a['B']@z-h*a['source'] if 'B' in a else a['C']*(p-before)+h*a['top'].B@z-h*a['source']
            defect=float((p-.01)@a['C']+Q.sum()-end*a['source'].sum())
            metrics=dict(pressure=metric(p,e['p'][i],.001,.05,a['top'].V0),content=metric(p@a['C'],e['content'][i],1e-10,.05),boundary=metric(Q.sum(),e['Q'][i].sum(),1e-10,.05),**{f'Q{k}':metric(Q[k],e['Q'][i,k],1e-10,.05) for k in range(3)})
            observed=bool(end in obs);fraction=max(x['absolute']/(.25*x['budget']) for x in metrics.values())
            good=bool(np.isfinite(p).all() and p.min()>=0 and D>=0 and abs(defect)<=1e-10 and (not observed or fraction<=1));allpass &=good
            row=dict(time_s=end,dt_s=h,physical_observation=observed,pressure_Pa=p.tolist(),flux_m3_s=z.tolist(),Q_by_axis_m3=Q.tolist(),content_m3=float(p@a['C']),mass_defect_m3=defect,cell_mass_residual_m3=residual.tolist(),dissipation_J=D,min_pressure_Pa=float(p.min()),max_pressure_Pa=float(p.max()),exact_min_pressure_Pa=float(e['p'][i].min()),metrics=metrics,time_budget_fraction=fraction,passed=good)
            rows.append(row)
            if i<=14:maxprefix=max(maxprefix,abs(row['min_pressure_Pa']-oldprefix[label]['rows'][i-1]['min_pressure_Pa']))
        records[label]=dict(status='passed_scoped' if all(r['passed'] for r in rows) else 'limited',steps=47,rows=rows,old_prefix_max_pressure_difference_Pa=maxprefix,min_pressure_Pa=min(r['min_pressure_Pa'] for r in rows),max_observation_time_budget_fraction=max(r['time_budget_fraction'] for r in rows if r['physical_observation']),dissipation_J=Dsum,exact_pressure=e['p'].tolist(),exact_boundary_by_axis=e['Q'].tolist())
        print('FULL_PRESSURE',label,records[label]['status'],records[label]['max_observation_time_budget_fraction'],records[label]['min_pressure_Pa'],flush=True)
    coarse=models['coarse']['top'];fine=models['fine']['top'];fp=restrict(exacts['fine']['p'],coarse,fine);grid=[]
    for t in obs[1:]:
        i=int(np.where(times==t)[0][0]);a=exacts['coarse'];b=exacts['fine'];metrics=dict(pressure=metric(a['p'][i],fp[i],.001,.05,coarse.V0),content=metric(a['content'][i],b['content'][i],1e-10,.05),boundary=metric(a['Q'][i].sum(),b['Q'][i].sum(),1e-10,.05),**{f'Q{k}':metric(a['Q'][i,k],b['Q'][i,k],1e-10,.05) for k in range(3)})
        grid.append(dict(time_s=float(t),metrics=metrics,passed=all(v['passed'] for v in metrics.values())))
    spacegood=all(r['passed'] for r in grid);passed=bool(allpass and spacegood)
    write(run/'S1/exact-discrete-reference.json',dict(status='passed_scoped',times_s=times.tolist(),observations_s=obs.tolist(),records={k:dict(p=exacts[k]['p'].tolist(),Q=exacts[k]['Q'].tolist(),content=exacts[k]['content'].tolist()) for k in models},continuous_truth=False))
    write(run/'S1/full-window-check.json',dict(status='passed_scoped' if allpass else 'limited',records=records,seconds=time.perf_counter()-started,actual_steps=94,full_corrected_time_window_qualified=bool(allpass),old_failed_protocol_sha256=sha(APP/'S2/fixed-time-protocol.json')))
    write(run/'S1/grid-comparison.json',dict(status='passed_scoped' if spacegood else 'limited',records=grid,physical_overlap_restriction=True,continuous_spatial_accuracy=False))
    write(run/'S1/pressure-scope-decision.json',dict(status='passed_scoped' if passed else 'limited',eligible_coupled=passed,full_corrected_time_window_qualified=bool(allpass),space_comparison_passed=spacegood,actual_steps=94,coupled_q5=False,production_C_E_integration=False,formal_default_changed=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
