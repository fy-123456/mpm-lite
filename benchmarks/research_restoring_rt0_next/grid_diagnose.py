"""Small exact pressure-matrix diagnostic; no new coupled trajectory."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .grid_transfer import restriction
from benchmarks.research_candidate_observable_next.coupling import algebra
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_phase_stress_next.time_study import history


def analyze(run):
    run=Path(run);p=read(APP/'S3/new-scene-protocol.json');models=[algebra(p['cuts'][k],p['parameters']) for k in ('coarse','fine')]
    hc=history(APP/'cases/observable-coarse-half');hf=history(run/'cases/grid32-half');times=np.array([h['state'].time for h in hf]);data=[]
    for a in models:
        n=a['top'].cells;initial=np.r_[np.full(n,p['parameters']['pressure0_Pa']),0.,1.];exact=np.array([la.expm(a['aug']*t)@initial for t in times]);Q=exact[:,n]
        data.append(dict(pressure=exact[:,:n],cumulative=Q,interval=np.diff(Q)/np.diff(times)))
    P,C,Z=restriction(models[0]['top'],models[1]['top']);records=[]
    for i,t in enumerate(times[1:],1):
        ac=hc[i]['state'].child_states['fluid'];af=hf[i]['state'].child_states['fluid'];coarse_flow=float(models[0]['top'].boundary_sign@np.array(ac['flux_interval_m3_s']));fine_flow=float(models[1]['top'].boundary_sign@np.array(af['flux_interval_m3_s']))
        records.append(dict(time_s=float(t),fixed_coarse_flow_m3_s=float(data[0]['interval'][i-1]),fixed_fine_flow_m3_s=float(data[1]['interval'][i-1]),
            actual_coarse_flow_m3_s=coarse_flow,actual_fine_flow_m3_s=fine_flow,
            fixed_flow_error=metric(data[0]['interval'][i-1],data[1]['interval'][i-1],1e-10,.05),actual_flow_error=metric(coarse_flow,fine_flow,1e-10,.05),
            fixed_cumulative_error=metric(data[0]['cumulative'][i],data[1]['cumulative'][i],1e-10,.05),
            pressure_error=metric(data[0]['pressure'][i],P@data[1]['pressure'][i],.001,.05,models[0]['top'].V0)))
    first=records[0];ratio=(first['fixed_fine_flow_m3_s']-first['fixed_coarse_flow_m3_s'])/(first['actual_fine_flow_m3_s']-first['actual_coarse_flow_m3_s'])
    write(run/'S3/grid-limitation-diagnostic.json',dict(status='limited',records=records,diagnostic='exact semidiscrete fixed-skeleton matrix exponential, same boundary/initial pressure and mobility; analytically integrated rest RT0, not a replacement for actual geometry',
        first_flow_difference_explained_ratio=float(ratio),initial_pressure_Pa=p['parameters']['pressure0_Pa'],reservoir_Pa=p['parameters']['reservoir_Pa'],
        first_cell_widths_m=[float(np.diff(a['top'].cuts[0])[0]) for a in models],
        cause='initial pressure/reservoir jump drives an unresolved early boundary layer; RT0 boundary gradients depend strongly on first-cell width',
        conclusion_supported=bool(.5<ratio<1.5),no_new_coupled_steps=True,no_parameter_or_budget_changes=True,
        next_experiment='refine only the boundary-pressure layer using a fixed-skeleton matrix study first; preserve the jump and physical parameters. A separate smooth-start fixture may be studied only as a separately labelled physical scenario.'))
    print('GRID_DIAGNOSIS first boundary-flow difference ratio',ratio,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):analyze(a.run)
