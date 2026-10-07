"""Physical comparisons of complete stored theta trajectories, no new dynamics."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .pressure import coarse_observations
from .coupling import case_name
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_stabilization_boundary_next.pressure import comparison
from benchmarks.research_restoring_rt0_next.reference import compare_pair
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction

def context(cuts,storage):
    top=ReferenceTopology(cuts)
    return dict(top=top,C=storage*top.V0,groups=np.array([top.boundary_sign*(top.axes==axis)*(top.boundary_sign==sign) for axis in range(3) for sign in (-1,1)]))

def fluid(folder):
    h=history(folder);rows=h[-1]['rows'];flux=np.asarray([r['flux_interval_m3_s'] for r in rows]);dt=np.asarray([r['dt'] for r in rows]);cumulative=np.vstack([np.zeros(flux.shape[1]),np.cumsum(flux*dt[:,None],axis=0)])
    return dict(pressure=np.asarray([v['state'].child_states['fluid']['pressure_Pa'] for v in h]),cumulative=cumulative,flux=flux,content=np.asarray([v['state'].child_states['fluid']['content_m3'] for v in h])),np.asarray([v['state'].time for v in h]),rows

def compare_fluid(ca,cb,a,b,times):
    result=comparison(ca,cb,a,b,times);_,C,_=restriction(ca['top'],cb['top'])
    for i,r in enumerate(result['records'],1):
        check=metric(a['content'][i],C@b['content'][i],1e-10,.05);r['checks']['actual_total_content']=check;r['budget_ratios']['actual_total_content']=check['absolute']/check['budget'];r['passed'] &=check['passed']
    result['max_budget_ratios']['actual_total_content']=max(r['budget_ratios']['actual_total_content'] for r in result['records'])
    result['status']='passed_scoped' if all(r['passed'] for r in result['records']) else 'limited'
    return result

def compare_fields(a,b,rows,rhs,stride):
    with np.load(a/'probes.npz') as aa,np.load(b/'probes.npz') as bb:
        out=compare_pair(aa,bb,rows,rhs,stride)
        fa={k:aa[k] for k in aa.files};fb={k:bb[k] for k in bb.files}
    stresses=[]
    for i,row in enumerate(rows,1):
        checks={region:{k:metric(fa[k][i],fb[k][i*stride],.02,.05,w) for k in ('PK1_total','Cauchy_skeleton','Cauchy_total')} for region,w in regions(fa['X']).items()}
        stresses.append(dict(time_s=row['time'],checks=checks,passed=all(v['passed'] for region in checks.values() for v in region.values())))
    out['total_and_cauchy_stress']=stresses;out['passed'] &=all(r['passed'] for r in stresses)
    return out

def review(run,complete=False):
    run=Path(run);verify(run);p=read(run/'S5/selected-protocol.json');contexts={g:context(p['cuts'][g],p['parameters']['storage']) for g in ('coarse','fine')};temporal={}
    if complete:raise ValueError('fine grid and full-window qualification deferred after source-binding repair')
    grids=('coarse','fine') if complete else ('coarse',)
    for grid in grids:
        a=run/'cases'/case_name(grid);b=run/'cases'/case_name(grid,True);va,t,rows=fluid(a);vb,tt,rhs=fluid(b)
        restricted=coarse_observations(vb,tt);restricted['content']=vb['content'][::2]
        flow=compare_fluid(contexts[grid],contexts[grid],va,restricted,t);solid=compare_fields(a,b,rows,rhs,2)
        energy=dict(h=read(a/'summary.json'),half=read(b/'summary.json'),Dnum_comparison=metric(sum(r['numerical_dissipation_J'] for r in rows),sum(r['numerical_dissipation_J'] for r in rhs),1e-9,.05),numerical_dissipation_not_physical_darcy=True)
        temporal[grid]=dict(status='passed_scoped' if flow['status']=='passed_scoped' and solid['passed'] else 'limited',fluid=flow,solid=solid,energy=energy)
    spatial=None
    if complete:
        a=run/'cases/startup-coarse-half';b=run/'cases/startup-fine-half';va,t,rows=fluid(a);vb,tt,rhs=fluid(b)
        if not np.array_equal(t,tt):raise ValueError('grid comparison times differ')
        spatial=dict(fluid=compare_fluid(contexts['coarse'],contexts['fine'],va,vb,t),solid=compare_fields(a,b,rows,rhs,1))
    result=dict(status='passed_scoped' if all(v['status']=='passed_scoped' for v in temporal.values()) else 'limited',temporal=temporal,spatial=spatial,first_intervals_included=True,fixed_skeleton_not_coupled_truth=True,method=p['method'],family=p['family'])
    write(run/('S5/coupled-comparison.json' if complete else 'S5/coarse-temporal.json'),result)
    if complete:
        write(run/'S5/coupling-decision.json',dict(status='passed_time_space_limited' if result['status']=='passed_scoped' else 'limited',temporal_passed=result['status']=='passed_scoped',grid_comparison_passed=spatial['fluid']['status']=='passed_scoped' and spatial['solid']['passed'],certified_spatial_accuracy=False,production_C_E_integration=False,coupled_q5=False,full_cycle=False,new_dynamic_steps=96,scope='original nested16/32 pressure grids, 0..2e-4 s, q7/M7, original formal solid space, backward Euler pressure'))
    print('COUPLED_REVIEW',{k:v['status'] for k,v in temporal.items()},'grid',None if spatial is None else spatial['fluid']['max_budget_ratios'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--complete',action='store_true');a=p.parse_args()
    with serial_lock(a.run):review(a.run,a.complete)
