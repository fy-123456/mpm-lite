"""Bounded original-grid fixed-skeleton diagnosis, never a coupled truth surrogate."""
import argparse,time
import numpy as np
from .provenance import *
from .review import fluid,context
from .coupling import case_name
from benchmarks.research_stabilization_boundary_next.pressure import algebra,exact,comparison,bisect
from benchmarks.research_sequential_next.compare import metric


def diagnose(run):
    run=Path(run);verify(run);decision=read(run/'S3/grid-decision.json')
    if decision['nested_grid_agreement']:raise ValueError('unnecessary reference study')
    register(run,'S3/diagnostic-protocol.json',dict(status='registered',family='original x-only',cells=[16,32,64,128],max_new_matrices=4,max_spectral_calls=4,max_matrix_steps=0,max_seconds=600,reference_fraction=.25,source_m3_s=0.,fixed_skeleton_only=True,no_new_GPU_steps=True))
    begun=time.perf_counter();p=read(run/'S0/input-contract.json');params=p['parameters'];cuts=p['cuts']['coarse'];times=np.array(p['fine_times_s']);models={};values={}
    for n in (16,32,64,128):
        if time.perf_counter()-begun>600:raise TimeoutError('CPU reference budget')
        a=algebra(cuts,params);models[n]=a;v=exact(a,params,times);values[n]=v
        np.savez_compressed(run/'S3'/f'fixed-original-{n}.npz',H=a['H'],C=a['C'],L=a['L'],rhs=a['rhs'],eigenvalues=a['lam'],pressure=v['pressure'],cumulative=v['cumulative'],flux=v['flux'],times=times)
        cuts=bisect(cuts)
    comparisons={f'{a}-{b}':comparison(models[a],models[b],values[a],values[b],times,.25 if a==64 else 1.) for a,b in ((16,32),(32,64),(64,128))}
    temporal_same_grid=read(APP/'S1/startup-screen.json');bound=[];method=decision['method']
    for g,n in [('coarse',16),('fine',32)]:
        v,t,rows=fluid(run/'cases'/case_name(g,True,method));groups=models[n]['groups'];flow=v['flux']@groups.T;fixed=values[n]['flux']@groups.T
        bound.append(dict(grid=n,side_order=['-x','+x','-y','+y','-z','+z'],first_interval_actual_m3_s=flow[0].tolist(),first_interval_fixed_m3_s=fixed[0].tolist(),all_intervals_actual_m3_s=flow.tolist(),cumulative_actual_m3=(v['cumulative']@groups.T).tolist(),last_cell_pressures_Pa=v['pressure'][-1].tolist(),actual_volume_and_pressure_both_evolve=True,actual_vs_fixed_difference_is_physics_not_accuracy=True))
    fixedpass=comparisons['64-128']['status']=='passed_scoped';coarsefail=comparisons['16-32']['status']!='passed_scoped';samepass=decision['coarse_time_passed'] and decision['fine_time_passed']
    reason='same-grid actual time checks pass; grid discrepancy persists in time-exact fixed-skeleton equations, indicating pressure spatial/boundary resolution limits' if samepass and coarsefail else 'see separated time/grid evidence; fixed-skeleton dynamics cannot certify deformable coupled accuracy'
    write(run/'S3/diagnosis.json',dict(status='limited_spatial_reference',reason=reason,actual_same_grid_time_passed=samepass,fixed_skeleton_comparisons=comparisons,fixed_skeleton_reference_reliable=fixedpass,fixed_skeleton_reference_fraction=.25,coupled_continuum_reference=False,boundary_attribution=bound,inherited_same_grid_time_screen=dict(path=str(APP/'S1/startup-screen.json'),sha256=sha(APP/'S1/startup-screen.json')),new_CPU_matrices=4,spectral_calls=4,matrix_steps=0,seconds=time.perf_counter()-begun,source_vector_zero=True,current_F_in_actual_operator=True,conservative_transfer_passed=read(run/'S3/transfer-check.json')['status']=='passed_scoped',skip_reason='do not extend GPU beyond32cells, add another grid family or rerun long trajectories'))
    decision.update(fixed_skeleton_reference_reliable=fixedpass,diagnosis=reason,spatial_accuracy=False,performance_grid='coarse');write(run/'S3/grid-decision.json',decision)
    print('DIAGNOSIS',reason,{k:v['max_budget_ratios'] for k,v in comparisons.items()},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):diagnose(a.run)
