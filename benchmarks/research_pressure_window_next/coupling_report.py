"""Rebuild coupled summaries from committed generations, without integration."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import read,write,sha
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology
from engine.aniso_phase1.research_pressure_window_next.coupled import time_tolerance


def summarize(run,label):
    folder=Path(run)/'cases'/f'coupled-{label}';identity=read(folder/'identity.json');h=GenerationStore(folder,identity).history();last=h[-1];rows=last['rows'];state=last['state'];f=state.child_states['fluid'];f0=h[0]['state'].child_states['fluid'];segments=read(folder/'segments.json')
    if state.step!=19 or len(rows)!=19:raise ValueError('complete 19-step history required')
    mass=float(np.sum(np.array(f['content_m3'])-f0['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']));frames=[]
    for item in h:
        path=item['folder']/'frame.npz'
        if path.exists():
            with np.load(path) as a:frames.append(dict(time_s=item['state'].time,u_max_m=float(np.max(abs(a['x']-a['X']))),v_max_m_s=float(np.max(abs(a['velocity']))),PK1_max_Pa=float(np.max(abs(a['PK1']))),PK1_total_max_Pa=float(np.max(abs(a['PK1_total'])))))
    maxu=max(x['u_max_m'] for x in frames);maxv=max(x['v_max_m_s'] for x in frames)
    passed=abs(mass)<=1e-10 and all(r['true_scaled_residual']<=1 and r['darcy_dissipation_J']>=0 and r['min_detF']>.1 for r in rows)
    result=dict(status='passed_scoped' if passed else 'limited',steps=19,end_s=state.time,cumulative_mass_defect_m3=mass,min_pressure_Pa=min(min(r['pressure_Pa']) for r in rows),min_detF=min(r['min_detF'] for r in rows),max_true_residual_fraction=max(r['true_scaled_residual'] for r in rows),max_force_residual_N=max(r['solid_force_residual_N'] for r in rows),max_energy_balance_J=max(abs(r['energy_balance_J']) for r in rows),max_pressure_work_defect_J=max(float(np.max(abs(np.asarray(r['pressure_work_defect_J'])))) for r in rows),max_displacement_m=maxu,max_velocity_m_s=maxv,mechanical_response_observable=bool(maxu>=5e-5 or maxv>=1e-4),frames=frames,actual_new_process=len({s['pid'] for s in segments})>=2,segments=segments,read_only_reaggregation=True,source_identity_sha256=sha(folder/'identity.json'),committed_pointer_sha256=sha(folder/'CURRENT.json'))
    recovery=Path(run)/'S2/resource-recovery.json'
    if label=='fine' and recovery.exists():
        result['resource_recovery']=dict(path=str(recovery.relative_to(Path(run))),sha256=sha(recovery),accepted_segments=[2,11,6],failed_attempts=1,process_count_at_least=3,failed_segment_wall_not_measured=True)
    write(folder/'summary.json',result);print('COUPLED_SUMMARY',label,result['status'],maxu,maxv,mass,flush=True)
    return result


def compare_fixed(run):
    run=Path(run);fixed=read(run/'S1/full-window-check.json')['records'];cuts=read(run/'S2/coupled-protocol.json')['cuts'];records={}
    for label in ('coarse','fine'):
        folder=run/'cases'/f'coupled-{label}';rows=read(folder/'ledger.json');top=CartesianTopology(cuts[label]);D=np.array([top.boundary_sign*(top.axes==i) for i in range(3)]);Q=np.zeros(3);comparisons=[]
        for row,reference in zip(rows,fixed[label]['rows']):
            if abs(row['time']-reference['time_s'])>time_tolerance(row['time'],reference['time_s']):raise ValueError('fixed/coupled physical nodes differ')
            Q+=row['dt']*(D@np.array(row['flux_interval_m3_s']));metrics=dict(pressure=metric(row['pressure_Pa'],reference['pressure_Pa'],.001,.05,top.V0),content=metric(sum(row['content_m3']),reference['content_m3'],1e-10,.05),**{f'Q{i}':metric(Q[i],reference['Q_by_axis_m3'][i],1e-10,.05) for i in range(3)})
            comparisons.append(dict(time_s=row['time'],metrics=metrics,delta_volume_m3=row['delta_volume_m3']))
        records[label]=comparisons
    result=dict(status='passed_scoped',records=records,scope='diagnostic difference from fixed-geometry trajectory, not coupled truncation error',dynamic_reference=False,all_differences_within_engineering_budget=all(v['passed'] for rows in records.values() for row in rows for v in row['metrics'].values()))
    write(run/'S2/coupled-vs-fixed.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--grid',choices=['coarse','fine'],required=True);a=p.parse_args();summarize(a.run,a.grid)
