"""Physical field, interval-flow and energy review; never advances coupling."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .coupling import setup,frame
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_phase_boundary_next.pressure_study import restrict
from benchmarks.research_sequential_next.compare import metric,impulse_average
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology


def report(run,grid):
    run=Path(run);c,m,cfg=setup(run,grid);cache=CachedProbes(m);a=history(run/f'cases/observable-{grid}-h');b=history(run/f'cases/observable-{grid}-half');p=read(run/'S3/new-scene-protocol.json');rows=[];frames=[]
    for i,x in enumerate(a):
        y=b[2*i]
        if abs(x['state'].time-y['state'].time)>1e-15:raise ValueError('coupled physical nodes differ')
        fa,fb=frame(c,cache,x['state']),frame(c,cache,y['state']);fld=fields(fa,fb,m.parent.params.fiber_direction)
        extra={k:metric(fa[k],fb[k],tol,.05) for k,tol in [('PK1_total',.02),('Cauchy_skeleton',.02),('Cauchy_total',.02),('cell_pressure_Pa',.001)]}
        sa=x['state'].child_states['fluid'];sb=y['state'].child_states['fluid']
        for k in ('content_m3','cumulative_source_m3','cumulative_boundary_m3'):extra[k]=metric(sa[k],sb[k],1e-10,.05)
        if i:
            rr=x['rows'][-1];frows=b[-1]['rows'][2*i-2:2*i];dt=rr['dt'];flux=sum(np.array(r['flux_interval_m3_s'])*r['dt'] for r in frows)/dt;reaction=sum(r['reaction_N']*r['dt'] for r in frows)/dt
            extra['flux_interval']=metric(sa['flux_interval_m3_s'],flux,1e-10,.05);extra['reaction_interval']=metric(rr['reaction_N'],reaction,1e-4,.05)
        rows.append(dict(time_s=x['state'].time,fields=fld,extra=extra,passed=good(fld) and all(v['passed'] for v in extra.values())))
        if x['state'].time in p['observations_s']:
            u=(fa['x']-fa['X']).reshape(-1,3);v=fa['velocity'].reshape(-1,3);N=len(v);frames.append(dict(time_s=x['state'].time,u_rms_m=float(np.linalg.norm(u)/np.sqrt(N)),v_rms_m_s=float(np.linalg.norm(v)/np.sqrt(N)),pressure_effect_N=max((z['pressure_force_max_N'] for z in x['rows']),default=0),volume_change_max_m3=max((float(np.max(abs(np.asarray(z['delta_volume_m3'])))) for z in x['rows']),default=0)))
    observed=[i for i,v in enumerate(frames) if v['u_rms_m']>=p['mechanical_u_rms_threshold_m'] or v['v_rms_m_s']>=p['mechanical_v_rms_threshold_m_s']];observable=any(j-i>=2 for i in observed for j in observed)
    ledger=a[-1]['rows'];initial=a[0]['state'];fluid=initial.child_states['fluid'];E0=m.kinetic(initial.velocity)+m.evaluate(initial.q)['U']+.5*np.sum(c.capacity*np.array(fluid['pressure_Pa'])**2);net=sum(r['darcy_dissipation_J']-r['external_work_J']-r['source_work_J']-r['reservoir_work_J'] for r in ledger);balance=ledger[-1]['total_energy_J']-E0+net;scale=max(E0,sum(abs(r['external_work_J'])+abs(r['source_work_J'])+abs(r['reservoir_work_J']) for r in ledger),1e-8);budget=1e-9+.01*scale
    phy=bool(all(r['min_detF']>.1 and r['true_scaled_residual']<=1 and r['darcy_dissipation_J']>=0 for r in ledger) and abs(balance)<=budget);timepass=all(x['passed'] for x in rows)
    write(run/'S3'/f'{grid}-time-comparison.json',dict(status='passed_scoped' if timepass else 'limited',records=rows,interval_flow_compared_as_impulse=True))
    write(run/'S3'/f'{grid}-physical-check.json',dict(status='passed_scoped' if observable and phy else 'limited',mechanically_observable=observable,nonadjacent_observation_indices=observed,frames=frames,physical_budgets_passed=phy,cumulative_energy_balance_J=balance,cumulative_energy_budget_J=budget,time_refinement_passed=timepass,pressure_force_nonzero=any(x['pressure_effect_N']>1e-7 for x in frames),volume_feedback_above_roundoff=any(x['volume_change_max_m3']>64*np.finfo(float).eps*max(c.geometry.V0) for x in frames),min_detF=min(x['min_detF'] for x in ledger)))
    # Export physical coarse probes for a later conservative grid comparison.
    out={k:[] for k in ('x','velocity','PK1','PK1_total')}
    for item in a:
        f=frame(c,cache,item['state'])
        for k in out:out[k].append(f[k])
    np.savez_compressed(run/'S3'/f'{grid}-common-probes.npz',X=f['X'],fiber=m.parent.params.fiber_direction,**{k:np.array(v) for k,v in out.items()})
    print('OBSERVABLE_REVIEW',grid,observable,phy,timepass,balance,flush=True)


def finish(run):
    run=Path(run);records={g:read(run/'S3'/f'{g}-physical-check.json') for g in ('coarse','fine') if (run/'S3'/f'{g}-physical-check.json').exists()};p=read(run/'S3/new-scene-protocol.json');grid=[]
    if len(records)==2:
        a=history(run/'cases/observable-coarse-h');b=history(run/'cases/observable-fine-h');ta=CartesianTopology(p['cuts']['coarse']);tb=CartesianTopology(p['cuts']['fine'])
        with np.load(run/'S3/coarse-common-probes.npz') as aa,np.load(run/'S3/fine-common-probes.npz') as bb:
            for i,(x,y) in enumerate(zip(a,b)):
                fld=fields(dict(X=aa['X'],**{k:aa[k][i] for k in ('x','velocity','PK1')}),dict(X=bb['X'],**{k:bb[k][i] for k in ('x','velocity','PK1')}),aa['fiber']);pa=np.array(x['state'].child_states['fluid']['pressure_Pa']);pb=restrict(np.array(y['state'].child_states['fluid']['pressure_Pa']),ta,tb);pe=metric(pa,pb,.001,.05,ta.V0);grid.append(dict(time_s=x['state'].time,fields=fld,pressure=pe,passed=good(fld) and pe['passed']))
    split=(run/'cases/observable-coarse-h/summary.json').exists() and read(run/'cases/observable-coarse-h/summary.json')['actual_new_process']
    write(run/'S3/grid-and-transaction-review.json',dict(status='passed_scoped' if grid and all(x['passed'] for x in grid) and split else 'limited',grid_comparison=grid,actual_new_process_restart=split,checkpoint_source_and_full_digest_validated=True,conservative_pressure_overlap=True,grid_cells=[16,32] if len(records)==2 else [16],new_fault_injection=False,inherited_faults_sha256=sha(APP/'S2/transaction-and-restart.json')))
    write(run/'S3/observable-coupled-check.json',dict(status='passed_scoped' if records and all(x['status']=='passed_scoped' for x in records.values()) else 'limited',records=records))
    write(run/'S3/coupled-time-comparison.json',dict(status='passed_scoped' if records and all(x['time_refinement_passed'] for x in records.values()) else 'limited',grids={g:dict(path=f'S3/{g}-time-comparison.json',sha256=sha(run/'S3'/f'{g}-time-comparison.json')) for g in records}))
    write(run/'S3/coupling-scope-decision.json',dict(status='passed_scoped' if len(records)==2 and all(x['status']=='passed_scoped' and x['time_refinement_passed'] for x in records.values()) and all(x['passed'] for x in grid) else 'limited',actual_observable_coupling=any(x['mechanically_observable'] for x in records.values()),complete_grids=len(records),full_coupled_cycle=False,coupled_q5=False,production_C_E_integration=False,formal_default_changed=False,physically_calibrated=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['coarse','fine','finish']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='finish':finish(a.run)
        else:report(a.run,a.phase)
