"""Fork the committed candidate prefix; compare registered nodes by index.

The old 42-step case remains immutable. A separately identified continuation
uses its exact state/history and unchanged solid integrator, so no step repeats.
"""
from pathlib import Path
import argparse,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .run import load_model
from . import config
from .candidate_study import fields,good
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,impulse_average
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.runtime_rules import advance_publish


def resume(run):
    run=Path(run);verify(run);original=run/'cases/candidate-h';old=history(original);origin=old[0]['state'];cfg=read(original/'execution-protocol.json');last=old[-1]
    if last['state'].step!=42 or len(old)!=43:raise ValueError('expected intact 42-step interrupted prefix')
    register(run,'S3/continuation-protocol.json',dict(status='passed_scoped',cause='round-to-ten-decimals key disagrees for one ulp at 1.07705078125; output comparison failed after committing step42',source_identity_sha256=sha(original/'identity.json'),source_state_sha256=sha(last['folder']/'state.json'),original_steps=42,remaining_h_steps=22,half_steps=128,total_dynamic_steps=192,no_repeated_steps=True,reference_matching='registered node index plus 1e-12s check',numeric_sources=source_files()))
    m,_=load_model(run,cfg);cache=CachedProbes(m);baseline,bcfg=load_model(run,read(APP/'cases/final-full/execution-protocol.json'));bcache=CachedProbes(baseline)
    # Diagnose mass conditioning without changing the projection or dynamics.
    mass=[]
    for model,state,probes in ((m,origin,cache),(baseline,history(APP/'cases/window0-half')[0]['state'],bcache)):
        a=model.boundary.unit*(-.005*2*np.pi**2*np.cos(2*np.pi*(state.time-.6)));f=model.evaluate(state.q)['force'];rhs=-f[model.free]-model.M[np.ix_(model.free,model.fixed)]@a[model.fixed];M=model.Mff;D=1/np.sqrt(np.diag(M));scaled=D[:,None]*M*D[None,:]
        raw=la.solve(M,rhs,assume_a='pos');equilibrated=D[:,None]*la.solve(scaled,D[:,None]*rhs,assume_a='pos');rawa=a.copy();newa=a.copy();rawa[model.free]=raw;newa[model.free]=equilibrated
        physical=probes.maps[0]@(rawa-newa);response=probes.maps[0]@rawa
        mass.append(dict(space=model.reduction.signature,raw_condition_2=float(np.linalg.cond(M)),scaled_condition_2=float(np.linalg.cond(scaled)),raw_relative_residual=float(la.norm(M@raw-rhs)/max(la.norm(rhs),1e-30)),scaled_relative_residual=float(la.norm(M@equilibrated-rhs)/max(la.norm(rhs),1e-30)),physical_acceleration_rms_scaling_difference_m_s2=float(la.norm(physical)/np.sqrt(len(physical))),physical_acceleration_rms_m_s2=float(la.norm(response)/np.sqrt(len(response))),dynamics_replaced=False))
    write(run/'S3/mass-conditioning-review.json',dict(status='passed_scoped',records=mass,no_added_mass=True,scope='zero-step diagnosis; compare original equation residual and physical output, not coefficient magnitude alone'))
    def piece(name,initial,times,protocol,source):
        folder=run/'cases'/name;identity=dict(schema='pressure-window-candidate-continuation-v1',model=m.identity,model_sha256=m.signature,initial_digest=initial.digest(),numerical_source_sha256=source_files(),driver_sha256=sha(__file__),source_checkpoint=str(source),source_checkpoint_sha256=sha(source),unchanged_integrator=True)
        write(folder/'identity.json',identity);write(folder/'execution-protocol.json',protocol);snapshot(folder/'source',source_files());store=GenerationStore(folder,identity);store.save(initial,[]);stepper=ValidatedAVF(m,protocol,initial);rows=[];begun=time.perf_counter()
        for target in times:
            n=64 if name.endswith('continuation') else 128;index=stepper.state.step+1
            row=advance_publish(stepper,store,rows,float(target-stepper.state.time),frame_builder=cache.frame if index in (n//2,n) else None);rows.append(row)
            if index%32==0:print('CANDIDATE_CONTINUED',name,index,flush=True)
            if time.perf_counter()-begun>1200:raise TimeoutError('bounded candidate window')
        write(folder/'ledger.json',rows);write(folder/'summary.json',dict(status='passed_scoped',accepted_steps=len(rows),global_step=stepper.state.step,end_s=stepper.state.time,seconds=time.perf_counter()-begun,min_detF=min(x['min_detF'] for x in rows),new_identity_for_continuation=True))
        return store.history()
    tail=piece('candidate-h-continuation',last['state'],cfg['times'][43:],cfg,last['folder']/'state.json');coarse=old+tail[1:]
    ts=np.linspace(1.075,1.078125,129);finecfg=config.make(read(run/'input-lock.json')['energy_scale_J'],space=cfg['physical_space'],start=float(ts[0]),end=float(ts[-1]),times=ts.tolist(),initial=cfg['initial_state'])
    fine=piece('candidate-half',origin,ts[1:],finecfg,old[0]['folder']/'state.json');formal=history(APP/'cases/window0-half');records=[]
    for label,trajectory in (('h',coarse),('half',fine)):
        comparisons=[]
        for i,item in enumerate(trajectory[1:],1):
            witness=formal[2*i if label=='h' else i]
            if abs(item['state'].time-witness['state'].time)>1e-12:raise ValueError('registered candidate/reference nodes differ')
            row=item['rows'][-1];fld=fields(cache.frame(item['state']),bcache.frame(witness['state']),m.parent.params.fiber_direction);R=metric(row['reaction_N'],impulse_average(formal[-1]['rows'],row['time']-row['dt'],row['time']),1e-4,.05)
            comparisons.append(dict(time_s=item['state'].time,fields=fld,reaction=R))
        passed=all(good(x['fields']) and x['reaction']['passed'] for x in comparisons)
        records.append(dict(case='candidate-'+label,status='passed_scoped' if passed else 'limited',steps=len(trajectory)-1,comparisons=comparisons))
    own=[]
    for i,item in enumerate(coarse[1:],1):
        other=fine[2*i];row=item['rows'][-1];fld=fields(cache.frame(item['state']),cache.frame(other['state']),m.parent.params.fiber_direction);R=metric(row['reaction_N'],impulse_average(fine[-1]['rows'],row['time']-row['dt'],row['time']),1e-4,.05)
        own.append(dict(time_s=item['state'].time,fields=fld,reaction=R,passed=good(fld) and R['passed']))
    selfgood=all(x['passed'] for x in own);crossgood=records[-1]['status']=='passed_scoped'
    vmax=lambda rows:max(v['velocity']['absolute']/v['velocity']['budget'] for row in rows for v in row['fields'].values())
    result=dict(status='passed_scoped' if selfgood else 'limited',actual_steps=192,candidate_time_refinement_passed=selfgood,cross_space_fine_passed=crossgood,self_comparison=own,records=records,max_velocity_budget_fraction_self=vmax(own),max_velocity_budget_fraction_cross_fine=vmax(records[-1]['comparisons']),formal_fine_is_not_continuum_truth=True,continuation='cases/candidate-h-continuation',prefix_not_recomputed=True)
    write(run/'S3/candidate-time-check.json',result)
    why='time refinement still differs; prioritize projected high-mode excitation and temporal reference' if not selfgood else 'same-space refinement passes; inspect spatial/initial-state discrepancy' if not crossgood else 'short-window fields pass; no continuum or full-cycle qualification'
    write(run/'S3/research-space-decision.json',dict(status='limited',candidate='balanced-direction-snapshot6',static_qualified=True,projection_passed=True,short_dynamic_stable=True,time_refinement=selfgood,cross_space_fine=crossgood,formal_space_changed=False,spatial_accuracy=False,full_dynamic_cycle=False,q5=False,pressure=False,actual_steps=192,reason=why,next_experiment='resolve candidate high-mode response against a separately justified time/space reference before modifying projection or training'))
    print('CANDIDATE_DECISION',selfgood,crossgood,result['max_velocity_budget_fraction_self'],result['max_velocity_budget_fraction_cross_fine'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):resume(a.run)
