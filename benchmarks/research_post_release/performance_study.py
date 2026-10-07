"""Same-state alternating two-step profiles; field work is counted separately."""
from pathlib import Path
import argparse
import copy
import gc
import time
import numpy as np
from .provenance import read,write,sha,digest,register,verify,serial_lock,utc
from .run import load_model,make_stepper
from . import config
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.run import probe_frame


def setup(run):
    run=Path(run);lock=verify(run);cert=run/'S4/qualification.json'
    cfg=config.make(lock['energy_scale_J'],rule_policy='q5_with_full_retry',
        qualification=dict(path=str(cert.resolve()),sha256=sha(cert)))
    model,_=load_model(run,cfg);control=make_stepper(run,cfg,model,model.rest())
    folder=run/'cases/full-q7-dt0125';hist=GenerationStore(folder,read(folder/'identity.json')).history()
    return cfg,model,control.full,{round(x['state'].time,10):x for x in hist}


def branch(item,model,full):
    # A new, independent performance experiment at the same physical q/v;
    # never relabel or resume the old numerical case.
    state=item['state'].clone();delta=float(model.evaluate(state.q)['U']-full.evaluate(state.q)['U'])
    state.child_states['identity']=copy.deepcopy(model.identity)
    provenance=dict(source_state_sha256=sha(item['folder']/'state.json'),source_state_digest=item['state'].digest(),
                    independent_performance_branch=True,q_v_unchanged=True,material_energy_difference_J=delta)
    state.child_states['performance_branch']=provenance
    return state,provenance


def trial(run,label,cfg,model,full,state,field_builder):
    from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry
    times={};counts={};original={}
    for name in ('evaluate','endpoint','wrench'):
        original[name]=getattr(model,name)
        def wrap(*args,_name=name,**kwargs):
            t=time.perf_counter()
            try:return original[_name](*args,**kwargs)
            finally:times[_name]=times.get(_name,0.)+time.perf_counter()-t;counts[_name]=counts.get(_name,0)+1
        setattr(model,name,wrap)
    model.operator.reset_profile();c=RuntimeRetry(model,full,cfg,state)
    folder=Path(run)/'S5/profiles'/label
    store=GenerationStore(folder,dict(schema='profile-experiment-v1',protocol_sha256=config.identity(cfg),initial=state.digest(),label=label))
    io=0.;rows=[];t=time.perf_counter();store.save(c.state,[]);io+=time.perf_counter()-t
    start=time.perf_counter();advance=0.
    try:
        for _ in range(2):
            t=time.perf_counter();rows.append(c.step(.0125));advance+=time.perf_counter()-t
            t=time.perf_counter();store.save(c.state,rows);io+=time.perf_counter()-t
        t=time.perf_counter();frame=field_builder(c.state);fields=time.perf_counter()-t
    finally:
        for name,method in original.items():setattr(model,name,method)
    report=dict(advance_seconds=advance,field_seconds=fields,checkpoint_seconds=io,
                total_seconds=advance+fields+io,model_nested_timings=times,model_counts=counts,
                gpu_nested_timings=dict(model.operator.timings),gpu_counts=dict(model.operator.counts),
                krylov_iterations=sum(r['krylov_iterations'] for r in rows),material_rule=model.rule.signature,
                q=c.state.q,v=c.state.velocity,predictor=c.state.predictor,frame=frame,rows=rows)
    return report


def baseline(run):
    run=Path(run)
    register(run,'S5/profile-protocol.json',dict(times=[.1,1.1],steps=2,repeats=3,
        selected_policy='q5_with_full_retry',material_order=5,mass_order=5,
        pairing='same explicit physical q/v branch; identical instrumentation',
        no_nested_timing_addition=True,display_cost='one 33x7x7 field per two-step microtrial, not daily cadence',
        final_daily_cadence='12 frames/128 steps; account cache build separately'))
    cfg,m,f,states=setup(run);records=[]
    for t in [.1,1.1]:
        state,origin=branch(states[t],m,f)
        result=trial(run,f'baseline-{t}',cfg,m,f,state,lambda s:probe_frame(m,s,cfg['probe_shape']))
        record={k:v for k,v in result.items() if k not in ('q','v','predictor','frame','rows')}
        record.update(time_s=t,initial=origin);records.append(record)
    write(run/'S5/baseline-cost.json',dict(records=records,decision='measure immutable probe-map cache; ordinary steps use no Krylov, keep tangent/LU caches off'))
    print('BASELINE_COST',records,flush=True)


def paired(run):
    from engine.aniso_phase1.research_post_release.fields import CachedProbes
    run=Path(run);register(run,'S5/paired-protocol.json',dict(windows=[.1,1.1],steps=2,alternating_repeats=3,
        immutable_probe_map=True,cache_bytes_limit=256<<20,minimum_end_to_end_gain=.10,
        daily_projection=dict(steps=128,frames=12),performance_claim_scope='matched microtrials; projected daily benefit is an estimate'))
    cfg,m,f,states=setup(run);cache=CachedProbes(m,tuple(cfg['probe_shape']));records=[]
    for t in [.1,1.1]:
        state,origin=branch(states[t],m,f)
        for repeat in range(3):
            samples={}
            for enabled in ([False,True] if repeat%2==0 else [True,False]):
                label=f'paired-{t}-{repeat}-'+('cached' if enabled else 'original')
                samples[enabled]=trial(run,label,cfg,m,f,state,cache.frame if enabled else lambda s:probe_frame(m,s,cfg['probe_shape']))
            a,b=samples[False],samples[True]
            state_error=max(float(np.max(abs(a[k]-b[k]))) for k in ('q','v','predictor'))
            errors={k:float(np.max(abs(a['frame'][k]-b['frame'][k]))) for k in a['frame']}
            if state_error>1e-12 or max(errors.values())>1e-7:raise ValueError('probe cache changed physical results')
            # Returned frame arrays must not alias cache owners.
            changed=cache.frame(state);changed['X'].fill(0.)
            if np.max(abs(cache.frame(state)['X']))==0:raise ValueError('probe cache owner leaked')
            record=dict(time_s=t,repeat=repeat,state_max_difference=state_error,field_max_differences=errors,
                        original={k:v for k,v in a.items() if k not in ('q','v','predictor','frame','rows')},
                        cached={k:v for k,v in b.items() if k not in ('q','v','predictor','frame','rows')})
            records.append(record)
    old=float(np.median([r['original']['total_seconds'] for r in records]));new=float(np.median([r['cached']['total_seconds'] for r in records]))
    old_field=float(np.median([r['original']['field_seconds'] for r in records]));new_field=float(np.median([r['cached']['field_seconds'] for r in records]))
    core=float(np.median([r['original']['advance_seconds'] for r in records]))/2
    io=float(np.median([r['original']['checkpoint_seconds'] for r in records]))/3
    estimate_old=128*core+129*io+12*old_field
    estimate_new=128*core+129*io+12*new_field+cache.build_seconds
    daily_gain=1-estimate_new/estimate_old
    enabled=daily_gain>.10 and new<old
    result=dict(status='passed_scoped',records=records,cache=dict(build_seconds=cache.build_seconds,bytes=cache.bytes),
                median_original_seconds=old,median_cached_seconds=new,microtrial_gain=1-new/old,
                field_original_seconds=old_field,field_cached_seconds=new_field,
                estimated_daily_old_seconds=estimate_old,estimated_daily_new_seconds=estimate_new,estimated_daily_gain=daily_gain,
                selected_field_cache=enabled,linearization_cache=False,preconditioner='original',
                selection='enable only if actual microtrial and build-amortized daily estimate show meaningful gain')
    write(run/'S5/performance-decision.json',result)
    print('PERFORMANCE_DECISION',{k:v for k,v in result.items() if k!='records'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['baseline','paired']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):(baseline if a.phase=='baseline' else paired)(a.run)
