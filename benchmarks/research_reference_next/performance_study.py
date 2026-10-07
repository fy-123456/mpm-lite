"""Same-state alternating two-step profiles; field work is counted separately."""
from pathlib import Path
import argparse
import copy
import gc
import time
import numpy as np
from .provenance import APP,read,write,sha,digest,register,verify,serial_lock,utc
from .run import load_model,make_stepper
from . import config
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.run import probe_frame


def setup(run):
    run=Path(run);lock=verify(run);cert=run/'Q4/qualification.json'
    cfg=config.make(lock['energy_scale_J'],rule_policy='q5_with_full_retry',
        qualification=dict(path=str(cert.resolve()),sha256=sha(cert)))
    model,_=load_model(run,cfg);control=make_stepper(run,cfg,model,model.rest())
    folder=APP/'cases/full-q7-dt0125';hist=GenerationStore(folder,read(folder/'identity.json')).history()
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
    folder=Path(run)/'Q4/performance/profiles'/label
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
    register(run,'Q4/performance/profile-protocol.json',dict(times=[.1,1.1],steps=2,repeats=3,
        selected_policy='q5_with_full_retry',material_order=5,mass_order=5,
        pairing='same explicit physical q/v branch; identical instrumentation',
        no_nested_timing_addition=True,display_cost='one 33x7x7 field per two-step microtrial, not daily cadence',
        final_daily_cadence='12 frames/128 steps; account cache build separately'))
    cfg,m,f,states=setup(run);records=[]
    from engine.aniso_phase1.research_post_release.fields import CachedProbes
    cache=CachedProbes(m,tuple(cfg['probe_shape']))
    for t in [.1,1.1]:
        state,origin=branch(states[t],m,f)
        result=trial(run,f'baseline-{t}',cfg,m,f,state,cache.frame)
        record={k:v for k,v in result.items() if k not in ('q','v','predictor','frame','rows')}
        record.update(time_s=t,initial=origin);records.append(record)
    write(run/'Q4/performance/baseline-cost.json',dict(records=records,decision='profile already cached fields; choose at most one actually measured material or mapping hotspot'))
    print('BASELINE_COST',records,flush=True)


def paired(run):
    from engine.aniso_phase1.research_post_release.fields import CachedProbes
    run=Path(run);register(run,'Q4/performance/paired-protocol.json',dict(windows=[.1,1.1],steps=2,repeats=3,
        baseline_field_cache=True,change='skip unused spectral output storage for force-only queries',minimum_gain=.10,
        tangent_path='unchanged exact tangent',warmup_separate=True,keep_original_if_below_gate=True))
    cfg,m,f,states=setup(run);newcfg=copy.deepcopy(cfg);newcfg['implementation']['force_only_responses']=True
    fast,_=load_model(run,newcfg);cache=CachedProbes(m,tuple(cfg['probe_shape']));records=[];checks=[]
    warm=time.perf_counter()
    for t in [.1,1.1]:
        q=states[t]['state'].q;d=np.zeros_like(q);d[m.free[0],0]=1.
        a=m.evaluate(q);b=fast.evaluate(q)
        error=max(abs(a['U']-b['U']),float(np.max(abs(a['force']-b['force']))))
        tangent=float(np.max(abs(m.evaluate(q,d)['tangent_action']-fast.evaluate(q,d)['tangent_action'])))
        if error>1e-8 or tangent>1e-7:raise ValueError('force-only/tangent differs')
        checks.append(dict(time=t,response_max=error,tangent_max=tangent))
    warmup=time.perf_counter()-warm
    for t in [.1,1.1]:
        state,origin=branch(states[t],m,f)
        for repeat in range(3):
            samples={}
            for enabled in ([False,True] if repeat%2==0 else [True,False]):
                samples[enabled]=trial(run,f'force-{t}-{repeat}-{int(enabled)}',newcfg if enabled else cfg,fast if enabled else m,f,state,cache.frame)
            a,b=samples[False],samples[True]
            error=max(float(np.max(abs(a[k]-b[k]))) for k in ('q','v','predictor'))
            field=max(float(np.max(abs(a['frame'][k]-b['frame'][k]))) for k in a['frame'])
            if error>1e-9 or field>1e-7:raise ValueError('optimization changed physical trajectory')
            records.append(dict(time=t,repeat=repeat,state_error=error,field_error=field,
                original={k:v for k,v in a.items() if k not in ('q','v','predictor','frame','rows')},
                force_only={k:v for k,v in b.items() if k not in ('q','v','predictor','frame','rows')}))
    old=float(np.median([x['original']['total_seconds'] for x in records]));new=float(np.median([x['force_only']['total_seconds'] for x in records]));gain=1-new/old
    # Extra warmup includes compilation and derivative checks; not presented as
    # the incremental build overhead or hidden within a speedup claim.
    result=dict(status='passed_scoped',records=records,operator_checks=checks,validation_and_warmup_seconds=warmup,
        median_original_seconds=old,median_force_only_seconds=new,warm_microtrial_gain=gain,
        selected_force_only=bool(gain>=.10),field_cache=True,tangent_cache=False,preconditioner='original',
        memory_saved_per_response_bytes=m.operator.count*(9+3)*8,
        selection='require >=10% warm representative task gain; cold compilation reported separately; no whole-cycle speedup claim')
    write(run/'Q4/performance-decision.json',result);print('PERFORMANCE',{k:v for k,v in result.items() if k not in ('records','operator_checks')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['baseline','paired']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; fork before running studies')
        (baseline if a.phase=='baseline' else paired)(a.run)
