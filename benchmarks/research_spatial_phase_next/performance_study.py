"""One optimization: authenticated exact space arrays instead of rebuilding R3."""
from pathlib import Path
import argparse,time,copy,gc
import numpy as np
from .provenance import read,write,sha,register,verify,serial_lock,source_files
from .run import load_model,make_stepper
from . import config
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_reference_next.performance_study import branch
from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_spatial_phase_next.performance import save as save_cache,load as load_cache


def construct(run,cfg):
    begin=time.perf_counter();m,_=load_model(run,cfg);c=make_stepper(run,cfg,m,m.rest());cache=CachedProbes(m,tuple(cfg['probe_shape']))
    return m,c.full,cache,time.perf_counter()-begin


def trial(run,label,cfg,m,full,cache,item):
    state,origin=branch(item,m,full);c=RuntimeRetry(m,full,cfg,state);rows=[];timings={};calls={};original={}
    for name in ('evaluate','endpoint','wrench'):
        original[name]=getattr(m,name)
        def wrap(*args,_name=name,**kwargs):
            started=time.perf_counter()
            try:return original[_name](*args,**kwargs)
            finally:timings[_name]=timings.get(_name,0.)+time.perf_counter()-started;calls[_name]=calls.get(_name,0)+1
        setattr(m,name,wrap)
    m.operator.reset_profile();folder=Path(run)/'N5/profiles'/label
    store=GenerationStore(folder,dict(schema='spatial-phase-performance-v1',config=config.identity(cfg),source=source_files(),initial=state.digest()))
    t=time.perf_counter();store.save(state,[]);io=time.perf_counter()-t;advance=0.
    try:
        for _ in range(2):
            t=time.perf_counter();row=c.step(.0125);advance+=time.perf_counter()-t;rows.append(row)
            t=time.perf_counter();store.save(c.state,rows);io+=time.perf_counter()-t
        t=time.perf_counter();frame=cache.frame(c.state);field=time.perf_counter()-t
    finally:
        for name,value in original.items():setattr(m,name,value)
    return dict(advance_seconds=advance,field_seconds=field,checkpoint_seconds=io,warm_seconds=advance+field+io,
        nested_model_seconds=timings,model_calls=calls,nested_gpu_seconds=dict(m.operator.timings),gpu_calls=dict(m.operator.counts),
        initial=origin,q=c.state.q,v=c.state.velocity,predictor=c.state.predictor,frame=frame)


def study(run):
    run=Path(run);verify(run);choice=read(run/'selected-space.json')
    if choice['selected']=='original144':raise ValueError('this bounded optimization addresses measured new-space reconstruction only')
    register(run,'N5/profile-protocol.json',dict(windows=[.1,1.1],steps=2,paired_repeats=2,
        profile='disjoint construction/advance/field/checkpoint; nested GPU counters not added to totals',
        maximum_changes=1,candidate='authenticated exact array archive',minimum_launch_gain=.1,
        counts='construction once per variant; representative independent-launch cost uses that measured build once per warm task',
        cold_scope='model creation with existing Warp kernel cache; not compiler cold start'))
    cert=run/'N3/qualification.json';cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],rule_policy='q5_with_full_retry',qualification=dict(path=str(cert.resolve()),sha256=sha(cert)),
        field_cache=True,space=choice['package'],mass_order=choice['mass_order'],full_order=choice['full_order'])
    source=run/'cases/common-full-prefix';states={round(x['state'].time,10):x for x in GenerationStore(source,read(source/'identity.json')).history()}
    m,full,cache,slow_build=construct(run,cfg);baseline=[]
    for t in [.1,1.1]:
        result=trial(run,f'baseline-{t}',cfg,m,full,cache,states[t]);baseline.append(dict(time=t,**{k:v for k,v in result.items() if k not in ('q','v','predictor','frame')}))
    write(run/'N5/baseline-cost.json',dict(construction_seconds=slow_build,records=baseline,
        chosen_hotspot='repeated R3 reconstruction dominates independent launches; preserve material and AVF kernels'))
    if slow_build<10*max(x['warm_seconds'] for x in baseline):raise ValueError('expected reconstruction hotspot not supported')
    started=time.perf_counter();entry=save_cache(m.reduction,run/'N5/space-cache',choice['package']);creation=time.perf_counter()-started
    cached=load_cache(entry,choice['package']);equal={}
    for k in ('M','K','P','offset','original_mass','original_stiffness'):equal[k]=bool(np.array_equal(getattr(cached,k),getattr(m.reduction,k)))
    if not all(equal.values()) or cached.signature!=m.reduction.signature:raise ValueError('archive changes mathematical operators')
    fastcfg=copy.deepcopy(cfg);fastcfg['physical_space']['cache']=entry
    fm,ff,fcache,fast_build=construct(run,fastcfg);checks=[]
    for t in [.1,1.1]:
        q=states[t]['state'].q;d=np.zeros_like(q);d[m.free[0],0]=1.
        a,b=m.evaluate(q,d),fm.evaluate(q,d)
        errs={k:float(np.max(abs(np.asarray(a[k])-np.asarray(b[k])))) for k in ('U','force','tangent_action')}
        if max(errs.values())>1e-9:raise ValueError('archive changes operator response')
        checks.append(dict(time=t,errors=errs))
    records=[]
    for t in [.1,1.1]:
        for repeat in range(2):
            pair={}
            for enabled in ([False,True] if repeat%2==0 else [True,False]):
                pair[enabled]=trial(run,f'pair-{t}-{repeat}-{int(enabled)}',fastcfg if enabled else cfg,fm if enabled else m,ff if enabled else full,fcache if enabled else cache,states[t])
            a,b=pair[False],pair[True];err=max(float(np.max(abs(a[k]-b[k]))) for k in ('q','v','predictor'))
            field=max(float(np.max(abs(a['frame'][k]-b['frame'][k]))) for k in a['frame'])
            if err>1e-9 or field>1e-7:raise ValueError('archive changes trajectory or fields')
            record=dict(time=t,repeat=repeat,state_error=err,field_error=field)
            for label,result in [('original',a),('cached',b)]:record[label]={k:v for k,v in result.items() if k not in ('q','v','predictor','frame')}
            records.append(record)
    oldwarm=float(np.median([x['original']['warm_seconds'] for x in records]));newwarm=float(np.median([x['cached']['warm_seconds'] for x in records]));gain=1-(fast_build+newwarm)/(slow_build+oldwarm)
    adopted=gain>=.1;cache_bytes=sum(p.stat().st_size for p in (run/'N5/space-cache').iterdir())
    decision=dict(status='passed_scoped',selected_space_cache=adopted,change='authenticated bit-preserved space/operator archive',
        source_sha256=source_files(),old_construction_seconds=slow_build,new_construction_seconds=fast_build,cache_write_seconds=creation,cache_bytes=cache_bytes,
        warm_original_median_s=oldwarm,warm_cached_median_s=newwarm,independent_launch_gain=gain,
        cache_generation_with_one_slow_build_seconds=slow_build+creation,conservative_amortization_launches=(slow_build+creation)/max(slow_build-fast_build,1e-12),
        matrices_bit_identical=equal,operator_checks=checks,records=records,cache_entry=entry,
        field_cache=True,tangent_cache=False,force_only=False,preconditioner='original',
        scope='construction measured once per variant; two warm windows, two paired repeats each; no whole-cycle speedup claim')
    write(run/'N5/performance-decision.json',decision)
    if adopted:
        choice['package']['cache']=entry;choice['performance']='authenticated exact array cache';write(run/'selected-space.json',choice)
    print('SPACE_CACHE',slow_build,fast_build,'launch gain',gain,'selected',adopted,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release')
        study(a.run)
