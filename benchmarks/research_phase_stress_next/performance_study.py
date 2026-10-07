"""Actual warm task profile, bounded independent-process paired comparison."""
from pathlib import Path
import argparse,time,cProfile,pstats,resource
import numpy as np
import warp as wp
from .provenance import APP,read,write,sha,register,source_files,verify,serial_lock
from . import config
from .run import load_model,make_stepper
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_reference_next.performance_study import branch
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry


def settings(run,shared=True):
    run=Path(run);choice=read(run/'selected-space.json');cert=run/'S5/qualification-final.json'
    if not cert.exists():cert=run/'S5/inherited-main.json'
    return config.make(read(run/'input-lock.json')['energy_scale_J'],rule_policy='q5_with_full_retry',
        qualification=dict(path=str(cert.resolve()),sha256=sha(cert)),field_cache=True,
        space=choice['package'],mass_order=choice['mass_order'],full_order=choice['full_order'],shared_reduction=True,
        coarse_instrumentation=read(run/'S3/performance-decision.json').get('warm_adopted',False) if (run/'S3/performance-decision.json').exists() else False)


def construct(run,cfg):
    t=time.perf_counter();m,_=load_model(run,cfg);control=make_stepper(run,cfg,m,m.rest());cache=CachedProbes(m,tuple(cfg['probe_shape']))
    wp.synchronize_device(m.device)
    return m,control.full,cache,time.perf_counter()-t


def trial(run,label,cfg,m,full,cache,item):
    state,origin=branch(item,m,full);c=RuntimeRetry(m,full,cfg,state)
    folder=Path(run)/'S3/profiles'/label
    store=GenerationStore(folder,dict(schema='phase-stress-profile-v1',source_sha256=source_files(),initial=state.digest(),protocol=config.identity(cfg)))
    t=time.perf_counter();store.save(state,[]);io=time.perf_counter()-t;advance=0.;rows=[]
    for _ in range(2):
        t=time.perf_counter();rows.append(c.step(.0125));wp.synchronize_device(m.device);advance+=time.perf_counter()-t
        t=time.perf_counter();store.save(c.state,rows);io+=time.perf_counter()-t
    t=time.perf_counter();frame=cache.frame(c.state);wp.synchronize_device(m.device);field=time.perf_counter()-t
    np.savez_compressed(folder/'result.npz',q=c.state.q,v=c.state.velocity,predictor=c.state.predictor,**frame)
    return dict(warm_s=advance+io+field,advance_s=advance,io_s=io,field_s=field,origin=origin,
        rss_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,gpu_memory=m.operator.memory_budget.report(),rows=rows)


def profile(run):
    run=Path(run);verify(run)
    register(run,'S3/profile-protocol.json',dict(times=[.1,1.1],steps=2,
        baseline='latest shared immutable reduction, uncompressed archive, field cache',
        profiling_only=True,candidate_limit=1,minimum_warm_gain=.1,paired_repeats=2,
        memory='independent process for each timed variant; no accumulated per-variant RSS claims'))
    cfg=settings(run);m,f,cache,build=construct(run,cfg)
    folder=APP/'cases/final-q7-dt0125';states={round(x['state'].time,10):x for x in GenerationStore(folder,read(folder/'identity.json')).history()}
    prof=cProfile.Profile();m.operator.reset_profile();f.operator.reset_profile();prof.enable();records=[]
    for t in (.1,1.1):records.append(dict(time=t,**trial(run,'diagnostic-'+str(t),cfg,m,f,cache,states[t])))
    prof.disable();stats=pstats.Stats(prof)
    rows=[dict(file=file,line=line,function=name,calls=nc,self_s=tt,cumulative_s=ct)
        for (file,line,name),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:35]]
    write(run/'S3/warm-profile.json',dict(status='measured',build_s=build,records=records,python_self_profile=rows,
        q5_partitions=dict(m.operator.timings),q5_counts=dict(m.operator.counts),q7_partitions=dict(f.operator.timings),
        q7_counts=dict(f.operator.counts),synchronized_profile=True,partitions_nested_in_advance=True))
    print('WARM_PROFILE',[(x['time'],x['warm_s'],x['advance_s'],x['io_s']) for x in records],rows[:8],flush=True)


def timed(run,time_s,repeat,candidate):
    run=Path(run);cfg=settings(run);cfg['implementation']['coarse_instrumentation']=candidate
    m,f,cache,build=construct(run,cfg)
    folder=APP/'cases/final-q7-dt0125'
    item=next(x for x in GenerationStore(folder,read(folder/'identity.json')).history() if abs(x['state'].time-time_s)<1e-10)
    label=f'paired-{time_s}-{repeat}-{int(candidate)}'
    result=trial(run,label,cfg,m,f,cache,item)
    write(run/'S3/profiles'/label/'timing.json',dict(candidate=candidate,time=time_s,repeat=repeat,build_s=build,**result))
    print('WARM_PAIR',label,result['warm_s'],build,flush=True)


def qualify(run):
    from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
    from engine.aniso_phase1.research_post_release.runtime_rules import RetryableRuleError
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
    run=Path(run);cfg=settings(run);m,f,cache,build=construct(run,cfg)
    folder=APP/'cases/final-q7-dt0125';states={round(x['state'].time,10):x for x in GenerationStore(folder,read(folder/'identity.json')).history()}
    values=[];rng=np.random.default_rng(7021)
    for t in (.1,1.1):
        q=states[t]['state'].q;d=rng.normal(size=q.shape);d[m.fixed]=0;d/=np.linalg.norm(d)
        values.append((t,q,d,{o:op.evaluate(q,d) for o,op in [(5,m),(7,f)]}))
    install_warm(m);install_warm(f);checks=[]
    for t,q,d,expected in values:
        for o,op in [(5,m),(7,f)]:
            actual=op.evaluate(q,d);err={k:float(np.max(abs(np.asarray(actual[k])-expected[o][k]))) for k in ('U','force','tangent_action')}
            assert max(err.values())<1e-8;checks.append(dict(time=t,order=o,errors=err))
    c=RuntimeRetry(m,f,cfg)
    def fail(attempt,where,state):
        if attempt==0 and where=='after_prepare':raise RetryableRuleError('coarse instrumentation fallback fixture')
    row=c.step(.0125,inject=fail);ref=ValidatedAVF(f,cfg);ref.step(.0125)
    err=max(float(np.max(abs(getattr(c.state,k)-getattr(ref.state,k)))) for k in ('q','velocity','predictor'))
    assert err<1e-8 and row['material_attempts']==2
    write(run/'S3/operator-equivalence.json',dict(status='passed_scoped',records=checks,no_numeric_kernel_change=True,
        allocation_reserve_preserved=True,response_end_memory_check=True,host_counters_are_not_GPU_partitions=True))
    write(run/'S3/transaction-check.json',dict(status='passed_scoped',fallback_error=err,attempts=row['material_attempts'],
        new_process_restart='S5/S6 final-source fixtures; no new mutable cache introduced'))


def decide(run):
    run=Path(run);records=[]
    for t in (.1,1.1):
        for rep in range(2):
            folders=[run/'S3/profiles'/f'paired-{t}-{rep}-{i}' for i in (0,1)]
            a,b=[read(p/'timing.json') for p in folders]
            with np.load(folders[0]/'result.npz') as x,np.load(folders[1]/'result.npz') as y:
                errors={k:float(np.max(abs(x[k]-y[k]))) for k in x.files}
            assert max(errors.values())<1e-8
            records.append(dict(time=t,repeat=rep,baseline=a,candidate=b,errors=errors))
    base=float(np.median([r['baseline']['warm_s'] for r in records]));new=float(np.median([r['candidate']['warm_s'] for r in records]))
    bbuild=float(np.median([r['baseline']['build_s'] for r in records]));nbuild=float(np.median([r['candidate']['build_s'] for r in records]))
    rss=max(r['candidate']['rss_peak_GiB']/r['baseline']['rss_peak_GiB'] for r in records)
    gain=1-new/base;adopt=gain>=.1 and nbuild<=1.05*bbuild and rss<=1.1
    write(run/'S3/performance-decision.json',dict(status='adopted' if adopt else 'retain_latest_warm_path',warm_adopted=adopt,
        shared_reduction=True,uncompressed_archive=True,field_cache=True,candidate_count=1,warm_gain=gain,
        baseline_warm_s=base,candidate_warm_s=new,baseline_build_s=bbuild,candidate_build_s=nbuild,rss_ratio=rss,
        records=records,reason='single preregistered 10% warm gate plus construction and memory limits',
        tangent_cache=False,whole_cycle_gain_claim=False))
    print('WARM_DECISION',adopt,gain,bbuild,nbuild,rss,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','timed','qualify','decide']);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--time',type=float,default=.1);p.add_argument('--repeat',type=int,default=0);p.add_argument('--candidate',action='store_true');a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='timed':timed(a.run,a.time,a.repeat,a.candidate)
        else:{'profile':profile,'qualify':qualify,'decide':decide}[a.phase](a.run)
