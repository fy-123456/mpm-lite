"""Independent startup and two-step warm measurements on the current q7 path."""
from pathlib import Path
import argparse,time,cProfile,pstats,resource
import numpy as np
from .provenance import APP,read,write,sha,register,source_files,verify,serial_lock
from . import config
from .run import load_model
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def settings(run):
    run=Path(run);choice=read(run/'selected-space.json')
    return config.make(read(run/'input-lock.json')['energy_scale_J'],space=choice['package'],mass_order=7,full_order=7,
        rule_policy='full_only',shared_reduction=True,coarse_instrumentation=True,field_cache=True)


def profile(run,repeat):
    import warp as wp
    run=Path(run)
    if not (run/'S4/profile-protocol.json').exists():register(run,'S4/profile-protocol.json',dict(repeats=2,states=[.1,1.1],steps_per_state=2,
        material='q7 before final main qualification',startup='fresh process each repeat; cached Warp kernels',candidate_limit=1,
        minimum_target_gain=.1,non_target_regression=.05,memory_ratio=1.1,coarse_timing_already_enabled=True))
    cfg=settings(run);trace=[]
    import scipy.sparse._csr as csr,scipy.sparse._csc as csc,hashlib,inspect
    original_csr=csr._csr_base.tocsc;original_csc=csc._csc_base.tocsr
    def wrapper(original,kind):
        def traced(matrix,*args,**kwargs):
            t0=time.perf_counter();out=original(matrix,*args,**kwargs);elapsed=time.perf_counter()-t0
            tick=time.perf_counter();h=hashlib.sha256()
            h.update(str((matrix.shape,str(matrix.dtype),kind)).encode())
            for array in (matrix.data,matrix.indices,matrix.indptr):h.update(memoryview(array).cast('B'))
            trace.append(dict(kind=kind,shape=list(matrix.shape),nnz=int(matrix.nnz),content=h.hexdigest(),seconds=elapsed,hash_seconds=time.perf_counter()-tick,caller=[f'{f.filename}:{f.lineno}:{f.function}' for f in inspect.stack()[1:4]]))
            return out
        return traced
    csr._csr_base.tocsc=wrapper(original_csr,'csr_to_csc');csc._csc_base.tocsr=wrapper(original_csc,'csc_to_csr')
    prof=cProfile.Profile();prof.enable();t=time.perf_counter();m,_=load_model(run,cfg);wp.synchronize_device(m.device);model_s=time.perf_counter()-t
    t=time.perf_counter();cache=CachedProbes(m,tuple(cfg['probe_shape']));wp.synchronize_device(m.device);field_build_s=time.perf_counter()-t;prof.disable()
    csr._csr_base.tocsc=original_csr;csc._csc_base.tocsr=original_csc
    stats=pstats.Stats(prof);rows=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:45]]
    src=run/'cases/selected-prefix' if (run/'cases/selected-prefix/identity.json').exists() else APP/'cases/final-q7-dt0125';lookup={round(x['state'].time,10):x for x in GenerationStore(src,read(src/'identity.json')).history()};warm=[]
    for ts in (.1,1.1):
        item=lookup[ts];m.validate(item['state'],material=True);c=ValidatedAVF(m,cfg,item['state']);folder=run/'S4/profiles'/f'baseline-{repeat}-{ts}'
        identity=dict(source=source_files(),source_state_sha256=sha(item['folder']/'state.json'),model=m.identity,protocol=cfg,label=f'baseline-{repeat}-{ts}')
        store=GenerationStore(folder,identity);t=time.perf_counter();store.save(c.state,[]);io=time.perf_counter()-t;advance=0.;records=[]
        for _ in range(2):
            t=time.perf_counter();records.append(c.step(.0125));wp.synchronize_device(m.device);advance+=time.perf_counter()-t
            t=time.perf_counter();store.save(c.state,records);io+=time.perf_counter()-t
        t=time.perf_counter();frame=cache.frame(c.state);wp.synchronize_device(m.device);fields=time.perf_counter()-t
        np.savez_compressed(folder/'result.npz',q=c.state.q,v=c.state.velocity,predictor=c.state.predictor,**frame)
        warm.append(dict(time_s=ts,advance_s=advance,io_s=io,field_s=fields,total_s=advance+io+fields,rows=records))
    report=dict(conversion_trace=trace,trace_hash_seconds=sum(x['hash_seconds'] for x in trace),repeat=repeat,model_build_s=model_s,field_build_s=field_build_s,build_s=model_s+field_build_s,python_profile=rows,warm=warm,
        peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,gpu_memory=m.operator.memory_budget.report(),
        timing='host elapsed with synchronization at measurement boundaries; nested cProfile costs cannot be summed',new_process=True)
    write(run/'S4/profiles'/f'profile-{repeat}.json',report);print('STARTUP',repeat,report['build_s'],rows[:10],flush=True)


def decide(run):
    run=Path(run);records=[read(run/'S4/profiles'/f'profile-{i}.json') for i in (0,1)]
    write(run/'S4/startup-profile.json',dict(status='measured',records=records,median_build_s=float(np.median([r['build_s'] for r in records]))))
    write(run/'S4/warm-profile.json',dict(status='measured',records=[dict(repeat=r['repeat'],warm=r['warm']) for r in records],rule='q7'))
    duplicate=[]
    for record in records:
        seen=set();repeated=[]
        for item in record['conversion_trace']:
            if item['content'] in seen:repeated.append(item)
            seen.add(item['content'])
        duplicate.append(dict(repeat=record['repeat'],repeated=repeated,optimistic_savings_s=sum(x['seconds'] for x in repeated),
            instrumented_build_s=record['build_s'],hash_overhead_s=record['trace_hash_seconds']))
    upper=max(x['optimistic_savings_s']/max(x['instrumented_build_s']-x['hash_overhead_s'],1e-9) for x in duplicate)
    write(run/'S4/reuse-opportunity.json',dict(status='measured_content_bound',records=duplicate,optimistic_repeated_conversion_fraction=upper,
        note='hash tracing adds measured overhead; these are diagnosis timings, not a speedup claim'))
    if upper>=.1:
        write(run/'S4/candidate-required.json',dict(reason='measured duplicate covers 10%; review call sites and implement at most one bounded immutable cache',upper_fraction=upper))
        print('PERFORMANCE_REVIEW',upper,flush=True);return
    write(run/'S4/performance-decision.json',dict(status='retain_current_implementation',warm_adopted=True,shared_reduction=True,uncompressed_archive=True,field_cache=True,candidate_count=0,tangent_cache=False,
        reason='content-bound duplicate-conversion upper bound below 10%; no justified new immutable cache',optimistic_gain_upper_fraction=upper,whole_cycle_gain_claim=False))
    write(run/'S4/operator-equivalence.json',dict(status='unchanged_backend',parent_kernel_sources_unchanged=True,new_space_operator_evidence='S1/candidates/global-snapshot6/operator-audit.json',no_candidate=True))
    write(run/'S4/transaction-check.json',dict(status='pending_final_check',no_performance_change=True,window_evidence='S3/window-runtime-check.json'))
    write(run/'S4/final-source-refresh-list.json',dict(numeric_source_sha256=source_files(),performance_numeric_changes=False))
    print('PERFORMANCE_RETAIN',upper,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','decide']);p.add_argument('--repeat',type=int,choices=[0,1],default=0);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='profile':profile(a.run,a.repeat)
        else:decide(a.run)
