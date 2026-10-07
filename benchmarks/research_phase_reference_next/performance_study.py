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
        rule_policy='full_only',shared_reduction=True,coarse_instrumentation=True,field_cache=True,reuse_transpose_buffers=True)


def profile(run,repeat):
    import warp as wp
    run=Path(run)
    if not (run/'S3/profile-protocol.json').exists():register(run,'S3/profile-protocol.json',dict(repeats=2,states=[.1,1.1],steps_per_state=2,
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
    src=run/'cases/selected-prefix' if (run/'cases/selected-prefix/identity.json').exists() else APP/'cases/final-full';lookup={round(x['state'].time,10):x for x in GenerationStore(src,read(src/'identity.json')).history()};warm=[];warm_prof=cProfile.Profile()
    for ts in (.1,1.1):
        item=lookup[ts];m.validate(item['state'],material=True);c=ValidatedAVF(m,cfg,item['state']);folder=run/'S3/profiles'/f'baseline-{repeat}-{ts}'
        identity=dict(source=source_files(),source_state_sha256=sha(item['folder']/'state.json'),model=m.identity,protocol=cfg,label=f'baseline-{repeat}-{ts}')
        store=GenerationStore(folder,identity);t=time.perf_counter();store.save(c.state,[]);io=time.perf_counter()-t;advance=0.;records=[]
        warm_prof.enable()
        for _ in range(2):
            t=time.perf_counter();records.append(c.step(.0125));wp.synchronize_device(m.device);advance+=time.perf_counter()-t
            t=time.perf_counter();store.save(c.state,records);io+=time.perf_counter()-t
        t=time.perf_counter();frame=cache.frame(c.state);wp.synchronize_device(m.device);fields=time.perf_counter()-t
        t=time.perf_counter();np.savez_compressed(folder/'result.npz',q=c.state.q,v=c.state.velocity,predictor=c.state.predictor,**frame);io+=time.perf_counter()-t;warm_prof.disable()
        warm.append(dict(time_s=ts,advance_s=advance,io_s=io,field_s=fields,total_s=advance+io+fields,rows=records))
    warm_stats=pstats.Stats(warm_prof);warm_rows=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(warm_stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:35]]
    report=dict(actual_model=type(m).__module__+'.'+type(m).__name__,reuse_transpose_buffers=cfg['implementation']['reuse_transpose_buffers'],warm_python_profile=warm_rows,conversion_trace=trace,trace_hash_seconds=sum(x['hash_seconds'] for x in trace),repeat=repeat,model_build_s=model_s,field_build_s=field_build_s,build_s=model_s+field_build_s,python_profile=rows,warm=warm,
        peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,gpu_memory=m.operator.memory_budget.report(),
        timing='host elapsed with synchronization at measurement boundaries; nested cProfile costs cannot be summed',new_process=True)
    write(run/'S3/profiles'/f'profile-{repeat}.json',report);print('STARTUP',repeat,report['build_s'],rows[:10],flush=True)


def decide(run):
    run=Path(run);records=[read(run/'S3/profiles'/f'profile-{i}.json') for i in (0,1)]
    write(run/'S3/startup-profile.json',dict(status='measured',records=records,median_build_s=float(np.median([r['build_s'] for r in records]))))
    write(run/'S3/warm-profile.json',dict(status='measured',records=[dict(repeat=r['repeat'],warm=r['warm']) for r in records],rule='q7'))
    duplicate=[]
    for record in records:
        seen=set();repeated=[]
        for item in record['conversion_trace']:
            if item['content'] in seen:repeated.append(item)
            seen.add(item['content'])
        duplicate.append(dict(repeat=record['repeat'],repeated=repeated,optimistic_savings_s=sum(x['seconds'] for x in repeated),
            instrumented_build_s=record['build_s'],hash_overhead_s=record['trace_hash_seconds']))
    upper=max(x['optimistic_savings_s']/max(x['instrumented_build_s']-x['hash_overhead_s'],1e-9) for x in duplicate)
    write(run/'S3/reuse-opportunity.json',dict(status='measured_content_bound',records=duplicate,optimistic_repeated_conversion_fraction=upper,
        note='hash tracing adds measured overhead; these are diagnosis timings, not a speedup claim'))
    share={key:float(np.median([w[key]/w['total_s'] for r in records for w in r['warm']])) for key in ('advance_s','io_s','field_s')}
    write(run/'S3/current-startup-profile.json',dict(status='measured',records=records,actual_reuse=True))
    write(run/'S3/current-warm-profile.json',dict(status='measured',shares=share,records=[dict(repeat=r['repeat'],warm=r['warm'],profile=r['warm_python_profile']) for r in records]))
    write(run/'S3/candidate-protocol.json',dict(status='review_required',duplicate_conversion_upper_fraction=upper,warm_shares=share,candidate_limit=1,minimum_gain=.1))
    print('PERFORMANCE_MEASURED',upper,share,flush=True)

def retain(run):
    run=Path(run);records=[read(run/f'S3/profiles/profile-{i}.json') for i in (0,1)]
    costs=[]
    for r in records:
        total=sum(x['total_s'] for x in r['warm']);json_self=sum(x['self_s'] for x in r['warm_python_profile'] if '/json/' in x['file']);zip_self=sum(x['self_s'] for x in r['warm_python_profile'] if 'compress' in x['function'])
        costs.append(dict(repeat=r['repeat'],total_s=total,json_self_s=json_self,compression_self_s=zip_self,json_fraction=json_self/total,compression_fraction=zip_self/total))
    upper=read(run/'S3/reuse-opportunity.json')['optimistic_repeated_conversion_fraction'];warm=read(run/'S3/current-warm-profile.json')
    if upper>=.1 or max(x['json_fraction'] for x in costs)>=.1:raise ValueError('measured opportunity requires candidate review')
    write(run/'S3/candidate-protocol.json',dict(status='condition_not_triggered',candidate_count=0,minimum_gain=.1,duplicate_conversion_upper_fraction=upper,
        component_costs=costs,field_share=warm['shares']['field_s'],reason='each identified isolated removal opportunity below 10% even if its measured work vanished; serialization cannot remove mandatory validation/fsync',
        material_hotspot='advance dominates; Warp.copy timing includes synchronization of GPU work, not proof of removable transfer; no content-equal redundant material operation identified',
        no_global_speedup_bound=True,tangent_cache=False))
    write(run/'S3/performance-decision.json',dict(status='retain_current_implementation',reuse_transpose_buffers=True,shared_reduction=True,coarse_instrumentation=True,field_cache=True,tangent_cache=False,candidate_count=0,whole_cycle_gain_claim=False))
    write(run/'S3/operator-equivalence.json',dict(status='unchanged_solid_backend',no_performance_candidate=True,compatibility='S0/compatibility.json',parent_operator_audit_sha256=sha(APP/'S1/candidates/global-snapshot6/operator-audit.json'),final_scene_pending=True))
    write(run/'S3/paired-performance.json',dict(status='condition_not_triggered',baseline_processes=2,candidate_processes=0,reason='no candidate with justified 10% net target gain',speedup_claim=False))
    write(run/'S3/qualification-refresh-map.json',dict(solid_numeric_algorithm_unchanged=True,new_pressure_solver=True,numerical_source_sha256=source_files(),scope_certificates_reissued_in=['S5','S6'],old_cases_unchanged=True))
    write(run/'S3/transaction-check.json',dict(status='pending_final_source',no_performance_change=True,evidence_expected=['S5/window-runtime-final.json','S6/cache-retry-check.json']))
    print('RETAIN_CURRENT_PERFORMANCE',costs,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','decide','retain']);p.add_argument('--repeat',type=int,choices=[0,1],default=0);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='profile':profile(a.run,a.repeat)
        else:globals()[a.phase](a.run)
