"""Two bounded accepted-step profiles on the latest vectorized baseline."""
from pathlib import Path
import argparse,time,cProfile,pstats,resource
import numpy as np
from .provenance import *
from .run import load_model
from . import config
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.runtime_rules import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes

def profile(run):
    import warp as wp
    run=Path(run);verify(run);hs=history(APP/'cases/final-full');lookup={round(x['state'].time,10):x for x in hs};times=read(run/'S1/time-decision.json')['times'];selected=set(np.linspace(0,len(times)-1,12,dtype=int))
    register(run,'S4/profile-protocol.json',dict(status='passed_scoped',states_s=[.5,1.1],actual_steps=2,vectorized_segment_metadata=True,display_frames_per_full_cycle=12,cProfile='discovery only; never speed ratio',timings='host wall-clock; coarse operator counters are not separate CUDA kernel timers',max_candidates=1,max_extra_steps=4))
    records=[]
    for ts in (.5,1.1):
        cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],times=times,space=read(run/'selected-space.json')['package'],field_cache=True)
        trace=cProfile.Profile();trace.enable();t=time.perf_counter();m,_=load_model(run,cfg);wp.synchronize_device('cuda:0');build=time.perf_counter()-t;t=time.perf_counter();cache=CachedProbes(m);probe_build=time.perf_counter()-t;item=lookup[ts];stepper=ValidatedAVF(m,cfg,item['state']);folder=run/'S4/profiles'/f'current-{ts}'
        identity=dict(model=m.identity,source_state_sha256=sha(item['folder']/'state.json'),numerical_sources=source_files());write(folder/'identity.json',identity);store=GenerationStore(folder,identity);t=time.perf_counter();store.save(stepper.state,[]);initial_io=time.perf_counter()-t
        parts={'field_s':0.,'commit_s':0.};originalsave=store.save
        def save(*args,**kwargs):
            t=time.perf_counter()
            try:return originalsave(*args,**kwargs)
            finally:parts['commit_s']+=time.perf_counter()-t
        store.save=save
        def frame(state):
            t=time.perf_counter()
            try:return cache.frame(state)
            finally:parts['field_s']+=time.perf_counter()-t
        idx=next(i for i,x in enumerate(times) if x>ts+1e-12);dt=times[idx]-ts;before=dict(m.operator.timings);counts=dict(m.operator.counts);t=time.perf_counter();row=advance_publish(stepper,store,[],dt,frame_builder=frame if idx in selected else None);wp.synchronize_device('cuda:0');advance=time.perf_counter()-t;trace.disable()
        raw=pstats.Stats(trace);stats=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(raw.stats.items(),key=lambda kv:kv[1][2],reverse=True)]
        record=dict(initial_s=ts,dt_s=dt,model_build_s=build,probe_build_s=probe_build,initial_io_s=initial_io,advance_s=advance,advance_excluding_field_commit_s=advance-sum(parts.values()),**parts,display_frame_written=idx in selected,operator_host_counter_delta={k:v-before.get(k,0) for k,v in m.operator.timings.items()},operator_call_delta={k:v-counts.get(k,0) for k,v in m.operator.counts.items()},total_s=build+probe_build+initial_io+advance,row=row,profile=stats[:70],peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,gpu_memory=m.operator.memory_budget.report())
        records.append(record);print('LATEST_PROFILE',ts,record['total_s'],row['krylov_iterations'],record['operator_host_counter_delta'],flush=True)
    actual=[row for n in range(2) for s in ('h','half') for row in read(run/f'cases/window{n}-{s}/ledger.json')];krylov=sum(x['krylov_iterations'] for x in actual)
    write(run/'S4/profile.json',dict(status='passed_scoped',records=records,actual_profile_steps=2,short_window_steps=len(actual),short_window_krylov=krylov,profiler_included=True,disjoint_host_partitions=True,operator_counters_nested_within_advance=True,not_a_speedup_comparison=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):profile(a.run)
