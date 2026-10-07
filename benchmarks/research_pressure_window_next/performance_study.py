"""Two real accepted coupled steps; preserve the actual physical output cadence."""
from pathlib import Path
import argparse,cProfile,pstats,time,resource,gc
from .provenance import *
from .coupling_study import setup,frame
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def profile(run):
    import warp as wp
    run=Path(run);verify(run);folder=run/'cases/coupled-coarse';hist=GenerationStore(folder,read(folder/'identity.json')).history();observations=read(run/'S2/coupled-protocol.json')['observations_s'];records=[]
    register(run,'S5/profile-protocol.json',dict(status='passed_scoped',path='actual coupled coarse grid',initial_indices=[2,14],steps=2,max_candidates=1,max_extra_steps=4,profiler_is_not_speed_ratio=True,preserve_original_observation_frames=True))
    for index in (2,14):
        source=hist[index];trace=cProfile.Profile();trace.enable();tick=time.perf_counter();c,m,cfg=setup(run,'coarse',source['state']);wp.synchronize_device('cuda:0');build=time.perf_counter()-tick;cache=CachedProbes(m)
        out=run/'S5/profiles'/str(index);identity=dict(coupling=c.identity,origin_sha256=sha(source['folder']/'state.json'),numerical_source_sha256=source_files());store=GenerationStore(out,identity);store.save(c.state,[]);parts={'field_s':0.,'commit_s':0.};save=store.save
        def save_timed(*a,**kw):
            t=time.perf_counter()
            try:return save(*a,**kw)
            finally:parts['commit_s']+=time.perf_counter()-t
        store.save=save_timed
        def fields(st):
            t=time.perf_counter()
            try:return frame(c,cache,st)
            finally:parts['field_s']+=time.perf_counter()-t
        emitted=c.times[index+1] in observations;tick=time.perf_counter();row=advance_publish(c,store,[],frame_builder=fields if emitted else None);wp.synchronize_device('cuda:0');advance=time.perf_counter()-tick;trace.disable();stats=pstats.Stats(trace)
        calls=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda z:z[1][2],reverse=True)[:60]]
        memory=m.operator.memory_budget.report();before=dict(free=int(wp.get_device('cuda:0').free_memory),used=int(wp.get_mempool_used_mem_current('cuda:0')),threshold=int(wp.get_mempool_release_threshold('cuda:0')));collected=gc.collect();wp.synchronize_device('cuda:0');after=dict(free=int(wp.get_device('cuda:0').free_memory),used=int(wp.get_mempool_used_mem_current('cuda:0')))
        records.append(dict(initial_index=index,time_s=source['state'].time,dt_s=row['dt'],model_build_s=build,advance_s=advance,advance_excluding_io_fields_s=advance-sum(parts.values()),**parts,frame_written=emitted,profile=calls,linear_calls=len(row['linear_scaling']),true_residual_fraction=row['true_scaled_residual'],peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,memory=memory,zero_step_gc_diagnostic=dict(before=before,after=after,collected=collected)));print('COUPLED_PROFILE',index,build,advance,flush=True)
        del c,m,cache;gc.collect();wp.synchronize_device('cuda:0')
    write(run/'S5/profile.json',dict(status='passed_scoped',records=records,actual_steps=2,profiler_included=True,not_a_speedup_comparison=True,device_synchronized=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):profile(a.run)
