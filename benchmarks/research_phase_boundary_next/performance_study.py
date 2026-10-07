"""Bounded current-solid cost measurement, with no unsupported speedup claim."""
from pathlib import Path
import argparse,time,cProfile,pstats
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
    run=Path(run);verify(run);source=APP/'cases/final-full';hs=history(source);lookup={round(x['state'].time,10):x for x in hs};times=read(run/'S1/time-decision.json')['times']
    register(run,'S5/profile-protocol.json',dict(states_s=[0.,1.1],actual_steps=2,scope='current formal solid; pressure coupling gate not bypassed',include_build_fields_commit=True,cProfile='cost discovery only; never a paired speed ratio',max_candidates=1))
    records=[];profiles=[]
    for ts in (0.,1.1):
        cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],times=times,space=read(run/'selected-space.json')['package'],field_cache=True,shared_reduction=True,coarse_instrumentation=True)
        trace=cProfile.Profile();trace.enable();t=time.perf_counter();m,_=load_model(run,cfg);cache=CachedProbes(m);build=time.perf_counter()-t;item=lookup[ts];stepper=ValidatedAVF(m,cfg,item['state']);folder=run/'S5/profiles'/f'current-{ts}'
        identity=dict(model=m.identity,source_state_sha256=sha(item['folder']/'state.json'),numerical_sources=source_files());write(folder/'identity.json',identity);store=GenerationStore(folder,identity);t=time.perf_counter();store.save(stepper.state,[]);initial_io=time.perf_counter()-t
        h=next(x-ts for x in times if x>ts+1e-12);t=time.perf_counter();row=advance_publish(stepper,store,[],h,frame_builder=cache.frame);advance=time.perf_counter()-t;trace.disable()
        raw=pstats.Stats(trace);stats=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(raw.stats.items(),key=lambda kv:kv[1][2],reverse=True)]
        records.append(dict(initial_s=ts,dt_s=h,build_and_probe_s=build,initial_io_s=initial_io,advance_fields_commit_s=advance,total_s=build+initial_io+advance,row=row));profiles.append(stats[:50]);print('CURRENT_SOLID_COST',ts,records[-1]['total_s'],row['krylov_iterations'],flush=True)
    actual=read(run/'cases/time-h/ledger.json')+read(run/'cases/time-half/ledger.json');krylov=sum(x['krylov_iterations'] for x in actual);material_calls=sum(x['material_point_calls'] for x in actual)
    tangent_self=sum(x['self_s'] for stats in profiles for x in stats if 'tangent' in x['function']);total=sum(x['total_s'] for x in records);upper=tangent_self/max(total,1e-30)
    write(run/'S5/profile.json',dict(status='measured',records=records,profile=profiles,time_study_steps=len(actual),time_study_krylov_iterations=krylov,material_point_calls=material_calls,phase_seconds={n:read(run/'cases'/n/'summary.json')['this_segment'] for n in ('time-h','time-half')},tangent_self_fraction=upper,profiler_included=True))
    write(run/'S5/candidate-decision.json',dict(status='not_triggered',candidate_count=0,reason='current true trajectories do not establish a recurring expensive tangent workload; material path states differ and are not legally cacheable as one F; no measured bounded equivalent candidate with supported >=5% end-to-end benefit',krylov_iterations=krylov,tangent_self_fraction=upper,pressure_optimization_deferred_to_qualified_coupling=True))
    write(run/'S5/equivalence.json',dict(status='not_triggered',reason='no new performance implementation selected; original operators retained'))
    write(run/'S5/paired-performance.json',dict(status='not_triggered',reason='cost discovery only; no new candidate or paired speedup claim',actual_steps=2))
    write(run/'S5/performance-decision.json',dict(status='retain_current',reuse_transpose_buffers=True,shared_reduction=True,field_cache=True,coarse_instrumentation=True,explicit_determinant=True,tangent_cache=False,candidate_count=0,no_new_speedup_claim=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):profile(a.run)
