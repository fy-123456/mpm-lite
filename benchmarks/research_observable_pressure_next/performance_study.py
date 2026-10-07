"""Two representative accepted pressure states; actual step plus publication costs."""
from pathlib import Path
import argparse,time,cProfile,pstats,resource
import numpy as np
from .provenance import read,write,sha,register,source_files,serial_lock
from .coupling_study import setup
from engine.aniso_phase1.research_observable_pressure_next.rt0 import BoundedGridCoupling
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish

def profile(run,kind):
    run=Path(run)
    if not (run/'S4/profile-protocol.json').exists():register(run,'S4/profile-protocol.json',dict(target='two-cell pressure actual step and committed output',states=[0,3],steps_per_state=1,max_candidates=1,
        minimum_gain=.1,max_other_regression=.05,include_model_and_geometry_build=True,independent_processes=True,no_solid_speedup_claim=True))
    original=run/'cases/pressure-2';h=GenerationStore(original,read(original/'identity.json')).history();protocol=read(run/'S3/pressure-time-protocol.json');times=protocol['times_s'];trace=cProfile.Profile();trace.enable();t=time.perf_counter();m,cfg=setup(run);cache=CachedProbes(m);build=time.perf_counter()-t;records=[]
    if kind=='candidate':
        from engine.aniso_phase1.research_observable_pressure_next.fast_rt0 import FastGridCoupling as Coupling
    else:Coupling=BoundedGridCoupling
    for index in (0,3):
        item=h[index];tick=time.perf_counter();source=np.asarray(read(run/'S3/fixed-2.json')['source_m3_s']);c=Coupling(m,cfg,times,cells=2,source_m3_s=source,state=item['state']);construction=time.perf_counter()-tick
        folder=run/'S4/profiles'/f'{kind}-{index}';identity=dict(source=source_files(),source_state_sha256=sha(item['folder']/'state.json'),coupling=c.identity,kind=kind)
        if (folder/'complete.json').exists():raise ValueError('profile already completed')
        write(folder/'identity.json',identity);store=GenerationStore(folder,identity);tick=time.perf_counter();store.save(c.state,[]);initial_io=time.perf_counter()-tick
        tick=time.perf_counter();row=advance_publish(c,store,[],frame_builder=lambda state:c.frame(cache));wall=time.perf_counter()-tick
        np.savez_compressed(folder/'final.npz',q=c.state.q,v=c.state.velocity,predictor=c.state.predictor)
        record=dict(index=index,model_build_s=build/2,geometry_build_s=construction,initial_io_s=initial_io,step_publish_s=wall,total_s=build/2+construction+initial_io+wall,
            row=row,fluid=c.state.child_states['fluid'],min_detF=row['min_detF']);write(folder/'complete.json',record);records.append(record);print('PRESSURE_PROFILE',kind,index,record['total_s'],flush=True)
    trace.disable();stats=pstats.Stats(trace);rows=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)[:40]]
    write(run/f'S4/profile-{kind}.json',dict(kind=kind,records=records,profile=rows,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,source_sha256=source_files(),timing='host wall including constructor, actual step, fields and publication; profile nesting not additive'))
    if kind=='baseline':write(run/'S4/current-profile.json',dict(status='measured',target='pressure',data=read(run/'S4/profile-baseline.json'),solid_evidence='parent actual ReusedSegmentedModel profile; no new solid optimization claim'))

def decide(run):
    run=Path(run);base=read(run/'S4/profile-baseline.json');cand=read(run/'S4/profile-candidate.json');records=[]
    for a,b in zip(base['records'],cand['records']):
        errors={};idx=a['index']
        with np.load(run/f'S4/profiles/baseline-{idx}/final.npz') as x,np.load(run/f'S4/profiles/candidate-{idx}/final.npz') as y:
            for k in x.files:errors[k]=float(np.max(abs(x[k]-y[k])))
        for k in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3'):errors[k]=float(np.max(abs(np.asarray(a['fluid'][k])-b['fluid'][k])))
        passed=max(errors.values())<=1e-8 and b['row']['true_scaled_residual']<=1
        records.append(dict(index=idx,errors=errors,passed=passed,gain=1-b['total_s']/a['total_s'],baseline_s=a['total_s'],candidate_s=b['total_s']))
    gain=float(np.median([x['gain'] for x in records]));passed=all(x['passed'] and x['gain']>=-.05 for x in records) and gain>=.1 and cand['peak_rss_GiB']<=16
    write(run/'S4/paired-performance.json',dict(status='passed_scoped' if passed else 'retain_baseline',records=records,median_gain=gain,baseline_rss_GiB=base['peak_rss_GiB'],candidate_rss_GiB=cand['peak_rss_GiB'],target='two-cell pressure; not solid full-cycle'))
    write(run/'S4/performance-decision.json',dict(status='adopt_pressure_scoped' if passed else 'retain_current',pressure_fast_RT0=passed,reuse_transpose_buffers=True,shared_reduction=True,field_cache=True,coarse_instrumentation=True,tangent_cache=False,candidate_count=1,solid_unchanged=True,pressure_scope='two-cell paired states; four-cell operator checked separately',median_pressure_gain=gain))
    print('PERFORMANCE_DECISION',passed,gain,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','decide']);p.add_argument('--kind',choices=['baseline','candidate'],default='baseline');p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='profile':profile(a.run,a.kind)
        else:decide(a.run)
