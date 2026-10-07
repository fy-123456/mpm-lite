"""Measure the inherited fast RT0 path before authorizing any new optimization."""
from pathlib import Path
import argparse,time,cProfile,pstats,resource
import numpy as np
from .provenance import *
from .physics import baseline_model
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_observable_pressure_next.fast_rt0 import FastGridCoupling
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish

def profile(run):
    run=Path(run);verify(run);register(run,'S4/profile-protocol.json',dict(states=[0,3],source=str(APP/'cases/pressure-2'),max_candidates=1,minimum_total_gain=.1,include_build_fields_and_commit=True,scope='two-cell BASELINE pressure'))
    folder=APP/'cases/pressure-2';hist=GenerationStore(folder,read(folder/'identity.json')).history();times=read(APP/'S3/pressure-time-protocol.json')['times_s'];src=read(APP/'S3/fixed-2.json')['source_m3_s']
    trace=cProfile.Profile();trace.enable();tick=time.perf_counter();m,cfg=baseline_model(run,pressure=True);cache=CachedProbes(m);build=time.perf_counter()-tick;rows=[]
    for index in (0,3):
        t=time.perf_counter();item=hist[index];c=FastGridCoupling(m,cfg,times,cells=2,source_m3_s=src,state=item['state']);construct=time.perf_counter()-t
        dest=run/'S4/profiles'/f'baseline-{index}';identity=dict(source_state_sha256=sha(item['folder']/'state.json'),numeric_sources=source_files(),coupling=c.identity);write(dest/'identity.json',identity);store=GenerationStore(dest,identity);t=time.perf_counter();store.save(c.state,[]);io=time.perf_counter()-t;t=time.perf_counter();row=advance_publish(c,store,[],frame_builder=lambda state:c.frame(cache));step=time.perf_counter()-t
        rows.append(dict(index=index,model_build_s=build/2,geometry_build_s=construct,initial_io_s=io,step_publish_s=step,total_s=build/2+construct+io+step,row=row));print('CURRENT_PRESSURE_PROFILE',index,rows[-1]['total_s'],flush=True)
    trace.disable();stats=pstats.Stats(trace);records=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda kv:kv[1][2],reverse=True)]
    total=sum(x['total_s'] for x in rows);field=sum(x['cumulative_s'] for x in records if x['function']=='field' and 'research_cost_phase_next/pressure.py' in x['file'])
    # Removing one duplicate F evaluation cannot save more than half the field time.
    upper=.5*field/total
    write(run/'S4/current-profile.json',dict(status='measured',records=rows,profile=records[:60],duplicate_F_optimistic_fraction=upper,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,inherited_fast_RT0=True,new_solid_smoke=read(run/'cases/space-smoke/summary.json') if (run/'cases/space-smoke/summary.json').exists() else None,pressure_target_reason='8-cell fixed-grid gate did not authorize coupling; two-cell FastRT0 remains the current inherited coupled research entry'))
    write(run/'S4/candidate-decision.json',dict(status='not_triggered' if upper<.1 else 'requires_candidate',reason='measured duplicate-field optimistic total saving',optimistic_fraction=upper,threshold=.1))
    if upper<.1:
        write(run/'S4/paired-performance.json',dict(status='not_triggered',reason='no registered equivalent candidate has supported >=10% end-to-end saving; two current accepted states measured',candidate_count=0))
        write(run/'S4/performance-decision.json',dict(status='retain_current',pressure_fast_RT0=True,reuse_transpose_buffers=True,shared_reduction=True,field_cache=True,coarse_instrumentation=True,tangent_cache=False,candidate_count=0,no_new_speedup_claim=True))
    print('DUPLICATE_F_MAX_GAIN',upper,flush=True)


def candidate(run):
    from engine.aniso_phase1.research_cross_direction_next.det_rt0 import FastGridCoupling as Candidate,FastTopology,det3
    from engine.aniso_phase1.research_observable_pressure_next.fast_rt0 import FastTopology as BaselineTopology
    from benchmarks.research_observable_pressure_next.coupling_study import affine_check
    from benchmarks.research_phase_reference_next.coupling_study import quadrature
    run=Path(run);verify(run);profile=read(run/'S4/current-profile.json');total=sum(x['total_s'] for x in profile['records']);det_time=sum(x['self_s'] for x in profile['profile'] if x['function']=='det')
    write(run/'S4/duplicate-field-screening.json',read(run/'S4/candidate-decision.json'))
    register(run,'S4/determinant-protocol.json',dict(candidate_count=1,method='explicit algebraic 3x3 determinant, same current F, no caching or diagonal mobility approximation',baseline_det_self_s=det_time,baseline_total_s=total,available_fraction=det_time/total,minimum_median_gain=.1,max_state_regression=.05,paired_states=[0,3],original_baseline_source=sha(run/'S4/profile-protocol.json')))
    write(run/'S4/candidate-decision.json',dict(status='registered_one_candidate',method='explicit determinant',duplicate_F_rejected=True,reason='profile shows general determinant consumes over 10% total'))
    checks=[]
    for cells in (2,4,8):
        top=FastTopology([[.25,.75],[-.125,.125],[-.1875,.1875]],cells)
        for F in (np.eye(3),np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]])):
            checks.append(affine_check(top,F));X,w,ids=quadrature(top);Fs=np.broadcast_to(F,(len(X),3,3));err=float(np.max(abs(det3(Fs)-np.linalg.det(Fs))))
            if err>1e-12:raise ValueError('explicit determinant differs')
            if cells in (2,4):
                old=BaselineTopology(top.bounds,cells);e=float(np.max(abs(top.assemble(X,w,ids,Fs)[0]-old.assemble(X,w,ids,Fs)[0])))
                if e>1e-10:raise ValueError('tensor assembly changed')
    write(run/'S4/operator-equivalence.json',dict(status='passed_scoped',records=checks,same_full_tensor=True,physical_identity_unchanged=True,current_F_always_used=True))
    source=APP/'cases/pressure-2';hist=GenerationStore(source,read(source/'identity.json')).history();times=read(APP/'S3/pressure-time-protocol.json')['times_s'];src=read(APP/'S3/fixed-2.json')['source_m3_s'];tick=time.perf_counter();m,cfg=baseline_model(run,pressure=True);cache=CachedProbes(m);build=time.perf_counter()-tick;records=[]
    for index in (0,3):
        item=hist[index];t=time.perf_counter();c=Candidate(m,cfg,times,cells=2,source_m3_s=src,state=item['state']);construct=time.perf_counter()-t;dest=run/'S4/profiles'/f'candidate-{index}';identity=dict(source_state_sha256=sha(item['folder']/'state.json'),numeric_sources=source_files(),coupling=c.identity)
        write(dest/'identity.json',identity);store=GenerationStore(dest,identity);t=time.perf_counter();store.save(c.state,[]);io=time.perf_counter()-t;t=time.perf_counter();row=advance_publish(c,store,[],frame_builder=lambda state:c.frame(cache));elapsed=time.perf_counter()-t
        records.append(dict(index=index,model_build_s=build/2,geometry_build_s=construct,initial_io_s=io,step_publish_s=elapsed,total_s=build/2+construct+io+elapsed,row=row));print('DETERMINANT_PROFILE',index,records[-1]['total_s'],flush=True)
    paired=[]
    for a,b in zip(profile['records'],records):
        i=a['index'];pa=run/f'S4/profiles/baseline-{i}';pb=run/f'S4/profiles/candidate-{i}';old=GenerationStore(pa,read(pa/'identity.json')).history()[-1];new=GenerationStore(pb,read(pb/'identity.json')).history()[-1];errors={k:float(np.max(abs(getattr(old['state'],k)-getattr(new['state'],k)))) for k in ('q','velocity','predictor')}
        for k in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3'):errors[k]=float(np.max(abs(np.asarray(old['state'].child_states['fluid'][k])-new['state'].child_states['fluid'][k])))
        with np.load(old['folder']/'frame.npz') as x,np.load(new['folder']/'frame.npz') as y:
            for k in ('x','velocity','solid_PK1','total_PK1','pressure_Pa'):errors['frame_'+k]=float(np.max(abs(x[k]-y[k])))
        passed=max(errors.values())<=1e-8 and b['row']['true_scaled_residual']<=1 and b['row']['darcy_dissipation_J']>=0
        paired.append(dict(index=i,errors=errors,passed=passed,gain=1-b['total_s']/a['total_s'],baseline_s=a['total_s'],candidate_s=b['total_s']))
    gain=float(np.median([x['gain'] for x in paired]));rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20;adopt=all(x['passed'] and x['gain']>=-.05 for x in paired) and gain>=.1 and rss<=16
    write(run/'S4/profile-candidate.json',dict(records=records,peak_rss_GiB=rss,numeric_sources=source_files()))
    write(run/'S4/paired-performance.json',dict(status='passed_scoped' if adopt else 'retain_baseline',records=paired,median_gain=gain,scope='two-cell inherited BASELINE pressure only; profiler overhead is included in baseline so speed ratio approximate',candidate_count=1))
    write(run/'S4/performance-decision.json',dict(status='pending_fair_timing',explicit_determinant=False,pressure_fast_RT0=True,reuse_transpose_buffers=True,shared_reduction=True,field_cache=True,coarse_instrumentation=True,tangent_cache=False,candidate_count=1,median_pressure_gain=gain,solid_speedup_claim=False))
    print('DIAGNOSTIC_TIMING_ONLY',gain,'fair baseline required',flush=True)


def fair_baseline(run):
    run=Path(run);verify(run)
    write(run/'S4/initial-unpaired-timing.json',read(run/'S4/paired-performance.json'))
    write(run/'S4/performance-decision.json',dict(status='pending_fair_timing',explicit_determinant=False,reuse_transpose_buffers=True,pressure_fast_RT0=True,candidate_count=1))
    register(run,'S4/timing-correction-protocol.json',dict(cause='initial cost-discovery baseline used cProfile; candidate wall timing did not',correction='two baseline-only accepted steps without cProfile on the same two immutable source states; reuse existing unprofiled candidate',extra_accepted_steps=2,candidate_repeats=0,no_more_repeats=True,decision='initial ratio is diagnostic only, not adoption evidence',user_authorized='ordinary identified problems may be corrected with bounded verification'))
    folder=APP/'cases/pressure-2';hist=GenerationStore(folder,read(folder/'identity.json')).history();times=read(APP/'S3/pressure-time-protocol.json')['times_s'];src=read(APP/'S3/fixed-2.json')['source_m3_s'];t=time.perf_counter();m,cfg=baseline_model(run,pressure=True);cache=CachedProbes(m);build=time.perf_counter()-t;records=[]
    for index in (0,3):
        item=hist[index];t=time.perf_counter();c=FastGridCoupling(m,cfg,times,cells=2,source_m3_s=src,state=item['state']);construct=time.perf_counter()-t;dest=run/'S4/profiles'/f'fair-baseline-{index}';identity=dict(source_state_sha256=sha(item['folder']/'state.json'),numeric_sources=source_files(),coupling=c.identity);write(dest/'identity.json',identity);store=GenerationStore(dest,identity);t=time.perf_counter();store.save(c.state,[]);io=time.perf_counter()-t;t=time.perf_counter();row=advance_publish(c,store,[],frame_builder=lambda state:c.frame(cache));elapsed=time.perf_counter()-t
        records.append(dict(index=index,model_build_s=build/2,geometry_build_s=construct,initial_io_s=io,step_publish_s=elapsed,total_s=build/2+construct+io+elapsed,row=row));print('FAIR_BASELINE',index,records[-1]['total_s'],flush=True)
    cand=read(run/'S4/profile-candidate.json');old=read(run/'S4/initial-unpaired-timing.json');paired=[]
    for a,b,x in zip(records,cand['records'],old['records']):
        i=a['index'];pa=run/f'S4/profiles/fair-baseline-{i}';pb=run/f'S4/profiles/candidate-{i}';sa=GenerationStore(pa,read(pa/'identity.json')).load()['state'];sb=GenerationStore(pb,read(pb/'identity.json')).load()['state'];err=max(float(np.max(abs(getattr(sa,k)-getattr(sb,k)))) for k in ('q','velocity','predictor'))
        paired.append(dict(index=i,baseline_s=a['total_s'],candidate_s=b['total_s'],gain=1-b['total_s']/a['total_s'],passed=x['passed'] and err<=1e-8,physical_errors=x['errors'],corrected_pair_state_error=err))
    gain=float(np.median([x['gain'] for x in paired]));adopt=all(x['passed'] and x['gain']>=-.05 for x in paired) and gain>=.1 and cand['peak_rss_GiB']<=16
    write(run/'S4/profile-fair-baseline.json',dict(records=records,cProfile=False,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    write(run/'S4/paired-performance.json',dict(status='passed_scoped' if adopt else 'retain_baseline',records=paired,median_gain=gain,same_instrumentation=True,includes_build_fields_commit=True,candidate_count=1,corrective_extra_steps=2,scope='two inherited BASELINE pressure states only'))
    write(run/'S4/performance-decision.json',dict(status='adopt_pressure_determinant_scoped' if adopt else 'retain_current',explicit_determinant=adopt,pressure_fast_RT0=True,reuse_transpose_buffers=True,shared_reduction=True,field_cache=True,coarse_instrumentation=True,tangent_cache=False,candidate_count=1,median_pressure_gain=gain,solid_speedup_claim=False))
    print('FAIR_PERFORMANCE_ADOPT',adopt,gain,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['profile','candidate','fair_baseline'],default='profile',nargs='?');p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
