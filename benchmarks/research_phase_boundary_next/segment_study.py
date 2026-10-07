"""One allocation-only candidate; two baseline plus two candidate accepted steps."""
from pathlib import Path
import argparse,cProfile,pstats,time,resource
import numpy as np
from .provenance import *
from .run import load_model
from . import config
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.runtime_rules import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_local_span_next.reuse import ReusedSegmentedCSR

def study(run):
    run=Path(run);verify(run);baseline=read(run/'S5/profile.json')
    register(run,'S5/segment-protocol.json',dict(candidate='vectorized ordered segment metadata',baseline='S5/profile.json',states_s=[0.,1.1],total_actual_steps=4,baseline_steps_already_completed=2,remaining_steps=2,paired_instrumentation='same cProfile enabled for both builds and accepted steps',minimum_median_gain=.05,scope='process load + one accepted step including probes and commit; does not establish steady-state/full-cycle speedup',selection_reason='from_csr Python self cost about .47s / 5.4-6.4s; upper bound above5%; no summation/kernel/precision change'))
    for name in ('candidate-decision','equivalence','paired-performance','performance-decision'):
        (run/f'S5/{name}-initial.json').write_bytes((run/f'S5/{name}.json').read_bytes())
    hs=history(APP/'cases/final-full');lookup={round(x['state'].time,10):x for x in hs};times=read(run/'S1/time-decision.json')['times'];records=[];equivalence=[]
    for ts in (0.,1.1):
        cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],times=times,space=read(run/'selected-space.json')['package'],field_cache=True,shared_reduction=True,coarse_instrumentation=True,vectorized_segment_metadata=True)
        trace=cProfile.Profile();trace.enable();t=time.perf_counter();m,_=load_model(run,cfg);cache=CachedProbes(m);build=time.perf_counter()-t;item=lookup[ts];stepper=ValidatedAVF(m,cfg,item['state']);folder=run/'S5/profiles'/f'candidate-{ts}'
        identity=dict(model=m.identity,source_state_sha256=sha(item['folder']/'state.json'),numerical_sources=source_files());write(folder/'identity.json',identity);store=GenerationStore(folder,identity);t=time.perf_counter();store.save(stepper.state,[]);initial_io=time.perf_counter()-t
        h=next(x-ts for x in times if x>ts+1e-12);t=time.perf_counter();row=advance_publish(stepper,store,[],h,frame_builder=cache.frame);advance=time.perf_counter()-t;trace.disable()
        raw=pstats.Stats(trace);stats=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(raw.stats.items(),key=lambda kv:kv[1][2],reverse=True)][:50]
        records.append(dict(initial_s=ts,dt_s=h,build_and_probe_s=build,initial_io_s=initial_io,advance_fields_commit_s=advance,total_s=build+initial_io+advance,row=row,profile=stats,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,gpu_memory=m.operator.memory_budget.report()))
        original_folder=run/'S5/profiles'/f'current-{ts}';original=GenerationStore(original_folder,read(original_folder/'identity.json')).history()[-1];actual=store.history()[-1]
        errors={key:float(np.max(abs(getattr(actual['state'],key)-getattr(original['state'],key)))) for key in ('q','velocity','predictor')}
        for key in ('reaction_N','energy_J','budget_defect_J','true_residual'):
            if key in row:errors[key]=float(np.max(abs(np.asarray(row[key])-np.asarray(original['rows'][-1][key]))))
        with np.load(actual['folder']/'frame.npz') as a,np.load(original['folder']/'frame.npz') as b:
            for key in a:errors['field_'+key]=float(np.max(abs(a[key]-b[key])))
        # Compare full U/force/tangent with exactly the original row algorithm
        # on the same GPU buffers. No extra accepted time steps are needed.
        maps=m.operator.maps;newpairs=[maps.old,maps.raw];oldpairs=[]
        for pair,source in zip(newpairs,(np.asarray(maps.space.oldA).T,maps.space.raw.T)):
            old=ReusedSegmentedCSR.from_csr(pair[1],source)
            for key in ('starts','stops','row_segments'):
                if not np.array_equal(getattr(old,key).numpy(),getattr(pair[1],key).numpy()):raise ValueError('segment order changed')
            if not all(getattr(old,key) is getattr(pair[1],key) for key in ('ptr','col','val')):raise ValueError('buffer ownership changed')
            oldpairs.append((pair[0],old))
        q=m.reduction.expand(item['state'].q);d=np.random.default_rng(31).normal(size=q.shape)*.001
        lin=m.operator.prepare(q);v=m.operator.action(lin,d)
        maps.old,maps.raw=oldpairs
        ref=m.operator.prepare(q);w=m.operator.action(ref,d)
        maps.old,maps.raw=newpairs
        for key in ('U','force','full_force','min_detF'):errors['operator_'+key]=float(np.max(abs(np.asarray(lin.response[key])-np.asarray(ref.response[key]))))
        errors['operator_tangent']=float(np.max(abs(v-w)))
        if max(errors.values())!=0.:raise ValueError('allocation-only candidate changed numerical results '+str(errors))
        equivalence.append(dict(initial_s=ts,errors=errors,segment_arrays_identical=True,original_buffers_reused=True,mass_unchanged=True))
        print('SEGMENT_CANDIDATE',ts,records[-1]['total_s'],'exact numerical equality',flush=True)
    ratios=[1-b['total_s']/a['total_s'] for a,b in zip(baseline['records'],records)];gain=float(np.median(ratios));adopt=gain>=.05 and min(ratios)>-.02
    write(run/'S5/equivalence.json',dict(status='passed',records=equivalence,scope='original summation, chunk128, kernels, precision, mass and equations; identical metadata and actual accepted states; pressure coupling not executed'))
    write(run/'S5/paired-performance.json',dict(status='measured',baseline=baseline['records'],candidate=records,gain_fractions=ratios,median_gain_fraction=gain,profiler_included_both=True,total_actual_steps=4,order='baseline then candidate; no repeated trials',scope='startup plus one solid accepted step, warm persistent Warp cache; small sample only'))
    write(run/'S5/candidate-decision.json',dict(status='evaluated',candidate_count=1,candidate='vectorized_segment_metadata',tangent_cache=False,baseline_initial_decision='S5/candidate-decision-initial.json',reason='full profile identified independent metadata construction hotspot after tangent cache was ruled out'))
    write(run/'S5/performance-decision.json',dict(status='adopt_scoped' if adopt else 'retain_current',reuse_transpose_buffers=True,shared_reduction=True,field_cache=True,coarse_instrumentation=True,explicit_determinant=True,tangent_cache=False,vectorized_segment_metadata=adopt,candidate_count=1,measured_gain_fraction=gain,scope='profiled startup plus one accepted step only; full-cycle gain not inferred'))
    print('SEGMENT_DECISION',adopt,ratios,gain,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
