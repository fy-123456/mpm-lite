"""Profile current device RT0, then qualify at most one local volume-gradient route."""
from pathlib import Path
import argparse,time,cProfile,pstats,resource
import numpy as np
from .provenance import *
from .physics import baseline_model
from benchmarks.research_restoring_rt0_next.performance import sharing
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_restoring_rt0_next.fixture import construct
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish

def origin(which):return history(OBS/'cases/observable-coarse-h')[(1,3)[which]]

def sources(candidate=False):
    result=dict(source_files(),**{str(Path(__file__).relative_to(ROOT)):sha(__file__)})
    if candidate:
        for p in (ROOT/'engine/aniso_phase1/research_stabilization_boundary_next').glob('*.py'):result[str(p.relative_to(ROOT))]=sha(p)
    return result

def prepare(run):
    run=Path(run);verify(run)
    register(run,'S5/profile-protocol.json',dict(status='registered',source_case=str(OBS/'cases/observable-coarse-h'),fallback_reason=read(run/'S2/coupling-decision.json')['reason'],source_states=[dict(path=str(origin(i)['folder']/'state.json'),sha256=sha(origin(i)['folder']/'state.json')) for i in (0,1)],backend='current-device-RT0-chunk-v1',profile_order=['A0','A1'],with_frames=[False,True],max_attempts=8,candidate_limit=1,minimum_hotspot_fraction=.2,minimum_gain=.05,max_setup_recovery_steps=16))

def setup(run,which,candidate=False):
    import warp as wp
    item=origin(which);p=read(Path(run)/'S5/profile-protocol.json')
    if sha(item['folder']/'state.json')!=p['source_states'][which]['sha256']:raise ValueError('profile state changed')
    m,cfg=baseline_model(run,pressure=True);c,bridge=construct(m,cfg,read(OBS/'S3/new-scene-protocol.json'),'coarse',state=item['state'],backend='device',bridge=True)
    extra=0.
    if candidate:
        from engine.aniso_phase1.research_stabilization_boundary_next.local_geometry import install
        tick=time.perf_counter();bridge=install(c,bridge);wp.synchronize_device(m.device);extra=time.perf_counter()-tick
    return item,m,c,bridge,extra

def one(run,which,kind,fault=False,restart=False):
    import warp as wp
    run=Path(run);verify(run);candidate=kind=='B'
    if candidate and not read(run/'S5/optimization-protocol.json')['eligible']:raise ValueError('candidate not eligible')
    start=time.perf_counter();item,m,c,bridge,extra=setup(run,which,candidate);wp.synchronize_device(m.device);build=time.perf_counter()-start
    folder=run/'S5'/f'{kind}{which}';cache=CachedProbes(m)
    identity=dict(schema='stabilization-performance-v1',coupling=c.identity,initial_state_sha256=sha(item['folder']/'state.json'),sources=sources(candidate),kind=kind)
    if restart:
        if read(folder/'identity.json')!=identity:raise ValueError('restart identity changed')
        store=GenerationStore(folder,identity);all_states=store.history();last=store.load(validator=c.validate);c.restore(last['state'])
        write(run/'S5/restart.json',dict(status='passed_scoped',full_digest=last['state'].digest(),same_digest=c.state.digest()==last['state'].digest(),generations=len(all_states),new_process=True,new_attempts=0,all_fluid_history_in_digest=True));return
    if fault:folder=run/'S5'/f'fault-{kind}{which}'
    if (folder/'identity.json').exists():raise ValueError('trial already exists')
    write(folder/'identity.json',identity);snapshot(folder/'source',sources(candidate));store=GenerationStore(folder,identity);store.save(c.state,[]);write(folder/'bridge.json',bridge)
    if fault:
        old=c.state.digest()
        def inject(stage,state):
            if stage=='before_commit':raise ValueError('registered failure before commit')
        write(folder/'attempt.json',dict(attempts=1,accepted=False))
        try:c.step(inject=inject)
        except ValueError as e:
            if 'registered failure' not in str(e):raise
            if c.state.digest()!=old or store.load()['state'].digest()!=old or len(c.geometry.cache):raise ValueError('rollback state/cache differs')
            write(folder/'failure.json',dict(status='passed_scoped',rollback_exact=True,cache_cleared=True,error=str(e),attempts=1));return
        raise ValueError('fault not triggered')
    parts=dict(field_s=0.,commit_s=0.,geometry_inclusive_s=0.,rt0_s=0.,volume_gradient_adjoint_s=0.,material_s=0.);calls={k:0 for k in parts}
    def timed(obj,method,key):
        method0=getattr(obj,method)
        def measured(*a,**kw):
            wp.synchronize_device(m.device);tick=time.perf_counter()
            try:return method0(*a,**kw)
            finally:wp.synchronize_device(m.device);parts[key]+=time.perf_counter()-tick;calls[key]+=1
        setattr(obj,method,measured)
    timed(store,'save','commit_s');timed(c.geometry,'evaluate','geometry_inclusive_s');timed(c.geometry.assembler,'assemble','rt0_s');timed(m,'evaluate','material_s')
    # Count only gradient adjoints inside the current geometry's evaluate call.
    evaluate=c.geometry.evaluate;inside=[False];adj=m.operator.maps.gradient_adjoint
    def geometry(*a,**kw):
        inside[0]=True
        try:return evaluate(*a,**kw)
        finally:inside[0]=False
    c.geometry.evaluate=geometry
    def measured_adjoint(*a,**kw):
        if not inside[0]:return adj(*a,**kw)
        wp.synchronize_device(m.device);tick=time.perf_counter()
        try:return adj(*a,**kw)
        finally:wp.synchronize_device(m.device);parts['volume_gradient_adjoint_s']+=time.perf_counter()-tick;calls['volume_gradient_adjoint_s']+=1
    m.operator.maps.gradient_adjoint=measured_adjoint
    def emit(s):
        tick=time.perf_counter()
        try:return frame(c,cache,s)
        finally:parts['field_s']+=time.perf_counter()-tick
    trace=cProfile.Profile();before=sharing();write(folder/'attempt.json',dict(attempts=1,accepted=False));wp.synchronize_device(m.device);tick=time.perf_counter();trace.enable()
    try:row=advance_publish(c,store,[],frame_builder=emit if which==1 else None)
    except Exception as e:
        write(folder/'failure.json',dict(status='limited',error=repr(e),rollback=store.load()['state'].digest()==c.state.digest()));raise
    finally:trace.disable()
    wp.synchronize_device(m.device);elapsed=time.perf_counter()-tick;stats=pstats.Stats(trace)
    top=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda x:x[1][2],reverse=True)[:40]]
    write(folder/'attempt.json',dict(attempts=1,accepted=True));write(folder/'measurement.json',dict(status='passed_scoped',advance_s=elapsed,build_s=build,additional_setup_s=extra,**parts,call_counts=calls,linear_solve_inclusive_s=sum(v[3] for (f,l,n),v in stats.stats.items() if n=='balanced_solve'),profile=top,nested_times_not_additive=True,same_instrumentation_for_AB=True,cold_first_evaluation=True,sharing_before=before,sharing_after=sharing(),frame_written=which==1,iterations=row['iterations'],residual_fraction=row['true_scaled_residual'],peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,source_sha256=sources(candidate)))
    print('PROFILE',kind,which,elapsed,parts,flush=True)

def decide(run):
    run=Path(run);records=[read(run/f'S5/A{i}/measurement.json') for i in (0,1)];fractions=[r['volume_gradient_adjoint_s']/r['advance_s'] for r in records];eligible=min(fractions)>=.2
    write(run/'S5/profile.json',dict(status='passed_scoped',records=records,baseline='already accelerated device RT0',real_steps=2,gradient_fractions=fractions,profile_is_not_noise_confidence=True))
    register(run,'S5/optimization-protocol.json',dict(status='registered' if eligible else 'not_triggered',eligible=bool(eligible),candidate='exact cell-local tensor adjoints, x-only bounded grid',fraction=fractions,physics_unchanged=True,cap_bytes=268435456,max_attempts=8))
    if not eligible:write(run/'S5/performance-decision.json',dict(status='not_triggered',selected=False,reason='volume gradient hotspot below registered 20% threshold',new_performance_steps=2,backend='current-device-RT0',production_default_changed=False))
    print('HOTSPOT_ELIGIBLE',eligible,fractions,flush=True)

def operators(run):
    import warp as wp,json
    from engine.aniso_phase1.research_stabilization_boundary_next.local_geometry import LocalGeometry
    run=Path(run);verify(run);item,m,c,bridge,_=setup(run,0);g=c.geometry;cap=min(256*2**20,int(.02*m.operator.memory_budget.initial_free));fast=LocalGeometry.from_device(g,cap_bytes=cap);records=[]
    rest=m.rest().q;direction=np.zeros_like(rest);direction[m.free]=np.random.default_rng(5).normal(size=direction[m.free].shape)
    physical_gradient=g.field(direction,True).numpy();direction*=.02/max(float(np.max(abs(physical_gradient))),1e-30);del physical_gradient
    manufactured=rest+direction
    for label,q in [('rest',rest),('manufactured',manufactured),('saved-1',item['state'].q),('saved-3',origin(1)['state'].q)]:
        g.cache.clear();fast.cache.clear();a=g.evaluate(q);b=fast.evaluate(q);checks={k:metric(a[k],b[k],1e-8,2e-5) for k in ('H','volume','gradient')};records.append(dict(state=label,checks=checks,passed=all(x['passed'] for x in checks.values())))
        b['gradient'][:]=0
        if not np.array_equal(fast.evaluate(q)['gradient'],a['gradient']) and not np.allclose(fast.evaluate(q)['gradient'],a['gradient'],atol=1e-8,rtol=2e-5):raise ValueError('cache ownership')
    q0=item['state'].q;q1=origin(1)['state'].q;da=g.discrete(q0,q1);db=fast.discrete(q0,q1);work=float(np.max(abs(np.einsum('cij,ij->c',da-db,q1-q0))));passed=all(r['passed'] for r in records) and work<1e-10 and json.loads(json.dumps(fast.identity))==fast.identity
    expected=fast.evaluate(q0);old=g.total_weights.copy()
    try:
        g.total_weights[:]*=2;fast.cache.clear();owned=fast.evaluate(q0)
        ownership=all(np.array_equal(expected[k],owned[k]) for k in ('H','volume','gradient'))
    finally:g.total_weights[:]=old
    passed=passed and ownership
    write(run/'S5/operator-equivalence.json',dict(status='passed_scoped' if passed else 'limited',records=records,pressure_work_error_J_per_Pa=work,metadata_bytes=fast.local_bytes,cap_bytes=cap,JSON_roundtrip=True,own_results=True,host_weight_source_mutation_isolated=ownership,current_F_always_used=True,sources=sources(True)))
    if not passed:raise ValueError('local gradient gate failed')
    print('LOCAL_OPERATOR',passed,fast.local_bytes,work,flush=True)

def finish(run):
    run=Path(run);records=[]
    for i in (0,1):
        a=history(run/f'S5/A{i}')[-1]['state'];b=history(run/f'S5/B{i}')[-1]['state'];checks={k:metric(getattr(a,k),getattr(b,k),1e-8,2e-5) for k in ('q','velocity','predictor')}
        for k,tol in [('pressure_Pa',1e-6),('flux_interval_m3_s',1e-10),('content_m3',1e-10),('cumulative_source_m3',1e-10),('cumulative_boundary_m3',1e-10)]:checks[k]=metric(a.child_states['fluid'][k],b.child_states['fluid'][k],tol,2e-5)
        A=read(run/f'S5/A{i}/measurement.json');B=read(run/f'S5/B{i}/measurement.json');gain=1-B['advance_s']/A['advance_s'];records.append(dict(index=i,A=A,B=B,gain=gain,checks=checks,passed=all(x['passed'] for x in checks.values()),setup_break_even_steps=B['additional_setup_s']/max(A['advance_s']-B['advance_s'],1e-30)))
    gains=[r['gain'] for r in records];median=float(np.median(gains));spread=float(np.ptp(gains));exclusive=all(r[k][s]['exclusive'] for r in records for k in ('A','B') for s in ('sharing_before','sharing_after'));selected=all(r['passed'] for r in records) and exclusive and min(gains)>=0 and median>=max(.05,spread) and max(r['setup_break_even_steps'] for r in records)<=16
    write(run/'S5/paired-performance.json',dict(status='passed_scoped' if selected else 'limited',records=records,median_gain=median,between_state_gain_range=spread,repeated_noise_estimate=False,exclusive=exclusive,paired_steps=4))
    attempts=sum(read(p)['attempts'] for p in (run/'S5').rglob('attempt.json'))
    write(run/'S5/performance-decision.json',dict(status='research_local_gradient_qualified' if selected else 'retain_device_rt0',selected=bool(selected),backend='local-cell-adjoint' if selected else 'current-device-RT0',formal_solid_changed=False,production_default_changed=False,preoptimization_trajectories_not_relabelled=True,scope='original16-cell observable short fixture; x-only layout; no new grid accuracy',attempts_completed_at_decision=attempts,max_attempts=8))
    print('PERFORMANCE_DECISION',selected,median,spread,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','A','B','decide','operators','fault','restart','finish']);p.add_argument('--run',type=Path,required=True);p.add_argument('--which',type=int,choices=[0,1],default=0);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase in ('A','B','fault','restart'):one(a.run,a.which,'B' if a.phase in ('B','fault','restart') else 'A',fault=a.phase=='fault',restart=a.phase=='restart')
        else:globals()[a.phase](a.run)
