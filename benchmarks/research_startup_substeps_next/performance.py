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

SCOPE='legacy'
def stage(run):return Path(run)/'S4'/SCOPE

def origin(run,which):
    if SCOPE=='legacy':return history(ZERO/'cases/zero-startup-coarse-h')[(1,3)[which]]
    from engine.aniso_phase1.research_startup_substeps_next.schedule import indices,OBSERVATIONS
    p=read(Path(run)/'S3/coupled-protocol.json');index=indices(p['times']['h'],OBSERVATIONS)[(2,4)[which]]
    return history(Path(run)/'cases/boundary32-h')[int(index)]

def sources(candidate=False):
    out=dict(source_files(),**{str(Path(__file__).relative_to(ROOT)):sha(__file__)})
    for key in ('benchmarks/research_startup_substeps_next/fixture.py','engine/aniso_phase1/research_boundary_reference_next/geometry.py'):out[key]=sha(ROOT/key)
    for key in ('benchmarks/research_startup_substeps_next/coupling.py','engine/aniso_phase1/research_startup_substeps_next/schedule.py'):out[key]=sha(ROOT/key)
    return out

def prepare(run):
    run=Path(run);mutable(run)
    if SCOPE=='boundary32' and (read(run/'S3/grid-decision.json')['status']!='qualified_engineering_window' or not read(run/'S4/legacy/performance-decision.json')['selected']):raise ValueError('new grid performance branch not eligible')
    base=stage(run).relative_to(run)
    register(run,str(base/'integration-protocol.json'),dict(status='registered',scope=SCOPE,source_states=[dict(path=str(origin(run,i)['folder']/'state.json'),sha256=sha(origin(run,i)['folder']/'state.json'),time_s=origin(run,i)['state'].time) for i in (0,1)],grid=SCOPE,method='startup',source_m3_s=0.,backend_A='BASE bounded raw transpose plus Gauss discrete gradient',backend_B='shared endpoint/midpoint exact Simpson volume gradient',order=['A0','B0','B1','A1'],max_all_S4_attempts=11,minimum_gain=.05,max_setup_recovery_steps=16,physics_unchanged=True,inherited_operator=dict(path=str(APP/'R4/operator-equivalence.json'),sha256=sha(APP/'R4/operator-equivalence.json'))))
    register(run,str(base/'optimization-protocol.json'),dict(status='registered',eligible=True,reason='unchanged quadratic cofactor Simpson identity, scope bound to original state',candidate_limit=1,additional_GPU_metadata_bytes=0,old_metadata_cap_bytes=268435456,max_RSS_GiB=16))
    write(stage(run)/'environment-check.json',sharing())

def setup(run,which,candidate=False):
    from .fixture import inherited_setup,new_setup
    import warp as wp
    item=origin(run,which);p=read(stage(run)/'integration-protocol.json')
    if sha(item['folder']/'state.json')!=p['source_states'][which]['sha256']:raise ValueError('profile source state changed')
    c,m,cfg,bridge=inherited_setup(run,item['state']) if SCOPE=='legacy' else new_setup(run,state=item['state']);extra=0.
    if candidate:
        from engine.aniso_phase1.research_boundary_reference_next.geometry import install
        tick=time.perf_counter();bridge=install(c,bridge);wp.synchronize_device(m.device);extra=time.perf_counter()-tick
    write(stage(run)/'model-shape.json',dict(nodal_shape=list(m.parent.shape),old_shape=list(m.parent.oldshape),parent_ndof=m.parent.ndof,reduced_shape=list(m.reduction.P.shape),raw_shape=list(m.parent.raw.shape),raw_nnz=m.parent.raw.nnz,transform_shape=list(m.parent.transform.shape),local_metadata_bytes=c.geometry.local_bytes))
    return item,m,c,bridge,extra

def one(run,which,kind,fault=False,restart=False):
    import warp as wp
    run=Path(run);mutable(run);candidate=kind=='B'
    if candidate and not read(stage(run)/'optimization-protocol.json')['eligible']:raise ValueError('candidate not eligible')
    start=time.perf_counter()
    if not (fault or restart) and not sharing()['exclusive']:
        write(stage(run)/'environment-block.json',dict(status='limited_shared_GPU',environment=sharing(),new_attempts=0));raise RuntimeError('exclusive GPU unavailable; stop performance branch')
    item,m,c,bridge,extra=setup(run,which,candidate);wp.synchronize_device(m.device);build=time.perf_counter()-start
    folder=stage(run)/f'{kind}{which}';cache=CachedProbes(m)
    identity=dict(schema='startup-substeps-performance-v1',scope=SCOPE,coupling=c.identity,initial_state_sha256=sha(item['folder']/'state.json'),sources=sources(candidate),kind=kind)
    if restart:
        if read(folder/'identity.json')!=identity:raise ValueError('restart identity changed')
        store=GenerationStore(folder,identity);all_states=store.history();last=store.load(validator=c.validate);c.restore(last['state'])
        write(stage(run)/'restart.json',dict(status='passed_scoped',total_process_seconds=time.perf_counter()-start,full_digest=last['state'].digest(),same_digest=c.state.digest()==last['state'].digest(),generations=len(all_states),new_process=True,new_attempts=0,all_fluid_history_in_digest=True));return
    if fault:folder=stage(run)/f'fault-{kind}{which}'
    if (folder/'identity.json').exists():raise ValueError('trial already exists')
    write(folder/'identity.json',identity);snapshot(folder/'source',sources(candidate));store=GenerationStore(folder,identity);store.save(c.state,[]);write(folder/'bridge.json',bridge)
    if sum(read(p)['attempts'] for p in (run/'S4').rglob('attempt.json'))>=11:raise RuntimeError('performance attempt budget')
    if fault:
        old=c.state.digest()
        def inject(stage,state):
            if stage=='before_commit':raise ValueError('registered failure before commit')
        write(folder/'attempt.json',dict(attempts=1,accepted=False))
        try:c.step(inject=inject)
        except ValueError as e:
            if 'registered failure' not in str(e):raise
            if c.state.digest()!=old or store.load()['state'].digest()!=old or len(c.geometry.cache):raise ValueError('rollback state/cache differs')
            write(folder/'failure.json',dict(status='passed_scoped',rollback_exact=True,cache_cleared=True,error=str(e),attempts=1,total_process_seconds=time.perf_counter()-start));return
        raise ValueError('fault not triggered')
    parts=dict(field_s=0.,commit_s=0.,geometry_inclusive_s=0.,rt0_s=0.,volume_gradient_adjoint_s=0.,nodal_to_full_adjoint_s=0.,material_s=0.);calls={k:0 for k in parts}
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
    def wrap_map(mp):
        grad0=mp.gradient_adjoint;adj0=mp.adjoint
        def grad(*args,**kwargs):
            if not inside[0]:return grad0(*args,**kwargs)
            wp.synchronize_device(m.device);t0=time.perf_counter()
            try:return grad0(*args,**kwargs)
            finally:wp.synchronize_device(m.device);parts['volume_gradient_adjoint_s']+=time.perf_counter()-t0;calls['volume_gradient_adjoint_s']+=1
        def adjoint(*args,**kwargs):
            if not inside[0]:return adj0(*args,**kwargs)
            wp.synchronize_device(m.device);t0=time.perf_counter()
            try:return adj0(*args,**kwargs)
            finally:wp.synchronize_device(m.device);parts['nodal_to_full_adjoint_s']+=time.perf_counter()-t0
        mp.gradient_adjoint=grad;mp.adjoint=adjoint
    for mp in [m.operator.maps,*getattr(c.geometry,'local_maps',[])]:wrap_map(mp)
    def emit(s):
        tick=time.perf_counter()
        try:return frame(c,cache,s)
        finally:parts['field_s']+=time.perf_counter()-tick
    trace=cProfile.Profile();before=sharing()
    if not before['exclusive']:raise RuntimeError('GPU became shared before timed step')
    write(folder/'attempt.json',dict(attempts=1,accepted=False));wp.synchronize_device(m.device);tick=time.perf_counter();trace.enable()
    try:row=advance_publish(c,store,[],frame_builder=emit if which==1 else None)
    except Exception as e:
        write(folder/'failure.json',dict(status='limited',error=repr(e),rollback=store.load()['state'].digest()==c.state.digest()));raise
    finally:trace.disable()
    wp.synchronize_device(m.device);elapsed=time.perf_counter()-tick;stats=pstats.Stats(trace)
    top=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda x:x[1][2],reverse=True)[:40]]
    write(folder/'attempt.json',dict(attempts=1,accepted=True));write(folder/'measurement.json',dict(status='passed_scoped',advance_s=elapsed,total_process_seconds=time.perf_counter()-start,build_s=build,additional_setup_s=extra,**parts,call_counts=calls,linear_solve_inclusive_s=sum(v[3] for (f,l,n),v in stats.stats.items() if n=='balanced_solve'),profile=top,nested_times_not_additive=True,same_instrumentation_for_AB=True,cold_first_evaluation=True,sharing_before=before,sharing_after=sharing(),frame_written=which==1,iterations=row['iterations'],residual_fraction=row['true_scaled_residual'],peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,source_sha256=sources(candidate)))
    print('PROFILE',kind,which,elapsed,parts,flush=True)

def operators(run):
    import warp as wp,json
    from engine.aniso_phase1.research_boundary_reference_next.geometry import SharedMidpointGeometry as LocalGeometry
    run=Path(run);verify(run);item,m,c,bridge,_=setup(run,0);g=c.geometry;cap=min(256*2**20,int(.02*m.operator.memory_budget.initial_free));fast=LocalGeometry.from_reduced(g);records=[]
    rest=m.rest().q;direction=np.zeros_like(rest);direction[m.free]=np.random.default_rng(5).normal(size=direction[m.free].shape)
    physical_gradient=g.field(direction,True).numpy();direction*=.02/max(float(np.max(abs(physical_gradient))),1e-30);del physical_gradient
    manufactured=rest+direction
    for label,q in [('rest',rest),('manufactured',manufactured),('saved-1',item['state'].q),('saved-3',origin(run,1)['state'].q)]:
        g.cache.clear();fast.cache.clear();a=g.evaluate(q);b=fast.evaluate(q);checks={k:metric(a[k],b[k],1e-8,2e-5) for k in ('H','volume','gradient')};records.append(dict(state=label,checks=checks,passed=all(x['passed'] for x in checks.values())))
        b['gradient'][:]=0
        if not np.array_equal(fast.evaluate(q)['gradient'],a['gradient']) and not np.allclose(fast.evaluate(q)['gradient'],a['gradient'],atol=1e-8,rtol=2e-5):raise ValueError('cache ownership')
    paths=[]
    for label,q0,q1 in [('actual',item['state'].q,origin(run,1)['state'].q),('manufactured',rest,manufactured),('reverse',manufactured,rest)]:
        da=g.discrete(q0,q1);db=fast.discrete(q0,q1);delta=fast.evaluate(q1)['volume']-fast.evaluate(q0)['volume'];work_error=float(np.max(abs(np.einsum('cij,ij->c',db,q1-q0)-delta)));check=metric(da,db,1e-8,2e-5)
        paths.append(dict(path=label,gradient=check,work_error_m3=work_error,passed=check['passed'] and work_error<1e-10))
    q0=item['state'].q;work=max(v['work_error_m3'] for v in paths);passed=all(r['passed'] for r in records+paths) and json.loads(json.dumps(fast.identity))==fast.identity
    expected=fast.evaluate(q0);old=g.total_weights.copy()
    try:
        try:g.total_weights[:]*=2
        except ValueError:pass
        fast.cache.clear();owned=fast.evaluate(q0)
        ownership=all(np.array_equal(expected[k],owned[k]) for k in ('H','volume','gradient'))
    finally:pass
    passed=passed and ownership
    write(stage(run)/'implementation-bridge.json',dict(status='passed_scoped',original_source_state_sha256=sha(item['folder']/'state.json'),source_identity=c.identity,theta=c.core.thetas.tolist(),source_vector=c.source.tolist(),physical_payload_exact_in_all_installs=True,per_trial_bridge_files='S4/'+SCOPE+'/A0,B0,B1,A1/bridge.json',Dnum_history=item['state'].child_states['fluid']['cumulative_numerical_dissipation_J']))
    write(stage(run)/'operator-equivalence.json',dict(status='passed_scoped' if passed else 'limited',records=records,paths=paths,pressure_work_error_J_per_Pa=work,metadata_bytes=fast.local_bytes,cap_bytes=cap,JSON_roundtrip=True,own_results=True,host_weight_source_mutation_isolated=ownership,current_F_always_used=True,sources=sources(True)))
    if not passed:raise ValueError('local gradient gate failed')
    print('LOCAL_OPERATOR',passed,fast.local_bytes,work,flush=True)

def finish(run):
    run=Path(run);mutable(run);records=[]
    for i in (0,1):
        a=history(stage(run)/f'A{i}')[-1]['state'];b=history(stage(run)/f'B{i}')[-1]['state'];checks={k:metric(getattr(a,k),getattr(b,k),1e-8,2e-5) for k in ('q','velocity','predictor')}
        for k,tol in [('pressure_Pa',1e-6),('flux_interval_m3_s',1e-10),('content_m3',1e-10),('cumulative_source_m3',1e-10),('cumulative_boundary_m3',1e-10),('cumulative_numerical_dissipation_J',1e-12)]:checks[k]=metric(a.child_states['fluid'][k],b.child_states['fluid'][k],tol,2e-5)
        la=history(stage(run)/f'A{i}')[-1]['rows'][-1];lb=history(stage(run)/f'B{i}')[-1]['rows'][-1]
        for k in ('darcy_dissipation_J','numerical_dissipation_J','energy_balance_J','source_work_J','reservoir_work_J'):checks[k]=metric(la[k],lb[k],1e-12,2e-5)
        checks['raw_residual']=dict(passed=max(la['true_scaled_residual'],lb['true_scaled_residual'])<=1)
        A=read(stage(run)/f'A{i}/measurement.json');B=read(stage(run)/f'B{i}/measurement.json');gain=1-B['advance_s']/A['advance_s'];records.append(dict(index=i,A=A,B=B,gain=gain,checks=checks,passed=all(x['passed'] for x in checks.values()),setup_break_even_steps=max(0.,B['build_s']-A['build_s'],B['additional_setup_s'])/max(A['advance_s']-B['advance_s'],1e-30)))
    gains=[r['gain'] for r in records];median=float(np.median(gains));spread=float(np.ptp(gains));exclusive=all(r[k][s]['exclusive'] for r in records for k in ('A','B') for s in ('sharing_before','sharing_after'));selected=all(r['passed'] for r in records) and exclusive and min(gains)>=0 and median>=max(.05,spread) and max(r['setup_break_even_steps'] for r in records)<=16
    write(stage(run)/'paired-performance.json',dict(status='passed_scoped' if selected else 'limited',records=records,median_gain=median,between_state_gain_range=spread,repeated_noise_estimate=False,exclusive=exclusive,paired_steps=4))
    attempts=sum(read(p)['attempts'] for p in stage(run).rglob('attempt.json'))
    write(stage(run)/'performance-decision.json',dict(status='shared_midpoint_qualified' if selected else 'retain_BASE_bounded_Gauss',selected=bool(selected),backend='shared-midpoint-exact-volume-gradient' if selected else 'BASE-bounded-Gauss',formal_solid_changed=False,production_default_changed=False,preoptimization_trajectories_not_relabelled=True,scope=SCOPE,origin_times_s=[origin(run,i)['state'].time for i in (0,1)],attempts_completed_at_decision=attempts,max_all_S4_attempts=11))
    write(stage(run)/'setup-cost.json',dict(records=[dict(index=r['index'],A_build_s=r['A']['build_s'],B_build_s=r['B']['build_s'],extra_s=r['B']['additional_setup_s'],recovery_steps=r['setup_break_even_steps']) for r in records],limit_steps=16))
    print('PERFORMANCE_DECISION',SCOPE,selected,median,spread,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','A','B','operators','fault','restart','finish']);p.add_argument('--run',type=Path,required=True);p.add_argument('--which',type=int,choices=[0,1],default=0);p.add_argument('--scope',choices=['legacy','boundary32'],default='legacy');a=p.parse_args();SCOPE=a.scope
    with serial_lock(a.run):
        if a.phase in ('A','B','fault','restart'):one(a.run,a.which,'B' if a.phase in ('B','fault','restart') else 'A',fault=a.phase=='fault',restart=a.phase=='restart')
        else:globals()[a.phase](a.run)
