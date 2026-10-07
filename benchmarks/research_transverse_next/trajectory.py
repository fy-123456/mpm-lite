"""Short source-bound trajectories use the frame-aware safe owner throughout."""
import argparse,time
import numpy as np
import warp as wp
from .provenance import *
from .fixture import setup
from .runtime import attempt,update
from .observables import modes,face_signals
from engine.aniso_phase1.research_transverse_next.initial import TAG
from engine.aniso_phase1.research_transverse_next.recovery import SafePublication
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_continuous_geometry_next.continuous import state_checks

FIELDS=('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')

def instrument(c,m):
    counters={}
    def hook(obj,name,label):
        original=getattr(obj,name)
        def call(*args,**kwargs):
            wp.synchronize_device(m.operator.device);t=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                wp.synchronize_device(m.operator.device);v=counters.setdefault(label,dict(calls=0,seconds=0.));v['calls']+=1;v['seconds']+=time.perf_counter()-t
        setattr(obj,name,call)
    hook(m,'evaluate','material_evaluate');hook(c.core.avf,'path','AVF_material_path_inclusive')
    return counters

def equivalence(a,b,row,reference):
    checks=state_checks(a,b)
    for key,tol in [('reaction_N',1e-8),('pressure_reaction_N',1e-8),('skeleton_reaction_N',1e-8),('inertial_reaction_N',1e-8),('total_energy_J',1e-12),('energy_balance_J',1e-12),('pressure_solid_work_J',1e-12),('pressure_fluid_work_J',1e-12),('true_scaled_residual',1e-3),('cell_volume_m3',1e-12),('darcy_dissipation_J',1e-12),('numerical_dissipation_J',1e-12)]:
        checks[key]=metric(row[key],reference[key],tol,2e-5)
    # Remaining physical histories are inherited verbatim. Identity extension
    # is permitted only for the explicitly documented legacy bridge.
    checks['physical_child_keys']=dict(passed=set(a.child_states)-{TAG}==set(b.child_states)-{TAG})
    checks['equation_identity']=dict(passed=a.child_states['fluid']['model']==b.child_states['fluid']['model'] and a.child_states['explicit_pressure_grid']==b.child_states['explicit_pressure_grid'])
    return checks

def bind(folder,ident):
    write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources'])

def cycle(run,case,stop,fine=False):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,cfg,ident=setup(run,case,fine);built=time.perf_counter()-tick
    folder=run/'cases'/(case+('-half' if fine else ''));store=GenerationStore(folder,ident);prior=store.load(validator=c.validate)
    if prior is None:
        bind(folder,ident);store.save(c.state,[])
        E0=m.evaluate(c.state.q)['U']+m.kinetic(c.state.velocity)+.5*np.sum(c.core.capacity*np.asarray(c.state.child_states['fluid']['pressure_Pa'])**2)
        write(folder/'execution-protocol.json',dict(initial_energy_J=E0,initial_digest=c.initial_digest,initial_contract_sha256=c.contract_id,full_times_s=c.times.tolist(),execution_end_s=37.5e-6 if fine else 75e-6,residual_window_s=c.core.window,mass_sha256=digest(m.M.tolist()),material_order=7,mass_order=7,source_zero=True,backend='D3',geometry=c.geometry.identity))
    else:
        if read(folder/'identity.json')!=ident:raise ValueError('executed source identity changed')
        store.history();c.restore(prior['state'])
        write(folder/f'restart-{c.state.step}.json',dict(status='passed_scoped',same_digest=c.state.digest()==prior['state'].digest(),step=c.state.step,time_s=c.state.time,new_process=True,new_attempts=0))
    limit=24 if fine else 18
    if stop>limit or stop<c.state.step:raise ValueError('unregistered short window')
    safe=SafePublication(c,store);cache=CachedProbes(m);samples={k:[] for k in FIELDS};times=[]
    observations={round(float(t),14) for t in np.arange(7)*12.5e-6};frame_times={round(25e-6,14),round(75e-6,14)} if not fine else set();probe_s=0.
    def probe(s):
        nonlocal probe_s
        t=time.perf_counter();f=frame(c,cache,s);times.append(s.time)
        for k in FIELDS:samples[k].append(f[k])
        probe_s+=time.perf_counter()-t;return f
    start=c.state.step;f=probe(c.state);c.geometry.profile={};c.geometry.cache_hits=c.geometry.cache_misses=0;timings=instrument(c,m);advance_s=0.
    while c.state.step<stop:
        with_frame=round(float(c.times[c.state.step+1]),14) in frame_times
        t=time.perf_counter();row=attempt(run,'S3' if fine else 'S2',folder.name,lambda:safe.advance(frame_builder=(lambda s:frame(c,cache,s)) if with_frame else None));advance_s+=time.perf_counter()-t
        if round(c.state.time,14) in observations:f=probe(c.state)
        print('TRANSVERSE_STEP',case,c.state.step,c.state.time,row['min_detF'],row['true_scaled_residual'],flush=True)
    rows=store.load()['rows'];h=store.history()
    np.savez_compressed(folder/f'probes-{start}-{stop}.npz',X=f['X'],fiber=m.parent.params.fiber_direction,times=times,**{k:np.asarray(v) for k,v in samples.items()})
    write(folder/'observables.json',[dict(step=v['state'].step,time_s=v['state'].time,modes=modes(c.geometry.topology,v['state'].child_states['fluid']['pressure_Pa']),faces=face_signals(c.geometry.topology,v['state'].child_states['fluid']['flux_interval_m3_s']),flux_is_completed_interval=v['state'].step>0) for v in h])
    write(folder/'ledger.json',rows);write(folder/f'profile-{start}-{stop}.json',dict(build_s=built,advance_s=advance_s,geometry=c.geometry.profile,material=timings,publication=safe.profile,probes_s=probe_s,cache_hits=c.geometry.cache_hits,cache_misses=c.geometry.cache_misses,process_s=time.perf_counter()-tick,timing_note='geometry and material inside physical_step; AVF path may contain evaluate; hooks synchronize, so diagnostic timings are not performance qualification'))
    write(folder/'summary.json',dict(status='passed_scoped' if stop==limit else 'partial',case=case,fine=fine,final_step=c.state.step,time_s=c.state.time,new_steps=len(rows),min_detF=min(r['min_detF'] for r in rows),max_residual_fraction=max(r['true_scaled_residual'] for r in rows),last_state_reload=store.load(validator=c.validate)['state'].digest()==c.state.digest()))
    update(run,f'S2 {case}：已到{c.state.time*1e6:g}微秒/第{stop}步，本进程{stop-start}步；安全发布和检查点回读通过。')

def legacy(run):
    run=Path(run);mutable(run);c,m,cfg,ident=setup(run,'LEGACY128');h=history(APP/'cases/pressure128-D3');a,b=h[8],h[9]
    old=a['state'].digest();s=c.bridge_legacy(a['state'],old);folder=run/'S1/legacy';bind(folder,ident);store=GenerationStore(folder,ident);store.save(s,[]);safe=SafePublication(c,store)
    stripped=s.clone();stripped.child_states.pop(TAG)
    if stripped.digest()!=old:raise ValueError('legacy bridge altered physical payload')
    row=attempt(run,'S2','legacy-bridge',safe.advance);checks=equivalence(c.state,b['state'],row,b['rows'][-1])
    passed=all(v['passed'] for v in checks.values())
    write(run/'S1/legacy-bridge-check.json',dict(status='passed_scoped' if passed else 'failed',source=str(a['folder']/'state.json'),source_sha256=sha(a['folder']/'state.json'),source_digest=old,tagged_digest=s.digest(),physical_payload_exact=True,only_added_child_tag=TAG,checks=checks,no_new_display_frame=True))
    if not passed:raise ValueError('legacy adapter equivalence failed')
    write(run/'S1/dynamic-entry-decision.json',dict(status='passed_scoped',legacy_one_step=True,static_grids=['Y64','Y128'],physics_unchanged=True))
    update(run,'S1继承桥接：BASE第8→9步通过新初态标签及安全提交入口，完整物理输入未变；状态、反力、压力功、能量和残差检查通过。')

def fault(run):
    run=Path(run);mutable(run);c,m,cfg,ident=setup(run,'YZ128');h=history(run/'cases/YZ128');a,b=h[8],h[9];c.restore(a['state']);old=c.state.digest();folder=run/'S2/fault';bind(folder,ident);store=GenerationStore(folder,ident);store.save(c.state,[]);safe=SafePublication(c,store)
    original=c.geometry.op.memory_budget.observe;blocked=False
    def observe(*args,**kwargs):
        if blocked:raise MemoryError('registered temporary resource refusal')
        return original(*args,**kwargs)
    c.geometry.op.memory_budget.observe=observe
    def inject(where,state):
        nonlocal blocked
        if where=='before_commit':blocked=True;raise MemoryError('registered temporary resource refusal')
    try:attempt(run,'S2','YZ128-resource-fault',lambda:safe.advance(inject_step=inject),fault=True)
    except MemoryError as e:
        if 'registered temporary' not in str(e):raise
    else:raise ValueError('fault not triggered')
    unchanged=c.state.digest()==old and store.load()['state'].digest()==old and safe.pending and not c.geometry.cache
    try:safe.prepare()
    except MemoryError:pending_blocked=True
    else:raise ValueError('unavailable device permitted advancement')
    blocked=False;row=attempt(run,'S2','YZ128-resource-recovery',safe.advance);checks=equivalence(c.state,b['state'],row,b['rows'][-1])
    passed=unchanged and all(v['passed'] for v in checks.values())
    write(run/'S2/transaction-check.json',dict(status='passed_scoped' if passed else 'failed',rollback_exact=unchanged,pending_revalidation_refused=pending_blocked,checks=checks,source_step=8,recomputed_step=9,no_failed_frame=not list(folder.rglob('frame.npz')),new_process_resume=read(run/'cases/YZ128/restart-8.json'),scope='controlled allocation refusal, not real driver loss'))
    if not passed:raise ValueError('resource recovery differs')
    update(run,'S2故障与恢复：第9步提交前受控资源拒绝，指针/完整状态/帧保持；设备受限时禁止推进，解除后重算与正常第9步等价。')

def load(run):
    c,m,cfg,ident=setup(run,'YZ128');folder=Path(run)/'cases/YZ128';store=GenerationStore(folder,ident);h=store.history();record=store.load(validator=c.validate);c.restore(record['state'])
    write(Path(run)/'S6/new-checkpoint-load.json',dict(status='passed_scoped',zero_steps=True,new_process=True,state_digest=c.state.digest(),step=c.state.step,history_generations=len(h),initial_contract_sha256=c.contract_id))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cycle','legacy','fault','load']);p.add_argument('--run',type=Path,required=True);p.add_argument('--case',choices=['Y64','Y128','YZ128'],default='Y64');p.add_argument('--stop',type=int,default=18);p.add_argument('--fine',action='store_true');a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='cycle':cycle(a.run,a.case,a.stop,a.fine)
        elif a.phase=='legacy':legacy(a.run)
        elif a.phase=='fault':fault(a.run)
        else:load(a.run)
