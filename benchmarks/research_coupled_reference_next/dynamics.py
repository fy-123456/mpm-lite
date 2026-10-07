"""Small BD bridge and profiled physical step, parent stores remain read-only."""
import argparse,time
import numpy as np
import warp as wp
from .provenance import *
from .fixture import setup,inputs
from .runtime import attempt,update
from benchmarks.research_transverse_reference_next.coupling import start_store,field_stats
from benchmarks.research_transverse_next.trajectory import equivalence,instrument
from benchmarks.research_continuous_geometry_next.review import balances
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_pressure_window_next.coupling_study import frame
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_transverse_next.recovery import SafePublication

def bridge(run):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,cfg,ident=setup(run);build=time.perf_counter()-tick;h=inputs();a,b=h[8],h[9]
    folder,store=start_store(run,'S2/bridge',c,ident,a);safe=SafePublication(c,store);cache=CachedProbes(m);expected=frame(c,cache,b['state'])
    c.geometry.cache.clear();c.geometry.profile={};timings=instrument(c,m)
    # Raw transpose use is timed separately, without claiming nested time is additive.
    raw_timing=dict(calls=0,seconds=0.);originals=[]
    for op in {id(mp.raw[1]):mp.raw[1] for _,_,mp,_ in c.geometry.local}.values():
        original=op.apply;originals.append((op,original))
        def measured(*args,_fn=original,**kwargs):
            wp.synchronize_device(m.operator.device);t=time.perf_counter()
            try:return _fn(*args,**kwargs)
            finally:wp.synchronize_device(m.operator.device);raw_timing['calls']+=1;raw_timing['seconds']+=time.perf_counter()-t
        op.apply=measured
    t=time.perf_counter();row=attempt(run,'S2','BD-bridge',safe.advance);advance=time.perf_counter()-t
    for op,original in originals:op.apply=original
    t=time.perf_counter();actual=frame(c,cache,c.state);probe_s=time.perf_counter()-t
    checks=equivalence(c.state,b['state'],row,b['rows'][-1]);checks.update({'field_'+k:metric(actual[k],expected[k],1e-8,2e-5) for k in ('x','velocity','PK1','PK1_total','Cauchy_total')})
    rejects={}
    for key in ('transverse_initial_contract','explicit_pressure_grid'):
        bad=c.state.clone();bad.child_states[key]='foreign'
        try:c.validate(bad)
        except ValueError:rejects[key]=True
        else:rejects[key]=False
    bal=balances(store.history());passed=all(x['passed'] for x in checks.values()) and all(rejects.values()) and bal['passed']
    write(run/'S2/bridge-check.json',dict(status='passed_scoped' if passed else 'failed',checks=checks,rejected=rejects,balances=bal,actual_backend=type(c.geometry).__name__,new_steps=1))
    write(run/'S2/backend-binding.json',dict(status='passed_scoped',execution_identity=ident,mass_sha256=digest(m.M.tolist()),material_order=7,mass_order=7,cells=c.geometry.cells,free_solid_dofs=len(m.ids),initial_digest=c.initial_digest))
    write(run/'S2/recovery-scope.json',dict(status='inherited',source=str(APP/'S3/buffer-lifecycle.json'),source_sha256=sha(APP/'S3/buffer-lifecycle.json'),reason='unchanged production BD transient buffers and SafePublication; reference does not persist solver state',new_fault_steps=0,real_driver_loss=False))
    write(run/'S3/BD-profile.json',dict(status='passed_scoped',build_s=build,advance_s=advance,probe_s=probe_s,geometry=c.geometry.profile,material=timings,raw_transpose=raw_timing,publication=safe.profile,extra_dynamic_steps=0,scope='synchronized diagnostic bridge, nested timers are not additive; use uninstrumented pairs for speed'))
    write(run/'S2/field-statistics.json',field_stats(actual,m.parent.params.fiber_direction))
    update(f'S2桥接：BD第8→9步、同位置场、含量/能量账本及错身份拒绝 {"通过" if passed else "失败"}。同步诊断推进{advance:.3f}s；局部raw转置{raw_timing["seconds"]:.3f}s，{raw_timing["calls"]}次。')
    print('BRIDGE',passed,advance,raw_timing,c.geometry.profile,flush=True)
    if not passed:raise ValueError('BD bridge differs from authenticated state')

def load(run):
    run=Path(run);mutable(run);c,m,cfg,ident=setup(run);folder=run/'S2/bridge';prior=read(folder/'identity.json');check(ROOT,prior['numerical_sources'])
    if ident['coupling']!=prior['coupling']:raise ValueError('load physical identity differs')
    store=GenerationStore(folder,prior);store.history();record=store.load(validator=c.validate);c.restore(record['state'])
    write(run/'S5/load-check.json',dict(status='passed_scoped',fresh_process=True,zero_steps=True,actual_backend=type(c.geometry).__name__,state_digest=c.state.digest(),step=c.state.step,execution_identity=ident,source=str(folder),source_identity_sha256=sha(folder/'identity.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['bridge','load']);p.add_argument('--run',type=Path,required=True);a=p.parse_args();globals()[a.phase](a.run)
