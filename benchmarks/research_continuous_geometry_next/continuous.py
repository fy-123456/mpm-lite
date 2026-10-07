"""Authenticated nonzero-origin full coupled steps with inherited reporting prefix."""
import argparse,time
import numpy as np
from .provenance import *
from .fixture import new_setup
from .runtime import attempt,update
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_pressure_window_next.coupling_study import frame
from engine.aniso_phase1.research_startup_substeps_next.schedule import OBSERVATIONS,interval_rows
from benchmarks.research_sequential_next.compare import metric

def sources():
    out=source_files()
    for f in ('fixture.py','continuous.py','runtime.py'):out['benchmarks/research_continuous_geometry_next/'+f]=sha(Path(__file__).parent/f)
    return out

def setup(run,kind='B'):
    parent=history(APP/'cases/boundary32-h')[4];p=read(Path(run)/'S0/state-contract.json')['states']['4']
    if sha(parent['folder']/'state.json')!=p['sha256']:raise ValueError('source mismatch')
    c,m,cfg,bridge=new_setup(run,state=parent['state'])
    if c.identity!=read(APP/'cases/boundary32-h/identity.json')['coupling']:raise ValueError('source physical fixture identity differs')
    if kind=='B':
        from engine.aniso_phase1.research_boundary_reference_next.geometry import install
        bridge=install(c,bridge)
    elif kind=='G1':
        from engine.aniso_phase1.research_boundary_reference_next.geometry import install
        bridge=install(c,bridge)
        from engine.aniso_phase1.research_continuous_geometry_next.geometry import install as optimized
        bridge=optimized(c,bridge)
    else:raise ValueError('unknown branch')
    src=sources()
    if kind=='G1':src['engine/aniso_phase1/research_continuous_geometry_next/geometry.py']=sha(ROOT/'engine/aniso_phase1/research_continuous_geometry_next/geometry.py')
    ident=dict(schema='continuous-geometry-branch-v1',coupling=c.identity,source=p,numerical_sources=src,kind=kind)
    return c,m,ident,bridge,parent

def cycle(run,stop=18,kind='B'):
    run=Path(run);mutable(run);start=time.perf_counter();c,m,ident,bridge,parent=setup(run,kind);folder=run/'cases'/('continuous-'+kind);store=GenerationStore(folder,ident);prior=store.load(validator=c.validate)
    if prior is None:
        write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources']);write(folder/'bridge.json',bridge);write(folder/'execution-protocol.json',dict(source=str(parent['folder']/'state.json'),source_sha256=sha(parent['folder']/'state.json'),global_times_s=c.times.tolist(),start_step=4,stop_step=18,inherited_prefix_rows=4,reported_windows_start_s=c.times[4]));store.save(c.state,[]);rows=[]
    else:
        if read(folder/'identity.json')!=ident:raise ValueError('branch source changed')
        store.history();c.restore(prior['state']);rows=prior['rows'];write(folder/f'restart-{c.state.step}.json',dict(status='passed_scoped',same_digest=c.state.digest()==prior['state'].digest(),step=c.state.step,time_s=c.state.time,new_process=True,new_attempts=0))
    cache=CachedProbes(m);probes={k:[] for k in ('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')};times=[]
    def probe(s):
        f=frame(c,cache,s);times.append(s.time)
        for k in probes:probes[k].append(f[k])
        return f
    chunk_start=c.state.step;f=probe(c.state)
    while c.state.step<stop:
        if time.perf_counter()-start>1200:raise TimeoutError('continuous case budget')
        last=c.state.step+1==18
        row=attempt(run,'S1' if kind=='B' else 'S2',folder.name,lambda:advance_publish(c,store,rows,frame_builder=(lambda s:frame(c,cache,s)) if last else None));rows.append(row)
        if c.state.step in (8,12,16,17,18):f=probe(c.state)
        print('CONTINUOUS',kind,c.state.step,c.state.time,row['min_detF'],row['true_scaled_residual'],flush=True)
    np.savez_compressed(folder/f'probes-{chunk_start}-{stop}.npz',X=f['X'],fiber=m.parent.params.fiber_direction,times=times,**{k:np.asarray(v) for k,v in probes.items()})
    write(folder/'ledger.json',rows);eng=interval_rows(parent['rows']+rows,OBSERVATIONS[:7],average=['reaction_N'],summed=['energy_balance_J','darcy_dissipation_J','numerical_dissipation_J','source_work_J','reservoir_work_J','external_work_J']);write(folder/'engineering-ledger.json',[r for r in eng if r['time']>c.times[4]])
    write(folder/'summary.json',dict(status='passed_scoped' if stop==18 else 'partial',start_step=4,final_step=c.state.step,new_steps=len(rows),time_s=c.state.time,min_detF=min(r['min_detF'] for r in rows),max_residual_fraction=max(r['true_scaled_residual'] for r in rows),seconds_current_process=time.perf_counter()-start,source_zero=all(r['source_work_J']==0 for r in rows),checkpoint_reload=store.load(validator=c.validate)['state'].digest()==c.state.digest()))
    print('CONTINUOUS_DONE',kind,stop,flush=True)

def transaction(run,kind='B'):
    run=Path(run);mutable(run);c,m,ident,bridge,parent=setup(run,kind);hist=history(run/'cases'/('continuous-'+kind));item=next(x for x in hist if x['state'].step==16);expected=next(x for x in hist if x['state'].step==17);c.restore(item['state']);old=c.state.digest();stage='S1' if kind=='B' else 'S2';folder=run/stage/'fault';store=GenerationStore(folder,ident);store.save(c.state,[])
    def inject(where,state):
        if where=='before_commit':raise ValueError('registered continuous failure')
    try:attempt(run,stage,'continuous-'+kind+'-fault',lambda:c.step(inject=inject),fault=True)
    except ValueError as e:
        if 'registered continuous failure' not in str(e):raise
        if c.state.digest()!=old or store.load()['state'].digest()!=old or len(c.geometry.cache):raise ValueError('fault rollback differs')
    else:raise ValueError('failure not triggered')
    write(run/stage/'transaction-check.json',dict(status='passed_scoped',rollback_exact=True,cache_cleared=True,source_step=16,target_step=17,digest=old))
    row=attempt(run,stage,'continuous-'+kind+'-recovery',lambda:advance_publish(c,store,[]));checks=state_checks(c.state,expected['state'])
    if not all(x['passed'] for x in checks.values()):raise ValueError('recovered state differs')
    write(run/stage/'recovery-comparison.json',dict(status='passed_scoped',checks=checks,step=c.state.step,time_s=c.state.time,expected_digest=expected['state'].digest(),actual_digest=c.state.digest()))
    print('TRANSACTION',kind,'passed',flush=True)

def state_checks(a,b):
    out={k:metric(getattr(a,k),getattr(b,k),1e-8,2e-5) for k in ('q','velocity','predictor')}
    for k,tol in [('pressure_Pa',1e-6),('flux_interval_m3_s',1e-10),('content_m3',1e-10),('cumulative_source_m3',1e-10),('cumulative_boundary_m3',1e-10),('cumulative_numerical_dissipation_J',1e-12)]:out[k]=metric(a.child_states['fluid'][k],b.child_states['fluid'][k],tol,2e-5)
    out['global_index']=dict(passed=a.step==b.step and abs(a.time-b.time)<1e-15)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cycle','fault']);p.add_argument('--run',type=Path,required=True);p.add_argument('--stop',type=int,choices=[8,18],default=18);p.add_argument('--kind',choices=['B','G1'],default='B');a=p.parse_args()
    with serial_lock(a.run):cycle(a.run,a.stop,a.kind) if a.phase=='cycle' else transaction(a.run,a.kind)
