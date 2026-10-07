"""Small actual zero-source two-way trajectories with owned checkpoints."""
import argparse,time
import numpy as np
from .provenance import *
from .fixture import setup
from .runtime import attempt,update
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_continuous_geometry_next.continuous import state_checks

FIELDS=('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')

def case_name(grid,fine=False):return 'pressure'+('32' if grid=='base' else '128')+'-D3'+('-half' if fine else '')

def cycle(run,grid,stop,fine=False):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,cfg,ident=setup(run,grid,fine);built=time.perf_counter()-tick
    folder=run/'cases'/case_name(grid,fine);store=GenerationStore(folder,ident);prior=store.load(validator=c.validate)
    if prior is None:
        write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources']);store.save(c.state,[]);rows=[]
        E0=m.evaluate(c.state.q)['U']+m.kinetic(c.state.velocity)+.5*np.sum(c.core.capacity*np.asarray(c.state.child_states['fluid']['pressure_Pa'])**2)
        write(folder/'execution-protocol.json',dict(initial_energy_J=E0,initial_digest=c.state.digest(),full_times_s=c.times.tolist(),execution_end_s=75e-6,residual_window_s=c.core.window,mass_sha256=digest(m.M.tolist()),material_order=7,source_zero=True,backend='D3',geometry=c.geometry.identity))
    else:
        if read(folder/'identity.json')!=ident:raise ValueError('executed source identity changed')
        store.history();c.restore(prior['state']);rows=prior['rows']
        write(folder/f'restart-{c.state.step}.json',dict(status='passed_scoped',same_digest=c.state.digest()==prior['state'].digest(),step=c.state.step,time_s=c.state.time,new_process=True))
    if stop>18*(2 if fine else 1) or stop<c.state.step:raise ValueError('registered 75us endpoint required')
    cache=CachedProbes(m);samples={k:[] for k in FIELDS};times=[];factor=2 if fine else 1
    observations={i*factor for i in (0,4,8,12,16,17,18)};frames={8,16,18} if not fine else set()
    def probe(s):
        f=frame(c,cache,s);times.append(s.time)
        for k in FIELDS:samples[k].append(f[k])
        return f
    start=c.state.step;f=probe(c.state);c.geometry.profile={}
    while c.state.step<stop:
        row=attempt(run,'S2',folder.name,lambda:advance_publish(c,store,rows,frame_builder=(lambda s:frame(c,cache,s)) if c.state.step+1 in frames else None));rows.append(row)
        if c.state.step in observations:f=probe(c.state)
        print('D3_STEP',grid,fine,c.state.step,c.state.time,row['min_detF'],row['true_scaled_residual'],flush=True)
    np.savez_compressed(folder/f'probes-{start}-{stop}.npz',X=f['X'],fiber=m.parent.params.fiber_direction,times=times,**{k:np.asarray(v) for k,v in samples.items()})
    write(folder/'ledger.json',rows);write(folder/f'profile-{start}-{stop}.json',dict(build_s=built,geometry=c.geometry.profile,process_s=time.perf_counter()-tick))
    write(folder/'summary.json',dict(status='passed_scoped' if stop==18*factor else 'partial',grid=grid,fine=fine,final_step=c.state.step,time_s=c.state.time,new_steps=len(rows),initial_source='same authenticated rest and constant pressure; no coarse state interpolation',min_detF=min(r['min_detF'] for r in rows),max_residual_fraction=max(r['true_scaled_residual'] for r in rows),last_state_reload=store.load(validator=c.validate)['state'].digest()==c.state.digest()))
    update(run,f'S2 {grid}{" half" if fine else ""}：已提交到第{stop}步/{c.state.time*1e6:g}微秒，本进程{stop-start}步，minJ={min(r["min_detF"] for r in rows):.9g}。')

def fault(run):
    run=Path(run);mutable(run);c,m,cfg,ident=setup(run,'yz');h=history(run/'cases'/case_name('yz'));item=h[8];expected=h[9];c.restore(item['state']);old=c.state.digest()
    folder=run/'S2/fault';store=GenerationStore(folder,ident);write(folder/'identity.json',ident);store.save(c.state,[])
    def inject(where,state):
        if where=='before_commit':raise ValueError('registered D3 precommit failure')
    try:attempt(run,'S2','yz-fault',lambda:advance_publish(c,store,[],inject_step=inject),fault=True)
    except ValueError as e:
        if 'registered D3' not in str(e):raise
        unchanged=c.state.digest()==old and store.load()['state'].digest()==old and len(c.geometry.cache)==0
        if not unchanged:raise ValueError('D3 rollback polluted state')
    else:raise ValueError('fault not injected')
    attempt(run,'S2','yz-recovery',lambda:advance_publish(c,store,[]));checks=state_checks(c.state,expected['state'])
    if not all(v['passed'] for v in checks.values()):raise ValueError('D3 fault recovery differs')
    write(run/'S2/transaction-check.json',dict(status='passed_scoped',rollback_exact=unchanged,checks=checks,committed_source_step=8,recomputed_step=9,original_digest=old,no_failed_frame=not list(folder.rglob('frame.npz'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['cycle','fault']);p.add_argument('--run',type=Path,required=True);p.add_argument('--grid',choices=['base','yz'],default='base');p.add_argument('--stop',type=int,default=18);p.add_argument('--fine',action='store_true');a=p.parse_args()
    with serial_lock(a.run):cycle(a.run,a.grid,a.stop,a.fine) if a.phase=='cycle' else fault(a.run)
