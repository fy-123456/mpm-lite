"""Only the missing 32-cell trajectories, with explicit owned implementation."""
from pathlib import Path
import argparse,time,os
import numpy as np
import scipy.linalg as la
from .provenance import *
from .physics import baseline_model
from .performance import numerical_sources
from engine.aniso_phase1.research_restoring_rt0_next.fixture import construct
from engine.aniso_phase1.research_candidate_observable_next.fixture import ObservableCoupling
from engine.aniso_phase1.research_restoring_rt0_next.device_rt0 import DeviceGeometry
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_candidate_observable_next.coupling import algebra,pressure_check

def protocol():return read(APP/'S3/new-scene-protocol.json')

def prepare(run):
    run=Path(run);verify(run);p=protocol();decision=read(run/'S2/performance-decision.json')
    register(run,'S3/grid-protocol.json',dict(status='registered',source=str(APP/'S3/new-scene-protocol.json'),source_sha256=sha(APP/'S3/new-scene-protocol.json'),pressure_grid='fine',pressure_grid_cells=32,time_refinements=[1,2],steps=[16,32],max_attempts=52,backend=decision['backend'],zero_grips=True,full_mass_order=7,material_order=7,single_model=True,hard_case_seconds=1800,hard_total_seconds=3600,frames_per_case=2))

def setup(run,fine=False,state=None):
    run=Path(run);m,cfg=baseline_model(run,pressure=True);p=read(run/'S3/grid-protocol.json');c,bridge=construct(m,cfg,protocol(),'fine',fine=fine,state=state,backend=p['backend']);return c,m,cfg

def operators(run):
    run=Path(run);verify(run);c,m,cfg=setup(run);g=c.geometry;q=c.state.q;v=g.evaluate(q);cache=CachedProbes(m);rng=np.random.default_rng(81);d=rng.normal(size=q.shape);d[m.fixed]=0;d/=max(float(np.max(abs(a@d))) for a in cache.maps[1:]);eps=1e-5
    plus=g.evaluate(q+eps*d);minus=g.evaluate(q-eps*d);G=g.discrete(q,q+eps*d);fd=(plus['volume']-minus['volume'])/(2*eps);analytic=np.einsum('cij,ij->c',v['gradient'],d);err=float(la.norm(fd-analytic));workerr=float(np.max(abs(plus['volume']-v['volume']-np.einsum('cij,ij->c',G,eps*d))))
    if err>1e-9+.001*la.norm(analytic) or workerr>1e-11:raise ValueError('32-cell volume work/gradient failed')
    records=[]
    if hasattr(g,'assembler'):
        for label,state in [('rest',q),('manufactured',q+eps*d)]:
            F=g.field(state);H,J=g.topology.assemble(g.X,g.total_weights,g.cell_ids,F.numpy(),g.mobility);fast=g.evaluate(state);passed=bool(np.allclose(H,fast['H'],atol=1e-8,rtol=2e-5));records.append(dict(state=label,H_max_error=float(np.max(abs(H-fast['H']))),passed=passed))
        if not all(x['passed'] for x in records):raise ValueError('32-cell H equivalence failed')
    # Exact discrete fixed-skeleton time check using the actual selected H.
    a=algebra(protocol()['cuts']['fine'],protocol()['parameters']);a['H']=v['H'];a['Z']=la.solve(a['H'],a['top'].B.T,assume_a='pos');a['z0']=-la.solve(a['H'],a['top'].boundary_term(protocol()['parameters']['reservoir_Pa']),assume_a='pos');a['L']=a['top'].B@a['Z'];a['rhs']=a['src']-a['top'].B@a['z0'];N=a['top'].cells;a['aug'][:]=0.;a['aug'][:N,:N]=-a['L']/a['C'][:,None];a['aug'][:N,-1]=a['rhs']/a['C'];a['aug'][N,:N]=np.sum(a['top'].B@a['Z'],axis=0);a['aug'][N,-1]=np.sum(a['top'].B@a['z0'])
    checks={str(n):pressure_check(a,protocol()['parameters'],np.linspace(0,.0002,n+1)) for n in (16,32)}
    if not all(x['status']=='passed_scoped' for x in checks.values()):raise ValueError('selected H fixed-pressure time gate failed')
    write(run/'S3/operator-check.json',dict(status='passed_scoped',volume_gradient_error=err,discrete_work_error=workerr,H_equivalence=records,current_H=True,numerical_source_sha256=numerical_sources(),geometry=g.identity))
    write(run/'S3/pressure-schedule-check.json',dict(status='passed_scoped',records=checks,actual_selected_H_used=True,fixed_matrix_steps=48))
    print('GRID32_OPERATOR',err,workerr,records,flush=True)

def cycle(run,fine=False,stop_after=None):
    run=Path(run);verify(run)
    if read(run/'S3/operator-check.json')['status']!='passed_scoped':raise ValueError('missing grid operator gate')
    name='grid32-half' if fine else 'grid32-h';folder=run/'cases'/name;loaded=None;start=time.perf_counter()
    if (folder/'identity.json').exists():
        identity=read(folder/'identity.json')
        if identity['numeric_sources']!=numerical_sources() or identity['driver_sha256']!=sha(__file__):raise ValueError('grid continuation source changed')
        store=GenerationStore(folder,identity);store.history();loaded=store.load()
    c,m,cfg=setup(run,fine,loaded['state'] if loaded else None);cache=CachedProbes(m);p=read(run/'S3/grid-protocol.json')
    if loaded is None:
        identity=dict(schema='restoring-grid32-v1',coupling=c.identity,numeric_sources=numerical_sources(),driver_sha256=sha(__file__),protocol_sha256=sha(run/'S3/grid-protocol.json'));write(folder/'identity.json',identity);write(folder/'execution-protocol.json',dict(solid=cfg,grid='fine',time_refinement=2 if fine else 1,coupling=c.identity));snapshot(folder/'source',dict(numerical_sources(),**{str(Path(__file__).relative_to(ROOT)):sha(__file__)}));store=GenerationStore(folder,identity);store.save(c.state,[],frame=frame(c,cache,c.state));rows=[]
    else:
        if identity['coupling']!=c.identity:raise ValueError('grid model identity changed')
        rows=loaded['rows']
    past=read(folder/'attempts.json') if (folder/'attempts.json').exists() else {};attempts=past.get('attempts',0);past_seconds=past.get('seconds',0.);initial=c.state.step;count=0
    while c.state.step<len(c.times)-1 and (stop_after is None or count<stop_after):
        others=[read(x) for x in (run/'cases').glob('grid32-*/attempts.json') if x.parent!=folder];seconds=past_seconds+time.perf_counter()-start
        if attempts+sum(x['attempts'] for x in others)>=p['max_attempts'] or seconds>p['hard_case_seconds'] or seconds+sum(x['seconds'] for x in others)>p['hard_total_seconds']:raise TimeoutError('registered grid budget exhausted')
        attempts+=1;write(folder/'attempts.json',dict(attempts=attempts,seconds=seconds,committed=c.state.step));tick=time.perf_counter()
        try:row=advance_publish(c,store,rows,frame_builder=(lambda s:frame(c,cache,s)) if c.state.step+1==len(c.times)-1 else None)
        except Exception as e:
            write(folder/'attempts.json',dict(attempts=attempts,seconds=past_seconds+time.perf_counter()-start,committed=c.state.step));write(folder/'failure.json',dict(status='limited',error=repr(e),rollback=store.load()['state'].digest()==c.state.digest(),state_step=c.state.step));raise
        rows.append(row);count+=1
        print(name,c.state.step,'seconds',round(time.perf_counter()-tick,3),'J',row['min_detF'],flush=True)
    seconds=past_seconds+time.perf_counter()-start;write(folder/'attempts.json',dict(attempts=attempts,seconds=seconds,committed=c.state.step));write(folder/'ledger.json',rows)
    segments=read(folder/'segments.json') if (folder/'segments.json').exists() else [];segments.append(dict(pid=os.getpid(),start_step=initial,end_step=c.state.step));write(folder/'segments.json',segments)
    write(folder/'summary.json',dict(status='passed_scoped' if c.state.step==len(c.times)-1 else 'in_progress',steps=c.state.step,end_s=c.state.time,min_detF=min(x['min_detF'] for x in rows),seconds=seconds,attempts=attempts,actual_new_process_restart=len({x['pid'] for x in segments})>1,checkpoint_reload=store.load()['state'].digest()==c.state.digest()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','operators','cycle']);p.add_argument('--run',type=Path,required=True);p.add_argument('--fine',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='cycle':cycle(a.run,a.fine,a.stop_after)
        else:globals()[a.phase](a.run)
