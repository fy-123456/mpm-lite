"""Actual full solid/fluid microsteps on the frozen boundary32 grid."""
import argparse,time,resource
import numpy as np
from .provenance import *
from .fixture import new_setup
from engine.aniso_phase1.research_startup_substeps_next.schedule import OBSERVATIONS,indices,interval_rows
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric

def sources():
    out=source_files()
    for key in ['benchmarks/research_startup_substeps_next/fixture.py','benchmarks/research_startup_substeps_next/coupling.py','engine/aniso_phase1/research_startup_substeps_next/schedule.py']:out[key]=sha(ROOT/key)
    return out

def identity(c):return dict(schema='startup-substeps-coupled-v1',coupling=c.identity,numerical_sources=sources())
def name(fine):return 'boundary32-'+('half' if fine else 'h')

def attempts(run):
    return sum(read(p)['attempts'] for p in (Path(run)/'cases').glob('*/attempts.json'))+sum(read(p)['attempts'] for p in (Path(run)/'S3').rglob('attempt.json')) if (Path(run)/'cases').exists() else 0

def operators(run):
    import warp as wp
    run=Path(run);mutable(run);start=time.perf_counter();c,m,cfg,bridge=new_setup(run);g=c.geometry;q=c.state.q;v=g.evaluate(q)
    d=np.zeros_like(q);d[m.free]=np.random.default_rng(83).normal(size=d[m.free].shape);d*=.02/max(float(np.max(abs(g.field(d,True).numpy()))),1e-30)
    eps=1e-4;plus=g.evaluate(q+eps*d);minus=g.evaluate(q-eps*d);derivative=(plus['volume']-minus['volume'])/(2*eps);actual=np.einsum('cij,ij->c',v['gradient'],d);error=float(np.linalg.norm(derivative-actual));work=float(np.max(abs(plus['volume']-v['volume']-np.einsum('cij,ij->c',g.discrete(q,q+eps*d),eps*d))))
    records=[]
    for label,qq in [('rest',q),('manufactured',q+d)]:
        H,_=g.topology.assemble(g.X,g.total_weights,g.cell_ids,g.field(qq).numpy(),g.mobility);cmp=metric(H,g.evaluate(qq)['H'],1e-8,2e-5);records.append(dict(state=label,H=cmp))
    rejected=[]
    for key in ('explicit_pressure_grid','time'):
        bad=c.state
        if key=='time':bad.time=1e-9
        else:bad.child_states[key]='foreign'
        try:c.validate(bad)
        except ValueError:rejected.append(key)
        else:raise ValueError('invalid state accepted')
    passed=error<=1e-9+.001*np.linalg.norm(actual) and work<1e-11 and all(r['H']['passed'] for r in records)
    write(run/'S3/operator-check.json',dict(status='passed_scoped' if passed else 'limited',gradient_error=error,volume_work_error_m3=work,records=records,bridge=bridge,invalid_states_rejected=rejected,mass_sha256=digest(m.M.tolist()),initial_source=c.source.tolist(),initial_pressure=c.state.child_states['fluid']['pressure_Pa'],geometry=g.identity,seconds=time.perf_counter()-start,sources=sources(),new_dynamic_steps=0))
    write(run/'S3/memory-estimate.json',dict(local_metadata_bytes=g.local_bytes,construction_peak_bytes=g.identity['construction_metadata_peak_bytes'],cap_bytes=g.identity['cap_bytes'],peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    if not passed:raise ValueError('new boundary32 operator check failed')
    print('OPERATORS',passed,'gradient',error,'work',work,'metadata',g.local_bytes,flush=True)

def cycle(run,fine=False):
    run=Path(run);mutable(run)
    if read(run/'S3/operator-check.json')['status']!='passed_scoped':raise ValueError('operator gate')
    begun=time.perf_counter();c,m,cfg,bridge=new_setup(run,fine);folder=run/'cases'/name(fine);ident=identity(c)
    if (folder/'identity.json').exists():raise ValueError('case already exists; explicit recovery required')
    write(folder/'identity.json',ident);snapshot(folder/'source',sources());write(folder/'execution-protocol.json',dict(coupling=c.identity,actual_times_s=c.times.tolist(),observations_s=OBSERVATIONS.tolist(),sources=sources(),fine=fine,stage='S3'));write(folder/'bridge.json',bridge)
    cache=CachedProbes(m);store=GenerationStore(folder,ident);store.save(c.state,[]);initial=c.state;E0=float(m.kinetic(initial.velocity)+m.evaluate(initial.q)['U']+.5*np.sum(c.core.capacity*np.array(initial.child_states['fluid']['pressure_Pa'])**2));rows=[];count=0
    ix=indices(c.times,OBSERVATIONS);probes={k:[] for k in ('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')};field=frame(c,cache,initial)
    for k in probes:probes[k].append(field[k])
    field_times=[0.]
    while c.state.step<len(c.times)-1:
        elapsed=time.perf_counter()-begun;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        others=sum(read(p).get('seconds',0.) for p in (run/'cases').glob('*/attempts.json') if p.parent!=folder)
        if rss>16 or elapsed>1200 or elapsed+others>2400 or attempts(run)>=89:raise TimeoutError('actual coupled resource budget')
        count+=1;write(folder/'attempts.json',dict(stage='S3',attempts=count,committed=len(rows),seconds=elapsed));last=c.state.step+1==len(c.times)-1
        try:row=advance_publish(c,store,rows,frame_builder=(lambda s:frame(c,cache,s)) if last else None)
        except BaseException as e:
            c.restore(store.load()['state']);c.geometry.cache.clear();write(folder/'failure.json',dict(status='limited',error=repr(e),rollback=store.load()['state'].digest()==c.state.digest(),step=c.state.step,time_s=c.state.time));raise
        rows.append(row)
        if c.state.step in ix:
            field=frame(c,cache,c.state)
            for k in probes:probes[k].append(field[k])
            field_times.append(c.state.time)
            print('COUPLED',name(fine),c.state.step,round(c.state.time*1e6,3),'us','J',row['min_detF'],'p',min(row['pressure_Pa']),'residual',row['true_scaled_residual'],flush=True)
    elapsed=time.perf_counter()-begun;write(folder/'attempts.json',dict(stage='S3',attempts=count,committed=len(rows),seconds=elapsed));write(folder/'ledger.json',rows)
    np.savez_compressed(folder/'probes.npz',X=field['X'],fiber=m.parent.params.fiber_direction,times=field_times,**{k:np.asarray(v) for k,v in probes.items()})
    eng=interval_rows(rows,OBSERVATIONS,average=['reaction_N'],summed=['darcy_dissipation_J','numerical_dissipation_J','energy_balance_J','source_work_J','reservoir_work_J','external_work_J']);write(folder/'engineering-ledger.json',eng)
    fluid=c.state.child_states['fluid'];first=initial.child_states['fluid'];mass=float(np.sum(np.array(fluid['content_m3'])-first['content_m3'])+fluid['cumulative_boundary_m3']-np.sum(fluid['cumulative_source_m3']));num=sum(r['numerical_dissipation_J'] for r in rows)
    if abs(mass)>1e-10 or not np.isclose(num,fluid['cumulative_numerical_dissipation_J'],rtol=1e-12,atol=1e-20):raise ValueError('cumulative mass/numerical ledger failed')
    balance=rows[-1]['total_energy_J']-E0+sum(r['darcy_dissipation_J']+r['numerical_dissipation_J']-r['external_work_J']-r['source_work_J']-r['reservoir_work_J'] for r in rows)
    if abs(balance)>1e-9+.01*abs(E0):raise ValueError('cumulative energy balance')
    write(folder/'summary.json',dict(status='passed_scoped',steps=count,end_s=c.state.time,initial_energy_J=E0,cumulative_energy_balance_J=balance,cumulative_mass_defect_m3=mass,Dnum_J=num,Darcy_J=sum(r['darcy_dissipation_J'] for r in rows),min_detF=min(r['min_detF'] for r in rows),minimum_pressure_Pa=min(min(r['pressure_Pa']) for r in rows),max_residual_fraction=max(r['true_scaled_residual'] for r in rows),max_pressure_work_error_J=max(float(np.max(abs(np.array(r['pressure_work_defect_J'])))) for r in rows),source_zero=all(r['source_work_J']==0 for r in rows),seconds=elapsed,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,checkpoint_reload=store.load(validator=c.validate)['state'].digest()==c.state.digest()))
    print('COMPLETE',name(fine),count,'steps',elapsed,flush=True)

def transaction(run,restart=False):
    run=Path(run);mutable(run);start=time.perf_counter();folder=run/'cases'/name(False);c,m,cfg,bridge=new_setup(run);ident=identity(c)
    if ident!=read(folder/'identity.json'):raise ValueError('transaction source changed')
    states=history(folder)
    if restart:
        c.restore(states[-1]['state']);write(run/'S3/restart.json',dict(status='passed_scoped',same_digest=c.state.digest()==states[-1]['state'].digest(),digest=c.state.digest(),step=c.state.step,time_s=c.state.time,new_process=True,new_attempts=0));return
    item=states[int(indices(c.times,OBSERVATIONS)[2])];c.restore(item['state']);old=c.state.digest();fault=run/'S3/fault';store=GenerationStore(fault,ident);store.save(c.state,[])
    if attempts(run)>=89:raise RuntimeError('fault attempt budget')
    write(fault/'attempt.json',dict(stage='S3',attempts=1,accepted=False))
    def inject(stage,state):
        if stage=='before_commit':raise ValueError('registered substep failure')
    try:c.step(inject=inject)
    except ValueError as e:
        if 'registered substep failure' not in str(e):raise
        if c.state.digest()!=old or len(c.geometry.cache) or store.load()['state'].digest()!=old:raise ValueError('rollback differs')
        write(run/'S3/fault.json',dict(status='passed_scoped',full_rollback=True,cache_cleared=True,seconds=time.perf_counter()-start,new_attempts=1));return
    raise ValueError('fault not reached')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['operators','cycle','fault','restart']);p.add_argument('--run',type=Path,required=True);p.add_argument('--fine',action='store_true');a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='operators':operators(a.run)
        elif a.phase=='cycle':cycle(a.run,a.fine)
        else:transaction(a.run,a.phase=='restart')
