"""Owned theta-pressure fixture: one nested grid family, bounded actual steps."""
from pathlib import Path
import argparse,copy,time,resource
import numpy as np
from .provenance import *
from .physics import baseline_model
from .continuation import prefix_audit, bridge_state
from engine.aniso_phase1.research_zero_source_next.coupled import ZeroSourceCoupling, explicit_zero
from engine.aniso_phase1.research_restoring_rt0_next.device_rt0 import DeviceGeometry
from engine.aniso_phase1.research_stabilization_boundary_next.local_geometry import install
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric,impulse_average

def numerical_sources():
    out=source_files()
    for name in ('coupled.py',):
        key='engine/aniso_phase1/research_zero_source_next/'+name;out[key]=sha(ROOT/key)
    return out

def source_for_protocol(params):
    return explicit_zero(params.get('source_density_s_inv'))

def stage_for(grid,method):
    return 'S3' if grid=='fine' else ('S1' if method=='backward-euler' else 'S2')

def case_name(grid,fine=False,method='backward-euler'):
    return 'zero-'+method+'-'+grid+('-half' if fine else '-h')

def setup(run,grid='coarse',fine=False,method='backward-euler'):
    p=read(Path(run)/'S0/input-contract.json');params=p['parameters'];ts=np.array(p['times_s'])
    source=source_for_protocol(params)
    if grid not in p['cuts'] or method not in p['methods']:raise ValueError('unregistered grid/method')
    if fine:ts=np.sort(np.r_[ts,.5*(ts[:-1]+ts[1:])])
    m,cfg=baseline_model(run,pressure=True)
    c=ZeroSourceCoupling(m,cfg,ts,method=method,source_m3_s=source,cuts=p['cuts'][grid],alpha=params['alpha'],storage=params['storage'],pressure0=params['pressure0_Pa'],reservoir=params['reservoir_Pa'],mobility=params['mobility_scale']*MOBILITY)
    if np.any(c.source!=0):raise ValueError('actual source differs from frozen zero source')
    cap=min(256*2**20,int(.02*m.operator.memory_budget.initial_free));state=c.state;before=physical_payload(state)
    geo=DeviceGeometry.from_owned(c.geometry,cap_bytes=cap);core=c.core;core.geometry=geo
    core.identity=dict(core.identity,geometry=geo.identity,implementation='current-device-RT0');core.signature=digest(core.identity)
    c.identity=dict(c.identity,core=core.identity);c.signature=digest(c.identity);state.child_states['fluid']['model']=core.signature;state.child_states['explicit_pressure_grid']=c.signature
    core._transaction=StateTransaction(state,validator=c.validate);install(c)
    if physical_payload(c.state)!=before:raise ValueError('geometry installation changed initial physics')
    return c,m,cfg

def operators(run,grid):
    run=Path(run);verify(run);c,m,cfg=setup(run,grid);g=c.geometry;q=c.state.q;v=g.evaluate(q);cache=CachedProbes(m)
    direction=np.random.default_rng(83).normal(size=q.shape);direction[m.fixed]=0
    direction*=.02/max(float(np.max(abs(g.field(direction,True).numpy()))),1e-30);eps=1e-4
    plus=g.evaluate(q+eps*direction);minus=g.evaluate(q-eps*direction)
    derivative=(plus['volume']-minus['volume'])/(2*eps);actual=np.einsum('cij,ij->c',v['gradient'],direction)
    de=float(np.linalg.norm(derivative-actual));work=float(np.max(abs(plus['volume']-v['volume']-np.einsum('cij,ij->c',g.discrete(q,q+eps*direction),eps*direction))))
    records=[]
    for label,qq in [('rest',q),('manufactured',q+eps*direction)]:
        F=g.field(qq);H,_=g.topology.assemble(g.X,g.total_weights,g.cell_ids,F.numpy(),g.mobility);fast=g.evaluate(qq)
        cmp=metric(H,fast['H'],1e-8,2e-5);records.append(dict(state=label,H=cmp))
    # Exact theta pressure blocks and the original midpoint matrix remain available.
    h=float(c.times[1]);half=c.core.theta_matrix(h,v['gradient'],v['gradient'],v['H'],.5);original=c.core.matrix(h,v['gradient'],v['gradient'],v['H'])
    if de>1e-9+.001*np.linalg.norm(actual) or work>1e-11 or not all(r['H']['passed'] for r in records) or not np.array_equal(half,original):raise ValueError('new fixture operator check failed')
    rejected=[]
    for field,value in [('cumulative_numerical_dissipation_J',float('nan')),('model','foreign')]:
        bad=c.state;bad.child_states['fluid'][field]=value
        try:c.validate(bad)
        except ValueError:rejected.append(field)
        else:raise ValueError('foreign history accepted')
    write(run/f'S0/operator-{grid}.json',dict(status='passed_scoped',gradient_error=de,volume_work_error=work,H=records,theta_half_matrix_equal=True,rejected_invalid_fields=rejected,geometry=g.identity,current_F=True,local_metadata_bytes=g.local_bytes,sources=numerical_sources(),quasi_newton_exact_jacobian=False,new_dynamic_steps=0,actual_source_m3_s=c.source.tolist(),actual_theta=c.core.thetas.tolist(),actual_initial_cumulative_source=c.state.child_states['fluid']['cumulative_source_m3']))
    print('COUPLED_OPERATORS',grid,de,work,flush=True)

def identity(c):
    return dict(schema='zero-source-theta-coupled-case-v1',coupling=c.identity,numerical_sources=numerical_sources(),driver_sha256=sha(__file__))

def cycle(run,grid,fine=False,method='backward-euler'):
    run=Path(run);verify(run)
    if read(run/f'S0/operator-{grid}.json')['status']!='passed_scoped':raise ValueError('missing operators')
    stage=stage_for(grid,method);name=case_name(grid,fine,method);folder=run/'cases'/name
    if (folder/'identity.json').exists():raise ValueError('case already started; no silent rerun')
    begun=time.perf_counter();c,m,cfg=setup(run,grid,fine,method);cache=CachedProbes(m);ident=identity(c)
    initial=c.state;p0=np.array(initial.child_states['fluid']['pressure_Pa'])
    E0=float(m.kinetic(initial.velocity)+m.evaluate(initial.q)['U']+.5*np.sum(c.core.capacity*p0*p0))
    inherited=[];audit=dict(status='not_triggered',reason='independent initial-state trajectory')
    if stage=='S1':
        original=APP/'cases'/('startup-zero-coarse-half' if fine else 'startup-zero-coarse-h')
        try:inherited,audit=prefix_audit(c,original)
        except (ValueError,KeyError) as e:audit=dict(status='replay_from_rest',reason=str(e),new_attempts_planned=len(c.times)-1)
    write(run/stage/(name+'-prefix-audit.json'),audit)
    write(folder/'identity.json',ident);write(folder/'execution-protocol.json',dict(solid=cfg,grid=grid,fine=fine,method=method,coupling=c.identity,stage=stage));snapshot(folder/'source',dict(numerical_sources(),**{str(Path(__file__).relative_to(ROOT)):sha(__file__),'benchmarks/research_zero_source_next/continuation.py':sha(Path(__file__).with_name('continuation.py'))}));store=GenerationStore(folder,ident)
    if inherited:
        for item in inherited:
            owned=bridge_state(c,item['state']);store.save(owned,item['rows'])
        c.restore(owned);rows=inherited[-1]['rows'].copy()
    else:store.save(c.state,[]);rows=[]
    inherited_steps=c.state.step;attempts=0
    while c.state.step<len(c.times)-1:
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        others=[read(f) for f in run.rglob('attempts.json') if f.parent!=folder and f.parent.name.startswith('zero-')]
        seconds=time.perf_counter()-begun
        if rss>16 or seconds>1200 or seconds+sum(v.get('seconds',0) for v in others)>2400 or attempts+sum(v['attempts'] for v in others if v['stage']==stage)>=read(run/'S0/experiment-budget.json')['stages'][stage]:raise TimeoutError('coupled resource budget')
        attempts+=1
        write(folder/'attempts.json',dict(stage=stage,attempts=attempts,committed=c.state.step,inherited_steps=inherited_steps,seconds=seconds));last=c.state.step+1==len(c.times)-1
        try:row=advance_publish(c,store,rows,frame_builder=(lambda s:frame(c,cache,s)) if last else None)
        except BaseException as e:
            c.restore(store.load()['state']);c.geometry.cache.clear()
            write(folder/'failure.json',dict(status='limited',error=repr(e),rollback=store.load()['state'].digest()==c.state.digest(),cache_cleared=not bool(c.geometry.cache),committed=c.state.step));raise
        rows.append(row)
        if c.state.step%4==0:print('COUPLED',grid,fine,c.state.step,'J',row['min_detF'],'p',min(row['pressure_Pa']),flush=True)
    write(folder/'attempts.json',dict(stage=stage,attempts=attempts,committed=c.state.step,inherited_steps=inherited_steps,seconds=time.perf_counter()-begun));write(folder/'ledger.json',rows)
    states=store.history();out={k:[] for k in ('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')}
    for item in states:
        f=frame(c,cache,item['state'])
        for k in out:out[k].append(f[k])
    np.savez_compressed(folder/'probes.npz',X=f['X'],fiber=m.parent.params.fiber_direction,times=[v['state'].time for v in states],**{k:np.asarray(v) for k,v in out.items()})
    fluid=c.state.child_states['fluid'];first=states[0]['state'].child_states['fluid'];mass=float(np.sum(np.asarray(fluid['content_m3'])-first['content_m3'])+fluid['cumulative_boundary_m3']-sum(fluid['cumulative_source_m3']))
    D=sum(r['numerical_dissipation_J'] for r in rows)
    if abs(mass)>1e-10 or D!=fluid['cumulative_numerical_dissipation_J']:raise ValueError('cumulative ledger mismatch')
    write(folder/'summary.json',dict(status='passed_scoped',steps=c.state.step,attempts=attempts,inherited_steps=inherited_steps,method=method,grid=grid,fine=fine,initial_energy_J=E0,direct_energy_change_J=rows[-1]['total_energy_J']-E0,min_detF=min(r['min_detF'] for r in rows),min_pressure_Pa=min(min(r['pressure_Pa']) for r in rows),max_residual_fraction=max(r['true_scaled_residual'] for r in rows),max_pressure_work_error_J=max(max(abs(np.asarray(r['pressure_work_defect_J']))) for r in rows),max_energy_balance_J=max(abs(r['energy_balance_J']) for r in rows),cumulative_energy_balance_J=sum(r['energy_balance_J'] for r in rows),numerical_dissipation_J=D,darcy_dissipation_J=sum(r['darcy_dissipation_J'] for r in rows),cumulative_mass_defect_m3=mass,seconds=time.perf_counter()-begun,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,checkpoint_reload=store.load(validator=c.validate)['state'].digest()==c.state.digest()))

def transaction(run,restart=False,method='backward-euler'):
    run=Path(run);verify(run);c,m,cfg=setup(run,'coarse',method=method);ident=identity(c);stage=stage_for('coarse',method)
    if restart:
        folder=run/'cases'/case_name('coarse',method=method)
        if ident!=read(folder/'identity.json'):raise ValueError('restart identity differs')
        store=GenerationStore(folder,ident);store.history();last=store.load(validator=c.validate);c.restore(last['state'])
        if c.state.digest()!=last['state'].digest():raise ValueError('restart payload differs')
        write(run/stage/'restart.json',dict(status='passed_scoped',same_digest=True,digest=c.state.digest(),numerical_dissipation_J=c.state.child_states['fluid']['cumulative_numerical_dissipation_J'],new_process=True,new_steps=0));return
    
    seed_index=2 if method=='startup' else 12
    c.restore(history(run/'cases'/case_name('coarse',method=method))[seed_index]['state'])
    folder=run/stage/'fault';store=GenerationStore(folder,ident);store.save(c.state,[]);old=c.state.digest();write(folder/'attempt.json',dict(attempts=1))
    def inject(stage,state):
        if stage=='before_commit':raise ValueError('registered theta transaction fault')
    try:c.step(inject=inject)
    except ValueError as e:
        if 'registered theta transaction fault' not in str(e):raise
        if old!=c.state.digest() or old!=store.load()['state'].digest() or c.geometry.cache:raise ValueError('theta rollback failed')
        write(run/stage/'fault.json',dict(status='passed_scoped',full_rollback=True,cache_cleared=True,numerical_ledger_rolled_back=True,new_attempts=1));return
    raise ValueError('fault was not reached')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['operators','cycle','fault','restart']);p.add_argument('--run',type=Path,required=True);p.add_argument('--grid',choices=['coarse','fine'],default='coarse');p.add_argument('--fine',action='store_true');p.add_argument('--method',choices=['backward-euler','startup'],default='backward-euler');a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='operators':operators(a.run,a.grid)
        elif a.phase=='cycle':cycle(a.run,a.grid,a.fine,a.method)
        else:transaction(a.run,a.phase=='restart',a.method)
