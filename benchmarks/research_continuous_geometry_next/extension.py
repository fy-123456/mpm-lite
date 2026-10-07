"""Two new time epochs from one authenticated 200us physical state."""
import argparse,time
import numpy as np
from .provenance import *
from .fixture import new_setup
from .runtime import attempt
from .continuous import sources,state_checks
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_pressure_window_next.coupling_study import frame
from engine.aniso_phase1.research_continuous_geometry_next.epoch import install,extension_times
from engine.aniso_phase1.research_startup_substeps_next.schedule import interval_rows,OBSERVATIONS

def prepare(run):
    run=Path(run);mutable(run)
    if read(run/'S1/backend-scope-decision.json')['status']!='continuous_B_qualified':raise ValueError('continuous B qualification required')
    old=read(run/'S0/coupled-protocol.json')['times']['h']
    decision=read(run/'S2/backend-decision.json');backend='G1' if decision.get('continuous_G1_qualified') else 'B'
    register(run,'S4/extension-protocol.json',dict(status='registered',backend=backend,source=read(run/'S0/state-contract.json')['states']['28'],times={k:extension_times(old,k=='half').tolist() for k in ('h','half')},observations_s=np.linspace(2e-4,3e-4,9).tolist(),inherited_steps=28,new_steps=[8,16],residual_normalization_window_s=2e-4,execution_end_s=3e-4,source_zero=True,material_q7=True,mass_M7=True))

def setup(run,fine=False):
    p=read(Path(run)/'S4/extension-protocol.json');item=history(APP/'cases/boundary32-h')[28]
    if sha(item['folder']/'state.json')!=p['source']['sha256']:raise ValueError('extension source changed')
    c,m,cfg,bridge=new_setup(run,state=item['state'],shared=True)
    if p['backend']=='G1':
        from engine.aniso_phase1.research_continuous_geometry_next.geometry import install as optimized
        bridge=optimized(c,bridge)
    epoch=install(c,fine,expected_prefix=read(Path(run)/'S0/coupled-protocol.json')['times']['h'])
    if not np.array_equal(c.times,p['times']['half' if fine else 'h']):raise ValueError('registered extension differs')
    src=sources()
    for f in ('benchmarks/research_continuous_geometry_next/extension.py','engine/aniso_phase1/research_continuous_geometry_next/epoch.py'):src[f]=sha(ROOT/f)
    if p['backend']=='G1':src['engine/aniso_phase1/research_continuous_geometry_next/geometry.py']=sha(ROOT/'engine/aniso_phase1/research_continuous_geometry_next/geometry.py')
    ident=dict(schema='continuous-geometry-extension-case-v1',coupling=c.identity,source=p['source'],numerical_sources=src,level='half' if fine else 'h',backend=p['backend'])
    return c,m,ident,epoch,item

def cycle(run,fine=False,restart=False):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,ident,bridge,parent=setup(run,fine);folder=run/'cases'/('extension-half' if fine else 'extension-h');store=GenerationStore(folder,ident);old=store.load(validator=c.validate)
    if old:
        if read(folder/'identity.json')!=ident:raise ValueError('extension identity changed')
        store.history();c.restore(old['state']);rows=old['rows']
        write(folder/f'restart-{c.state.step}.json',dict(status='passed_scoped',same_digest=c.state.digest()==old['state'].digest(),step=c.state.step,time_s=c.state.time,zero_step=restart))
    else:
        if restart:raise ValueError('restart source missing')
        write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources']);write(folder/'bridge.json',bridge);write(folder/'execution-protocol.json',read(run/'S4/extension-protocol.json'));store.save(c.state,[]);rows=[]
    if restart:print('EXTENSION_RESTART',c.state.step,c.state.time,flush=True);return
    cache=CachedProbes(m);probes={k:[] for k in ('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')};times=[]
    def probe(s):
        f=frame(c,cache,s);times.append(s.time)
        for k in probes:probes[k].append(f[k])
        return f
    f=probe(c.state)
    while c.state.step<len(c.times)-1:
        if time.perf_counter()-tick>1200:raise TimeoutError('extension time cap')
        last=c.state.step+1==len(c.times)-1
        row=attempt(run,'S4',folder.name,lambda:advance_publish(c,store,rows,frame_builder=(lambda s:frame(c,cache,s)) if last else None));rows.append(row)
        if not fine or (c.state.step-28)%2==0:f=probe(c.state)
        print('EXTENSION',ident['level'],c.state.step,c.state.time,row['min_detF'],row['true_scaled_residual'],flush=True)
    np.savez_compressed(folder/'probes.npz',X=f['X'],fiber=m.parent.params.fiber_direction,times=times,**{k:np.asarray(v) for k,v in probes.items()})
    obs=np.r_[OBSERVATIONS,np.linspace(2e-4,3e-4,9)[1:]];eng=interval_rows(parent['rows']+rows,obs,average=['reaction_N'],summed=['energy_balance_J','darcy_dissipation_J','numerical_dissipation_J','source_work_J','reservoir_work_J','external_work_J'])
    write(folder/'engineering-ledger.json',[r for r in eng if r['time']>2e-4]);write(folder/'ledger.json',rows);write(folder/'summary.json',dict(status='passed_scoped',new_steps=len(rows),first_step=28,final_step=c.state.step,time_s=c.state.time,seconds=time.perf_counter()-tick,source_zero=not np.any(c.source),residual_normalization_window_s=c.core.window,new_Dnum_J=sum(r['numerical_dissipation_J'] for r in rows),inherited_Dnum_J=parent['state'].child_states['fluid']['cumulative_numerical_dissipation_J']))

def fault(run):
    run=Path(run);mutable(run);c,m,ident,bridge,parent=setup(run);h=history(run/'cases/extension-h');old=next(x for x in h if x['state'].step==32);expected=next(x for x in h if x['state'].step==33);c.restore(old['state']);digest0=c.state.digest();folder=run/'S4/fault';write(folder/'identity.json',ident);store=GenerationStore(folder,ident);store.save(c.state,[])
    def inject(stage,state):
        if stage=='before_commit':raise ValueError('registered extension failure')
    try:attempt(run,'S4','extension-fault',lambda:c.step(inject=inject),fault=True)
    except ValueError as e:
        if 'registered extension failure' not in str(e):raise
        if c.state.digest()!=digest0 or store.load()['state'].digest()!=digest0 or c.geometry.cache:raise ValueError('extension rollback failed')
    else:raise ValueError('fault was not injected')
    attempt(run,'S4','extension-recovery',lambda:advance_publish(c,store,[]));checks=state_checks(c.state,expected['state'])
    if not all(v['passed'] for v in checks.values()):raise ValueError('extension recovery state differs')
    write(run/'S4/transaction-check.json',dict(status='passed_scoped',rollback_exact=True,cache_cleared=True,source_step=32,target_step=33,source_time_s=old['state'].time,checks=checks));print('EXTENSION_TRANSACTION passed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cycle','restart','fault']);p.add_argument('--run',type=Path,required=True);p.add_argument('--fine',action='store_true');a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='prepare':prepare(a.run)
        elif a.phase=='fault':fault(a.run)
        else:cycle(a.run,a.fine,a.phase=='restart')
