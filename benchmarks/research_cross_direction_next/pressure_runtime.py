"""Conditional four-step 4/8 coupling with complete state rollback and restart."""
from pathlib import Path
import argparse,time
import numpy as np
from .provenance import *
from .physics import baseline_model
from .pressure_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_cross_direction_next.rt0 import BoundedGridCoupling,BoundedTopology
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish

def build(run,cells,state=None):
    run=Path(run)
    if cells not in (4,8) or not read(run/'S3/fixed-grid-comparison.json')['eligible_coupled']:raise ValueError('complete fixed 4/8 pressure/content/outflow gate required')
    m,cfg=baseline_model(run,pressure=True);times=read(run/'S3/fixed-time-protocol.json')['times_s'];top=BoundedTopology([[e[0],e[-1]] for e in m.parent.edges],cells);src=top.V0*np.array([.001 if j<cells//2 else 0. for j in range(cells)])
    return BoundedGridCoupling(m,cfg,times,cells=cells,source_m3_s=src,state=state),m,cfg

def start(run,cells):
    run=Path(run);c,m,cfg=build(run,cells);folder=run/'cases'/f'pressure-{cells}';identity=dict(coupling=c.identity,numerical_source_sha256=source_files(),constructor_sha256=sha(Path(__file__)),baseline=read(run/'baseline-space.json'))
    if (folder/'identity.json').exists():raise ValueError('pressure case exists')
    write(folder/'identity.json',identity);write(folder/'execution-protocol.json',dict(config=cfg,times=c.times.tolist(),coupling=c.identity));cache=CachedProbes(m);store=GenerationStore(folder,identity);store.save(c.state,[],frame=c.frame(cache));rows=[]
    for i in range(2 if cells==4 else 4):
        t=time.perf_counter();rows.append(advance_publish(c,store,rows,frame_builder=(lambda st:c.frame(cache)) if i==3 else None));print('GRID_COUPLED',cells,c.state.step,time.perf_counter()-t,flush=True)
    if cells==4:
        before=c.state.digest();pointer=read(store.pointer);faults=[]
        def precommit(stage,state):raise ValueError('controlled complete pressure precommit')
        def prepointer(stage):
            if stage=='before_pointer':raise OSError('controlled pressure publish')
        for name,kw in [('before_commit',dict(inject_step=precommit)),('before_pointer',dict(inject_store=prepointer))]:
            try:advance_publish(c,store,rows,**kw)
            except (ValueError,OSError):pass
            else:raise AssertionError('injected pressure fault not observed')
            if c.state.digest()!=before or read(store.pointer)!=pointer:raise ValueError('partial pressure rollback')
            faults.append(dict(stage=name,complete_state_and_pointer_unchanged=True))
        for _ in range(2):c.step()
        np.savez_compressed(run/'S3/restart-witness.npz',q=c.state.q,v=c.state.velocity,predictor=c.state.predictor)
        write(run/'S3/restart-witness.json',dict(fluid=c.state.child_states['fluid'],base_digest=before,faults=faults,extra_steps=2))

def resume(run):
    run=Path(run);folder=run/'cases/pressure-4';identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or identity['constructor_sha256']!=sha(Path(__file__)):raise ValueError('pressure source drift')
    store=GenerationStore(folder,identity);loaded=store.load();c,m,cfg=build(run,4,loaded['state']);rows=loaded['rows'];cache=CachedProbes(m);w=read(run/'S3/restart-witness.json')
    if c.state.digest()!=w['base_digest'] or c.state.step!=2:raise ValueError('wrong restart state')
    while c.state.step<4:
        def post(stage):
            if stage=='after_pointer':raise OSError('controlled postcommit observer')
        rows.append(advance_publish(c,store,rows,frame_builder=(lambda st:c.frame(cache)) if c.state.step==3 else None,inject_store=post if c.state.step==3 else None))
    with np.load(run/'S3/restart-witness.npz') as z:errors={k:float(np.max(abs(getattr(c.state,k)-z[n]))) for k,n in [('q','q'),('velocity','v'),('predictor','predictor')]}
    for key in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_boundary_m3','cumulative_source_m3'):errors[key]=float(np.max(abs(np.asarray(c.state.child_states['fluid'][key])-w['fluid'][key])))
    if max(errors.values())>1e-8:raise ValueError('pressure restart witness mismatch')
    write(run/'S3/transaction-check.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,faults=w['faults'],after_pointer_accepted_once=True,complete_fluid_history=True))

def analyze(run):
    run=Path(run);hist={c:history(run/f'cases/pressure-{c}') for c in (4,8)};physics={};raw={}
    for c,h in hist.items():
        m,cfg=baseline_model(run,pressure=True)
        from engine.aniso_phase1.research_cross_direction_next.rt0 import BoundedGeometry
        from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
        geo=BoundedGeometry(m,c);react=[]
        for a,b in zip(h[:-1],h[1:]):
            x,y=a['state'],b['state'];dt=y.time-x.time;W=(y.q-x.q)/dt;avf=ValidatedAVF(m,cfg,x);force=avf.path(x.q,W,dt)['force'];grad=geo.discrete(x.q,y.q);pbar=.5*(np.array(x.child_states['fluid']['pressure_Pa'])+y.child_states['fluid']['pressure_Pa']);impulse=2*m.M@(W-x.velocity)+dt*(force-.8*np.einsum('k,kij->ij',pbar,grad));react.append(float(np.sum(impulse*m.boundary.unit)/dt))
            if np.linalg.norm(impulse[m.free])>1e-8:raise ValueError('reconstructed original momentum differs')
        raw[c]=react;f=h[-1]['state'].child_states['fluid'];f0=h[0]['state'].child_states['fluid'];rows=h[-1]['rows'];mass=float(sum(np.array(f['content_m3'])-f0['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
        v=dict(steps=len(rows),end_s=h[-1]['state'].time,mass_defect_m3=mass,max_residual_fraction=max(x['true_scaled_residual'] for x in rows),min_detF=min(x['min_detF'] for x in rows),min_dissipation_J=min(x['darcy_dissipation_J'] for x in rows),max_pressure_work_defect_J=max(max(abs(np.asarray(x['pressure_work_defect_J']))) for x in rows),max_energy_balance_J=max(abs(x['energy_balance_J']) for x in rows),general_LU_calls=sum(len(x['linear_scaling']) for x in rows))
        if len(rows)!=4 or abs(mass)>1e-10 or v['max_residual_fraction']>1 or v['min_dissipation_J']<0 or v['min_detF']<=.1:raise ValueError('coupled physical budget failed')
        physics[c]=v
    a,b=hist[4][-1],hist[8][-1]
    if abs(a['state'].time-b['state'].time)>1e-14:raise ValueError('different physical endpoints')
    f,g=a['state'].child_states['fluid'],b['state'].child_states['fluid'];fluids={k:metric(sum(np.atleast_1d(f[k])),sum(np.atleast_1d(g[k])),1e-10,.05) for k in ('content_m3','cumulative_boundary_m3','cumulative_source_m3')};fluids['pressure']=metric(f['pressure_Pa'],np.array(g['pressure_Pa']).reshape(4,2).mean(axis=1),.001,.05)
    fields={}
    with np.load(a['folder']/'frame.npz') as x,np.load(b['folder']/'frame.npz') as y:
        for reg,w in regions(x['X']).items():fields[reg]={k:metric(x[k]-x['X'] if k=='x' else x[k],y[k]-y['X'] if k=='x' else y[k],at,.05,w) for k,at in [('x',5e-5),('velocity',1e-4),('solid_PK1',.02),('total_PK1',.02)]}
    reactions=[metric(x,y,1e-4,.05) for x,y in zip(raw[4],raw[8])];passed=all(x['passed'] for x in fluids.values()) and all(x['passed'] for row in fields.values() for x in row.values()) and all(x['passed'] for x in reactions)
    write(run/'S3/coupled-grid-check.json',dict(status='passed_scoped' if passed else 'grid_sensitive',physics=physics,fluid=fluids,fields=fields,raw_reactions=raw,reaction_comparison=reactions,solid='BASELINE',new_solid_integration=False))
    decision=read(run/'S3/pressure-scope-decision.json');decision.update(status='passed_scoped' if passed else 'limited_research_scope',four_eight_coupled=passed,coupled_end_s=a['state'].time,coupled_steps=4);write(run/'S3/pressure-scope-decision.json',decision)
    print('COUPLED_4_8',passed,fluids,flush=True)


def inspect_current(run):
    run=Path(run);verify(run);decision=read(run/'S4/performance-decision.json')
    if decision['status']=='pending_fair_timing':raise ValueError('performance decision not frozen')
    if decision.get('explicit_determinant',False):
        from engine.aniso_phase1.research_cross_direction_next.det_rt0 import FastGridCoupling as Coupling
    else:
        from engine.aniso_phase1.research_observable_pressure_next.fast_rt0 import FastGridCoupling as Coupling
    folder=APP/'cases/pressure-2';item=history(folder)[-1];m,cfg=baseline_model(run,pressure=True);times=read(APP/'S3/pressure-time-protocol.json')['times_s'];src=read(APP/'S3/fixed-2.json')['source_m3_s'];c=Coupling(m,cfg,times,cells=2,source_m3_s=src,state=item['state']);c.validate(c.state)
    report=dict(status='passed_scoped',implementation=Coupling.__module__,saved_time_s=c.state.time,pressure_cells=2,new_integration_steps=0,complete_state_digest=c.state.digest(),source_state_sha256=sha(item['folder']/'state.json'),solid='BASELINE',new_solid_coupling=False,paired_runtime_evidence='S4/paired-performance.json')
    if not (run/'release.json').exists():write(run/'S4/runtime-check.json',report)
    print(report,flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['start','resume','analyze','inspect-current']);p.add_argument('--cells',type=int,choices=[4,8],default=4);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase!='inspect-current' and (a.run/'release.json').exists():raise ValueError('sealed run is immutable')
        if a.phase=='start':start(a.run,a.cells)
        elif a.phase=='inspect-current':inspect_current(a.run)
        else:globals()[a.phase](a.run)
