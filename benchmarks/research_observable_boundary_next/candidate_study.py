"""Existing static candidate: physical mass projection and at most 16 q7 steps."""
from pathlib import Path
import argparse,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .spaces import load_selected
from .run import load_model
from . import config
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions,impulse_average
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.runtime_rules import advance_publish
from engine.aniso_phase1.research_phase_boundary_next.segments import VectorizedModel
from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
from engine.aniso_phase1.research_sequential.condensation import CondensedModel

NAME='balanced-direction-snapshot6'
EXPECTED='2afebf44c452409f8a3c6273ef7c53791db4c86c560658b9be9ff267dda1f587'

def fields(a,b,d):
    if not np.array_equal(a['X'],b['X']):raise ValueError('physical evaluation points differ')
    out={}
    for region,w in regions(a['X']).items():
        values={}
        for k,at in [('x',5e-5),('velocity',1e-4),('PK1',.02)]:values[k]=metric(a[k]-a['X'] if k=='x' else a[k],b[k]-b['X'] if k=='x' else b[k],at,.05,w)
        values['fiber']=metric(np.einsum('i,...ij,j->...',d,a['PK1'],d),np.einsum('i,...ij,j->...',d,b['PK1'],d),.02,.05,w);out[region]=values
    return out

def good(data):return all(x['passed'] for row in data.values() for x in row.values())

def study(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');folder=APP/'S2/candidates'/NAME;p=folder/'space-package.json'
    if sha(p)!=EXPECTED:raise ValueError('wrong delivered candidate')
    register(run,'S3/candidate-protocol.json',dict(status='passed_scoped',package=dict(path=str(p),sha256=EXPECTED),new_spaces=0,new_training_solves=0,rest_steps=8,mapped_steps=8,formal_replacement=False,mass_order=7,material_order=7,material_angle=45,static_decision=dict(path=str(APP/'S2/research-space-decision.json'),sha256=sha(APP/'S2/research-space-decision.json'))))
    entry=dict(path=str(p),sha256=EXPECTED);r,package=load_selected(entry);cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],end=.1,space=entry);m=install_warm(VectorizedModel(r,order=7,device='cuda:0'));cpu=CondensedModel(r,order=7,device='cpu');rng=np.random.default_rng(14);direction=rng.normal(size=m.rest().q.shape);direction[m.fixed]=0;direction/=la.norm(direction)
    a,b=m.evaluate(m.rest().q,direction),cpu.evaluate(cpu.rest().q,direction);operator={k:metric(a[k],b[k],1e-8,2e-5) for k in ('U','force','tangent_action')}
    if not all(x['passed'] for x in operator.values()):raise ValueError('candidate CPU/GPU rest mismatch')
    eig,V=la.eigh(m.rest_K[np.ix_(m.ids,m.ids)],m.M3ff,subset_by_index=[0,5]);rank=r.audit
    if eig[0]<=0 or rank['artificial_mass']:raise ValueError('candidate dynamic mass/modes invalid')
    write(run/'S3/operator-check.json',dict(status='passed_scoped',cpu_gpu=operator,reused_static_operator=dict(path=str(folder/'operator-audit.json'),sha256=sha(folder/'operator-audit.json')),package_sha256=EXPECTED))
    write(run/'S3/mass-modal-check.json',dict(status='passed_scoped',rank=rank,low_eigenvalues=eig.tolist(),periods_s=(2*np.pi/np.sqrt(eig)).tolist(),mass_orthogonality=float(la.norm(V.T@m.M3ff@V-np.eye(6))),no_mode_index_transfer=True,full_cross_terms=True))
    del cpu
    baseline,_=load_model(run,read(run/'cases/compatibility/execution-protocol.json'));hist=history(APP/'cases/final-full');bytime={round(x['state'].time,10):x for x in hist};origin=bytime[1.075];s=origin['state'];cache=CachedProbes(m);bcache=CachedProbes(baseline)
    with np.load(folder/'space.npz') as z:T=z['T'].copy()
    with np.load(Path(read(run/'baseline-space.json')['package']['path']).parent/'space.npz') as z:Tb=z['T'].copy()
    with np.load(REFERENCE/'Q1/R3/data.npz') as z:Ma=z['M'].copy()
    A=T@r.P[:,r.free];G=A.T@Ma@A;state=m.rest();state.time=s.time;state.q=m.boundary.lift(s.time);state.velocity=m.boundary.speed(s.time)
    targetq=Tb@baseline.reduction.expand(s.q);targetv=Tb@baseline.reduction.velocity(s.velocity)
    liftq=T@(r.offset+r.P@state.q);liftv=T@(r.P@state.velocity)
    state.q[m.free]=la.solve(G,A.T@Ma@(targetq-liftq),assume_a='pos');state.velocity[m.free]=la.solve(G,A.T@Ma@(targetv-liftv),assume_a='pos');state.predictor=state.velocity.copy();state.step=0
    state.child_states['projection_provenance']=dict(source=str(origin['folder']/'state.json'),source_sha256=sha(origin['folder']/'state.json'),prior_step=s.step,prior_history_is_provenance_only=True,cumulative_ledgers_reset=True,predictor='mapped velocity; new model identity')
    m.validate(state,material=True);err=fields(cache.frame(state),bcache.frame(s),m.parent.params.fiber_direction);allowed=good(err)
    energy=dict(old_material_plus_stabilization_J=baseline.evaluate(s.q)['U'],new_material_plus_stabilization_J=m.evaluate(state.q)['U'],old_kinetic_J=baseline.kinetic(s.velocity),new_kinetic_J=m.kinetic(state.velocity))
    write(run/'S3/projection-check.json',dict(status='passed_scoped' if allowed else 'limited',fields=err,energy_reset=energy,method='constrained full ambient R3 mass projection of physical displacement and velocity',mass_equivalence_max=float(np.max(abs(G-m.Mff))),eligible_mapped_window=allowed,old_history_not_copied=True))
    records=[];actual_steps=0
    branches=[('candidate-rest',m.rest(),np.linspace(0,.1,9))]
    if allowed:branches.append(('candidate-mapped',state,np.linspace(1.075,1.078125,9)))
    for name,initial,ts in branches:
        cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],space=entry,start=float(ts[0]),end=float(ts[-1]),times=ts.tolist(),initial=None if ts[0]==0 else dict(path=str(origin['folder']/'state.json'),sha256=sha(origin['folder']/'state.json')))
        case=run/'cases'/name;identity=dict(schema='observable-boundary-candidate-v1',model=m.identity,model_sha256=m.signature,initial_digest=initial.digest(),numerical_source_sha256=source_files(),projection='S3/projection-check.json' if ts[0] else None);write(case/'identity.json',identity);write(case/'execution-protocol.json',cfg);store=GenerationStore(case,identity);store.save(initial,[],frame=cache.frame(initial));stepper=ValidatedAVF(m,cfg,initial);rows=[];comparisons=[];t0=time.perf_counter()
        for i,t in enumerate(ts[1:],1):
            row=advance_publish(stepper,store,rows,float(t-stepper.state.time),frame_builder=cache.frame if i in (4,8) else None);rows.append(row);actual_steps+=1;witness=bytime[round(float(t),10)];fm=fields(cache.frame(stepper.state),bcache.frame(witness['state']),m.parent.params.fiber_direction);reaction=metric(row['reaction_N'],impulse_average(hist[-1]['rows'],row['time']-row['dt'],row['time']),1e-4,.05);comparisons.append(dict(time_s=float(t),fields=fm,reaction=reaction))
        passed=all(good(x['fields']) and x['reaction']['passed'] for x in comparisons);summary=dict(status='passed_scoped' if passed else 'limited',steps=8,end_s=float(ts[-1]),min_detF=min(x['min_detF'] for x in rows),max_energy_closure_J=max(abs(x['budget_defect_J']) for x in rows),seconds=time.perf_counter()-t0,rows=rows,comparisons=comparisons,full_mass=True,material_order=7)
        write(case/'summary.json',summary);write(case/'ledger.json',rows);records.append(dict(case=name,**summary));print('CANDIDATE_DYNAMIC',name,passed,summary['min_detF'],flush=True)
    qualified=allowed and len(records)==2 and all(x['status']=='passed_scoped' for x in records)
    write(run/'S3/dynamic-check.json',dict(status='passed_scoped' if qualified else 'limited',actual_candidate_steps=actual_steps,records=records))
    write(run/'S3/research-space-decision.json',dict(status='limited_short_dynamic_qualified' if qualified else 'static_qualified_dynamic_limited',candidate=NAME,static_qualified=True,short_dynamic=qualified,projection_passed=allowed,formal_space_changed=False,spatial_accuracy=False,full_dynamic_cycle=False,q5=False,pressure=False,actual_steps=actual_steps))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
