"""Existing candidate, force/acceleration diagnosis and one 64/128-step window."""
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
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');folder=ROOT/'docs/results/phase-boundary/20261001T183820Z-phase-boundary/S2/candidates'/NAME;p=folder/'space-package.json'
    if sha(p)!=EXPECTED:raise ValueError('wrong delivered candidate')
    register(run,'S3/candidate-protocol.json',dict(status='passed_scoped',package=dict(path=str(p),sha256=EXPECTED),new_spaces=0,new_training_solves=0,rest_steps=0,mapped_steps=[64,128],formal_replacement=False,mass_order=7,material_order=7,material_angle=45,static_decision=dict(path=str(folder.parent.parent/'research-space-decision.json'),sha256=sha(folder.parent.parent/'research-space-decision.json'))))
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
    write(run/'S3/projection-audit.json',read(run/'S3/projection-check.json'))
    accel=[];forces=[]
    # t=1.075 is in the smooth unloading branch, not an endpoint jump.
    boundary_acc=-.005*2*np.pi**2*np.cos(2*np.pi*(s.time-.6))
    for model,initial,probe in ((baseline,s,bcache),(m,state,cache)):
        response=model.evaluate(initial.q);acc=model.boundary.unit*boundary_acc
        acc[model.free]=la.solve(model.Mff,-response['force'][model.free]-model.M[np.ix_(model.free,model.fixed)]@acc[model.fixed],assume_a='pos')
        accel.append(probe.maps[0]@acc);forces.append(dict(material_J=response['material_U'],stabilization_J=response['stabilization_U'],force_norm_N=float(la.norm(response['force'][model.free])),acceleration_norm=float(la.norm(acc))))
    acc_error=float(la.norm(accel[1]-accel[0])/np.sqrt(len(accel[0])));window=.003125
    with np.load(APP/'S1/modal-basis.npz') as z:BV=z['vectors'][:,:6].copy()
    ca=[];ba=[]
    AB=Tb@baseline.reduction.P[:,baseline.free]
    for j in range(6):ca.append(A@V[:,j].reshape(-1,3));ba.append(AB@BV[:,j].reshape(-1,3))
    overlap=np.array([[np.sum(x*(Ma@y)) for y in ba] for x in ca]);sv=la.svdvals(overlap)
    write(run/'S3/force-acceleration-modal-diagnostic.json',dict(status='passed_scoped',state_time_s=s.time,models=forces,boundary_acceleration_m_s2=boundary_acc,physical_acceleration_rms_difference_m_s2=acc_error,local_velocity_difference_estimate_m_s=acc_error*window,estimate_is_not_uniform_bound=True,low_mode_subspace_singular_values=sv.tolist(),potential_jump_J=energy['new_material_plus_stabilization_J']-energy['old_material_plus_stabilization_J'],kinetic_jump_J=energy['new_kinetic_J']-energy['old_kinetic_J'],hypotheses=['space/initial projection changes elastic force','coarse temporal phase amplifies trajectory difference','remaining continuum-reference uncertainty'],mode_indices_not_matched=True))
    if not allowed:
        write(run/'S3/candidate-time-check.json',dict(status='not_triggered',reason='initial physical field projection gate failed',steps=0))
        write(run/'S3/research-space-decision.json',dict(status='limited',formal_space_changed=False,q5=False,pressure=False,actual_steps=0));return
    formal=history(APP/'cases/window0-half');fine_lookup={round(x['state'].time,10):x for x in formal};records=[];trajectories={};actual_steps=0
    for label,N in (('h',64),('half',128)):
        name='candidate-'+label;ts=np.linspace(1.075,1.078125,N+1)
        cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],space=entry,start=float(ts[0]),end=float(ts[-1]),times=ts.tolist(),initial=dict(path=str(origin['folder']/'state.json'),sha256=sha(origin['folder']/'state.json')))
        case=run/'cases'/name;identity=dict(schema='pressure-window-candidate-v1',model=m.identity,model_sha256=m.signature,initial_digest=state.digest(),numerical_source_sha256=source_files(),driver_sha256=sha(__file__),projection_sha256=sha(run/'S3/projection-check.json'))
        write(case/'identity.json',identity);write(case/'execution-protocol.json',cfg);snapshot(case/'source',source_files());store=GenerationStore(case,identity);store.save(state,[],frame=cache.frame(state));stepper=ValidatedAVF(m,cfg,state);rows=[];comparisons=[];started=time.perf_counter()
        for i,t in enumerate(ts[1:],1):
            row=advance_publish(stepper,store,rows,float(t-stepper.state.time),frame_builder=cache.frame if i in (N//2,N) else None);rows.append(row);actual_steps+=1
            witness=formal[(128//N)*i]
            if abs(witness['state'].time-float(t))>1e-12:raise ValueError('registered physical nodes differ')
            comparison=fields(cache.frame(stepper.state),bcache.frame(witness['state']),m.parent.params.fiber_direction)
            reaction=metric(row['reaction_N'],impulse_average(formal[-1]['rows'],row['time']-row['dt'],row['time']),1e-4,.05)
            comparisons.append(dict(time_s=float(t),fields=comparison,reaction=reaction))
            if i%32==0:print('CANDIDATE_WINDOW',label,i,N,flush=True)
            if time.perf_counter()-started>1200:raise TimeoutError('candidate bounded window exceeded')
        passed=all(good(x['fields']) and x['reaction']['passed'] for x in comparisons)
        summary=dict(status='passed_scoped' if passed else 'limited',steps=N,end_s=float(ts[-1]),min_detF=min(x['min_detF'] for x in rows),max_energy_closure_J=max(abs(x['budget_defect_J']) for x in rows),seconds=time.perf_counter()-started,comparisons=comparisons,full_mass=True,material_order=7)
        write(case/'summary.json',summary);write(case/'ledger.json',rows);records.append(dict(case=name,**summary));trajectories[label]=store.history()
    fine={round(x['state'].time,10):x for x in trajectories['half']};self_compare=[]
    for i,x in enumerate(trajectories['h'][1:],1):
        y=trajectories['half'][2*i];row=x['rows'][-1];fld=fields(cache.frame(x['state']),cache.frame(y['state']),m.parent.params.fiber_direction);R=metric(row['reaction_N'],impulse_average(trajectories['half'][-1]['rows'],row['time']-row['dt'],row['time']),1e-4,.05)
        self_compare.append(dict(time_s=x['state'].time,fields=fld,reaction=R,passed=good(fld) and R['passed']))
    selfgood=all(x['passed'] for x in self_compare);crossgood=records[-1]['status']=='passed_scoped'
    vmax=lambda comp:max(v['velocity']['absolute']/v['velocity']['budget'] for x in comp for v in x['fields'].values())
    result=dict(status='passed_scoped' if selfgood else 'limited',actual_steps=actual_steps,candidate_time_refinement_passed=selfgood,cross_space_fine_passed=crossgood,self_comparison=self_compare,records=records,max_velocity_budget_fraction_self=vmax(self_compare),max_velocity_budget_fraction_cross_fine=vmax(records[-1]['comparisons']),formal_fine_is_not_continuum_truth=True)
    write(run/'S3/candidate-time-check.json',result)
    reason='same-space refinement passes; residual cross-space/initial-state discrepancy remains' if selfgood and not crossgood else 'short-window engineering comparisons pass; no full-cycle or continuum certificate' if selfgood else 'candidate temporal refinement still differs; prioritize temporal/reference diagnosis'
    write(run/'S3/research-space-decision.json',dict(status='limited',candidate=NAME,static_qualified=True,projection_passed=True,short_dynamic_stable=True,time_refinement=selfgood,cross_space_fine=crossgood,formal_space_changed=False,spatial_accuracy=False,full_dynamic_cycle=False,q5=False,pressure=False,actual_steps=actual_steps,reason=reason,next_experiment='compare constrained projection force/energy against an independently converged physical-space reference'))
    print('CANDIDATE_DECISION',selfgood,crossgood,result['max_velocity_budget_fraction_self'],result['max_velocity_budget_fraction_cross_fine'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
