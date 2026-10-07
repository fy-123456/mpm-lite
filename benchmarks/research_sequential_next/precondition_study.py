"""Exact AVF Jacobian probes and bounded static-LU reuse on existing states."""
from pathlib import Path
import argparse
import copy
import time
import numpy as np
import scipy.linalg as la
from .provenance import read,write,serial_lock,register_study,utc,PREVIOUS,sha
from .checkpoint import GenerationStore
from .run import load_model
from engine.aniso_phase1.research_d.common_state import CommonState
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
from engine.aniso_phase1.research_sequential_next.preconditioner import ReusedAVF


def study(run):
    run=Path(run);cfg=read(run/'cases/gpu-q7-dt0025/execution-protocol.json')
    cfg['implementation']=dict(operator='segmented',linearization_cache=False,preconditioner='original',cache_bytes=3<<30,cache_entries=4)
    old_trajectory=PREVIOUS/'cpu-q7/trajectory.npz'
    register_study(run,'N09/protocol.json',dict(utc=utc(),solver=cfg['solver'],states=[0.,.5,1.1],jacobian_dt=.025,
        probe_directions=4,seed=909,finite_difference_epsilon=1e-5,warm_repeats=3,
        normal_window=[.45,.5],difficult_window=[.5,.6],old_trajectory_sha256=sha(old_trajectory),
        difficult_provenance='original CPU strict run reached 0.5 then hold step was expensive; replay on GPU at practical tolerance, without claiming it must remain difficult',
        physical_equations_unchanged=True,factor_entries=8,factor_bytes=64<<20))
    model,_=load_model(run,cfg);folder=run/'cases/gpu-q7-dt0025'
    states={round(v['state'].time,10):v['state'] for v in GenerationStore(folder,read(folder/'identity.json')).history()}
    rng=np.random.default_rng(909);vectors=la.qr(rng.normal(size=(len(model.ids),4)),mode='economic')[0]
    diagnostics=[]
    for t in [0.,.5,1.1]:
        state=states[t];dt=.025;stepper=ValidatedAVF(model,cfg,state)
        W=state.velocity.copy();W[model.fixed]=((model.boundary.lift(t+dt)-model.boundary.lift(t))/dt)[model.fixed]
        def residual(velocity):return (2*model.M@(velocity-state.velocity)+dt*stepper.path(state.q,velocity,dt)['force'])[model.free].ravel()
        actions=[];finite=[]
        for i in range(4):
            d=np.zeros_like(W);d[model.free]=vectors[:,i].reshape(-1,3)
            p=stepper.path(state.q,W,dt,d)
            action=(2*model.M@d+dt*dt*p['action'])[model.free].ravel();actions.append(action)
            eps=1e-5;fd=(residual(W+eps*d)-residual(W-eps*d))/(2*eps)
            finite.append(float(la.norm(action-fd)/max(la.norm(action),1e-12)))
        projected=vectors.T@np.column_stack(actions)
        symmetry=float(la.norm(projected-projected.T)/max(la.norm(projected),1e-12))
        if max(finite+[symmetry])>2e-5:raise ValueError('AVF Jacobian finite-difference/symmetry gate failed')
        diagnostics.append(dict(time=t,dt=dt,finite_difference_relative=finite,projected_symmetry_relative=symmetry,
            projected_eigenvalues=la.eigvalsh(.5*(projected+projected.T)).tolist(),
            scope='four-dimensional Euclidean projection only; no full-system condition number or global SPD claim'))
        print('J probes',t,'max finite-difference',max(finite),flush=True)
    with np.load(old_trajectory) as z:
        ids=np.flatnonzero(np.isclose(z['times'],.5));assert len(ids)==1;i=int(ids[0])
        difficult=CommonState(z['q'][i].copy(),z['velocity'][i].copy(),time=.5,step=2,
            predictor=(z['q'][i]-z['q'][i-1])/(z['times'][i]-z['times'][i-1]),child_states={'identity':copy.deepcopy(model.identity)})
    # Predictor is reconstructed from the exact AVF position update, because
    # the legacy trajectory stores q/v but no intermediate predictor arrays.
    model.validate(difficult,material=True)
    samples=[];comparisons=[]
    for case,initial,dts in [('normal',states[.45],[.025,.025]),('old_hold',difficult,[.1])]:
        for repeat in range(3):
            finals={}
            variants=[('original',ValidatedAVF),('reuse_static',ReusedAVF)]
            if repeat%2:variants.reverse()
            for name,cls in variants:
                model._cached_q=None;model._cached_response=None
                start=time.perf_counter();integrator=cls(model,cfg,initial)
                rows=[integrator.step(h) for h in dts];elapsed=time.perf_counter()-start
                finals[name]=integrator.state
                samples.append(dict(case=case,repeat=repeat,variant=name,seconds=elapsed,rows=rows,
                    factor=integrator.factors.report() if hasattr(integrator,'factors') else None))
                print(case,name,round(elapsed,5),'Newton',[r['newton_iterations'] for r in rows],
                      'Krylov',[r['krylov_iterations'] for r in rows],flush=True)
            delta=max(float(np.max(abs(finals['original'].q-finals['reuse_static'].q))),
                      float(np.max(abs(finals['original'].velocity-finals['reuse_static'].velocity))))
            if delta>1e-8:raise ValueError('factor reuse changed physical state')
            comparisons.append(dict(case=case,repeat=repeat,q_v_max_difference=delta))
    control=ValidatedAVF(model,cfg,states[.45]);cached=ReusedAVF(model,cfg,states[.45]);before=cached.state.digest()
    def fail(where,state):
        if where=='after_prepare':raise ValueError('intentional rejected reused-factor trial')
    try:cached.step(.025,inject=fail)
    except StepRejected:pass
    rollback=cached.state.digest()==before;cached.step(.025);control.step(.025)
    retry=float(np.max(abs(cached.state.q-control.state.q)))
    if not rollback or retry>1e-8:raise ValueError('preconditioner failure retry mismatch')
    stats={}
    for case in ['normal','old_hold']:
        variants={name:[x['seconds'] for x in samples if x['case']==case and x['variant']==name] for name in ['original','reuse_static']}
        med={name:float(np.median(x)) for name,x in variants.items()}
        spread={name:float(np.ptp(x)) for name,x in variants.items()}
        stats[case]=dict(median_seconds=med,range_seconds=spread,relative_saving=1-med['reuse_static']/med['original'],
            gain_exceeds_spread=med['original']-med['reuse_static']>max(spread.values()))
    # Default promotion requires useful savings in both representative windows,
    # including the one-step hold where constructing a factor is still needed.
    promote=all(v['gain_exceeds_spread'] and v['relative_saving']>.05 for v in stats.values())
    result=dict(utc=utc(),status='passed_scoped',jacobian=diagnostics,samples=samples,comparisons=comparisons,
        costs=stats,rollback=rollback,retry_q_max_difference=retry,
        default_preconditioner='reuse_static' if promote else 'original',
        optimization_status='measurable_gain' if promote else 'no_measurable_gain',
        state_updated_preconditioner='not warranted unless practical-case iterations or residual failures justify its cost',
        precision='float64',equations_mass_stabilization_unchanged=True,gpu_memory=model.operator.memory_budget.report())
    write(run/'N09/result.json',result)
    caps=read(run/'capabilities.json');caps['N09']=dict(status='passed_scoped',evidence='N09/result.json',performance=result['optimization_status']);write(run/'capabilities.json',caps)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
