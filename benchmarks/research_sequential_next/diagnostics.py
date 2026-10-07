"""Bounded one-factor diagnostics; no tolerance is changed retrospectively."""
from pathlib import Path
import argparse
import copy
import time
import numpy as np
import scipy.linalg as la
from .provenance import read,write,serial_lock,source_files,utc,sha
from .checkpoint import GenerationStore
from .run import load_model,probe_frame
from .compare import metric,regions
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_sequential_next.model import PracticalModel


def _history(run,name):
    p=Path(run)/'cases'/name
    return GenerationStore(p,read(p/'identity.json')).history()


def weak_moments(model,q):
    """Independent per-slab constant/linear probes, matching B's definition.

    The existing GPU evaluate API does not export these diagnostics. Reuse
    its exact response kernel and apply quadrature weights exactly once.
    """
    import warp as wp
    from engine.aniso_phase1.research_d.stage2.gpu_operator import response_kernel
    op=model.operator;lin=op.prepare(model.reduction.expand(q));p=op.params
    stress=wp.empty(op.count,dtype=wp.mat33d,device=op.device)
    psi=wp.empty(op.count,dtype=wp.float64,device=op.device)
    Q=wp.empty_like(lin.Q);c=wp.empty_like(lin.c)
    det=wp.empty_like(psi);bad=wp.zeros(1,dtype=wp.int32,device=op.device)
    wp.launch(response_kernel,dim=op.count,inputs=[lin.F,op.A,p.mu,p.lam,p.k_f,psi,stress,Q,c,det,bad],device=op.device)
    if bad.numpy()[0]:raise ValueError('invalid weak-moment material state')
    shape=tuple(len(x) for x in op.points);P=stress.numpy().reshape(*shape,3,3)
    V=op.weights.numpy().reshape(shape);order=op.rule.orders[0];out=[]
    for cell in range(len(op.space.edges[0])-1):
        index=slice(cell*order,(cell+1)*order);w=V[index];field=P[index]
        probes=[1.,op.points[0][index,None,None],op.points[1][None,:,None],op.points[2][None,None,:]]
        out.append([np.einsum('abc,abcij->ij',w*probe,field) for probe in probes])
    return np.asarray(out)


def time_factors(run):
    run=Path(run);cfg=read(run/'cases/window-load-turn-dt00125/execution-protocol.json')
    model,_=load_model(run,cfg);hist=_history(run,'window-load-turn-dt00125')
    item=next(x for x in hist if abs(x['state'].time-.6125)<1e-12);initial=item['state'];dt=.0125
    cases={}
    for name in ('control','tighter_solve','path3'):
        c=copy.deepcopy(cfg)
        if name=='tighter_solve':c['solver'].update(residual_atol=1e-9,residual_rtol=1e-7)
        if name=='path3':c['path_order']=3
        stepper=ValidatedAVF(model,c,initial);row=stepper.step(dt)
        cases[name]=(stepper.state,row,probe_frame(model,stepper.state,cfg['probe_shape']))
    weights=regions(cases['control'][2]['X']);factors={}
    for name in ('tighter_solve','path3'):
        aa,bb=cases[name][2],cases['control'][2]
        factors[name]=dict(true_residual=cases[name][1]['true_residual'],regions={})
        for region,w in weights.items():
            factors[name]['regions'][region]={k:metric(aa[k]-(aa['X'] if k=='x' else 0),
                bb[k]-(bb['X'] if k=='x' else 0),a,.05,w) for k,a in [('x',5e-5),('velocity',1e-4),('PK1',.02)]}
    full=_history(run,'gpu-q7-dt005')
    reference=PracticalModel(model.reduction,order=8,device=cfg['device'])
    rng=np.random.default_rng(20260930);d=rng.normal(size=initial.q.shape);d[model.fixed]=0.;d/=la.norm(d)
    integration=[]
    for target in (.5,1.1):
        state=next(x['state'] for x in full if abs(x['state'].time-target)<1e-12)
        seven=model.evaluate(state.q,d);eight=reference.evaluate(state.q,d)
        seven['weak_moments']=weak_moments(model,state.q);eight['weak_moments']=weak_moments(reference,state.q)
        values={}
        for key in ('material_U','material_force','weak_moments','tangent_action'):
            a=np.asarray(seven[key]);b=np.asarray(eight[key]);absdiff=float(la.norm((a-b).ravel()));norm=float(la.norm(b.ravel()))
            rtol=.03 if key=='tangent_action' else .02
            values[key]=dict(absolute=absdiff,reference_norm=norm,relative=absdiff/max(norm,1e-14),
                passed=absdiff<=1e-10+rtol*norm)
        integration.append(dict(time=target,comparisons=values,passed=all(v['passed'] for v in values.values())))
    # The static modes are diagnostic, not a proof about the nonlinear orbit.
    r=model.reduction;M=np.kron(r.M[np.ix_(r.free,r.free)],np.eye(3));K=r.K[np.ix_(r.ids,r.ids)]
    eig,phi=la.eigh(K,M)
    coarse=_history(run,'window-load-turn-dt0025')
    co=next(x['state'] for x in coarse if abs(x['state'].time-.625)<1e-12)
    fi=next(x['state'] for x in hist if abs(x['state'].time-.625)<1e-12)
    dv=(co.velocity[r.free]-fi.velocity[r.free]).ravel();modal=phi.T@(M@dv);energy=.5*modal**2
    indices=np.argsort(energy)[-8:][::-1];total=float(energy.sum())
    modes=[]
    for j in indices:
        omega=float(np.sqrt(max(eig[j],0.)));period=2*np.pi/omega if omega else None
        modes.append(dict(index=int(j),eigenvalue=float(eig[j]),period_s=period,
            velocity_difference_kinetic_fraction=float(energy[j]/max(total,1e-300)),
            midpoint_frequency_coarse=2*np.arctan(.025*omega/2)/.025,
            midpoint_frequency_fine=2*np.arctan(.0125*omega/2)/.0125))
    result=dict(utc=utc(),status='diagnosed',source_sha256=source_files(),
        initial_state_sha256=sha(item['folder']/'state.json'),window_step=[.6125,.625],
        control_residual=cases['control'][1]['true_residual'],one_factor_comparisons=factors,
        material_7_8=integration,linearized_modal_diagnostic=dict(total_difference_kinetic_J=total,modes=modes,
            caveat='rest modes only; nonlinear trajectory and boundary excitation are not diagonalized'),
        hypotheses=['solve tolerance sensitivity','AVF path quadrature sensitivity','material quadrature sensitivity'],
        no_global_threshold_relaxation=True)
    write(run/'N03/diagnostics.json',result)
    for name,(_,row,_) in cases.items():print(name,'residual',row['true_residual'],flush=True)
    print('material_7_8_passed',all(x['passed'] for x in integration),'leading difference modes',modes[:3],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):time_factors(a.run)
