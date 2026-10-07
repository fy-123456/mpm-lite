"""Bounded stress-response design; immutable R4 reload and real equilibria."""
from pathlib import Path
import argparse, time, gc, resource, threading, os
import numpy as np
import scipy.linalg as la
from .provenance import APP, SPACE_PARENT, REFERENCE, read, write, sha, register, verify, serial_lock
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_reference_next.field_audit import compare_nodal, invariants
from benchmarks.research_sequential_next.spatial import static_solve, compare_fields
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_reference_next.reference import interior_h
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_spatial_phase_next.space import combine
from engine.aniso_phase1.research_post_release.fields import kron3
from engine.aniso_phase1.tensor_metrics import sampling, quadrature_axis
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.research_d.common_kinetic import PointInertia


def design_maps(s):
    # Low-order design surrogate only; all promotion errors use common-cell q7.
    x,wx=quadrature_axis(s.edges[0],3)
    axes=[x];weights=[wx]
    for e in s.edges[1:]:
        xx,ww=quadrature_axis(np.array([e[0],e[-1]]),3)
        axes.append(xx);weights.append(ww)
    B=[sampling(e,s.p,x) for e,x in zip(s.edges,axes)]
    D=[sampling(e,s.p,x,True) for e,x in zip(s.edges,axes)]
    maps=[]
    for d in range(3):
        a=[D[k] if k==d else B[k] for k in range(3)]
        maps.append(np.column_stack((kron3([v@p for v,p in zip(a,s.prolong)])@s.oldA,(kron3(a)@s.raw)@s.transform)))
    w=(weights[0][:,None,None]*weights[1][None,:,None]*weights[2][None,None,:]).ravel()
    w/=w.sum()
    return np.stack(maps,axis=-1),w


def response(F,s):
    return material_response(F,np.broadcast_to(s.A,F.shape),s.params)[1]


def study(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    if not (run/'S2/design-protocol.json').exists():register(run,'S2/design-protocol.json',dict(training_peaks_m=[.005,.00375],heldout_peak_m=.00425,
        heldout_accessed=False,max_candidates=2,first_candidate='stress-response6',budget=144,
        fixed_supports=[[.25,.5],[.5,.75]],replace=6,PK1_scale_Pa=.02,fiber_scale_Pa=.02,eta=1,
        design_sampling='x actual-cell Gauss3; y/z global Gauss3, surrogate only',
        method='full Gram of finite-difference material stress response; two known snapshots; nonlinear rebalance',
        design_regularization_relative=1e-10,physical_regularization=False,
        second_candidate_condition='first candidate beats reference-resolvable main stress/fiber gain; otherwise no evidence for more replacement',
        final_quadrature=7,reference_uncertainty='parent conservative R2/R3,R3/R4 maximum',
        soft_solve_seconds=600,hard_process_seconds=1200,max_rss_GiB=16))
    start=time.perf_counter()
    def guard():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>16 or time.perf_counter()-start>1200:
                write(run/'S2/resource-stop.json',dict(reason='16 GiB or 1200 second hard cap'));os._exit(75)
            time.sleep(.5)
    threading.Thread(target=guard,daemon=True).start()
    r,rm,rs,reload3=reopen(REFERENCE/'Q1/R3');s=r.parent
    r4s,definition=interior_h(s);folder=APP/'P3/R4';package=read(folder/'space-package.json')
    assert sha(folder/'data.npz')==package['data_sha256'] and sha(folder/'definition.json')==package['definition_sha256']
    with np.load(folder/'data.npz',allow_pickle=False) as z:
        r4=Condensation(r4s,z['M'],z['K']);q4=z['q'].copy();full4=z['full'].copy();nodes4=z['nodes'].copy()
    assert r4s.signature==package['space_sha256'] and r4.signature==package['reduction_sha256']
    coordinate_error=float(np.max(abs(r4.expand(q4)-full4)))
    node_error=float(np.max(abs(r4s.nodes(full4)-nodes4)))
    assert max(coordinate_error,node_error)<1e-8
    # The nested mass projection measures physical displacement-span loss.
    # Its nonlinear stress error is a diagnostic upper estimate, not a best-stress lower bound.
    n=s.ndof;basis=r.P[:,r.free];fixed=np.zeros((r.P.shape[1],3))
    fixed[r.fixed]=full4[s.fixed_scalar_ids];lift=r.expand(fixed)
    target4=full4.copy();target4[:n]-=lift
    coeff=la.solve(basis.T@r4.original_mass[:n,:n]@basis,basis.T@r4.original_mass[:n]@target4,assume_a='pos')
    projected=lift+basis@coeff
    refnodal=(r4s.edges,r4s.p,nodes4)
    loss=compare_nodal((s.edges,s.p,s.nodes(projected)),refnodal,s.A,s.params)
    write(run/'S2/reference-reload.json',dict(status='passed',source=str(folder),source_sha256=sha(folder/'space-package.json'),
        R3_probe_error=reload3,R4_coordinate_error=coordinate_error,R4_node_error=node_error,
        space_signature=r4s.signature,reduction_signature=r4.signature,definition=definition,mass_order=7,original_Ks=True))
    decision=read(APP/'P3/reference-decision.json');write(run/'S2/reference-decision.json',decision)
    write(run/'S2/span-diagnostic.json',dict(method='nested full-mass projection into R3; fixed boundary restored',
        nonlinear_stress_loss=loss,scope='one main state, not an optimal stress lower bound',
        use_R3_for_first_candidate=loss['interior']['PK1']['absolute']<.0069746))
    del r4,r4s;gc.collect()
    with np.load(SPACE_PARENT/'N1/candidates/nonlinear-modes-swap6/space.npz') as z:oldT=z['T'].copy()
    with np.load(SPACE_PARENT/'N1/candidates/nonlinear-modes-swap6/static.npz') as z:old=oldT@z['full']
    ref=r.expand(rs.q)
    with np.load(SPACE_PARENT/'N1/heldout/R3/static.npz') as z:small=z['full'].copy()
    with np.load(SPACE_PARENT/'N1/heldout/nonlinear-modes-swap6/static.npz') as z:oldsmall=oldT@z['full']
    baseline=compare_nodal((s.edges,s.p,s.nodes(old)),refnodal,s.A,s.params)
    write(run/'S2/current-vs-R4.json',baseline)
    if not decision['reliable_for_candidates'] or loss['interior']['PK1']['absolute']>=.0069746:
        raise ValueError('R3 span diagnosis requires explicit R4 candidate adapter before design')
    maps,w=design_maps(s);direction=np.asarray(s.params.fiber_direction);eps=1e-7
    groups=[list(range(144,150))+list(range(156,168))+list(range(180,192))+list(range(204,210)),
            list(range(150,156))+list(range(168,180))+list(range(192,204))+list(range(210,216))]
    C=np.zeros((s.ndof-s.n,144));C[:138,:138]=np.eye(138);reports=[]
    for half,group in enumerate(groups):
        ix=s.n+np.asarray(group);rows=[]
        for full in (ref,small):
            F=np.eye(3)+np.einsum('naj,ai->nij',maps,full)
            for component in range(3):
                responses=[]
                for j in ix:
                    d=np.zeros_like(F);d[:,component,:]=maps[:,j,:]
                    delta=(response(F+eps*d,s)-response(F-eps*d,s))/(2*eps)
                    fiber=np.einsum('i,nij,j->n',direction,delta,direction)
                    values=np.column_stack((delta.reshape(-1,9),fiber))*np.sqrt(w[:,None])/.02
                    responses.append(values.ravel())
                rows.append(np.stack(responses,axis=1))
        D=np.vstack(rows);G=D.T@D
        eig=la.eigvalsh(G);regularization=max(float(eig[-1])*1e-10,1e-16)
        L=la.cholesky(G+regularization*np.eye(len(ix)),lower=True)
        snapshots=np.column_stack((ref[ix],small[ix]/.75,(ref-old)[ix],(small-oldsmall)[ix]/.75))
        U,sv,_=la.svd(L.T@snapshots,full_matrices=False);modes=la.solve_triangular(L.T,U[:,:3],lower=False)
        C[np.ix_(group,np.arange(138+3*half,141+3*half))]=modes
        reports.append(dict(group=group,gram_eigen_range=[float(eig[0]),float(eig[-1])],design_regularization=regularization,
            singular_values=sv.tolist(),retained_fraction=float(sum(sv[:3]**2)/sum(sv**2)),
            off_diagonal_fraction=float(la.norm(G-np.diag(np.diag(G)))/la.norm(G)),finite_difference_epsilon=eps))
    name='stress-response6';cr,T=combine(r,C,name);target=run/'S2/candidates'/name;target.mkdir(parents=True,exist_ok=False)
    initial=la.lstsq(T,ref)[0]
    np.savez_compressed(target/'space.npz',C=C,T=T,M=cr.original_mass,K=cr.original_stiffness,initial=initial)
    write(target/'space-package.json',dict(schema='spatial-phase-combination-v1',name=name,budget=144,
        reference_sha256=sha(REFERENCE/'Q1/R3/space-package.json'),data_sha256=sha(target/'space.npz'),
        reduction_sha256=cr.signature,space_sha256=cr.parent.signature,mass_order=7,full_order=7,removed=list(range(138,144)),
        supports=[[.25,.5],[.5,.75]],rank=cr.audit,promoted=False))
    write(run/'S2/joint-scores.json',dict(method='stress Gram includes all scalar cross terms and all three vector components',modes=reports,
        fixed_deletion='same six removed carrier-enrichment functions as current swap6; no diagonal deletion approximation',
        design_only=True))
    inv=invariants(cr.parent);rng=np.random.default_rng(20261001);d=rng.normal(size=(cr.parent.ndof,3));d/=la.norm(d)
    direct=PointInertia(cr.parent,order=7).apply(d);high=PointInertia(cr.parent,order=8).apply(d)
    mass=dict(transformed=metric(cr.original_mass@d,direct,1e-10,2e-5),adjacent=metric(direct,high,1e-10,2e-5))
    assert all(x['passed'] for x in mass.values())
    model=SegmentedModel(cr,order=7,device='cuda:0',hold=.005);higher=SegmentedModel(cr,order=8,device='cuda:0',hold=.005)
    def materials(q):
        d=rng.normal(size=q.shape);d[model.fixed]=0;d/=la.norm(d);a,b=model.evaluate(q,d),higher.evaluate(q,d)
        out={k:metric(a[k],b[k],at,rt) for k,at,rt in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]}
        assert all(x['passed'] for x in out.values());return out
    pre=materials(cr.project(initial));state,stats=static_solve(model,initial,max_seconds=600);post=materials(state.q)
    errors=compare_nodal((cr.parent.edges,cr.parent.p,cr.parent.nodes(cr.expand(state.q))),refnodal,s.A,s.params)
    reasons=[];unc=decision['regional']
    for region in ('global_domain','interior'):
        for key in ('PK1','fiber_PK1'):
            gain=baseline[region][key]['absolute']-errors[region][key]['absolute']
            threshold=max(.15*baseline[region][key]['absolute'],2*unc[region][key]['empirical_uncertainty_Pa'])
            if gain<=threshold:reasons.append(f'{region}/{key}: gain {gain:.8g} <= resolvable {threshold:.8g}')
    for region in errors:
        for key in ('PK1','fiber_PK1'):
            if errors[region][key]['absolute']>baseline[region][key]['absolute']+.02:reasons.append(region+'/'+key+' regression')
    fields,_,_=compare_fields(model,state,rm,rs)
    reaction=metric(stats['reaction_N'],read(folder/'result.json')['solve']['reaction_N'],1e-4,.05)
    if not reaction['passed'] or not all(x['displacement']['passed'] for x in fields.values()):reasons.append('displacement/reaction mismatch')
    np.savez_compressed(target/'static.npz',q=state.q,full=cr.expand(state.q))
    write(target/'operator-audit.json',dict(status='passed',invariants=inv,mass=mass,material_before=pre,material_after=post,rank=cr.audit))
    result=dict(status='passed_scoped',eligible=not reasons,reasons=reasons,errors=errors,solve=stats,reaction=reaction,
        displacement_vs_R3={k:v['displacement'] for k,v in fields.items()})
    write(target/'result.json',result);write(run/'S2/main-decision.json',result)
    if not reasons:raise RuntimeError('main candidate eligible: heldout must be completed before promotion')
    write(run/'S2/heldout-check.json',dict(status='condition_not_triggered',heldout_accessed=False,reason='main candidate has no resolved gain'))
    write(run/'S2/space-decision.json',dict(status='retain_latest_swap6',selected='nonlinear-modes-swap6',candidates_tested=1,
        second_candidate='not triggered: first stress-targeted candidate gives no reference-resolvable gain',
        heldout_accessed=False,spatial_accuracy=False,records=[result],seconds=time.perf_counter()-start,
        peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    write(run/'S2/dynamic-qualification.json',dict(status='inherited_same_space',
        source_sha256=sha(SPACE_PARENT/'N1/dynamic-qualification.json'),new_prefix_not_needed=True))
    print('SPACE_RETAIN',errors['interior'],reasons,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
