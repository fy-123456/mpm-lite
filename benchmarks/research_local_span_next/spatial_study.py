"""New nonlinear snapshot span at a fixed 144 budget; exact mass and equilibrium."""
from pathlib import Path
import argparse,gc,time,resource,threading,os
import numpy as np
import scipy.linalg as la
from .provenance import APP,PHASE,SPACE_PARENT,COST_PARENT,REFERENCE,read,write,sha,register,verify,serial_lock
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_reference_next.field_audit import compare_nodal,invariants
from benchmarks.research_sequential_next.spatial import static_solve,compare_fields
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_basis_allocation_next.spatial_study import projected
from engine.aniso_phase1.research_spatial_phase_next.space import combine
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_reference_next.reference import interior_h
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.tensor_metrics import quadrature_axis,evaluate_gradient
from engine.aniso_phase1.consistent_transfer import material_response
NAME='global-snapshot6'

def setup(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');begin=time.perf_counter()
    def guard():
        while True:
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
            if rss>16 or time.perf_counter()-begin>1200:
                write(run/'S1/resource-stop.json',dict(reason='16GiB/1200s cap',rss_GiB=rss,seconds=time.perf_counter()-begin));os._exit(75)
            time.sleep(.5)
    threading.Thread(target=guard,daemon=True).start();return run,begin

def reference4(r):
    s,_=interior_h(r.parent);folder=COST_PARENT/'P3/R4';p=read(folder/'space-package.json')
    if sha(folder/'data.npz')!=p['data_sha256'] or s.signature!=p['space_sha256']:raise ValueError('R4 content changed')
    with np.load(folder/'data.npz') as z:
        rr=Condensation(s,z['M'],z['K']);full=z['full'].copy();nodes=z['nodes'].copy()
        err=max(float(np.max(abs(rr.expand(z['q'])-full))),float(np.max(abs(s.nodes(full)-nodes))))
    if err>1e-8:raise ValueError('R4 reconstruction mismatch')
    return rr,full,nodes,err

def independent_basis(r,C):
    s=r.parent;car=s.free_scalar_ids[s.free_scalar_ids<s.n];Q=la.null_space(r.N[car].T)
    B=np.zeros((s.ndof,Q.shape[1]+C.shape[1]));B[car,:Q.shape[1]]=Q;B[s.n:,Q.shape[1]:]=C
    return B-r.N@la.solve(r.G,r.N[:s.n].T@s.Ks@B[:s.n],assume_a='pos')

def score(s,refs,candidates,order):
    qs=[quadrature_axis(e,order) for e in s.edges];totals=np.zeros((len(candidates),2,2,3));G=np.zeros((len(candidates),len(candidates)))
    refnodes=[s.nodes(q) for q in refs];nodes=[[s.nodes(q) for q in states] for states in candidates];a=np.asarray(s.params.fiber_direction)
    for start in range(0,len(qs[0][0]),order):
        sl=slice(start,start+order);axes=[qs[0][0][sl],qs[1][0],qs[2][0]];W=qs[0][1][sl,None,None]*qs[1][1][None,:,None]*qs[2][1][None,None,:];w=W.ravel()
        frac=(axes[0]-s.edges[0][0])/(s.edges[0][-1]-s.edges[0][0]);inside=np.broadcast_to(((frac>.25)&(frac<.75))[:,None,None],W.shape).ravel()
        for j in range(2):
            F=np.eye(3)+evaluate_gradient((s.edges,s.p,refnodes[j]),axes).reshape(-1,3,3);P=material_response(F,np.broadcast_to(s.A,F.shape),s.params)[1];D=[]
            for i in range(len(candidates)):
                Fc=np.eye(3)+evaluate_gradient((s.edges,s.p,nodes[i][j]),axes).reshape(-1,3,3);delta=material_response(Fc,np.broadcast_to(s.A,Fc.shape),s.params)[1]-P;fiber=np.einsum('i,nij,j->n',a,delta,a)
                for k,mask in enumerate((np.ones(len(w),bool),inside)):
                    ww=w*mask;totals[i,j,k]+=np.array([np.sum(ww*np.sum(delta**2,axis=(1,2))),np.sum(ww*fiber**2),sum(ww)])
                D.append((np.column_stack((delta.reshape(-1,9),fiber))*np.sqrt(w[:,None])/.02).ravel())
            D=np.stack(D,axis=1);G+=D.T@D
    errors=np.sqrt(totals[...,:2]/totals[...,2,None]);return dict(order=order,errors=errors.tolist(),scores=np.sum(errors**2,axis=(1,2,3)).tolist(),full_Gram=G.tolist(),streamed=True,proxy_only=True)

def main_candidate(run):
    run,begin=setup(run)
    register(run,'S1/design-protocol.json',dict(training_peaks_m=[.005,.00375],heldout_peak_m=.00425,heldout_accessed=False,removed=list(range(138,144)),
        name=NAME,candidate_limit=2,method='mass-Schur weighted nonlinear snapshots from two seen amplitudes; keep 138 old columns; six mixed directions from remaining R3 pool',
        support_change='allow the six new directions to couple existing support groups; measure actual support, do not claim compact original half-support',
        reason='prior three-per-half compression and fixed six directions left unresolved nonlinear residual; testing new span instead of another deletion list',
        no_outside_R3=True,rank_relative_cutoff=1e-9,second_candidate_condition='localized remaining error and resolvable support-only benefit',design_rules=[3,5]))
    r,rm,rs,err=reopen(REFERENCE/'Q1/R3');s=r.parent;refs=[r.expand(rs.q)]
    with np.load(SPACE_PARENT/'N1/heldout/R3/static.npz') as z:refs.append(z['full'].copy())
    with np.load(SPACE_PARENT/'N1/candidates/nonlinear-modes-swap6/space.npz') as z:C0=z['C'].copy()
    keep=np.zeros((s.ndof-s.n,138));keep[:138]=np.eye(138);B=independent_basis(r,keep);M=r.original_mass;ids=np.arange(s.n+138,s.ndof)
    S=M[np.ix_(ids,ids)]-(M[ids]@B)@la.solve(B.T@M@B,B.T@M[:,ids],assume_a='pos');S=.5*(S+S.T);L=la.cholesky(S,lower=True)
    snapshots=np.column_stack((refs[0][ids],refs[1][ids]/.75));U,sv,_=la.svd(L.T@snapshots,full_matrices=False);rank=int(sum(sv>sv[0]*1e-9));vectors=[u for u in U[:,:min(rank,6)].T]
    for v in (L.T@C0[138:,138:]).T:
        if len(vectors)==6:break
        v=v.copy()
        for u in vectors:v-=u*(u@v)
        if la.norm(v)>1e-10:vectors.append(v/la.norm(v))
    if len(vectors)!=6:raise ValueError('six independent directions unavailable')
    C=np.zeros_like(C0);C[:138,:138]=np.eye(138);C[138:,138:]=la.solve_triangular(L.T,np.stack(vectors,axis=1),lower=False)
    oldB=independent_basis(r,C0);Z=np.zeros((s.ndof,6));Z[s.n:]=C[:,138:];perp=Z-oldB@la.solve(oldB.T@M@oldB,oldB.T@M@Z,assume_a='pos')
    angles=la.eigvalsh(perp.T@M@perp,Z.T@M@Z);new_rank=int(sum(angles>1e-10))
    support=[]
    from engine.aniso_phase1.tensor_reference import coordinates
    xs=coordinates(s.edges,s.p)[0]
    for j in range(138,144):
        nodal=np.asarray(s.raw@(s.transform@C[:,j])).reshape(s.shape);active=np.where(np.max(abs(nodal),axis=(1,2))>np.max(abs(nodal))*1e-10)[0]
        support.append(dict(column=j,x_nonzero_range=xs[active[[0,-1]]].tolist(),boundary_max=float(max(np.max(abs(nodal[0])),np.max(abs(nodal[-1]))))))
    write(run/'S1/novelty-audit.json',dict(status='new_span' if new_rank else 'same_span',new_rank=new_rank,principal_sine_squared=angles.tolist(),snapshot_singular_values=sv.tolist(),retained_snapshot_rank=rank,
        mass_Schur_eigenvalues=la.eigvalsh(S).tolist(),supports=support,no_full_mass_inverse=True,no_global_projection_used_to_claim_local_support=True))
    if new_rank==0:raise ValueError('candidate gives no new span')
    predictions=[[projected(r,A,f) for f in refs] for A in (C0,C)];scores=[score(s,refs,predictions,o) for o in (3,5)]
    if np.argsort(scores[0]['scores']).tolist()!=np.argsort(scores[1]['scores']).tolist():raise ValueError('design ranking unresolved')
    write(run/'S1/score-resolution-check.json',dict(status='ranking_stable',records=scores))
    r4,full4,nodes4,err4=reference4(r);refnodal=(r4.parent.edges,r4.parent.p,nodes4);del r4;gc.collect()
    reference=read(PHASE/'S2/reference-decision.json');baseline=read(PHASE/'S2/current-vs-R4.json')
    write(run/'S1/reference-audit.json',dict(status='passed_scoped',R3_reload=err,R4_reload=err4,reference=reference,R4_sha256=sha(COST_PARENT/'P3/R4/space-package.json'),scope='F45 .005 static; empirical only'))
    write(run/'S1/error-localization.json',dict(baseline=baseline,prior_failure=read(APP/'S1/main-decision.json'),mass_projection_not_stress_lower_bound=True))
    cr,T=combine(r,C,NAME);target=run/'S1/candidates'/NAME;target.mkdir(parents=True);initial=la.lstsq(T,refs[0])[0]
    np.savez_compressed(target/'space.npz',C=C,T=T,M=cr.original_mass,K=cr.original_stiffness,initial=initial)
    write(target/'space-package.json',dict(schema='spatial-phase-combination-v1',name=NAME,budget=144,reference_sha256=sha(REFERENCE/'Q1/R3/space-package.json'),data_sha256=sha(target/'space.npz'),reduction_sha256=cr.signature,space_sha256=cr.parent.signature,mass_order=7,full_order=7,removed=list(range(138,144)),supports=support,rank=cr.audit,promoted=False))
    rng=np.random.default_rng(20261001);d=rng.normal(size=(cr.parent.ndof,3));d/=la.norm(d);direct=PointInertia(cr.parent,order=7).apply(d);higher=PointInertia(cr.parent,order=8).apply(d)
    mass=dict(transformed=metric(cr.original_mass@d,direct,1e-10,2e-5),adjacent=metric(direct,higher,1e-10,2e-5))
    if not all(x['passed'] for x in mass.values()):raise ValueError('M7 not sufficient or not complete')
    model=SegmentedModel(cr,order=7,device='cuda:0',hold=.005);high=SegmentedModel(cr,order=8,device='cuda:0',hold=.005)
    def materials(q):
        d=rng.normal(size=q.shape);d[model.fixed]=0;d/=la.norm(d);a,b=model.evaluate(q,d),high.evaluate(q,d)
        out={k:metric(a[k],b[k],at,rt) for k,at,rt in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]}
        if not all(x['passed'] for x in out.values()):raise ValueError('full material rule insufficient')
        return out
    pre=materials(cr.project(initial));state,stats=static_solve(model,initial,max_seconds=600);post=materials(state.q)
    errors=compare_nodal((cr.parent.edges,cr.parent.p,cr.parent.nodes(cr.expand(state.q))),refnodal,s.A,s.params);reasons=[]
    for region in ('global_domain','interior'):
        for key in ('PK1','fiber_PK1'):
            gain=baseline[region][key]['absolute']-errors[region][key]['absolute'];limit=max(.15*baseline[region][key]['absolute'],2*reference['regional'][region][key]['empirical_uncertainty_Pa'])
            if gain<=limit:reasons.append(f'{region}/{key}: gain {gain:g} <= {limit:g}')
    for region in errors:
        for key in ('PK1','fiber_PK1'):
            if errors[region][key]['absolute']>baseline[region][key]['absolute']+.02:reasons.append(region+'/'+key+' regression')
    fields,_,_=compare_fields(model,state,rm,rs);reaction=metric(stats['reaction_N'],read(COST_PARENT/'P3/R4/result.json')['solve']['reaction_N'],1e-4,.05)
    if not reaction['passed'] or not all(x['displacement']['passed'] for x in fields.values()):reasons.append('displacement/reaction')
    np.savez_compressed(target/'static.npz',q=state.q,full=cr.expand(state.q));write(target/'operator-audit.json',dict(status='passed',mass=mass,invariants=invariants(cr.parent),rank=cr.audit,material_before=pre,material_after=post))
    report=dict(eligible=not reasons,reasons=reasons,errors=errors,baseline=baseline,reaction=reaction,solve=stats,seconds=time.perf_counter()-begin,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
    write(target/'result.json',report);write(run/'S1/main-decision.json',report)
    write(run/'S1/space-decision.json',dict(status='pending_heldout' if not reasons else 'retain_swap6',selected='nonlinear-modes-swap6',candidate=NAME if not reasons else None,candidates_tested=1,heldout_accessed=False,spatial_accuracy=False,second_candidate='not triggered without independent localized support evidence'))
    if reasons:
        write(run/'S1/heldout-check.json',dict(status='condition_not_triggered',heldout_accessed=False));write(run/'S1/dynamic-qualification.json',dict(status='inherited_unchanged_space',source_sha256=sha(APP/'S6/final-scene.json')))
    print('SPACE_MAIN',not reasons,errors['interior'],reasons,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main_candidate(a.run)
