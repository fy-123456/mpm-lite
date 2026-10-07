"""Two-material mass-weighted span study with immutable R5 inputs and bounded solves."""
from pathlib import Path
from dataclasses import replace
import argparse,copy,gc,time,resource,threading,os
import numpy as np
import scipy.linalg as la
from .provenance import *
from .spaces import load_selected
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_phase_reference_next.spatial_study import reload_level,qcheck
from benchmarks.research_local_span_next.spatial_study import independent_basis,reference4
from benchmarks.research_reference_next.field_audit import compare_nodal,invariants
from benchmarks.research_observable_pressure_next.spatial_study import bounded_static
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_spatial_phase_next.space import combine
from engine.aniso_phase1.research_sequential.condensation import Condensation,CondensedModel
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.research_post_release.ambient_reference import embedding_audit
NAME='cross-direction-snapshot6'

def material(r,angle):
    if angle==45:return r
    s=copy.copy(r.parent);s.params=replace(s.params,fiber_direction=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.]));s.A=s.params.A0
    s.signature=digest(dict(parent=r.parent.signature,angle=angle,Ks='original',purpose='cross direction registered static'))
    s.metadata=dict(s.metadata,material_variation_fiber_angle_degrees=angle,original_Ks=True)
    return Condensation(s,r.original_mass,r.original_stiffness)

def nodal(r,full):return r.parent.edges,r.parent.p,r.parent.nodes(full)
def passed(errors):return all(v['passed'] for row in errors.values() for v in row.values())
def mass_project(r,r5,full):
    n=r.parent.ndof;B=r.P[:,r.free];lift=r.offset+r.P[:,r.fixed]@full[r.parent.fixed_scalar_ids]
    B5=np.pad(B,((0,r5.parent.ndof-n),(0,0)));lift5=np.pad(lift,((0,r5.parent.ndof-n),(0,0)))
    M=r5.original_mass;coeff=la.solve(B5.T@M@B5,B5.T@M@(full-lift5),assume_a='pos')
    return lift+B@coeff

def materials(r,q):
    from benchmarks.research_sequential_next.diagnostics import weak_moments
    from benchmarks.research_sequential_next.material_study import material_metrics
    a=SegmentedModel(r,order=7,device='cuda:0',hold=.005);b=SegmentedModel(r,order=8,device='cuda:0',hold=.005)
    wa,wb=weak_moments(a,q),weak_moments(b,q);out={}
    random=np.random.default_rng(20261002).normal(size=q.shape)
    for name,d in [('mixed',random),('state',q.copy())]:
        d[r.fixed]=0;d/=max(la.norm(d),1e-30);out[name]=material_metrics(a.evaluate(q,d),b.evaluate(q,d),wa,wb)
    if not all(v['passed'] for v in out.values()):raise ValueError('q7/q8 material budget failed')
    return out

def study(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');start=time.perf_counter()
    def guard():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>16 or time.perf_counter()-start>1200:
                write(run/'S1/resource-stop.json',dict(reason='16GiB/1200s cap'));os._exit(75)
            time.sleep(.5)
    threading.Thread(target=guard,daemon=True).start()
    if not (run/'S1/design-protocol.json').exists():register(run,'S1/design-protocol.json',dict(train_angles=[45.,60.],reserved_angle=52.5,peak_m=.005,mu=10,lam=20,k_f=200,hidden=False,
        reserved='not used for selection; historical code/plan search only, no exhaustive hidden-set claim',functions=144,keep=138,ambient='R3 p6 unchanged',
        method='full mass Schur snapshots of R5 states physically projected to nested R3; equal material weights and amplitudes',
        normalization=dict(stress_atol_Pa=.02,rtol=.05,displacement_atol_m=5e-5,reaction_atol_N=1e-4),
        scoring='whole nonlinear field on common cells with actual material per case; retains overlap cross effects',
        rank_cutoff=1e-9,fill_order='published old last-six columns',second_candidate='only if actual residual localized to fixed columns with independent new-span evidence; never after reserved solve',max_candidates=2))
    r,rm,rs,err=reopen(REFERENCE/'Q1/R3');del rm,rs;gc.collect()
    r4,f4,n4,e4=reference4(r);r5,f5,n5,e5=reload_level(PHASE_REFERENCE,5)
    emb=[embedding_audit(r.parent,r4.parent),embedding_audit(r4.parent,r5.parent)]
    refs={45:f5}
    with np.load(APP/'S2/direction/R5/state-fields.npz') as z:refs[60]=z['full'].copy()
    baseline,_=load_selected(read(run/'baseline-space.json')['package'])
    with np.load(LOCAL/'S1/candidates/global-snapshot6/static.npz') as z:base45=z['full'].copy()
    with np.load(APP/'S2/direction/formal/state-fields.npz') as z:base60=z['full'].copy()
    audits={};projections={};ambient={};old_errors={}
    for angle in (45,60):
        rr=material(r,angle);params=rr.parent.params
        if angle==45:full4=f4;fullbase=base45
        else:
            with np.load(APP/'S2/direction/R4/state-fields.npz') as z:full4=z['full'].copy()
            fullbase=base60
        adjacent=compare_nodal(nodal(r4,full4),nodal(r5,refs[angle]),params.A0,params)
        old_errors[angle]=compare_nodal(nodal(baseline,fullbase),nodal(r5,refs[angle]),params.A0,params)
        expected=read(PHASE_REFERENCE/'S2/current-vs-R5.json')['errors'] if angle==45 else read(APP/'S2/direction-check.json')['errors']
        delta=max(abs(old_errors[angle][k][v]['absolute']-expected[k][v]['absolute']) for k in expected for v in ('PK1','fiber_PK1'))
        if delta>1e-8:raise ValueError('reference evidence does not reproduce')
        reliable=all(v['absolute']<=.25*v['budget'] for row in adjacent.values() for v in row.values())
        audits[angle]=dict(adjacent=adjacent,reliable=reliable,old_errors=old_errors[angle],reproduction_difference=delta,actual_material=dict(mu=params.mu,lam=params.lam,k_f=params.k_f,fiber=params.fiber_direction.tolist()))
        projections[angle]=mass_project(r,r5,refs[angle]);ambient[angle]=compare_nodal(nodal(r,projections[angle]),nodal(r5,refs[angle]),params.A0,params)
    write(run/'S1/reference-audit.json',dict(status='passed_scoped',records=audits,embedding=emb,R3_reload=err,source_R5=sha(PHASE_REFERENCE/'S2/R5/space-package.json'),source_F60=sha(APP/'S2/direction/R5/state-fields.npz')))
    write(run/'S1/ambient-projection-review.json',dict(status='diagnostic',projection='constrained full-mass projection through exact nested embedding; no coefficient truncation',records=ambient,not_a_stress_lower_bound=True))
    with np.load(LOCAL/'S1/candidates/global-snapshot6/space.npz') as z:C0=z['C'].copy()
    s=r.parent;M=r.original_mass;keep=np.zeros((s.ndof-s.n,138));keep[:138]=np.eye(138);B=independent_basis(r,keep);ids=np.arange(s.n+138,s.ndof)
    S=M[np.ix_(ids,ids)]-(M[ids]@B)@la.solve(B.T@M@B,B.T@M[:,ids],assume_a='pos');L=la.cholesky((S+S.T)/2,lower=True)
    U,sv,_=la.svd(L.T@np.column_stack([projections[a][ids] for a in (45,60)]),full_matrices=False);rank=int(sum(sv>sv[0]*1e-9));vectors=list(U[:,:min(rank,6)].T)
    for v in (L.T@C0[138:,138:]).T:
        if len(vectors)==6:break
        v=v.copy()
        for u in vectors:v-=u*(u@v)
        if la.norm(v)>1e-10:vectors.append(v/la.norm(v))
    if len(vectors)!=6:raise ValueError('six independent columns unavailable')
    C=np.zeros_like(C0);C[:138,:138]=np.eye(138);C[138:,138:]=la.solve_triangular(L.T,np.stack(vectors,axis=1),lower=False)
    oldB=independent_basis(r,C0);oldB/=np.sqrt(np.diag(oldB.T@M@oldB));Z=np.zeros((s.ndof,6));Z[s.n:]=C[:,138:];perp=Z-oldB@la.solve(oldB.T@M@oldB,oldB.T@M@Z,assume_a='pos');angles=la.eigvalsh(perp.T@M@perp,Z.T@M@Z)
    from engine.aniso_phase1.tensor_reference import coordinates
    xs=coordinates(s.edges,s.p)[0];supports=[]
    for j in range(144):
        v=np.asarray(s.raw@(s.transform@C[:,j])).reshape(s.shape);active=np.where(np.max(abs(v),axis=(1,2))>np.max(abs(v))*1e-10)[0]
        supports.append(dict(column=j,x=xs[active[[0,-1]]].tolist(),boundary_max=float(max(np.max(abs(v[0])),np.max(abs(v[-1]))))))
    write(run/'S1/novelty-audit.json',dict(status='new_span',principal_sine_squared=angles.tolist(),new_rank=int(sum(angles>1e-10)),snapshot_singular_values=sv.tolist(),snapshot_rank=rank,supports=supports))
    if not np.any(angles>1e-10):raise ValueError('no new physical span')
    cr,T=combine(r,C,NAME);target=run/'S1/candidates'/NAME;target.mkdir(parents=True);initial=la.lstsq(T,projections[45])[0]
    np.savez_compressed(target/'space.npz',C=C,T=T,M=cr.original_mass,K=cr.original_stiffness,initial=initial)
    write(target/'space-package.json',dict(schema='spatial-phase-combination-v1',name=NAME,budget=144,reference_sha256=sha(REFERENCE/'Q1/R3/space-package.json'),data_sha256=sha(target/'space.npz'),reduction_sha256=cr.signature,space_sha256=cr.parent.signature,mass_order=7,full_order=7,supports=supports,rank=cr.audit,promoted=False))
    rng=np.random.default_rng(20261002);d=rng.normal(size=(cr.parent.ndof,3));d/=la.norm(d);p7=PointInertia(cr.parent,order=7).apply(d);p8=PointInertia(cr.parent,order=8).apply(d)
    mass=dict(transform=metric(cr.original_mass@d,p7,1e-10,2e-5),sufficiency=metric(p7,p8,1e-10,2e-5))
    gpu=SegmentedModel(cr,order=7,device='cuda:0',hold=.005);cpu=CondensedModel(cr,order=7,device='cpu',hold=.005);q=cr.project(initial);dq=rng.normal(size=q.shape);dq[cr.fixed]=0;dq/=la.norm(dq)
    a,b=gpu.evaluate(q,dq),cpu.evaluate(q,dq);hardware={k:metric(a[k],b[k],1e-8,2e-5) for k in ('U','force','tangent_action')}
    if not all(v['passed'] for v in [*mass.values(),*hardware.values()]):raise ValueError('candidate operator validation failed')
    write(target/'operator-audit.json',dict(status='passed_scoped',mass=mass,cpu_gpu=hardware,invariants=invariants(cr.parent),rank=cr.audit,no_artificial_mass=True))
    del gpu,cpu;gc.collect();results={};reasons=[]
    for angle in (45,60):
        rr=material(cr,angle);m=SegmentedModel(rr,order=7,device='cuda:0',hold=.005);ini=la.lstsq(T,projections[angle])[0]
        pre=materials(rr,rr.project(ini));state,stats=bounded_static(m,ini,target/f'F{angle}-search.json',600)
        post=materials(rr,state.q);errors=compare_nodal(nodal(rr,rr.expand(state.q)),nodal(r5,refs[angle]),rr.parent.A,rr.parent.params)
        frame=probe_frame(m,state,[33,7,7]);refpath=PHASE_REFERENCE/'S2/R5/result.json' if angle==45 else APP/'S2/direction/R5/result.json';reaction=metric(stats['reaction_N'],read(refpath)['solve']['reaction_N'],1e-4,.05)
        axes=[np.linspace(e[0],e[-1],n) for e,n in zip(r5.parent.edges,(33,7,7))]
        refu,_=r5.parent._sample(r5.parent.nodes(refs[angle]),axes)
        disp={k:metric(frame['x']-frame['X'],refu,5e-5,.05,w) for k,w in regions(frame['X']).items()}
        local=[]
        if not audits[angle]['reliable']:local.append('reference_limited')
        if not passed(errors):local.append('regional_stress_budget')
        if not reaction['passed'] or not all(v['passed'] for v in disp.values()):local.append('displacement_or_reaction')
        for region in errors:
            for key in ('PK1','fiber_PK1'):
                new=errors[region][key];old=old_errors[angle][region][key];adj=audits[angle]['adjacent'][region][key]['absolute']
                if angle==60 and region in ('interior','transition') and old['absolute']-new['absolute']<=max(.2*old['absolute'],2*adj):local.append(region+'/'+key+' gain')
                if angle==45 and new['absolute']-old['absolute']>max(2*adj,.1*new['budget']):local.append(region+'/'+key+' regression')
        np.savez_compressed(target/f'F{angle}.npz',q=state.q,full=rr.expand(state.q),nodes=rr.parent.nodes(rr.expand(state.q)),**frame)
        results[angle]=dict(errors=errors,old=old_errors[angle],reaction=reaction,displacement=disp,solve=stats,material_before=pre,material_after=post,reasons=local);reasons+=local
        print('TRAIN',angle,errors['interior'],local,flush=True);del m;gc.collect()
    write(run/'S1/training-comparison.json',dict(status='passed_scoped' if not reasons else 'space_limited',records=results,reasons=reasons,seconds=time.perf_counter()-start))
    # A second deletion cannot be justified from a scalar stress error or a mass projection alone.
    write(run/'S1/second-candidate-decision.json',dict(status='not_triggered',reason='no demonstrated localization to fixed columns plus independent support-change benefit; do not infer this from mass projection stress loss',candidate_count=1))
    if reasons:
        write(run/'S1/reserved-direction-check.json',dict(status='not_triggered',angle=52.5,accessed=False,reason='training gate failed'))
        write(run/'S1/dynamic-smoke.json',dict(status='not_triggered',reason='new space not eligible'))
        write(run/'S1/space-decision.json',dict(status='retain_baseline',selected='global-snapshot6',candidates_tested=1,spatial_accuracy=False,reasons=reasons))
        write(run/'selected-solid-space.json',read(run/'baseline-space.json'))
    else:write(run/'S1/space-decision.json',dict(status='pending_reserved',candidate=NAME,spatial_accuracy=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
