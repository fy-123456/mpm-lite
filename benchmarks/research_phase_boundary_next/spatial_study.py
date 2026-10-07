"""Three-direction static research; the qualified main space never changes."""
from pathlib import Path
import argparse,gc,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .spaces import load_selected
from .solver import bounded_static
from benchmarks.research_cross_direction_next.spatial_study import material,materials,mass_project,nodal
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_phase_reference_next.spatial_study import reload_level
from benchmarks.research_local_span_next.spatial_study import independent_basis,reference4
from benchmarks.research_reference_next.field_audit import compare_nodal,invariants
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_spatial_phase_next.space import combine
from engine.aniso_phase1.research_sequential.condensation import CondensedModel
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.research_post_release.ambient_reference import embedding_audit
NAME='balanced-direction-snapshot6'
OBS=ROOT/'docs/results/observable-pressure/20261001T154616Z-observable-pressure'

def study(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');started=time.perf_counter()
    register(run,'S2/design-protocol.json',dict(train_angles=[45.,52.5,60.],reserved_angle=48.75,peak_m=.005,functions=144,keep=138,ambient='R3 p6 unchanged',method='three-direction mass-Schur snapshots, equal case weight, top six directions',rank_cutoff=1e-9,scoring='worst normalized regional stress first, then mean',reserved_not_used_for_design=True,max_candidates=2,promote_to_main=False))
    r,rm,rs,err=reopen(REFERENCE/'Q1/R3');del rm,rs;gc.collect()
    r4,f4,_,_=reference4(r);r5,f5,_,_=reload_level(PHASE_REFERENCE,5)
    emb=[embedding_audit(r.parent,r4.parent),embedding_audit(r4.parent,r5.parent)]
    refs={45:f5};refs4={45:f4};base={};reactions={45:PHASE_REFERENCE/'S2/R5/result.json',52.5:APP/'S1/reserved/R5/result.json',60:OBS/'S2/direction/R5/result.json'}
    basefolder=Path(read(run/'baseline-space.json')['package']['path']).parent
    for angle in (45,60):
        with np.load(basefolder/f'F{angle}.npz') as z:base[angle]=z['full'].copy()
    for angle in (52.5,60):
        for label,store in [('R4',refs4),('R5',refs)]:
            path=APP/'S1/reserved'/label/'state.npz' if angle==52.5 else OBS/'S2/direction'/label/'state-fields.npz'
            with np.load(path) as z:store[angle]=z['full'].copy()
    with np.load(APP/'S1/reserved/candidate/state.npz') as z:base[52.5]=z['full'].copy()
    baseline,_=load_selected(read(run/'baseline-space.json')['package']);audits={};projections={};ambient={};old_errors={}
    for angle in (45,52.5,60):
        rr=material(r,angle);params=rr.parent.params
        adjacent=compare_nodal(nodal(r4,refs4[angle]),nodal(r5,refs[angle]),params.A0,params)
        old_errors[angle]=compare_nodal(nodal(baseline,base[angle]),nodal(r5,refs[angle]),params.A0,params)
        expected=read(APP/'S1/reserved-direction-check.json')['new'] if angle==52.5 else read(APP/'S1/training-comparison.json')['records'][str(angle)]['errors']
        delta=max(abs(old_errors[angle][k][v]['absolute']-expected[k][v]['absolute']) for k in expected for v in ('PK1','fiber_PK1'))
        if delta>1e-8:raise ValueError('formal-space reference comparison does not reproduce')
        reliable=all(v['absolute']<=.25*v['budget'] for row in adjacent.values() for v in row.values())
        audits[angle]=dict(adjacent=adjacent,reliable=reliable,old_errors=old_errors[angle],reproduction_difference=delta)
        projections[angle]=mass_project(r,r5,refs[angle]);ambient[angle]=compare_nodal(nodal(r,projections[angle]),nodal(r5,refs[angle]),params.A0,params)
    write(run/'S2/reference-audit.json',dict(status='passed_scoped' if all(v['reliable'] for v in audits.values()) else 'reference_limited',records=audits,embedding=emb,R3_reload=err))
    if not all(v['reliable'] for v in audits.values()):raise ValueError('registered references not resolved')
    write(run/'S2/ambient-projection-review.json',dict(status='diagnostic',records=ambient,not_a_stress_lower_bound=True))
    with np.load(basefolder/'space.npz') as z:C0=z['C'].copy()
    s=r.parent;M=r.original_mass;keep=np.zeros((s.ndof-s.n,138));keep[:138]=np.eye(138);B=independent_basis(r,keep);ids=np.arange(s.n+138,s.ndof)
    S=M[np.ix_(ids,ids)]-(M[ids]@B)@la.solve(B.T@M@B,B.T@M[:,ids],assume_a='pos');L=la.cholesky((S+S.T)/2,lower=True)
    U,sv,_=la.svd(L.T@np.column_stack([projections[a][ids] for a in (45,52.5,60)]),full_matrices=False);rank=int(sum(sv>sv[0]*1e-9));vectors=list(U[:,:min(rank,6)].T)
    for v in (L.T@C0[138:,138:]).T:
        if len(vectors)==6:break
        v=v.copy()
        for u in vectors:v-=u*(u@v)
        if la.norm(v)>1e-10:vectors.append(v/la.norm(v))
    if len(vectors)!=6:raise ValueError('six independent directions unavailable')
    C=np.zeros_like(C0);C[:138,:138]=np.eye(138);C[138:,138:]=la.solve_triangular(L.T,np.stack(vectors,axis=1),lower=False)
    oldB=independent_basis(r,C0);oldB/=np.sqrt(np.diag(oldB.T@M@oldB));Z=np.zeros((s.ndof,6));Z[s.n:]=C[:,138:];perp=Z-oldB@la.solve(oldB.T@M@oldB,oldB.T@M@Z,assume_a='pos');angles=la.eigvalsh(perp.T@M@perp,Z.T@M@Z)
    orth=float(la.norm(oldB.T@M@perp)/max(la.norm(oldB.T@M@Z),1e-30))
    if orth>1e-8 or not np.any(angles>1e-6):raise ValueError('no resolved new span')
    from engine.aniso_phase1.tensor_reference import coordinates
    xs=coordinates(s.edges,s.p)[0];supports=[]
    for j in range(144):
        v=np.asarray(s.raw@(s.transform@C[:,j])).reshape(s.shape);active=np.where(np.max(abs(v),axis=(1,2))>np.max(abs(v))*1e-10)[0]
        supports.append(dict(column=j,x=xs[active[[0,-1]]].tolist(),boundary_max=float(max(np.max(abs(v[0])),np.max(abs(v[-1]))))))
    write(run/'S2/novelty-audit.json',dict(status='new_span',principal_sine_squared=angles.tolist(),snapshot_singular_values=sv.tolist(),mass_orthogonality=orth,supports=supports))
    cr,T=combine(r,C,NAME);target=run/'S2/candidates'/NAME;target.mkdir(parents=True);initial=la.lstsq(T,projections[45])[0]
    np.savez_compressed(target/'space.npz',C=C,T=T,M=cr.original_mass,K=cr.original_stiffness,initial=initial)
    write(target/'space-package.json',dict(schema='spatial-phase-combination-v1',name=NAME,budget=144,reference_sha256=sha(REFERENCE/'Q1/R3/space-package.json'),data_sha256=sha(target/'space.npz'),reduction_sha256=cr.signature,space_sha256=cr.parent.signature,mass_order=7,full_order=7,supports=supports,rank=cr.audit,promoted=False))
    rng=np.random.default_rng(20261002);d=rng.normal(size=(cr.parent.ndof,3));d/=la.norm(d);p7=PointInertia(cr.parent,order=7).apply(d);p8=PointInertia(cr.parent,order=8).apply(d)
    mass=dict(transform=metric(cr.original_mass@d,p7,1e-10,2e-5),sufficiency=metric(p7,p8,1e-10,2e-5))
    gpu=SegmentedModel(cr,order=7,device='cuda:0',hold=.005);cpu=CondensedModel(cr,order=7,device='cpu',hold=.005);q=cr.project(initial);dq=rng.normal(size=q.shape);dq[cr.fixed]=0;dq/=la.norm(dq)
    a,b=gpu.evaluate(q,dq),cpu.evaluate(q,dq);hardware={k:metric(a[k],b[k],1e-8,2e-5) for k in ('U','force','tangent_action')}
    if not all(v['passed'] for v in [*mass.values(),*hardware.values()]):raise ValueError('operator validation failed')
    op=dict(status='passed_scoped',mass=mass,cpu_gpu=hardware,invariants=invariants(cr.parent),rank=cr.audit,no_artificial_mass=True)
    write(target/'operator-audit.json',op);write(run/'S2/operator-check.json',op);del gpu,cpu;gc.collect();results={};reasons=[]
    for angle in (45,52.5,60):
        rr=material(cr,angle);m=SegmentedModel(rr,order=7,device='cuda:0',hold=.005);ini=la.lstsq(T,projections[angle])[0]
        state,stats=bounded_static(m,ini,target/f'F{angle}-search.json',600);post=materials(rr,state.q)
        errors=compare_nodal(nodal(rr,rr.expand(state.q)),nodal(r5,refs[angle]),rr.parent.A,rr.parent.params)
        frame=probe_frame(m,state,[33,7,7]);reaction=metric(stats['reaction_N'],read(reactions[angle])['solve']['reaction_N'],1e-4,.05)
        axes=[np.linspace(e[0],e[-1],n) for e,n in zip(r5.parent.edges,(33,7,7))];refu,_=r5.parent._sample(r5.parent.nodes(refs[angle]),axes)
        disp={k:metric(frame['x']-frame['X'],refu,5e-5,.05,w) for k,w in regions(frame['X']).items()};local=[]
        if not all(v['passed'] for row in errors.values() for v in row.values()):local.append('regional_stress_budget')
        if not reaction['passed'] or not all(v['passed'] for v in disp.values()):local.append('displacement_or_reaction')
        for region in errors:
            for key in ('PK1','fiber_PK1'):
                new=errors[region][key];old=old_errors[angle][region][key];adj=audits[angle]['adjacent'][region][key]['absolute']
                if angle==52.5 and region=='interior' and key=='fiber_PK1' and new['absolute']>.8*old['absolute']:local.append('52.5 fiber gain below 20%')
                if angle==60 and region=='interior' and new['absolute']>.1*(.06633 if key=='PK1' else .06554):local.append('F60 retains less than 90% original gain')
                if angle==45 and new['absolute']-old['absolute']>max(2*adj,.1*new['budget']):local.append(region+'/'+key+' regression')
        np.savez_compressed(target/f'F{angle}.npz',q=state.q,full=rr.expand(state.q),nodes=rr.parent.nodes(rr.expand(state.q)),**frame)
        results[angle]=dict(errors=errors,old=old_errors[angle],reaction=reaction,displacement=disp,solve=stats,material_after=post,reasons=local);reasons+=local
        write(run/'S2/training-comparison.json',dict(status='in_progress',records=results,reasons=reasons));print('TRAIN',angle,errors['interior'],local,flush=True);del m;gc.collect()
    write(run/'S2/training-comparison.json',dict(status='passed_scoped' if not reasons else 'space_limited',records=results,reasons=reasons,seconds=time.perf_counter()-started))
    write(run/'S2/second-candidate-decision.json',dict(status='not_triggered',reason='no independent evidence that fixed columns cause remaining error; no blind scan',candidate_count=1))
    write(run/'S2/research-space-decision.json',dict(status='pending_reserved' if not reasons else 'space_limited',candidate=NAME,training_passed=not reasons,formal_space_changed=False,reasons=reasons,spatial_accuracy=False))
    if reasons:write(run/'S2/reserved-direction-check.json',dict(status='not_triggered',reason='training gate failed',accessed=False))
    print('SPACE_RESEARCH',not reasons,reasons,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
