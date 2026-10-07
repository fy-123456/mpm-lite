"""Four preregistered joint deletion predictions, one exact-budget equilibrium."""
from pathlib import Path
import argparse, time, resource, threading, os, gc
import numpy as np
import scipy.linalg as la
from .provenance import APP,SPACE_PARENT,COST_PARENT,REFERENCE,read,write,sha,register,verify,serial_lock
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_reference_next.field_audit import compare_nodal,invariants
from benchmarks.research_sequential_next.spatial import static_solve,compare_fields
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_spatial_phase_next.space import combine
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_reference_next.reference import interior_h
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.tensor_metrics import quadrature_axis,evaluate_gradient
from engine.aniso_phase1.consistent_transfer import material_response


def projected(r,C,full):
    """Rest-stiffness projection after original Ks null elimination, not full M inverse."""
    s=r.parent;car=s.free_scalar_ids[s.free_scalar_ids<s.n];Q=la.null_space(r.N[car].T)
    B=np.zeros((s.ndof,Q.shape[1]+144));B[car,:Q.shape[1]]=Q;B[s.n:,Q.shape[1]:]=C
    B-=r.N@la.solve(r.G,r.N[:s.n].T@s.Ks@B[:s.n],assume_a='pos')
    lift=np.zeros_like(full);lift[s.fixed_scalar_ids]=full[s.fixed_scalar_ids]
    lift-=r.N@la.solve(r.G,r.N[:s.n].T@s.Ks@(lift[:s.n]+s.reference[:s.n]),assume_a='pos')
    B3=np.kron(B,np.eye(3));K=B3.T@r.original_stiffness@B3
    a=la.solve(K,B3.T@r.original_stiffness@(full-lift).ravel(),assume_a='pos')
    return lift+(B3@a).reshape(full.shape)


def study(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');begin=time.perf_counter()
    def guard():
        while True:
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>16 or time.perf_counter()-begin>1200:
                write(run/'S1/resource-stop.json',dict(reason='16 GiB or 1200s cap',seconds=time.perf_counter()-begin,rss_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20));os._exit(75)
            time.sleep(.5)
    threading.Thread(target=guard,daemon=True).start()
    oldscores=read(SPACE_PARENT/'N1/nonlinear-block-scores.json')['records'][:24]
    removes=[x['indices'] for x in sorted(oldscores,key=lambda z:z['removal_energy_J'])[:4]]
    if not (run/'S1/design-protocol.json').exists():register(run,'S1/design-protocol.json',dict(training=[.005,.00375],heldout=.00425,heldout_accessed=False,
        removals=removes,selection='four best already published rest-Schur deletion blocks; no new parameter sweep',
        unchanged_added_modes='latest nonlinear-modes-swap6 last six columns',design_rules=[3,5],regions=['global_domain','interior'],
        PK1_scale=.02,fiber_scale=.02,eta=1.,independent_projection='full rest K with original Ks null elimination',
        candidate_limit=2,second_condition='both design states resolve gain, and a distinct support change has a justified prediction'))
    r,rm,rs,err=reopen(REFERENCE/'Q1/R3');s=r.parent
    with np.load(SPACE_PARENT/'N1/candidates/nonlinear-modes-swap6/space.npz') as z:C0=z['C'].copy();T0=z['T'].copy()
    refs=[r.expand(rs.q)]
    with np.load(SPACE_PARENT/'N1/heldout/R3/static.npz') as z:refs.append(z['full'].copy())
    olds=[]
    for p in [SPACE_PARENT/'N1/candidates/nonlinear-modes-swap6/static.npz',SPACE_PARENT/'N1/heldout/nonlinear-modes-swap6/static.npz']:
        with np.load(p) as z:olds.append(T0@z['full'])
    arrays=[];pred=[];support=[]
    from engine.aniso_phase1.tensor_reference import coordinates
    xs=coordinates(s.edges,s.p)[0]
    for ids in removes:
        C=np.zeros_like(C0);keep=sorted(set(range(144))-set(ids));C[keep,np.arange(138)]=1.;C[:,138:]=C0[:,138:]
        arrays.append(C);pred.append([projected(r,C,full) for full in refs])
        details=[]
        for j in ids:
            nodal=np.asarray(s.raw@s.transform[:,j]).reshape(s.shape);active=np.where(np.max(abs(nodal),axis=(1,2))>1e-12)[0]
            details.append(dict(local_index=j,full_scalar_index=s.n+j,x_nonzero_node_range=xs[active[[0,-1]]].tolist()))
        support.append(dict(removed=ids,columns=details))
    write(run/'S1/deletion-support-map.json',dict(coordinates='full local indices, not condensed vector DOFs',records=support))
    # Do not materialize point x all-basis gradients. Stream physical fields.
    # This computes the same full joint D^T W D, accumulating one x-cell at a time.
    if (run/'S1/resource-stop.json').exists() and not (run/'S1/design-memory-recovery.json').exists():
        register(run,'S1/design-memory-recovery.json',dict(cause='dense tensor design maps exceed 16 GiB; process terminated before candidate solve',
            fix='stream x-cell physical gradients for four unchanged coupled proposals',preserved_protocol_sha256=sha(run/'S1/design-protocol.json'),
            numerical_formula_unchanged=True,repeated_candidate_solves=0))
    scores=[];a0=np.asarray(s.params.fiber_direction)
    refnodes=[s.nodes(q) for q in refs];prednodes=[[s.nodes(q) for q in pair] for pair in pred]
    for order in (3,5):
        qs=[quadrature_axis(e,order) for e in s.edges];totals=np.zeros((4,2,2,3));gram=np.zeros((4,4));count=0
        for start in range(0,len(qs[0][0]),order):
            sl=slice(start,start+order);axes=[qs[0][0][sl],qs[1][0],qs[2][0]]
            weights=qs[0][1][sl,None,None]*qs[1][1][None,:,None]*qs[2][1][None,None,:]
            w=weights.ravel();frac=(axes[0]-s.edges[0][0])/(s.edges[0][-1]-s.edges[0][0]);inside=np.broadcast_to(((frac>.25)&(frac<.75))[:,None,None],weights.shape).ravel()
            count+=len(w)
            for state_id in range(2):
                Fr=np.eye(3)+evaluate_gradient((s.edges,s.p,refnodes[state_id]),axes).reshape(-1,3,3)
                Pr=material_response(Fr,np.broadcast_to(s.A,Fr.shape),s.params)[1];columns=[]
                for i in range(4):
                    F=np.eye(3)+evaluate_gradient((s.edges,s.p,prednodes[i][state_id]),axes).reshape(-1,3,3)
                    P=material_response(F,np.broadcast_to(s.A,F.shape),s.params)[1];delta=P-Pr;fiber=np.einsum('i,nij,j->n',a0,delta,a0)
                    for region,mask in enumerate((np.ones(len(w),bool),inside)):
                        ww=w*mask;totals[i,state_id,region]+=np.array([np.sum(ww*np.sum(delta**2,axis=(1,2))),np.sum(ww*fiber**2),sum(ww)])
                    eps=1e-5;dF=F-Fr
                    dp=(material_response(Fr+eps*dF,np.broadcast_to(s.A,Fr.shape),s.params)[1]-material_response(Fr-eps*dF,np.broadcast_to(s.A,Fr.shape),s.params)[1])/(2*eps)
                    ff=np.einsum('i,nij,j->n',a0,dp,a0);columns.append((np.column_stack((dp.reshape(-1,9),ff))*np.sqrt(w[:,None])/.02).ravel())
                D=np.stack(columns,axis=1);gram+=D.T@D
        record=[]
        for i in range(4):
            states=[]
            for j in range(2):
                reg={}
                for k,name in enumerate(('global_domain','interior')):
                    v=totals[i,j,k];pk,ff=np.sqrt(v[:2]/v[2]);reg[name]=dict(PK1=float(pk),fiber=float(ff),score=float((pk/.02)**2+(ff/.02)**2))
                states.append(reg)
            record.append(dict(removed=removes[i],states=states,score=sum(a[b]['score'] for a in states for b in a)))
        scores.append(dict(order=order,points=count,records=record,ranking=np.argsort([x['score'] for x in record]).tolist(),
            full_Gram=gram.tolist(),offdiag_fraction=float(la.norm(gram-np.diag(np.diag(gram)))/la.norm(gram)),streamed=True))
        print('DESIGN_RULE',order,scores[-1]['ranking'],[v['score'] for v in record],flush=True)
    stable=scores[0]['ranking']==scores[1]['ranking']
    chosen=next(i for i in scores[-1]['ranking'] if removes[i]!=list(range(138,144)))
    write(run/'S1/score-resolution-check.json',dict(status='ranking_stable' if stable else 'use_actual_cell_Gauss5',records=scores,
        full_cross_response=True,proxy_only=True,nonlinear_tangent_SPD_assumed=False))
    write(run/'S1/joint-removal-scores.json',dict(selected=chosen,removed=removes[chosen],predictions=scores[-1],
        baseline_prediction_index=removes.index(list(range(138,144))),reason='best changed deletion; fixed current six added directions'))
    reference=read(APP/'S2/reference-decision.json');baseline=read(APP/'S2/current-vs-R4.json')
    rs4,definition=interior_h(s);folder=COST_PARENT/'P3/R4';pack=read(folder/'space-package.json')
    if sha(folder/'data.npz')!=pack['data_sha256'] or rs4.signature!=pack['space_sha256']:raise ValueError('R4 source mismatch')
    with np.load(folder/'data.npz') as z:
        rr4=Condensation(rs4,z['M'],z['K']);q4=z['q'];full4=z['full'];nodes4=z['nodes'].copy()
        reload4=max(float(np.max(abs(rr4.expand(q4)-full4))),float(np.max(abs(rs4.nodes(full4)-nodes4))))
    if reload4>1e-8:raise ValueError('R4 reload mismatch')
    write(run/'S1/reference-audit.json',dict(status='passed_scoped',R3_reload_error=err,R4_reload_error=reload4,
        reference_sha256=sha(folder/'space-package.json'),reference=reference,old_mass_projection=read(APP/'S2/span-diagnostic.json'),
        scope='F45 .005 static; mass projection is not best nonlinear stress bound'))
    refnodal=(rs4.edges,rs4.p,nodes4);del rr4,rs4;gc.collect()
    name='joint-deletion6';C=arrays[chosen];cr,T=combine(r,C,name);target=run/'S1/candidates'/name;target.mkdir(parents=True)
    initial=la.lstsq(T,pred[chosen][0])[0]
    np.savez_compressed(target/'space.npz',C=C,T=T,M=cr.original_mass,K=cr.original_stiffness,initial=initial)
    write(target/'space-package.json',dict(schema='spatial-phase-combination-v1',name=name,budget=144,
        reference_sha256=sha(REFERENCE/'Q1/R3/space-package.json'),data_sha256=sha(target/'space.npz'),reduction_sha256=cr.signature,
        space_sha256=cr.parent.signature,mass_order=7,full_order=7,removed=removes[chosen],supports=[[.25,.5],[.5,.75]],rank=cr.audit,promoted=False))
    rng=np.random.default_rng(20261001);d=rng.normal(size=(cr.parent.ndof,3));d/=la.norm(d)
    direct=PointInertia(cr.parent,order=7).apply(d);higher=PointInertia(cr.parent,order=8).apply(d)
    mass=dict(transformed=metric(cr.original_mass@d,direct,1e-10,2e-5),adjacent=metric(direct,higher,1e-10,2e-5))
    if not all(x['passed'] for x in mass.values()):raise ValueError('mass action mismatch')
    model=SegmentedModel(cr,order=7,device='cuda:0',hold=.005);high=SegmentedModel(cr,order=8,device='cuda:0',hold=.005)
    def materials(q):
        dr=rng.normal(size=q.shape);dr[model.fixed]=0;dr/=la.norm(dr);a,b=model.evaluate(q,dr),high.evaluate(q,dr)
        out={k:metric(a[k],b[k],at,rt) for k,at,rt in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]}
        if not all(x['passed'] for x in out.values()):raise ValueError('material quadrature insufficient')
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
    fields,_,_=compare_fields(model,state,rm,rs);reaction=metric(stats['reaction_N'],read(folder/'result.json')['solve']['reaction_N'],1e-4,.05)
    if not reaction['passed'] or not all(x['displacement']['passed'] for x in fields.values()):reasons.append('reaction/displacement')
    np.savez_compressed(target/'static.npz',q=state.q,full=cr.expand(state.q))
    write(target/'operator-audit.json',dict(status='passed',invariants=invariants(cr.parent),mass=mass,material_before=pre,material_after=post,rank=cr.audit))
    report=dict(eligible=not reasons,reasons=reasons,errors=errors,baseline=baseline,reaction=reaction,solve=stats)
    write(target/'result.json',report);write(run/'S1/main-decision.json',report)
    write(run/'S1/bottleneck-diagnostic.json',dict(proxy=scores[-1]['records'][chosen],actual=errors,
        implication='joint deletion and fixed six directions tested separately; projection/score does not replace equilibrium',
        no_new_reference=True,no_mass_inverse=True))
    if not reasons:raise RuntimeError('candidate eligible; heldout solve required before promotion')
    write(run/'S1/heldout-check.json',dict(status='condition_not_triggered',heldout_accessed=False))
    write(run/'S1/space-decision.json',dict(status='retain_swap6',selected='nonlinear-modes-swap6',candidates_tested=1,
        second_candidate='not triggered: no resolved first-candidate gain, no evidence justifying new support',
        heldout_accessed=False,spatial_accuracy=False,seconds=time.perf_counter()-begin,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    write(run/'S1/dynamic-qualification.json',dict(status='inherited_same_space',source=str(APP/'S6/final-scene.json'),sha256=sha(APP/'S6/final-scene.json')))
    print('SPACE',removes[chosen],errors['interior'],reasons,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
