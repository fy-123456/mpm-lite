"""One further local h reference before spending the fixed function budget."""
from pathlib import Path
import argparse,time,resource,threading,os,traceback
import numpy as np
from .provenance import APP,REFERENCE,read,write,sha,register,verify,serial_lock,resources
from benchmarks.research_reference_next.reference_study import reopen
from engine.aniso_phase1.research_reference_next.reference import interior_h
from engine.aniso_phase1.research_post_release.ambient_reference import embedding_audit,operators
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from benchmarks.research_sequential_next.spatial import static_solve
from benchmarks.research_reference_next.field_audit import compare_nodal
from benchmarks.research_sequential_next.compare import metric

def reference(run):
    run=Path(run);verify(run)
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    register(run,'P3/reference-protocol.json',dict(parent_R3=str(REFERENCE/'Q1/R3'),new_level='R4',
        construction='one more local x bisection at existing internal .375/.625 coverage, degree6 unchanged',
        functions='reference enrichment only; production stays swap6 until independently qualified',
        soft_seconds=600,hard_seconds=1200,max_rss_GiB=16,original_Ks=True,full_cross_mass=True,
        old_main_error_Pa=.02722719361,old_uncertainty_Pa=.0081861035,heldout_static_m=.00425,max_candidates=2))
    start=time.perf_counter();folder=run/'P3/R4';folder.mkdir(exist_ok=False);stop=threading.Event()
    def guard():
        while not stop.wait(.5):
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
            if peak>16 or time.perf_counter()-start>1200:
                write(folder/'failure.json',dict(type='resource_limit',seconds=time.perf_counter()-start,peak_rss_GiB=peak))
                os._exit(75)
    threading.Thread(target=guard,daemon=True).start()
    try:
        parent,pm,ps,reload_error=reopen(REFERENCE/'Q1/R3')
        space,definition=interior_h(parent.parent)
        # Pin the refinement definition separately; inherited helper builds an exact nested family.
        write(folder/'definition.json',dict(**definition,parent_space=parent.parent.signature,parent_reference=str(REFERENCE/'Q1/R3')))
        embedding=embedding_audit(parent.parent,space)
        initial=np.vstack((parent.expand(ps.q),np.zeros((space.ndof-parent.parent.ndof,3))))
        rr,op=operators(parent,space,max_seconds=max(1,600-(time.perf_counter()-start)),progress=lambda *x:print('R4',*x,flush=True))
        model=SegmentedModel(rr,order=7,device='cuda:0',hold=.005)
        state,solve=static_solve(model,initial,max_seconds=max(1,600-(time.perf_counter()-start)))
        oldnodes=parent.parent.nodes(parent.expand(ps.q));newnodes=space.nodes(rr.expand(state.q))
        adjacent=compare_nodal((parent.parent.edges,parent.parent.p,oldnodes),(space.edges,space.p,newnodes),space.A,space.params)
        higher=SegmentedModel(rr,order=8,device='cuda:0',hold=.005)
        d=np.random.default_rng(20261001).normal(size=state.q.shape);d[model.fixed]=0;d/=np.linalg.norm(d)
        a,b=model.evaluate(state.q,d),higher.evaluate(state.q,d)
        material={k:metric(a[k],b[k],at,rt) for k,at,rt in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]}
        if not all(x['passed'] for x in material.values()):raise ValueError('R4 q7/q8 insufficient')
        prev=read(REFERENCE/'Q1/common-cell-comparisons.json')['pairs']['R2-R3'];regions={}
        for region in adjacent:
            regions[region]={}
            for key in ('PK1','fiber_PK1'):
                previous=prev[region][key]['absolute'];current=adjacent[region][key]['absolute']
                regions[region][key]=dict(previous_difference_Pa=previous,new_difference_Pa=current,
                    contracted=current<=max(.9*previous,1e-8),empirical_uncertainty_Pa=max(previous,current,1e-8))
        reliable=all(v['contracted'] for region in ('interior','transition','global_domain') for v in regions[region].values())
        np.savez_compressed(folder/'data.npz',M=rr.original_mass,K=rr.original_stiffness,q=state.q,full=rr.expand(state.q),nodes=newnodes)
        write(folder/'space-package.json',dict(schema='cost-phase-R4-v1',parent_reference_sha256=sha(REFERENCE/'Q1/R3/space-package.json'),
            data_sha256=sha(folder/'data.npz'),space_sha256=space.signature,reduction_sha256=rr.signature,definition_sha256=sha(folder/'definition.json'),
            mass_order=7,material_order=7,scope='F45 .005 static; one local h extension'))
        write(folder/'result.json',dict(status='passed_scoped',embedding=embedding,reload_error=reload_error,operators=op,solve=solve,adjacent=adjacent,
            material=material,seconds=time.perf_counter()-start,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
        write(run/'P3/reference-decision.json',dict(status='reference_usable_scoped' if reliable else 'reference_limited',
            reliable_for_candidates=reliable,regional=regions,scope='empirical F45 .005, not continuum certification',
            reason='last local-h difference contracted' if reliable else 'at least one relevant region did not contract under the single permitted local-h refinement'))
        print('R4_DECISION',reliable,regions['interior'],flush=True)
    except (ValueError,TimeoutError,MemoryError) as exc:
        write(folder/'failure.json',dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
        write(run/'P3/reference-decision.json',dict(status='reference_limited',reliable_for_candidates=False,reason=str(exc)))
        print('R4_LIMIT',type(exc).__name__,str(exc),flush=True)
    finally:stop.set()

def retain(run):
    run=Path(run);decision=read(run/'P3/reference-decision.json')
    if decision['reliable_for_candidates']:raise ValueError('reliable reference requires candidate evaluation, not unconditional retain')
    write(run/'P3/space-decision.json',dict(status='retain_latest_swap6_reference_limited',selected='nonlinear-modes-swap6',
        reason=decision['reason'],new_candidates=0,heldout_accessed=False,spatial_accuracy=False,
        conditional_steps={'P3.2':'candidate generation not opened','P3.3':'not applicable, no new candidate','P3.4':'no promotion','P3.5':'same physical space; P1/P2 identities unchanged'}))
    write(run/'P3/dynamic-qualification.json',dict(status='inherited_same_space',source_sha256=sha(APP/'N1/dynamic-qualification.json'),
        final_scene='P6 checks final implementation',new_prefix_not_needed=True))

def candidates(run):
    import scipy.linalg as la
    import gc
    from engine.aniso_phase1.research_spatial_phase_next.space import combine
    from engine.aniso_phase1.research_d.common_kinetic import PointInertia
    from benchmarks.research_reference_next.field_audit import invariants
    from benchmarks.research_sequential_next.spatial import compare_fields
    run=Path(run);verify(run)
    if not read(run/'P3/reference-decision.json')['reliable_for_candidates']:raise ValueError('reference not qualified')
    register(run,'P3/design-protocol.json',dict(candidates=['stress-modes6','stress-modes12'],budget=144,
        training_peaks=[.005,.00375],heldout_peak=.00425,heldout_accessed=False,
        method='reference-correction and nonlinear residual snapshots, static stiffness metric, joint removal cost',
        minimum_relative_gain=.15,reference_uncertainty_multiple=2,reference='R4',no_extra_candidates=True))
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    start=time.perf_counter();r,rm,rs,_=reopen(REFERENCE/'Q1/R3');s=r.parent;nlocal=s.ndof-s.n
    with np.load(APP/'N1/candidates/nonlinear-modes-swap6/space.npz') as z:oldT=z['T'].copy()
    with np.load(APP/'N1/candidates/nonlinear-modes-swap6/static.npz') as z:oldfull=z['full'].copy()
    embedded=oldT@oldfull;ref=r.expand(rs.q)
    with np.load(APP/'N1/heldout/R3/static.npz') as z:refsmall=z['full'].copy()
    with np.load(APP/'N1/heldout/nonlinear-modes-swap6/static.npz') as z:oldsmall=oldT@z['full']
    rich=SegmentedModel(r,order=7,device='cuda:0',hold=.005)
    free=s.free_scalar_ids;ids=(3*free[:,None]+np.arange(3)).ravel();K=r.original_stiffness[np.ix_(ids,ids)];K=(K+K.T)/2
    inverse=la.cho_solve(la.cho_factor(K),np.eye(len(ids)))
    residual=rich.evaluate(r.project(embedded))['original_full_force'][free].ravel()
    relax=np.zeros_like(ref);relax[free]=(-inverse@residual).reshape(-1,3)
    mapping={int(v):i for i,v in enumerate(free)}
    def removal(block):
        idx=(3*np.array([mapping[s.n+j] for j in block])[:,None]+np.arange(3)).ravel();a=ref[s.n+np.array(block)].ravel()
        force=la.solve(inverse[np.ix_(idx,idx)],a,assume_a='pos');delta=-inverse[:,idx]@force
        energy=float(.5*a@force);direct=float(.5*delta@K@delta)
        assert abs(energy-direct)<=1e-12+2e-5*abs(direct)
        return energy
    blocks=[dict(indices=list(range(i,i+6)),joint_removal_J=removal(list(range(i,i+6)))) for i in range(0,138,6)]
    additional=min(blocks,key=lambda x:x['joint_removal_J'])['indices']
    groups=[list(range(144,150))+list(range(156,168))+list(range(180,192))+list(range(204,210)),
            list(range(150,156))+list(range(168,180))+list(range(192,204))+list(range(210,216))]
    modes=[];reports=[]
    for group in groups:
        ix=s.n+np.array(group);vix=(3*ix[:,None]+np.arange(3)).ravel();Kb=r.original_stiffness[np.ix_(vix,vix)].reshape(len(ix),3,len(ix),3)
        gram=sum(Kb[:,j,:,j] for j in range(3))/3;L=la.cholesky((gram+gram.T)/2,lower=True)
        snapshots=np.column_stack((ref[ix],refsmall[ix]/.75,(ref-embedded)[ix],(refsmall-oldsmall)[ix]/.75,relax[ix]))
        U,sv,_=la.svd(L.T@snapshots,full_matrices=False);modes.append(la.solve_triangular(L.T,U,lower=False))
        reports.append(dict(indices=group,singular_values=sv.tolist(),stiffness_metric_orthogonality=float(la.norm(U.T@U-np.eye(U.shape[1])))))
    for count in (6,12):
        removed=list(range(138,144))+(additional if count==12 else []);keep=sorted(set(range(144))-set(removed));C=np.zeros((nlocal,144));C[keep,np.arange(len(keep))]=1
        for half,(group,mode) in enumerate(zip(groups,modes)):
            width=count//2;C[np.ix_(group,np.arange(len(keep)+half*width,len(keep)+(half+1)*width))]=mode[:,:width]
        name=f'stress-modes{count}';cr,T=combine(r,C,name);folder=run/'P3/candidates'/name;folder.mkdir(parents=True,exist_ok=False)
        initial=la.lstsq(T,ref)[0];np.savez_compressed(folder/'space.npz',C=C,T=T,M=cr.original_mass,K=cr.original_stiffness,initial=initial)
        write(folder/'space-package.json',dict(schema='spatial-phase-combination-v1',name=name,budget=144,reference_sha256=sha(REFERENCE/'Q1/R3/space-package.json'),
            data_sha256=sha(folder/'space.npz'),reduction_sha256=cr.signature,space_sha256=cr.parent.signature,mass_order=7,full_order=7,
            removed=removed,joint_removal_J=removal(removed),supports=[[.25,.5],[.5,.75]],rank=cr.audit,promoted=False))
    write(run/'P3/joint-scores.json',dict(blocks=blocks,modes=reports,nonlinear_residual_N=float(la.norm(residual)),normalization='stiffness metric; no nonlinear SPD assumption'))
    definition=read(run/'P3/R4/definition.json');e=s.edges[0];r4edges=(np.unique(np.r_[e,*[(e[j]+e[j+1])/2 for j in definition['cells']]]),*s.edges[1:])
    with np.load(run/'P3/R4/data.npz') as z:r4nodes=z['nodes'].copy()
    refnodal=(r4edges,s.p,r4nodes);baseline=compare_nodal((s.edges,s.p,s.nodes(embedded)),refnodal,s.A,s.params)
    write(run/'P3/current-vs-R4.json',baseline);unc=read(run/'P3/reference-decision.json')['regional'];results=[]
    del rich;gc.collect()
    for count in (6,12):
        name=f'stress-modes{count}';folder=run/'P3/candidates'/name
        register(run,f'P3/candidates/{name}/solve-protocol.json',dict(reference='R4',peak=.005,heldout=False,orders=[7,8]))
        with np.load(folder/'space.npz') as z:cr,T=combine(r,z['C'],name);initial=z['initial'].copy()
        inv=invariants(cr.parent);rng=np.random.default_rng(20261001);direction=rng.normal(size=(cr.parent.ndof,3));direction/=la.norm(direction)
        direct=PointInertia(cr.parent,order=7).apply(direction);high=PointInertia(cr.parent,order=8).apply(direction)
        mass=dict(transformed=metric(cr.original_mass@direction,direct,1e-10,2e-5),adjacent=metric(direct,high,1e-10,2e-5))
        assert all(x['passed'] for x in mass.values())
        model=SegmentedModel(cr,order=7,device='cuda:0',hold=.005);state,stats=static_solve(model,initial)
        d=rng.normal(size=state.q.shape);d[model.fixed]=0;d/=la.norm(d);higher=SegmentedModel(cr,order=8,device='cuda:0',hold=.005)
        a,b=model.evaluate(state.q,d),higher.evaluate(state.q,d);materials={k:metric(a[k],b[k],at,rt) for k,at,rt in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]}
        assert all(x['passed'] for x in materials.values())
        errors=compare_nodal((cr.parent.edges,cr.parent.p,cr.parent.nodes(cr.expand(state.q))),refnodal,s.A,s.params);reasons=[]
        for region in ('global_domain','interior'):
            for key in ('PK1','fiber_PK1'):
                gain=baseline[region][key]['absolute']-errors[region][key]['absolute'];threshold=max(.15*baseline[region][key]['absolute'],2*unc[region][key]['empirical_uncertainty_Pa'])
                if gain<=threshold:reasons.append(f'{region}/{key}: gain {gain:.8g} <= resolvable {threshold:.8g}')
        for region in errors:
            for key in ('PK1','fiber_PK1'):
                if errors[region][key]['absolute']>baseline[region][key]['absolute']+.02:reasons.append(region+'/'+key+' regression')
        fields,_,_=compare_fields(model,state,rm,rs)
        reaction=metric(stats['reaction_N'],read(run/'P3/R4/result.json')['solve']['reaction_N'],1e-4,.05)
        if not reaction['passed'] or not all(v['displacement']['passed'] for v in fields.values()):reasons.append('displacement/reaction mismatch')
        np.savez_compressed(folder/'static.npz',q=state.q,full=cr.expand(state.q))
        result=dict(status='passed_scoped',eligible=not reasons,reasons=reasons,errors=errors,mass=mass,material=materials,invariants=inv,solve=stats,
            reaction=reaction,displacement_vs_R3={k:v['displacement'] for k,v in fields.items()},displacement_scope='R3; main stress and reaction use R4')
        write(folder/'result.json',result);results.append(dict(name=name,eligible=not reasons,reasons=reasons,errors=errors));print('CANDIDATE',name,not reasons,errors['interior'],flush=True)
        del model,higher,cr;gc.collect()
    eligible=[x for x in results if x['eligible']];winner=min(eligible,key=lambda x:x['errors']['interior']['PK1']['absolute'])['name'] if eligible else None
    write(run/'P3/space-decision.json',dict(status='pending_heldout' if winner else 'retain_latest_swap6_no_resolved_gain',candidate=winner,
        selected='nonlinear-modes-swap6',records=results,heldout_accessed=False,spatial_accuracy=False,seconds=time.perf_counter()-start))
    if not winner:write(run/'P3/dynamic-qualification.json',dict(status='inherited_same_space',source_sha256=sha(APP/'N1/dynamic-qualification.json'),new_prefix_not_needed=True))
    print('SPACE_DECISION',winner or 'retain swap6',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['reference','retain','candidates']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'reference':reference,'retain':retain,'candidates':candidates}[a.phase](a.run)
