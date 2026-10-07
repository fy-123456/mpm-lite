"""Nonlinear residual modes on the sealed R3, with two frozen 144 budgets."""
from pathlib import Path
import argparse,time,resource,threading,os,gc
import numpy as np
import scipy.linalg as la
from .provenance import APP,PARENT,read,write,sha,digest,verify,register,serial_lock,resources
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_reference_next.field_audit import compare_nodal,invariants
from benchmarks.research_sequential_next.model_package import load_reduction
from benchmarks.research_sequential_next.spatial import static_solve,compare_fields
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_spatial_phase_next.space import combine
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.model import PracticalModel
from engine.aniso_phase1.research_d.common_kinetic import PointInertia


def setup(run):
    verify(run)
    import warp as wp
    wp.config.kernel_cache_dir=str(Path(run).resolve()/'warp-cache')
    def guard():
        while True:
            peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
            if peak>16:
                write(Path(run)/'N1/resource-stop.json',dict(peak_rss_GiB=peak,reason='16 GiB limit'));os._exit(75)
            time.sleep(.2)
    threading.Thread(target=guard,daemon=True).start()


def reference(level='R3'):
    return reopen(APP/'Q1'/level)


from .spaces import reopen_candidate


def design(run):
    run=Path(run);setup(run)
    register(run,'N1/design-protocol.json',dict(main_peak_m=.005,heldout_peak_m=.00375,swaps=[6,12],budget=144,
        reference_sha256=sha(APP/'Q1/R3/space-package.json'),promotion=dict(relative_gain=.2,uncertainty_multiple=2,regional_atol_Pa=.02),
        method='nonlinear rich-space residual; full rest-Schur block scores; mass-normalized compact residual-relaxation/reference-difference snapshot modes',
        support='two existing reference halves, no extension outside R3 span',heldout_not_used=True))
    begin=time.perf_counter();r,cpu,state,reload_error=reference();s=r.parent;nlocal=s.ndof-s.n
    original=load_reduction(PARENT)
    with np.load(PARENT/'reference/original144-static.npz',allow_pickle=False) as z:oldfull=z['full'].copy()
    full=np.vstack((oldfull,np.zeros((s.ndof-len(oldfull),3))))
    axes=[np.linspace(e[0]+1e-8,e[-1]-1e-8,n) for e,n in zip(s.edges,[29,5,5])]
    a,ga=original.parent._sample(original.parent.nodes(oldfull),axes);b,gb=s._sample(s.nodes(full),axes)
    embedding=max(float(np.max(abs(a-b))),float(np.max(abs(ga-gb))))
    if embedding>1e-8:raise ValueError('original144 embedding failed')
    model=SegmentedModel(r,order=7,device='cuda:0',hold=.005);out=model.evaluate(r.project(full));res=out['original_full_force']
    free=s.free_scalar_ids;ids=(3*free[:,None]+np.arange(3)).ravel();K=r.original_stiffness[np.ix_(ids,ids)];K=.5*(K+K.T)
    inv=la.cho_solve(la.cho_factor(K),np.eye(len(ids)));rv=res[free].ravel();delta=-inv@rv
    mapping={int(x):i for i,x in enumerate(free)};relax=np.zeros_like(full);relax[free]=delta.reshape(-1,3)
    ref_full=r.expand(state.q);difference=ref_full-full
    def dofs(indices):return (3*np.array([mapping[s.n+i] for i in indices])[:,None]+np.arange(3)).ravel()
    def block_score(indices):
        ix=dofs(indices);S=la.solve(inv[np.ix_(ix,ix)],np.eye(len(ix)),assume_a='pos');rb=-S@delta[ix]
        predicted=float(.5*rb@la.solve(S,rb,assume_a='pos'))
        direct=float(.5*delta[ix]@S@delta[ix])
        if abs(predicted-direct)>1e-12+2e-5*abs(direct):raise ValueError('Schur residual score mismatch')
        return predicted
    def removal(indices):
        ix=dofs(indices);a=ref_full[s.n+np.array(indices)].ravel();force=la.solve(inv[np.ix_(ix,ix)],a,assume_a='pos');d=-inv[:,ix]@force
        energy=float(.5*a@force);direct=float(.5*d@K@d)
        if abs(energy-direct)>1e-12+2e-5*abs(energy):raise ValueError('joint removal mismatch')
        return energy
    blocks=[dict(indices=list(range(i,i+6)),residual_energy_J=block_score(list(range(i,i+6))),removal_energy_J=removal(list(range(i,i+6)))) for i in range(0,nlocal,6)]
    old=sorted(blocks[:24],key=lambda x:x['removal_energy_J'])
    # The inherited 36 h functions alternate the two half-domain supports.
    groups=[list(range(144,150))+list(range(156,168))+list(range(180,192))+list(range(204,210)),
            list(range(150,156))+list(range(168,180))+list(range(192,204))+list(range(210,216))]
    if nlocal!=216 or sorted(groups[0]+groups[1])!=list(range(144,nlocal)):raise ValueError('reference pool layout changed')
    modes=[];mode_reports=[]
    for group in groups:
        ix=s.n+np.array(group);M=r.original_mass[np.ix_(ix,ix)];L=la.cholesky(M,lower=True)
        snapshots=np.column_stack((relax[ix],difference[ix]));U,sv,_=la.svd(L.T@snapshots,full_matrices=False)
        modes.append(la.solve_triangular(L.T,U,lower=False));mode_reports.append(dict(indices=group,singular_values=sv.tolist(),mass_orthogonality=float(la.norm(U.T@U-np.eye(6)))))
    # Both proposals are committed before their nonlinear results are observed.
    for swap in [6,12]:
        remove=[j for x in old[:swap//6] for j in x['indices']];keep=sorted(set(range(144))-set(remove));C=np.zeros((nlocal,144))
        C[keep,np.arange(len(keep))]=1
        for half,(group,mode) in enumerate(zip(groups,modes)):
            width=swap//2;C[np.ix_(group,np.arange(len(keep)+half*width,len(keep)+(half+1)*width))]=mode[:,:width]
        name=f'nonlinear-modes-swap{swap}';cr,T=combine(r,C,name);folder=run/'N1/candidates'/name;folder.mkdir(parents=True,exist_ok=False)
        initial=la.lstsq(T,ref_full)[0]
        np.savez_compressed(folder/'space.npz',C=C,T=T,M=cr.original_mass,K=cr.original_stiffness,initial=initial)
        package=dict(schema='spatial-phase-combination-v1',name=name,budget=144,reference_sha256=sha(APP/'Q1/R3/space-package.json'),data_sha256=sha(folder/'space.npz'),
            reduction_sha256=cr.signature,space_sha256=cr.parent.signature,mass_order=7,full_order=7,removed=remove,joint_removal_J=removal(remove),
            supports=[[.25,.5],[.5,.75]],rank=cr.audit,meaning='same reference R3 span, compact vector-component snapshot combinations',promoted=False)
        write(folder/'space-package.json',package);print('DESIGNED',name,cr.audit['independent_mass_condition'],flush=True)
    olderrors=compare_nodal((original.parent.edges,original.parent.p,original.parent.nodes(oldfull)),(s.edges,s.p,s.nodes(ref_full)),s.A,s.params)
    previous=read(APP/'Q2/space-decision.json')['original_errors']
    if abs(olderrors['interior']['PK1']['absolute']-previous['interior']['PK1']['absolute'])>1e-7:raise ValueError('reference metric changed')
    write(run/'N1/reference-use.json',dict(status='passed_scoped',embedding_max=embedding,reference_reload_error=reload_error,original_errors=olderrors,
        nonlinear_rich_residual_N=float(la.norm(rv)),original_subspace_residual_N=float(la.norm(res[s.free_scalar_ids[s.free_scalar_ids<len(oldfull)]])),
        main_peak_m=.005,reference_scope=read(APP/'Q1/reference-decision.json')['scope']))
    write(run/'N1/nonlinear-block-scores.json',dict(records=blocks,compact_modes=mode_reports,rest_SPD_checked=True,nonlinear_SPD_not_assumed=True,
        seconds=time.perf_counter()-begin,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))


def solve(run,name):
    run=Path(run);setup(run);begin=time.perf_counter();folder=run/'N1/candidates'/name
    register(run,f'N1/candidates/{name}/solve-protocol.json',dict(peak_m=.005,material=[7,8],mass=[7,8],reference='R3',heldout=False))
    r,ref,rs,_=reference();refnodes=r.parent.nodes(r.expand(rs.q));cr,p=reopen_candidate(folder,r);s=cr.parent
    with np.load(folder/'space.npz',allow_pickle=False) as z:initial=z['initial'].copy();T=z['T'].copy()
    invchecks=invariants(s);rng=np.random.default_rng(20261001);d=rng.normal(size=(s.ndof,3));d/=la.norm(d)
    M7=PointInertia(s,order=7).apply(d);M8=PointInertia(s,order=8).apply(d)
    mass=dict(transformed=metric(cr.original_mass@d,M7,1e-10,2e-5),adjacent=metric(M7,M8,1e-10,2e-5),all_cross_terms=True)
    if not mass['transformed']['passed'] or not mass['adjacent']['passed']:raise ValueError('candidate mass action mismatch')
    model=SegmentedModel(cr,order=7,device='cuda:0',hold=.005);state,stats=static_solve(model,initial)
    q=state.q;direction=rng.normal(size=q.shape);direction[model.fixed]=0;direction/=la.norm(direction)
    higher=SegmentedModel(cr,order=8,device='cuda:0',hold=.005);aa=model.evaluate(q,direction);bb=higher.evaluate(q,direction)
    material={k:metric(aa[key],bb[key],atol,rtol) for k,key,atol,rtol in [('energy','material_U',1e-10,.02),('force','material_force',1e-8,.02),('tangent','tangent_action',1e-8,.03)]}
    if not all(x['passed'] for x in material.values()):raise ValueError('candidate sufficient rule not qualified')
    errors=compare_nodal((s.edges,s.p,s.nodes(cr.expand(q))),(r.parent.edges,r.parent.p,refnodes),s.A,s.params)
    baseline=read(run/'N1/reference-use.json')['original_errors'];uncertainty=read(APP/'Q1/reference-decision.json')['regional'];reasons=[]
    for reg in ['global_domain','interior']:
        for key in ['PK1','fiber_PK1']:
            e0=baseline[reg][key]['absolute'];gain=e0-errors[reg][key]['absolute'];threshold=max(.2*e0,2*uncertainty[reg][key]['empirical_uncertainty'])
            if gain<=threshold:reasons.append(f'{reg}/{key}: gain {gain:g} <= {threshold:g}')
    for reg in errors:
        for key in ['PK1','fiber_PK1']:
            if errors[reg][key]['absolute']>baseline[reg][key]['absolute']+.02:reasons.append(f'{reg}/{key}: regression')
    fields,_,_=compare_fields(model,state,ref,rs)
    reaction=metric(stats['reaction_N'],float(np.sum(ref.evaluate(rs.q)['force']*ref.boundary.unit)),1e-4,.05)
    if not reaction['passed'] or not all(v['displacement']['passed'] for v in fields.values()):reasons.append('reaction or displacement outside budget')
    np.savez_compressed(folder/'static.npz',q=q,full=cr.expand(q));write(folder/'result.json',dict(status='passed_scoped',solve=stats,errors=errors,material=material,mass=mass,invariants=invchecks,
        reaction=reaction,displacement={k:v['displacement'] for k,v in fields.items()},eligible=not reasons,reasons=reasons,seconds=time.perf_counter()-begin,
        peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,new_process_reload=True))
    print('CANDIDATE',name,'eligible',not reasons,'interior',errors['interior'],'seconds',time.perf_counter()-begin,flush=True)


def decision(run):
    run=Path(run);reports=[]
    for name in ['nonlinear-modes-swap6','nonlinear-modes-swap12']:
        rec=read(run/'N1/candidates'/name/'result.json');reports.append(dict(name=name,eligible=rec['eligible'],errors=rec['errors'],reasons=rec['reasons']))
    eligible=[x for x in reports if x['eligible']];winner=min(eligible,key=lambda x:x['errors']['interior']['PK1']['absolute'])['name'] if eligible else None
    write(run/'N1/space-decision.json',dict(status='pending_heldout_and_dynamics' if winner else 'not_promoted_no_resolved_gain',candidate=winner,selected='original144',records=reports,heldout_accessed=False,spatial_certified=False))
    if not winner:write(run/'selected-space.json',dict(selected='original144',package=None,mass_order=5,full_order=7,status='retained_after_N1'))
    print('SPACE_DECISION',winner or 'original144',flush=True)


def heldout(run,name):
    run=Path(run);setup(run)
    dec=read(run/'N1/space-decision.json')
    if not dec['candidate']:raise ValueError('heldout not authorized without main gain')
    if name not in ['R2','R3','original144',dec['candidate']]:raise ValueError('unknown heldout space')
    folder=run/'N1/heldout'/name
    register(run,f'N1/heldout/{name}/protocol.json',dict(peak_m=.00375,selection_frozen=sha(run/'N1/space-decision.json'),actual_nonlinear_solve=True))
    if name in ['R2','R3']:
        r,_,old,_=reference(name);initial=r.expand(old.q)*.75
    elif name=='original144':
        r=load_reduction(PARENT)
        with np.load(PARENT/'reference/original144-static.npz') as z:initial=z['full']*.75
    else:
        r,_=reopen_candidate(run/'N1/candidates'/name)
        with np.load(run/'N1/candidates'/name/'static.npz') as z:initial=z['full']*.75
    m=SegmentedModel(r,order=7,device='cuda:0',hold=.00375);state,stats=static_solve(m,initial)
    out=m.evaluate(state.q)
    np.savez_compressed(folder/'static.npz',q=state.q,full=r.expand(state.q),nodes=r.parent.nodes(r.expand(state.q)),**{f'e{i}':e for i,e in enumerate(r.parent.edges)})
    write(folder/'result.json',dict(status='passed_scoped',solve=stats,p=r.parent.p,scaled_initial_only=True,min_detF=out['min_detF'],reduction_sha256=r.signature))
    print('HELDOUT',name,stats['nonlinear_residual_N'],flush=True)


def heldout_decision(run):
    run=Path(run);setup(run);dec=read(run/'N1/space-decision.json');name=dec['candidate'];data={}
    for label in ['R2','R3','original144',name]:
        folder=run/'N1/heldout'/label
        with np.load(folder/'static.npz') as z:data[label]=([z[f'e{i}'].copy() for i in range(3)],read(folder/'result.json')['p'],z['nodes'].copy())
    original=load_reduction(PARENT);s=original.parent
    unc=compare_nodal(data['R2'],data['R3'],s.A,s.params)
    old=compare_nodal(data['original144'],data['R3'],s.A,s.params);new=compare_nodal(data[name],data['R3'],s.A,s.params)
    reasons=[]
    for region in old:
        for key in ['PK1','fiber_PK1']:
            if not unc[region][key]['passed']:reasons.append(region+'/'+key+' reference limited')
            if region in ['global_domain','interior'] and old[region][key]['absolute']-new[region][key]['absolute']<=unc[region][key]['absolute']:reasons.append(region+'/'+key+' gain below reference difference')
            if new[region][key]['absolute']>old[region][key]['absolute']+.02:reasons.append(region+'/'+key+' regression')
    refreaction=read(run/'N1/heldout/R3/result.json')['solve']['reaction_N']
    for label in ['R2',name]:
        if not metric(read(run/'N1/heldout'/label/'result.json')['solve']['reaction_N'],refreaction,1e-4,.05)['passed']:reasons.append(label+' reaction')
    write(run/'N1/heldout-result.json',dict(status='passed_scoped' if not reasons else 'not_promoted',reference_difference=unc,original=old,candidate=new,reasons=reasons,scope='new static .00375 loading, adjacent reference difference only'))
    dec.update(heldout_accessed=True,heldout_passed=not reasons,status='pending_short_dynamics' if not reasons else 'heldout_not_promoted')
    write(run/'N1/space-decision.json',dec)
    if not reasons:
        from .run import create_config
        package=run/'N1/candidates'/name/'space-package.json';entry=dict(path=str(package.resolve()),sha256=sha(package))
        create_config(run,'candidate-full-short',dt=.0125,end=.05,field_cache=True,space=entry,mass_order=7,full_order=7,display_frames=3)
    print('HELDOUT_DECISION',not reasons,reasons,flush=True)


def promote(run):
    from .run import load_model
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
    from benchmarks.research_sequential_next.checkpoint import GenerationStore
    run=Path(run);setup(run);dec=read(run/'N1/space-decision.json')
    if not dec.get('heldout_passed'):raise ValueError('no qualified heldout candidate')
    folder=run/'cases/candidate-full-short';cfg=read(folder/'execution-protocol.json');summary=read(folder/'summary.json')
    if summary['status']!='passed_scoped' or summary['steps']!=4:raise ValueError('missing short dynamic qualification')
    m,_=load_model(run,cfg);zero=SegmentedModel(m.reduction,order=7,device='cuda:0',hold=0.);solver=ValidatedAVF(zero,cfg);initial=solver.state
    for _ in range(4):solver.step(.0125)
    zerr=max(float(la.norm(solver.state.q-initial.q)),float(la.norm(solver.state.velocity)))
    if zerr>1e-8:raise ValueError('zero load dynamic drift')
    hist=GenerationStore(folder,read(folder/'identity.json')).history();rng=np.random.default_rng(44);v=rng.normal(size=m.rest().q.shape)
    massfull=m.reduction.velocity(v);energy_full=.5*float(np.sum(massfull*(m.reduction.original_mass@massfull)))
    mass_error=abs(energy_full-m.kinetic(v))
    if mass_error>1e-10*max(1.,abs(energy_full)):raise ValueError('full kinetic cross terms not preserved')
    write(run/'N1/dynamic-qualification.json',dict(status='passed_scoped',summary=summary,zero_load_four_step_error=zerr,kinetic_cross_term_error=mass_error,
        fixed_free_mass_cross_norm=float(la.norm(m.M[np.ix_(m.free,m.fixed)])),generations=len(hist),old_q_v_not_transplanted=True,
        original_space_static_reference_scope_only=True,source_sha256=read(folder/'identity.json')['numerical_source_sha256']))
    dec.update(status='promoted_scoped',selected=dec['candidate'],spatial_certified=False,dynamic_qualified=True)
    write(run/'N1/space-decision.json',dec)
    write(run/'selected-space.json',dict(selected=dec['candidate'],package=cfg['physical_space'],mass_order=7,full_order=7,status='promoted_scoped',
        scope='F45 .005 and heldout .00375 static plus eight dynamic qualification steps; no continuum accuracy certificate'))
    print('PROMOTED',dec['candidate'],zerr,mass_error,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['design','solve','decision','heldout','heldout-decision','promote']);p.add_argument('--run',type=Path,required=True);p.add_argument('--candidate');a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release')
        if a.phase=='solve':solve(a.run,a.candidate)
        elif a.phase=='heldout':heldout(a.run,a.candidate)
        else:{'design':design,'decision':decision,'heldout-decision':heldout_decision,'promote':promote}[a.phase](a.run)
