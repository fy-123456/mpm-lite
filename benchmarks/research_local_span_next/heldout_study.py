"""One reserved-amplitude solve per space; no new design after accessing it."""
from pathlib import Path
import argparse,gc,time
import numpy as np
import scipy.linalg as la
from .provenance import APP,SPACE_PARENT,REFERENCE,read,write,sha,register,serial_lock
from .spatial_study import setup,reference4,NAME
from .spaces import load_selected
from .run import create_config
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_reference_next.field_audit import compare_nodal
from benchmarks.research_spatial_phase_next.spaces import reopen_candidate
from benchmarks.research_sequential_next.spatial import static_solve
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel

def solve(run,label):
    run,begin=setup(run)
    if not read(run/'S1/main-decision.json')['eligible']:raise ValueError('heldout gated by main result')
    target=run/'S1/heldout'/label;register(run,f'S1/heldout/{label}/protocol.json',dict(peak_m=.00425,design_frozen_sha256=sha(run/'S1/design-protocol.json'),main_result_sha256=sha(run/'S1/main-decision.json'),actual_nonlinear_solve=True))
    if label in ('R3','R4'):
        r,unused,state,_=reopen(REFERENCE/'Q1/R3');initial=r.expand(state.q)
        del unused,state;gc.collect()
        if label=='R4':r,initial,_,_=reference4(r)
    elif label=='baseline':
        r,_=load_selected(read(APP/'selected-space.json')['package'])
        with np.load(SPACE_PARENT/'N1/candidates/nonlinear-modes-swap6/static.npz') as z:initial=z['full'].copy()
    elif label=='candidate':
        r,_=reopen_candidate(run/'S1/candidates'/NAME)
        with np.load(run/'S1/candidates'/NAME/'static.npz') as z:initial=z['full'].copy()
    else:raise ValueError('unknown heldout space')
    initial*=.85;m=SegmentedModel(r,order=7,device='cuda:0',hold=.00425);state,stats=static_solve(m,initial,max_seconds=600)
    high=SegmentedModel(r,order=8,device='cuda:0',hold=.00425);rng=np.random.default_rng(1001);d=rng.normal(size=state.q.shape);d[m.fixed]=0;d/=la.norm(d);a,b=m.evaluate(state.q,d),high.evaluate(state.q,d)
    checks={k:metric(a[k],b[k],at,rt) for k,at,rt in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]}
    if not all(x['passed'] for x in checks.values()):raise ValueError('heldout sufficient material not qualified')
    full=r.expand(state.q);np.savez_compressed(target/'static.npz',q=state.q,full=full,nodes=r.parent.nodes(full),**{f'e{i}':e for i,e in enumerate(r.parent.edges)})
    write(target/'result.json',dict(status='passed_scoped',p=r.parent.p,solve=stats,material=checks,reduction_sha256=r.signature,seconds=time.perf_counter()-begin));print('HELDOUT',label,stats,flush=True)

def decide(run):
    run=Path(run);data={};s=None
    for label in ('R3','R4','baseline','candidate'):
        folder=run/'S1/heldout'/label
        with np.load(folder/'static.npz') as z:data[label]=([z[f'e{i}'].copy() for i in range(3)],read(folder/'result.json')['p'],z['nodes'].copy())
    r,_=load_selected(read(APP/'selected-space.json')['package']);s=r.parent
    unc=compare_nodal(data['R3'],data['R4'],s.A,s.params);old=compare_nodal(data['baseline'],data['R4'],s.A,s.params);new=compare_nodal(data['candidate'],data['R4'],s.A,s.params);reasons=[]
    for reg in old:
        for key in ('PK1','fiber_PK1'):
            if not unc[reg][key]['passed']:reasons.append(reg+'/'+key+' reference limited')
            if reg in ('global_domain','interior') and old[reg][key]['absolute']-new[reg][key]['absolute']<=max(.15*old[reg][key]['absolute'],2*unc[reg][key]['absolute']):reasons.append(reg+'/'+key+' unresolved gain')
            if new[reg][key]['absolute']>old[reg][key]['absolute']+.02:reasons.append(reg+'/'+key+' regression')
    reaction=metric(read(run/'S1/heldout/candidate/result.json')['solve']['reaction_N'],read(run/'S1/heldout/R4/result.json')['solve']['reaction_N'],1e-4,.05)
    if not reaction['passed']:reasons.append('reaction')
    write(run/'S1/heldout-check.json',dict(status='passed_scoped' if not reasons else 'not_promoted',heldout_accessed=True,reference_difference=unc,baseline=old,candidate=new,reaction=reaction,reasons=reasons,scope='reserved .00425 static, empirical reference only'))
    dec=read(run/'S1/space-decision.json');dec.update(heldout_accessed=True,heldout_passed=not reasons,status='pending_short_dynamics' if not reasons else 'retained_baseline');write(run/'S1/space-decision.json',dec)
    if not reasons:
        from engine.aniso_phase1.research_spatial_phase_next.performance import save
        cr,_=reopen_candidate(run/'S1/candidates'/NAME);package=run/'S1/candidates'/NAME/'space-package.json';entry=dict(path=str(package.resolve()),sha256=sha(package))
        entry['cache']=save(cr,run/'S1/candidates'/NAME/'qualified-array-cache',entry)
        # Reuse the validated uncompressed archive representation rather than new cache semantics.
        import shutil,scipy.sparse as sp
        original=Path(entry['cache']['path']);target=run/'S1/candidates'/NAME/'uncompressed-space-cache';target.mkdir()
        metadata=read(original)
        for filename in metadata['files']:
            if filename!='raw.npz':shutil.copyfile(original.parent/filename,target/filename)
        sp.save_npz(target/'raw.npz',cr.parent.raw,compressed=False)
        metadata['files']={filename:sha(target/filename) for filename in metadata['files']}
        metadata['construction']='exact qualified arrays; uncompressed raw CSR, inherited representation'
        metadata['source_cache_sha256']=entry['cache']['sha256'];write(target/'cache.json',metadata)
        entry['cache']=dict(path=str((target/'cache.json').resolve()),sha256=sha(target/'cache.json'))
        loaded,_=load_selected(entry)
        if loaded.signature!=cr.signature or not np.array_equal(loaded.M,cr.M):raise ValueError('archive reconstruction differs')
        write(run/'S1/provisional-space.json',dict(selected=NAME,package=entry,mass_order=7,full_order=7,status='pending_dynamic',spatial_accuracy=False))
        create_config(run,'selected-prefix',end=1.2,space=entry,field_cache=True,display_frames=3)
    print('HELDOUT_DECISION',not reasons,new['interior'],reasons,flush=True)

def promote(run):
    run=Path(run);folder=run/'cases/selected-prefix';summary=read(folder/'summary.json')
    if not read(run/'S1/space-decision.json')['heldout_passed'] or summary['steps']<4 or summary['steps']>8:raise ValueError('short dynamic gate missing')
    cfg=read(folder/'execution-protocol.json');r,_=load_selected(cfg['physical_space']);m=SegmentedModel(r,order=7,device='cuda:0',hold=0.)
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
    c=ValidatedAVF(m,cfg);initial=c.state
    for _ in range(4):c.step(.0125)
    zero=max(float(la.norm(c.state.q-initial.q)),float(la.norm(c.state.velocity)))
    rng=np.random.default_rng(41);v=rng.normal(size=c.state.q.shape);full=r.velocity(v);kin=.5*float(np.sum(full*(r.original_mass@full)));ke=abs(kin-m.kinetic(v))
    if zero>1e-8 or ke>1e-10*max(1.,abs(kin)):raise ValueError('new space inertial invariant failed')
    selected=read(run/'S1/provisional-space.json');selected.update(status='promoted_scoped',scope='two seen static snapshots, reserved .00425 static and eight dynamic steps; no continuum certificate')
    write(run/'selected-space.json',selected);dec=read(run/'S1/space-decision.json');dec.update(status='promoted_scoped',selected=NAME);write(run/'S1/space-decision.json',dec)
    write(run/'S1/dynamic-qualification.json',dict(status='passed_scoped',loaded_steps=summary['steps'],zero_load_steps=4,zero_load_error=zero,complete_kinetic_error=ke,summary=summary))
    write(run/'S1/dependency-invalidation.json',dict(space_changed=True,recompute=['modal basis','time references','q5 certificates','performance state sources'],new_cache_bound=True,old_checkpoints_not_projected=True))
    print('SPACE_PROMOTED',NAME,zero,ke,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['solve','decide','promote']);p.add_argument('--label',choices=['baseline','candidate','R3','R4']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='solve':solve(a.run,a.label)
        else:{'decide':decide,'promote':promote}[a.phase](a.run)
