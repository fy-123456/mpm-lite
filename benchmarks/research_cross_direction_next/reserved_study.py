"""Conditional one-shot reserved-angle check, never fed back to design."""
from pathlib import Path
import argparse,gc
import numpy as np
import scipy.linalg as la
from .provenance import *
from .spatial_study import material,materials,NAME,nodal
from .spaces import load_selected
from .run import create_config,load_model
from benchmarks.research_spatial_phase_next.spaces import reopen_candidate
from benchmarks.research_phase_reference_next.spatial_study import reload_level
from .solver import bounded_static
from benchmarks.research_reference_next.field_audit import compare_nodal
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel

def solve(run,label):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    if read(run/'S1/space-decision.json')['status']!='pending_reserved':raise ValueError('training gate required')
    target=run/'S1/reserved'/label
    if not (target/'protocol.json').exists():register(run,f'S1/reserved/{label}/protocol.json',dict(angle=52.5,peak_m=.005,training_sha256=sha(run/'S1/training-comparison.json'),design_sha256=sha(run/'S1/design-protocol.json'),no_retraining=True))
    if label in ('R4','R5'):r,initial,_,_=reload_level(PHASE_REFERENCE,int(label[1:]))
    elif label=='baseline':
        r,_=load_selected(read(run/'baseline-space.json')['package'])
        with np.load(LOCAL/'S1/candidates/global-snapshot6/static.npz') as z:initial=z['full'].copy()
    else:
        r,_=reopen_candidate(run/'S1/candidates'/NAME)
        with np.load(run/'S1/candidates'/NAME/'F45.npz') as z:initial=z['full'].copy()
    r=material(r,52.5);m=SegmentedModel(r,order=7,device='cuda:0',hold=.005);state,stats=bounded_static(m,initial,target/'search.json',600);checks=materials(r,state.q);frame=probe_frame(m,state,[33,7,7]);full=r.expand(state.q)
    np.savez_compressed(target/'state.npz',q=state.q,full=full,nodes=r.parent.nodes(full),**{f'e{i}':e for i,e in enumerate(r.parent.edges)},**frame)
    write(target/'result.json',dict(status='passed_scoped',p=r.parent.p,solve=stats,material=checks,space=r.signature));print('RESERVED',label,stats['nonlinear_residual_N'],flush=True)

def decide(run):
    run=Path(run);z={n:np.load(run/'S1/reserved'/n/'state.npz') for n in ('R4','R5','baseline','candidate')};r,_=load_selected(read(run/'baseline-space.json')['package']);r=material(r,52.5)
    def field(n):return [z[n][f'e{i}'] for i in range(3)],read(run/'S1/reserved'/n/'result.json')['p'],z[n]['nodes']
    adj=compare_nodal(field('R4'),field('R5'),r.parent.A,r.parent.params);old=compare_nodal(field('baseline'),field('R5'),r.parent.A,r.parent.params);new=compare_nodal(field('candidate'),field('R5'),r.parent.A,r.parent.params)
    reasons=[]
    for reg in new:
        for key,v in new[reg].items():
            if adj[reg][key]['absolute']>.25*adj[reg][key]['budget']:reasons.append('reference_limited '+reg+'/'+key)
            if not v['passed'] or v['absolute']-old[reg][key]['absolute']>max(2*adj[reg][key]['absolute'],.1*v['budget']):reasons.append('space_limited '+reg+'/'+key)
    reaction=metric(read(run/'S1/reserved/candidate/result.json')['solve']['reaction_N'],read(run/'S1/reserved/R5/result.json')['solve']['reaction_N'],1e-4,.05)
    disp={k:metric(z['candidate']['x']-z['candidate']['X'],z['R5']['x']-z['R5']['X'],5e-5,.05,w) for k,w in regions(z['R5']['X']).items()}
    if not reaction['passed'] or not all(v['passed'] for v in disp.values()):reasons.append('reaction/displacement')
    write(run/'S1/reserved-direction-check.json',dict(status='passed_scoped' if not reasons else 'limited',angle=52.5,accessed=True,adjacent=adj,old=old,new=new,displacement=disp,reaction=reaction,reasons=reasons))
    if reasons:
        write(run/'S1/space-decision.json',dict(status='retain_baseline',selected='global-snapshot6',spatial_accuracy=False,reasons=reasons));write(run/'selected-solid-space.json',read(run/'baseline-space.json'));write(run/'S1/dynamic-smoke.json',dict(status='not_triggered',reason='reserved check failed'));return
    from engine.aniso_phase1.research_spatial_phase_next.performance import save
    cr,_=reopen_candidate(run/'S1/candidates'/NAME);pkg=run/'S1/candidates'/NAME/'space-package.json';entry=dict(path=str(pkg.resolve()),sha256=sha(pkg));entry['cache']=save(cr,run/'S1/candidates'/NAME/'qualified-array-cache',entry)
    # Keep the known uncompressed archive layout; all arrays still hash-bound.
    import shutil,scipy.sparse as sp
    original=Path(entry['cache']['path']);dest=run/'S1/candidates'/NAME/'uncompressed-space-cache';dest.mkdir();meta=read(original)
    for name in meta['files']:
        if name!='raw.npz':shutil.copyfile(original.parent/name,dest/name)
    sp.save_npz(dest/'raw.npz',cr.parent.raw,compressed=False);meta['files']={n:sha(dest/n) for n in meta['files']};meta['source_cache_sha256']=entry['cache']['sha256'];write(dest/'cache.json',meta);entry['cache']=dict(path=str((dest/'cache.json').resolve()),sha256=sha(dest/'cache.json'))
    loaded,_=load_selected(entry)
    if loaded.signature!=cr.signature or not np.array_equal(loaded.M,cr.M):raise ValueError('new cache mismatches physical space')
    write(run/'S1/provisional-space.json',dict(selected=NAME,package=entry,mass_order=7,full_order=7,spatial_accuracy=False))
    times=read(APP/'S1/time-decision.json')['times'][:9];create_config(run,'space-smoke',end=times[-1],times=times,space=entry,field_cache=True,display_frames=3)
    write(run/'S1/space-decision.json',dict(status='pending_smoke',selected=NAME,spatial_accuracy=False))

def promote(run):
    run=Path(run);report=read(run/'cases/space-smoke/summary.json')
    if report['status']!='passed_scoped' or report['steps']!=8 or report['min_detF']<=.1:raise ValueError('eight-step scene gate failed')
    cfg=read(run/'cases/space-smoke/execution-protocol.json');m,_=load_model(run,cfg);K=m.rest_K[np.ix_(m.ids,m.ids)];M=m.M3ff;lam,V=la.eigh(K,M);orth=float(la.norm(V.T@M@V-np.eye(len(lam))));res=float(la.norm(K@V-(M@V)*lam)/la.norm(K@V))
    if orth>1e-6 or res>1e-7 or lam[0]<=0:raise ValueError('new modal basis invalid')
    from benchmarks.research_phase_stress_next.time_study import history
    from engine.aniso_phase1.research_post_release.fields import CachedProbes
    h=history(run/'cases/space-smoke');state=h[-1]['state'];cache=CachedProbes(m);force=(m.rest_K@m.boundary.unit.ravel()).reshape(state.q.shape);forced=np.abs(V.T@force[m.free].ravel());energy=np.array([.5*(V.T@M@x['state'].velocity[m.free].ravel())**2 for x in h]);ids=sorted(set([int(np.argmax(forced)),int(np.argmax(np.max(energy,axis=0))),0]));records=[]
    for j in ids:
        d=np.zeros_like(state.q);d[m.free]=V[:,j].reshape(-1,3);eps=1e-7/max(1.,la.norm(d));a,b=state.clone(),state.clone();a.q+=eps*d;b.q-=eps*d;stress=float(np.sqrt(np.mean(((cache.frame(a)['PK1']-cache.frame(b)['PK1'])/(2*eps))**2)));amp=max(abs(float(V[:,j]@M@x['state'].q[m.free].ravel())) for x in h)
        records.append(dict(mode=j,period_s=float(2*np.pi/np.sqrt(lam[j])),estimated_output_peak_Pa=stress*amp,scope='eight-step loading smoke only'))
    np.savez_compressed(run/'S1/modal-basis.npz',values=lam,vectors=V);write(run/'S1/modal-observable-map.json',dict(space=m.reduction.signature,selected_modes=records,mass_orthogonality=orth,eigen_residual=res,new_modal_numbers=True))
    choice=read(run/'S1/provisional-space.json');choice['status']='provisional_until_full_cycle';write(run/'selected-space.json',choice);write(run/'selected-solid-space.json',choice);write(run/'S1/space-decision.json',dict(status='provisional_new_space',selected=NAME,spatial_accuracy=False));write(run/'S1/dynamic-smoke.json',dict(status='passed_scoped',summary=report,from_rest=True,old_coordinates_not_copied=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['solve','decide','promote']);p.add_argument('--name',choices=['R4','R5','baseline','candidate']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):solve(a.run,a.name) if a.phase=='solve' else globals()[a.phase](a.run)
