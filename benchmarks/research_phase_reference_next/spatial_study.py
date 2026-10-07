"""Bounded nested physical reference; production basis and original Ks stay fixed."""
from pathlib import Path
import argparse,gc,time,resource,threading,os,traceback
import numpy as np
from .provenance import APP,REFERENCE,COST_PARENT,read,write,sha,register,verify,serial_lock
from .spaces import load_selected
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_local_span_next.spatial_study import reference4
from benchmarks.research_reference_next.field_audit import compare_nodal
from benchmarks.research_sequential_next.spatial import static_solve
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_reference_next.reference import interior_h
from engine.aniso_phase1.research_post_release.ambient_reference import embedding_audit,operators
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential.condensation import Condensation

def qcheck(r,q):
    a=SegmentedModel(r,order=7,device='cuda:0',hold=.005);b=SegmentedModel(r,order=8,device='cuda:0',hold=.005)
    d=np.random.default_rng(20261001).normal(size=q.shape);d[a.fixed]=0;d/=np.linalg.norm(d)
    x,y=a.evaluate(q,d),b.evaluate(q,d)
    out={k:metric(x[k],y[k],at,rt) for k,at,rt in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]}
    if not all(v['passed'] for v in out.values()):raise ValueError('reference q7/q8 insufficient')
    return out

def reload4():
    r,m,s,e=reopen(REFERENCE/'Q1/R3');rr,f,n,e4=reference4(r);del r,m,s;gc.collect();return rr,f,n,e4

def reload_level(run,level):
    r,f,n,e=reload4()
    for k in range(5,level+1):
        space,_=interior_h(r.parent);p=Path(run)/f'S2/R{k}';package=read(p/'space-package.json')
        if sha(p/'data.npz')!=package['data_sha256'] or space.signature!=package['space_sha256']:raise ValueError('new reference changed')
        with np.load(p/'data.npz') as z:rr=Condensation(space,z['M'],z['K']);f=z['full'].copy();n=z['nodes'].copy()
        del r;r=rr;gc.collect()
    return r,f,n,e

def audit(run):
    run=Path(run);verify(run)
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    r,m,s,e=reopen(REFERENCE/'Q1/R3');r4,f4,n4,e4=reference4(r)
    adj=compare_nodal((r.parent.edges,r.parent.p,r.parent.nodes(r.expand(s.q))),(r4.parent.edges,r4.parent.p,n4),r.parent.A,r.parent.params)
    selected=read(run/'selected-space.json');formal,_=load_selected(selected['package'])
    with np.load(APP/'S1/candidates/global-snapshot6/static.npz') as z:full=z['full'].copy()
    current=compare_nodal((formal.parent.edges,formal.parent.p,formal.parent.nodes(full)),(r4.parent.edges,r4.parent.p,n4),formal.parent.A,formal.parent.params)
    previous=read(APP/'S1/candidates/global-snapshot6/result.json')['errors']
    mismatch=max(abs(current[a][b]['absolute']-previous[a][b]['absolute']) for a in current for b in ('PK1','fiber_PK1'))
    if mismatch>1e-8:raise ValueError('latest static evidence does not reconstruct')
    write(run/'S2/reference-audit.json',dict(status='passed',R3_reload=e,R4_reload=e4,recomputed_difference=mismatch,adjacent_R3_R4=adj,source_sha256=sha(COST_PARENT/'P3/R4/space-package.json')))
    write(run/'S2/error-vs-reference.json',dict(current=current,empirical_uncertainty_Pa=.0034873,strict_bound=False,static_peak_m=.005,stress='PK1',prior_equilibrium=read(APP/'S1/candidates/global-snapshot6/result.json')['solve']))
    raw=r4.parent.raw;rawbytes=sum(a.nbytes for a in (raw.data,raw.indices,raw.indptr));nodes=int(np.prod(r4.parent.shape));estimate=3*rawbytes+nodes*(12*8+12*8+10*8)+2*1024**3
    register(run,'S2/refinement-protocol.json',dict(route='nested local h at existing .375/.625 residual support, p6 unchanged',max_levels=2,levels=[5,6],
        first_reference='R4',formal_space='global-snapshot6',formal_change=False,current_raw_nnz=int(raw.nnz),current_raw_bytes=rawbytes,
        estimated_peak_GiB=estimate/2**30,estimate_formula='3 sparse copies + 12 basis + 12 coordinate/work arrays + 10 field arrays + 2 GiB runtime',
        estimated_scalar_dofs=[r4.parent.ndof+12,r4.parent.ndof+24],max_rss_GiB=16,soft_seconds=600,hard_seconds=1200,
        second_level_condition='R4-R5 fails to contract relative to R3-R4 or exceeds .005 Pa interior/global',original_Ks=True,mass_order=7,material_orders=[7,8],
        material_test=dict(k_f=220.,peak_m=.005,label='pre-registered material variation; no independent hidden-test claim')))
    if estimate/2**30>16:raise MemoryError('reference estimate exceeds budget')
    print('REFERENCE_AUDIT',mismatch,'estimated_GiB',estimate/2**30,flush=True)

def reference(run,level):
    run=Path(run);verify(run);folder=run/f'S2/R{level}';folder.mkdir(exist_ok=False);start=time.perf_counter();stop=threading.Event()
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    def guard():
        while not stop.wait(.5):
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
            if rss>16 or time.perf_counter()-start>1200:
                write(folder/'failure.json',dict(type='resource_limit',seconds=time.perf_counter()-start,peak_rss_GiB=rss));os._exit(75)
    threading.Thread(target=guard,daemon=True).start()
    try:
        r,f,n,e=reload_level(run,level-1);space,definition=interior_h(r.parent)
        write(folder/'definition.json',dict(**definition,parent_space=r.parent.signature));embedding=embedding_audit(r.parent,space)
        initial=np.vstack((f,np.zeros((space.ndof-r.parent.ndof,3))))
        rr,op=operators(r,space,max_seconds=max(1,600-(time.perf_counter()-start)),progress=lambda *a:print('REFERENCE',level,*a,flush=True))
        model=SegmentedModel(rr,order=7,device='cuda:0',hold=.005)
        state,stats=static_solve(model,initial,max_seconds=max(1,600-(time.perf_counter()-start)))
        nodes=space.nodes(rr.expand(state.q));adj=compare_nodal((r.parent.edges,r.parent.p,n),(space.edges,space.p,nodes),space.A,space.params)
        del model;gc.collect();material=qcheck(rr,state.q)
        np.savez_compressed(folder/'data.npz',M=rr.original_mass,K=rr.original_stiffness,q=state.q,full=rr.expand(state.q),nodes=nodes)
        write(folder/'space-package.json',dict(data_sha256=sha(folder/'data.npz'),space_sha256=space.signature,reduction_sha256=rr.signature,definition_sha256=sha(folder/'definition.json'),mass_order=7,material_order=7,scope='main .005 F45 static, original Ks'))
        write(folder/'result.json',dict(status='passed_scoped',operators=op,embedding=embedding,solve=stats,adjacent=adj,material=material,seconds=time.perf_counter()-start,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
        print('REFERENCE_DONE',level,adj['interior'],flush=True)
    except (ValueError,TimeoutError,MemoryError) as exc:
        write(folder/'failure.json',dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start));print('REFERENCE_LIMIT',str(exc),flush=True)
    finally:stop.set()

def decision(run):
    run=Path(run);old=read(run/'S2/reference-audit.json')['adjacent_R3_R4'];rows=[];eligible=4;need=False
    for level in (5,6):
        folder=run/f'S2/R{level}'
        if not (folder/'result.json').exists():continue
        result=read(folder/'result.json');current=result['adjacent'];reasons=[]
        for region in ('global_domain','interior','transition'):
            for key in ('PK1','fiber_PK1'):
                if current[region][key]['absolute']>max(1.1*old[region][key]['absolute'],.001):reasons.append(region+'/'+key+' not contracted within engineering slack')
        good=not reasons;need=not good or any(current[x]['PK1']['absolute']>.005 for x in ('interior','global_domain'))
        if good:eligible=level
        rows.append(dict(level=level,qualified=good,reasons=reasons,adjacent=current));old=current
    failures={f'R{k}':read(run/f'S2/R{k}/failure.json') for k in (5,6) if (run/f'S2/R{k}/failure.json').exists()}
    write(run/'S2/reference-convergence.json',dict(status='passed_scoped' if eligible>4 else 'reference_limited',qualified_level=eligible,second_level_needed=need and len(rows)==1,records=rows,failures=failures,strict_continuum_accuracy=False))
    print('REFERENCE_DECISION',eligible,need,flush=True)

def finish(run):
    run=Path(run);d=read(run/'S2/reference-convergence.json');g=read(run/'S2/generalization-check.json')
    write(run/'S2/space-decision.json',dict(status='retain_current_formal_space',selected='global-snapshot6',package=read(run/'selected-space.json'),new_formal_candidates=0,reference=d['qualified_level'],generalization_status=g['status'],spatial_accuracy=False))
    write(run/'S2/capability-scope.json',dict(formal='same 144 functions, original M7 and Ks',reference='empirical static only',continuous_spatial_accuracy=False,material_variation=g['status'],time_study_invalidated=False,production_material_unchanged=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['audit','reference','decision','finish']);p.add_argument('--level',type=int,default=5);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='reference':reference(a.run,a.level)
        else:globals()[a.phase](a.run)
