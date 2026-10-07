"""One actual nested h reference beyond authenticated R5, same original Ks."""
import argparse,gc,time,resource,threading,os,traceback
import numpy as np
from .provenance import *
from .runtime import update
from .spaces import load_selected
from benchmarks.research_phase_reference_next.spatial_study import reload_level
from benchmarks.research_reference_next.field_audit import compare_nodal,invariants
from engine.aniso_phase1.research_reference_next.reference import interior_h
from engine.aniso_phase1.research_post_release.ambient_reference import embedding_audit,operators
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from benchmarks.research_sequential_next.spatial import static_solve
from benchmarks.research_sequential_next.compare import metric

def main(run):
    run=Path(run);mutable(run);tick=time.perf_counter();folder=run/'S3';folder.mkdir(exist_ok=True);stop=threading.Event()
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    def guard():
        while not stop.wait(.5):
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
            if rss>16 or time.perf_counter()-tick>900:
                write(folder/'resource-stop.json',dict(status='limited',peak_rss_GiB=rss,seconds=time.perf_counter()-tick));os._exit(75)
    threading.Thread(target=guard,daemon=True).start()
    try:
        old=PHASE_REFERENCE/'S2/R5';old_result=read(old/'result.json')
        r,f,n,e=reload_level(PHASE_REFERENCE,5);raw=r.parent.raw;rawbytes=sum(a.nbytes for a in (raw.data,raw.indices,raw.indptr));nodes=int(np.prod(r.parent.shape));estimate=3*rawbytes+nodes*34*8+2*2**30
        write(folder/'solid-reference-audit.json',dict(status='passed_scoped',reference='R5, actual p6 ambient local h',source=dict(path=str(old/'space-package.json'),sha256=sha(old/'space-package.json')),R5_solve=old_result['solve'],R4_R5=old_result['adjacent'],same_problem='F45 pure-solid hold 0.005m, original carrier Ks; not zero-grip fluid release',new_formal_space=False))
        register(run,'S3/refinement-protocol.json',dict(route='one nested internal h beyond R5 near .375/.625; p6 unchanged',max_new_levels=2,first='R6',estimated_peak_GiB=estimate/2**30,raw_nnz=int(raw.nnz),max_rss_GiB=16,original_Ks=True,pressure_problem_unchanged=True,second_level_only_if_first_not_resolved=True))
        if estimate>16*2**30:raise MemoryError('R6 estimated memory exceeds cap')
        space,definition=interior_h(r.parent);embedding=embedding_audit(r.parent,space);patch=invariants(space)
        write(folder/'R6-definition.json',dict(**definition,parent_space=r.parent.signature,embedding=embedding,patch=patch))
        initial=np.vstack((f,np.zeros((space.ndof-r.parent.ndof,3))));rr,op=operators(r,space,max_seconds=350,progress=lambda *x:print('REFERENCE',*x,flush=True))
        m=SegmentedModel(rr,order=7,device='cuda:0',hold=.005);state,stats=static_solve(m,initial,max_seconds=180);response=m.evaluate(state.q)
        newnodes=space.nodes(rr.expand(state.q));newedges=space.edges;degree=space.p;A=space.A;params=space.params
        adjacent=compare_nodal((r.parent.edges,r.parent.p,n),(newedges,degree,newnodes),A,params)
        reaction=metric(stats['reaction_N'],old_result['solve']['reaction_N'],1e-4,.05)
        np.savez_compressed(folder/'R6.npz',M=rr.original_mass,K=rr.original_stiffness,q=state.q,full=rr.expand(state.q),nodes=newnodes)
        del m;gc.collect()
        high=SegmentedModel(rr,order=9,device='cuda:0',hold=.005);v=high.evaluate(state.q)
        material={k:metric(response[k],v[k],at,.02) for k,at in [('material_U',1e-10),('material_force',1e-8)]}
        if not all(x['passed'] for x in material.values()):raise ValueError('new reference q7/q9 inadequate')
        del high,r,rr,space,raw,initial,n,f;gc.collect()
        selected=read(run/'selected-space.json')['package'];formal,_=load_selected(selected);source=Path(selected['path']).parent/'F45.npz'
        with np.load(source) as z:
            full=z['full'].copy() if 'full' in z.files else formal.expand(z['q'])
        formal_nodes=formal.parent.nodes(full);current=compare_nodal((formal.parent.edges,formal.parent.p,formal_nodes),(newedges,degree,newnodes),A,params)
        previous=old_result['adjacent'];contraction={region:{k:adjacent[region][k]['absolute']/max(previous[region][k]['absolute'],1e-14) for k in ('PK1','fiber_PK1')} for region in adjacent}
        resolved=all(x['absolute']<=x['budget']*.25 for region in adjacent.values() for x in region.values()) and reaction['absolute']<=reaction['budget']*.25
        needs_reallocation=resolved and any(not x['passed'] for region in current.values() for x in region.values())
        result=dict(status='passed_scoped' if resolved else 'limited',new_reference='R6',old_reference='R5',adjacent=adjacent,contraction=contraction,formal_vs_R6=current,reaction=reaction,solve=stats,material=material,operators=op,embedding=embedding,patch=patch,seconds=time.perf_counter()-tick,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,new_equilibrium_solves=1,formal_source=dict(path=str(source),sha256=sha(source)),spatial_continuum_certificate=False,pressure_dynamic_reference=False)
        write(folder/'solid-reference-comparison.json',result);write(folder/'R6-package.json',dict(data_sha256=sha(folder/'R6.npz'),definition_sha256=sha(folder/'R6-definition.json'),scope='F45 .005m static, nested R5->R6, original Ks and full cross terms',new_formal_space=False))
        write(folder/'allocation-entry-decision.json',dict(status='retain_current_144' if not needs_reallocation else 'future_candidate_needed',reference_resolved=resolved,new_training_solves=0,new_spaces=0,reason='reference adjacent change and formal regional error reported; no dynamic-solid certificate',formal_regions=current,second_level='not_triggered' if resolved else 'requires bounded follow-up decision'))
        update(run,f'S3：新增真实h参考R6完成，R5/R6参考分辨结果{result["status"]}；新平衡1次，原Ks与充分q7/q9通过。正式144函数保持；结果仅针对F45/0.005m静态。')
        print('SOLID_REFERENCE',result['status'],resolved,current,flush=True)
    except (ValueError,MemoryError,TimeoutError) as e:
        write(folder/'failure.json',dict(status='limited',error=repr(e),traceback=traceback.format_exc(),seconds=time.perf_counter()-tick));raise
    finally:stop.set()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
