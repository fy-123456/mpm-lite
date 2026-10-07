"""Actual ambient refinement with loadable packages and bounded reliability gates."""
from pathlib import Path
import argparse
import gc
import time
import traceback
import numpy as np
import scipy.linalg as la
from .provenance import PARENT,read,write,sha,register,verify,serial_lock,utc
from benchmarks.research_sequential_next.model_package import load_reduction
from benchmarks.research_sequential_next.spatial import static_solve,compare_fields
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_sequential_next.reference_space import extend
from engine.aniso_phase1.research_sequential_next.model import PracticalModel
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_d.common_state import CommonState
from engine.aniso_phase1.research_post_release.ambient_reference import ambient_space,embedding_audit,operators


def base_reference():
    r=load_reduction(PARENT)
    with np.load(PARENT/'reference/level2.npz',allow_pickle=False) as z:
        s=extend(r.parent,z['basis'],label='local-h-level2');rr=Condensation(s,z['M'],z['K'])
        q=np.array(z['q']);full=np.array(z['full'])
    model=PracticalModel(rr,order=7,device='cpu',hold=.005)
    state=CommonState(q,np.zeros_like(q),time=.5,child_states={'identity':model.identity})
    return rr,model,state,full


def reopen(folder):
    folder=Path(folder);p=read(folder/'space-package.json')
    if sha(folder/'data.npz')!=p['data_sha256']:raise ValueError('reference data changed')
    r,_,_,_=base_reference();s,_=ambient_space(r.parent,p['level'],mode=p['mode'])
    with np.load(folder/'data.npz',allow_pickle=False) as z:
        rr=Condensation(s,z['M'],z['K']);q=np.array(z['q']);expected=np.array(z['PK1'])
    if rr.signature!=p['reduction_sha256']:raise ValueError('reference reconstruction changed')
    m=PracticalModel(rr,order=p['material_order'],device='cpu',hold=.005)
    state=CommonState(q,np.zeros_like(q),time=.5,child_states={'identity':m.identity})
    frame=probe_frame(m,state,[33,7,7]);error=float(np.max(abs(frame['PK1']-expected)))
    if error>1e-8:raise ValueError('reloaded reference fields changed')
    return rr,m,state,error


def reference(run,mode):
    run=Path(run);verify(run)
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    levels=[1,2] if mode=='h' else [1]
    register(run,f'S2/{mode}-protocol.json',dict(ambient_refinement=mode,levels=levels,
       old_reference='parent/reference/level2.npz; bounded enriched original Q4',
       physical_model='same nonlinear Hencky+fiber, original carrier Ks, same grips and static hold .005m',
       soft_seconds=600,hard_seconds=1200,max_rss_GiB=16,
       reliability='contraction by region plus engineering budgets; empirical uncertainty, no continuum certificate',
       p_fallback='one local degree-5 family, diagnostic only without further p levels'))
    begun=time.perf_counter();r,base,state,full=base_reference();previous_model,previous_state=base,state
    results=[];failure=None
    print('AMBIENT_BASE',r.parent.shape,[e.tolist() for e in r.parent.edges],flush=True)
    try:
        for level in levels:
            start=time.perf_counter();space,definition=ambient_space(r.parent,level,mode=mode)
            estimate=8*((space.ndof)**2)*10+int(np.prod(space.shape))*3*8*12+int(np.prod([len(e)-1 for e in space.edges]))*max(7,space.p+1)**3*600
            folder=run/f'S2/{mode}{level}'
            write(folder/'definition.json',dict(definition,predicted_core_bytes=estimate))
            if estimate>12*(1<<30):raise MemoryError('reference estimate exceeds reserved workspace')
            invariant=embedding_audit(r.parent,space)
            rr,ops=operators(r,space,max_seconds=max(1,600-(time.perf_counter()-begun)),progress=lambda *v:print(*v,flush=True))
            current=SegmentedModel(rr,order=max(7,space.p+1),device='cuda:0',hold=.005)
            initial=np.vstack((full,np.zeros((space.ndof-r.parent.ndof,3))))
            solved,stats=static_solve(current,initial,max_seconds=max(1,600-(time.perf_counter()-begun)))
            comparison,_,frame=compare_fields(previous_model,previous_state,current,solved)
            a=current.evaluate(solved.q)
            check=SegmentedModel(rr,order=max(8,space.p+2),device='cuda:0',hold=.005)
            b=check.evaluate(solved.q)
            rng=np.random.default_rng(20261001);direction=rng.normal(size=solved.q.shape);direction[rr.fixed]=0;direction/=la.norm(direction)
            da=current.evaluate(solved.q,direction)['tangent_action'];db=check.evaluate(solved.q,direction)['tangent_action']
            integral=dict(energy=metric(a['material_U'],b['material_U'],1e-10,.02),
                force=metric(a['material_force'],b['material_force'],1e-8,.02),tangent=metric(da,db,1e-8,.03))
            record=dict(status='passed_scoped',definition=definition,invariants=invariant,operators=ops,solve=stats,
                        previous_level_difference=comparison,material_adjacent=integral,seconds=time.perf_counter()-start,
                        original_material_and_Ks=True,continuum_certified=False)
            np.savez_compressed(folder/'data.npz',q=solved.q,M=rr.original_mass,K=rr.original_stiffness,
                                full=rr.expand(solved.q),X=frame['X'],x=frame['x'],PK1=frame['PK1'])
            write(folder/'space-package.json',dict(schema='post-release-reference-v1',mode=mode,level=level,
                parent_release_sha256=read(run/'parent-release-lock.json')['parent_release_sha256'],
                parent_reference_sha256=sha(PARENT/'reference/level2.npz'),data_sha256=sha(folder/'data.npz'),
                space_sha256=space.signature,reduction_sha256=rr.signature,material_order=max(7,space.p+1),mass_order=ops['mass_order'],
                original_stabilization='carrier Ks energy exactly preserved; enrichment enters material only',scope='static .005m same-model local ambient reference'))
            write(folder/'result.json',record);results.append(record)
            previous_model=PracticalModel(rr,order=7,device='cpu',hold=.005);previous_state=solved.clone()
            previous_state.child_states['identity']=previous_model.identity
            del current,check;gc.collect()
            print('REFERENCE',mode,level,'seconds',record['seconds'],'PK1 interior',comparison['interior']['PK1']['relative'],flush=True)
            if time.perf_counter()-begun>600:raise TimeoutError('reference soft budget; retain completed levels')
    except (ValueError,TimeoutError,MemoryError) as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc())
        write(run/f'S2/{mode}-failure.json',failure)
    reliable=False;contraction={}
    if len(results)==2:
        for region in results[-1]['previous_level_difference']:
            contraction[region]={}
            for key in ('PK1','fiber_PK1'):
                a=results[0]['previous_level_difference'][region][key];b=results[1]['previous_level_difference'][region][key]
                contraction[region][key]=dict(ratio=b['absolute']/max(a['absolute'],1e-14),
                    contracted=b['absolute']<=max(a['absolute'],1e-8),engineering_passed=b['passed'],last_difference=b)
        reliable=all(x['contracted'] and x['engineering_passed'] for region in contraction.values() for x in region.values())
        reliable=reliable and all(x['passed'] for rr in results for x in rr['material_adjacent'].values())
    outcome=dict(status='reference_reliable_scoped' if reliable else 'reference_limited',mode=mode,completed_levels=len(results),
       empirical_contraction=contraction,reliable_for_candidate_comparison=reliable,failure=failure,
       continuum_certified=False,scope='local ambient static comparison; unchanged material/Ks; no dynamic truth claim',seconds=time.perf_counter()-begun)
    write(run/f'S2/{mode}-result.json',outcome)
    if mode=='h':write(run/'S2/reference-decision.json',outcome)
    print('REFERENCE_DECISION',outcome,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['h','p','reload']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.mode=='reload':
            reports=[]
            for folder in sorted((a.run/'S2').glob('[hp][12]')):
                if (folder/'space-package.json').exists():
                    r,m,s,error=reopen(folder);reports.append(dict(package=str(folder),PK1_max_difference=error,reduction_sha256=r.signature))
            write(a.run/'S2/reload-check.json',dict(status='passed_scoped',actual_new_process=True,references=reports))
        else:reference(a.run,a.mode)
