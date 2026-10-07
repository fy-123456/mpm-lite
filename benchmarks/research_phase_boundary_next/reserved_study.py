"""One reserved material direction after training; no reselection or promotion."""
from pathlib import Path
import argparse,gc,time
import numpy as np
from .provenance import *
from .spaces import load_selected
from .spatial_study import NAME,material,materials,nodal
from .solver import bounded_static
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_local_span_next.spatial_study import reference4
from benchmarks.research_reference_next.field_audit import compare_nodal
from benchmarks.research_sequential_next.run import probe_frame
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_reference_next.reference import interior_h
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_spatial_phase_next.space import combine
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel

def study(run):
    import warp as wp
    run=Path(run);verify(run)
    if read(run/'S2/research-space-decision.json')['status']!='pending_reserved':raise ValueError('training gate required')
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');angle=read(run/'S2/design-protocol.json')['reserved_angle']
    register(run,'S2/reserved-protocol.json',dict(angle=angle,training_sha256=sha(run/'S2/training-comparison.json'),max_solves=4,no_retraining=True,formal_space_unchanged=True))
    r,m,s,_=reopen(REFERENCE/'Q1/R3');del m,s;gc.collect();r4,f4,_,_=reference4(r)
    space,_=interior_h(r4.parent);p=PHASE_REFERENCE/'S2/R5';pkg=read(p/'space-package.json')
    if sha(p/'data.npz')!=pkg['data_sha256'] or space.signature!=pkg['space_sha256']:raise ValueError('R5 reference identity changed')
    with np.load(p/'data.npz') as z:r5=Condensation(space,z['M'],z['K']);f5=z['full'].copy()
    current,_=load_selected(read(run/'selected-space.json')['package']);cp=Path(read(run/'selected-space.json')['package']['path']).parent
    with np.load(cp/'F45.npz') as z:old_initial=z['full'].copy()
    folder=run/'S2/candidates'/NAME;pkg=read(folder/'space-package.json')
    if sha(folder/'space.npz')!=pkg['data_sha256']:raise ValueError('candidate data changed')
    with np.load(folder/'space.npz') as z:candidate,_=combine(r,z['C'],NAME)
    if candidate.signature!=pkg['reduction_sha256']:raise ValueError('candidate reconstruction changed')
    with np.load(folder/'F45.npz') as z:new_initial=z['full'].copy()
    reductions={'R4':r4,'R5':r5,'baseline':current,'candidate':candidate};initials={'R4':f4,'R5':f5,'baseline':old_initial,'candidate':new_initial};values={};results={}
    for label in ('R4','R5','baseline','candidate'):
        rr=material(reductions[label],angle);model=SegmentedModel(rr,order=7,device='cuda:0',hold=.005);target=run/'S2/reserved'/label;target.mkdir(parents=True)
        state,stats=bounded_static(model,initials[label],target/'search.json',600);checks=materials(rr,state.q);full=rr.expand(state.q);frame=probe_frame(model,state,[33,7,7]);nodes=rr.parent.nodes(full)
        np.savez_compressed(target/'state.npz',full=full,q=state.q,nodes=nodes,**{f'e{i}':e for i,e in enumerate(rr.parent.edges)},**frame)
        results[label]=dict(status='passed_scoped',p=rr.parent.p,space=rr.signature,solve=stats,material=checks);write(target/'result.json',results[label]);values[label]=(rr.parent.edges,rr.parent.p,nodes,frame)
        print('RESERVED',angle,label,stats['nonlinear_residual_N'],flush=True);del model;gc.collect()
    params=material(current,angle).parent.params
    adjacent=compare_nodal(values['R4'][:3],values['R5'][:3],params.A0,params);old=compare_nodal(values['baseline'][:3],values['R5'][:3],params.A0,params);new=compare_nodal(values['candidate'][:3],values['R5'][:3],params.A0,params);reasons=[]
    for reg in new:
        for key,v in new[reg].items():
            ref=adjacent[reg][key]
            if ref['absolute']>.25*ref['budget']:reasons.append('reference_limited '+reg+'/'+key)
            if not v['passed'] or v['absolute']-old[reg][key]['absolute']>max(2*ref['absolute'],.1*v['budget']):reasons.append('space_limited '+reg+'/'+key)
    reaction=metric(results['candidate']['solve']['reaction_N'],results['R5']['solve']['reaction_N'],1e-4,.05);a,b=values['candidate'][3],values['R5'][3]
    disp={k:metric(a['x']-a['X'],b['x']-b['X'],5e-5,.05,w) for k,w in regions(a['X']).items()}
    if not reaction['passed'] or not all(v['passed'] for v in disp.values()):reasons.append('reaction/displacement')
    write(run/'S2/reserved-direction-check.json',dict(status='passed_scoped' if not reasons else 'limited',angle=angle,accessed=True,adjacent=adjacent,old=old,new=new,reaction=reaction,displacement=disp,reasons=reasons,independent_of_training=True))
    write(run/'S2/research-space-decision.json',dict(status='qualified_static_research' if not reasons else 'limited',candidate=NAME,training_passed=True,reserved_passed=not reasons,formal_space_changed=False,spatial_accuracy=False,reasons=reasons,dynamics=False,coupling=False,q5=False))
    print('RESERVED_DECISION',not reasons,new['interior'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
