"""Joint-block ranking and conditional nonlinear candidate validation."""
from pathlib import Path
import argparse,time,gc
import numpy as np
import scipy.linalg as la
from .provenance import PARENT,read,write,sha,register,serial_lock
from .reference_study import reopen
from .field_audit import compare_nodal
from benchmarks.research_sequential_next.model_package import load_reduction
from benchmarks.research_sequential_next.spatial import static_solve
from engine.aniso_phase1.research_sequential_next.reference_space import select_locals
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel


def study(run):
    run=Path(run);decision=read(run/'Q1/reference-decision.json')
    register(run,'Q2/protocol.json',dict(block_size=6,candidate_swaps=[6,12],budget=144,heldout_m=.00375,
       promotion=dict(relative_gain=.2,gain_over_uncertainty=2.,regional_atol_Pa=.02),
       reference_qualified=decision['reliable_for_candidates'],score='joint Schur removal with all cross terms'))
    r,reference,state,_=reopen(run/'Q1/R3');s=r.parent;full=r.expand(state.q)
    free=s.free_scalar_ids;ids=(3*free[:,None]+np.arange(3)).ravel();K=r.original_stiffness[np.ix_(ids,ids)];K=.5*(K+K.T)
    inverse=la.cho_solve(la.cho_factor(K),np.eye(len(ids)));mapping={int(x):i for i,x in enumerate(free)}
    def score(block):
        pos=np.array([mapping[s.n+int(x)] for x in block]);dofs=(3*pos[:,None]+np.arange(3)).ravel();a=full[s.n+np.array(block)].ravel()
        dual=la.solve(inverse[np.ix_(dofs,dofs)],a,assume_a='pos');delta=-inverse[:,dofs]@dual
        value=float(.5*a@dual);direct=float(.5*delta@K@delta)
        if abs(value-direct)>1e-12+2e-5*abs(value):raise ValueError('joint Schur identity failed')
        return value
    blocks=[dict(indices=list(range(i,i+6)),conditional_energy_J=score(list(range(i,i+6)))) for i in range(0,s.ndof-s.n,6)]
    old=sorted([x for x in blocks if max(x['indices'])<144],key=lambda x:x['conditional_energy_J']);extra=sorted([x for x in blocks if min(x['indices'])>=144],key=lambda x:x['conditional_energy_J'],reverse=True)
    write(run/'Q2/block-scores.json',dict(records=blocks,rest_K_SPD=True,nonlinear_SPD_not_assumed=True))
    proposals=[]
    for count in (1,2):
        remove=[j for x in old[:count] for j in x['indices']];add=[j for x in extra[:count] for j in x['indices']]
        indices=sorted(set(range(144))-set(remove))+sorted(add);name=f'joint-swap{6*count}'
        cs,allids=select_locals(s,indices,label=name);ix=(3*allids[:,None]+np.arange(3)).ravel();cr=Condensation(cs,r.original_mass[np.ix_(allids,allids)],r.original_stiffness[np.ix_(ix,ix)])
        folder=run/'Q2/candidates'/name;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'space.npz',indices=indices,M=cr.original_mass,K=cr.original_stiffness,initial=full[allids])
        record=dict(name=name,indices=indices,removed=remove,added=add,joint_removal_energy_J=score(remove),rank=cr.audit,
            local_budget=144,space_sha256=cs.signature,reduction_sha256=cr.signature,data_sha256=sha(folder/'space.npz'),
            reference_sha256=sha(run/'Q1/R3/space-package.json'),promoted=False)
        write(folder/'space-package.json',record);proposals.append(record)
    if not decision['reliable_for_candidates']:
        write(run/'Q2/space-decision.json',dict(status='not_promoted_reference_limited',selected='original144',candidate_designs=proposals,
            nonlinear_screen='not run: registered reference gate failed',dynamic_qualification='conditional_not_required',spatial_certified=False))
        write(run/'selected-space.json',dict(selected='original144',parent=str(PARENT/'model/model-package.json')))
        print('SPACE original144; reference gate blocks promotion',flush=True);return
    refnodal=(s.edges,s.p,s.nodes(full));original=load_reduction(PARENT)
    with np.load(PARENT/'reference/original144-static.npz',allow_pickle=False) as z:oldfull=z['full'].copy()
    olderrors=compare_nodal((original.parent.edges,original.parent.p,original.parent.nodes(oldfull)),refnodal,s.A,s.params)
    records=[]
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    for item in proposals:
        start=time.perf_counter();folder=run/'Q2/candidates'/item['name'];cs,allids=select_locals(s,item['indices'],label=item['name']);ix=(3*allids[:,None]+np.arange(3)).ravel()
        cr=Condensation(cs,r.original_mass[np.ix_(allids,allids)],r.original_stiffness[np.ix_(ix,ix)]);m=SegmentedModel(cr,order=7,device='cuda:0',hold=.005)
        solved,stats=static_solve(m,full[allids]);errors=compare_nodal((cs.edges,cs.p,cs.nodes(cr.expand(solved.q))),refnodal,s.A,s.params)
        reasons=[]
        for region in ('global_domain','interior'):
            for key in ('PK1','fiber_PK1'):
                baseline=olderrors[region][key]['absolute'];gain=baseline-errors[region][key]['absolute'];unc=decision['regional'][region][key]['empirical_uncertainty']
                if gain<.2*baseline or gain<=2*unc:reasons.append(f'{region}/{key}: gain below registered uncertainty or practical improvement')
        for region in errors:
            for key in ('PK1','fiber_PK1'):
                if errors[region][key]['absolute']>olderrors[region][key]['absolute']+.02:reasons.append(f'{region}/{key}: regional regression')
        rec=dict(name=item['name'],solve=stats,errors=errors,eligible=not reasons,reasons=reasons,seconds=time.perf_counter()-start)
        np.savez_compressed(folder/'static.npz',q=solved.q,full=cr.expand(solved.q));write(folder/'result.json',rec);records.append(rec)
        print('CANDIDATE',item['name'],'eligible',not reasons,'interior',errors['interior'],flush=True)
        del m,cr;gc.collect()
    eligible=[x for x in records if x['eligible']]
    write(run/'Q2/space-decision.json',dict(status='candidate_pending_heldout_and_dynamics' if eligible else 'not_promoted_no_resolved_gain',
          selected=min(eligible,key=lambda x:x['errors']['interior']['PK1']['absolute'])['name'] if eligible else 'original144',
          original_errors=olderrors,candidates=records,spatial_certified=False,heldout_accessed=False))
    if not eligible:write(run/'selected-space.json',dict(selected='original144',parent=str(PARENT/'model/model-package.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; fork before running studies')
        study(a.run)
