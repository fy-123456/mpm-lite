"""Interaction-aware block ranking; unresolved reference prevents promotion."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import read,write,sha,register,serial_lock,utc
from .reference_study import reopen
from engine.aniso_phase1.research_sequential_next.reference_space import select_locals
from engine.aniso_phase1.research_sequential.condensation import Condensation


def study(run):
    run=Path(run);decision=read(run/'S2/reference-decision.json')
    register(run,'S3/protocol.json',dict(budget=144,source_reference='S2/h2/space-package.json',
        score='conditional block removal energy with all free-coordinate cross terms in unchanged rest K',
        block_size_scalar=6,max_candidates=2,required_gain_over_empirical_uncertainty=2.,
        promotion_requires_credible_global_reference=True,
        heldout='not evaluated or claimed unseen while reference qualification is limited'))
    r,model,state,_=reopen(run/'S2/h2');s=r.parent;full=r.expand(state.q)
    free=s.free_scalar_ids;ids=(3*free[:,None]+np.arange(3)).ravel()
    K=r.original_stiffness[np.ix_(ids,ids)];K=.5*(K+K.T)
    factor=la.cho_factor(K);inverse=la.cho_solve(factor,np.eye(len(ids)))
    mapping={int(x):i for i,x in enumerate(free)}
    scores=[];audits=[]
    for start in range(0,s.ndof-s.n,6):
        block=np.arange(start,min(start+6,s.ndof-s.n))
        if len(block)!=6:continue
        positions=np.array([mapping[s.n+int(i)] for i in block]);dofs=(3*positions[:,None]+np.arange(3)).ravel()
        sub=inverse[np.ix_(dofs,dofs)];sub=.5*(sub+sub.T)
        values=full[s.n+block].ravel();dual=la.solve(sub,values,assume_a='pos')
        cost=float(.5*values@dual)
        delta=-inverse[:,dofs]@dual
        direct=float(.5*delta@K@delta)
        error=abs(direct-cost)/max(abs(cost),1e-14)
        if error>2e-5:raise ValueError('block Schur energy identity failed')
        scores.append(dict(indices=block.tolist(),conditional_energy_J=cost,
                           diagonal_energy_J=float(.5*values@K[np.ix_(dofs,dofs)]@values),
                           linear_relaxation_identity_relative=error))
    old=sorted([x for x in scores if max(x['indices'])<144],key=lambda x:x['conditional_energy_J'])
    extra=sorted([x for x in scores if min(x['indices'])>=144],key=lambda x:x['conditional_energy_J'],reverse=True)
    proposals=[]
    for count in (1,2):
        remove=[j for x in old[:count] for j in x['indices']];add=[j for x in extra[:count] for j in x['indices']]
        selected=sorted(set(range(144))-set(remove))+sorted(add)
        name=f'interaction-swap{6*count}'
        candidate,allids=select_locals(s,selected,label=name);ids3=(3*allids[:,None]+np.arange(3)).ravel()
        cr=Condensation(candidate,r.original_mass[np.ix_(allids,allids)],r.original_stiffness[np.ix_(ids3,ids3)])
        record=dict(name=name,indices=selected,removed=remove,added=add,local_function_budget=144,
                    parent_reference_sha256=sha(run/'S2/h2/space-package.json'),space_sha256=candidate.signature,
                    independent_rank=cr.audit,predicted_removed_conditional_energy_J=sum(x['conditional_energy_J'] for x in old[:count]),
                    nonlinear_gain_not_measured=True,promoted=False)
        write(run/'S3/candidate-designs'/f'{name}.json',record);proposals.append(record)
    write(run/'S3/block-scores.json',dict(scores=scores,mathematics='S_b=(K_ff^-1)_bb^-1; eta=.5 a_b^T S_b a_b',
          positive_rest_K=True,nonlinear_tangent_SPD_not_assumed=True,full_cross_terms=True))
    write(run/'S3/space-decision.json',dict(status='not_promoted_reference_limited',selected='original144',
        reason='Global reference uncertainty remains unresolved; ranking is a design diagnostic, not evidence of physical accuracy',
        reference_decision_sha256=sha(run/'S2/reference-decision.json'),candidate_designs=proposals,
        dynamic_requalification='not required: formal space unchanged',heldout='not run; no unseen-generalization claim',
        spatial_certified=False,old_candidate_claims_not_promoted=True))
    print('SPACE original144 retained; 2 interaction-aware designs, rank valid',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
