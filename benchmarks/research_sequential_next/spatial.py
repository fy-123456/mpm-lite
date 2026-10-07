"""N04 bounded, nested same-model local h references; no continuum claim."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import resource
import numpy as np
import scipy.linalg as la
from .provenance import read,write,serial_lock,source_files,utc
from .run import load_model,probe_frame,log
from .compare import metric,regions
from engine.aniso_phase1.research_sequential_next.reference_space import local_h_family,extend,extended_operators
from engine.aniso_phase1.research_sequential_next.model import PracticalModel
from engine.aniso_phase1.research_d.common_state import CommonState


def static_solve(model,full_initial,*,hold=.005,max_seconds=600):
    started=time.perf_counter();r=model.reduction;w=r.project(full_initial)
    w[model.fixed]=model.boundary.lift(.5)[model.fixed]
    factor=la.lu_factor(r.K[np.ix_(model.ids,model.ids)])
    records=[]
    for iteration in range(10):
        out=model.evaluate(w);force=out['force'][model.free].ravel();norm=float(la.norm(force))
        records.append(dict(iteration=iteration,true_residual_N=norm,energy_J=out['U'],min_detF=out['min_detF']))
        if norm<=1e-7:break
        if iteration==9:raise ValueError('bounded nonlinear static solve did not converge')
        update=la.lu_solve(factor,-force).reshape(-1,3)
        for ls in range(12):
            trial=w.copy();trial[model.free]+=2.**(-ls)*update
            new=model.evaluate(trial)
            if new['min_detF']>.1 and la.norm(new['force'][model.free])<norm:
                w=trial;break
        else:raise ValueError('local reference static line search failed')
        if time.perf_counter()-started>max_seconds:raise TimeoutError('bounded nonlinear reference solve time exhausted')
    state=CommonState(w,np.zeros_like(w),time=.5,child_states={'identity':model.identity})
    model.validate(state,material=True)
    reaction=float(np.sum(out['force']*model.boundary.unit))
    return state,dict(accepted=True,iterations=records,reaction_N=reaction,energy_J=out['U'],
        nonlinear_residual_N=norm,seconds=time.perf_counter()-started)


def compare_fields(model_a,state_a,model_b,state_b):
    a=probe_frame(model_a,state_a,[33,7,7]);b=probe_frame(model_b,state_b,[33,7,7])
    if not np.array_equal(a['X'],b['X']):raise ValueError('reference probes differ')
    weights=regions(a['X']);direction=np.array(model_b.parent.params.fiber_direction,copy=True)
    result={}
    for name,w in weights.items():
        result[name]=dict(displacement=metric(a['x']-a['X'],b['x']-b['X'],5e-5,.05,w),
            PK1=metric(a['PK1'],b['PK1'],.02,.05,w),
            fiber_PK1=metric(np.einsum('i,...ij,j->...',direction,a['PK1'],direction),
                np.einsum('i,...ij,j->...',direction,b['PK1'],direction),.02,.05,w))
    return result,a,b


def reference(run):
    run=Path(run);cfg=read(run/'cases/gpu-q7-dt0025/execution-protocol.json')
    base,_=load_model(run,cfg);r=base.reduction;s=r.parent
    # Freeze the work definition before producing reference values or candidates.
    protocol=dict(utc=utc(),source_sha256=source_files(),physics='current nonlinear Hencky+fiber and original carrier Ks',
        parent_space_sha256=s.signature,levels=[dict(level=1,extra_scalar=12),dict(level=2,extra_scalar=36)],
        refinement='nested compact x bubbles on 2 then 4 subintervals, six transverse modes; fixed conforming Q4 ambient',
        continuum_certified=False,developer_case=dict(fiber_degrees=45,displacement_m=.005),
        heldout=dict(displacement_m=.0035,usage='only after candidate choice; same material'),
        budgets=dict(max_seconds_per_level=600,max_rss_GiB=16,reference_engineering_rtol=.05),
        old_A_reference='quadratic small-strain model without the same carrier Ks: different_model, diagnostic only')
    p=run/'N04/protocol.json'
    if p.exists():raise ValueError('reference run already registered; inspect its result before rerunning')
    write(p,protocol)
    model=PracticalModel(r,order=7,device=cfg['device'],hold=.005)
    state,stats=static_solve(model,s.expand(s.q0));references=[(model,state,stats)]
    output=run/'reference';output.mkdir(exist_ok=True)
    np.savez_compressed(output/'original144-static.npz',q=state.q,full=r.expand(state.q))
    write(output/'original144-static.json',stats)
    for level in (1,2):
        begun=time.perf_counter();basis,labels=local_h_family(s,level)
        estimate=dict(nodal_shape=list(s.shape),basis_bytes=basis.nbytes,extra_scalar=basis.shape[1],
            predicted_dense_operator_bytes=8*((s.ndof+basis.shape[1])**2)*(1+9),
            original_nodes=int(np.prod(s.shape)),fixed_ambient=True)
        write(output/f'level{level}-resource.json',estimate)
        space=extend(s,basis,label=f'local-h-level{level}')
        reduction,operators=extended_operators(r,space,device=cfg['device'],progress=log)
        current=PracticalModel(reduction,order=7,device=cfg['device'],hold=.005)
        full=np.vstack((r.expand(state.q),np.zeros((basis.shape[1],3))))
        solved,solve=static_solve(current,full,max_seconds=max(1,600-(time.perf_counter()-begun)))
        fields,_,f=compare_fields(model,state,current,solved)
        # New basis belongs to the same ambient Q4; independently check q7/q8 force at the solved state.
        check=PracticalModel(reduction,order=8,device=cfg['device'],hold=.005)
        a=current.evaluate(solved.q);b=check.evaluate(solved.q)
        material=dict(energy=metric(a['material_U'],b['material_U'],1e-10,.02),
            force=metric(a['material_force'],b['material_force'],1e-8,.02))
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        if peak>16:raise MemoryError('reference RSS budget exceeded')
        np.savez_compressed(output/f'level{level}.npz',basis=basis,q=solved.q,full=reduction.expand(solved.q),M=reduction.original_mass,K=reduction.original_stiffness,
            X=f['X'],x=f['x'],PK1=f['PK1'])
        result=dict(level=level,operators=operators,solve=solve,labels=labels,original_vs_reference=fields,
            material_7_8=material,seconds=time.perf_counter()-begun,peak_rss_GiB=peak,spatial_certified=False)
        write(output/f'level{level}.json',result);references.append((current,solved,solve))
        log('reference level',level,'seconds',result['seconds'],'residual',solve['nonlinear_residual_N'])
    comparison,_,_=compare_fields(*references[1][:2],*references[2][:2])
    reaction=metric(references[1][2]['reaction_N'],references[2][2]['reaction_N'],1e-4,.05)
    result=dict(utc=utc(),status='reference_limited',nested_levels=comparison,reaction=reaction,
        original_static=stats,all_regions_engineering_passed=all(m['passed'] for v in comparison.values() for m in v.values()) and reaction['passed'],
        model_consistent=True,original_Ks_preserved=True,continuum_certified=False,
        limitation='Two local h-enrichment levels test only the finite ambient Q4 space; neither an independent continuum solution nor a uniform error bound.',
        next='N05 may use these regional diagnostics for two bounded budget-144 candidates; retain reference qualification limits.')
    write(run/'N04/result.json',result)
    return result


def candidates(run):
    from engine.aniso_phase1.research_sequential_next.reference_space import select_locals
    from engine.aniso_phase1.research_sequential.condensation import Condensation
    run=Path(run);cfg=read(run/'cases/gpu-q7-dt0025/execution-protocol.json')
    base,_=load_model(run,cfg);r=base.reduction;s=r.parent
    data=np.load(run/'reference/level2.npz',allow_pickle=False)
    space=extend(s,data['basis'],label='local-h-level2')
    rr=Condensation(space,data['M'],data['K']);ref=PracticalModel(rr,order=7,device=cfg['device'],hold=.005)
    refstate=CommonState(data['q'],np.zeros_like(data['q']),time=.5,child_states={'identity':ref.identity})
    coeff=data['full'][s.n:];K=data['K'].reshape(space.ndof,3,space.ndof,3)
    scores=np.array([float(coeff[j]@K[s.n+j,:,s.n+j,:]@coeff[j]) for j in range(len(coeff))])
    old_count=s.ndof-s.n;old_rank=np.argsort(scores[:old_count]);new_rank=old_count+np.argsort(scores[old_count:])[::-1]
    registered=[]
    for swap in (6,12):
        keep=sorted(set(range(old_count))-set(old_rank[:swap].tolist()))
        selected=np.array(keep+sorted(new_rank[:swap].tolist()),dtype=int)
        registered.append(dict(name=f'budget144-local-h-swap{swap}',swapped=swap,indices=selected.tolist()))
    protocol=dict(utc=utc(),source_sha256=source_files(),budget=144,candidates=registered,
        score='diagonal elastic energy of level2 development coefficients; heuristic, cross energies not assigned to individual functions',
        development='F45 nonlinear static extension 0.005 m',heldout='F45 static extension 0.0035 m, accessed only after decision',
        promotion='global and interior PK1/fiber error reduction vs level2 must exceed level1/level2 uncertainty; no other region regresses by more than 0.02 Pa; mass rank and boundary must pass',
        reference_uncertainty=read(run/'N04/result.json')['nested_levels'])
    if (run/'N05/protocol.json').exists():raise ValueError('candidate study already registered')
    write(run/'N05/protocol.json',protocol)
    with np.load(run/'reference/original144-static.npz') as z:oldq=z['q'];oldfull=z['full']
    oldmodel=PracticalModel(r,order=7,device=cfg['device'],hold=.005)
    oldstate=CommonState(oldq,np.zeros_like(oldq),time=.5,child_states={'identity':oldmodel.identity})
    old_errors,_,_=compare_fields(oldmodel,oldstate,ref,refstate)
    result=[]
    for item in registered:
        begun=time.perf_counter();selected=np.array(item['indices']);candidate,ids=select_locals(space,selected,label=item['name'])
        ids3=(3*ids[:,None]+np.arange(3)).ravel()
        cr=Condensation(candidate,data['M'][np.ix_(ids,ids)],data['K'][np.ix_(ids3,ids3)])
        cm=PracticalModel(cr,order=7,device=cfg['device'],hold=.005)
        initial=data['full'][ids]
        state,solve=static_solve(cm,initial)
        errors,_,field=compare_fields(cm,state,ref,refstate)
        reasons=[]
        for region in ('global_domain','interior'):
            for key in ('PK1','fiber_PK1'):
                gain=old_errors[region][key]['absolute']-errors[region][key]['absolute']
                uncertainty=protocol['reference_uncertainty'][region][key]['absolute']
                if gain<=uncertainty:reasons.append(f'{region}/{key}: improvement not larger than reference uncertainty')
        for region in errors:
            for key in ('PK1','fiber_PK1'):
                if errors[region][key]['absolute']>old_errors[region][key]['absolute']+.02:
                    reasons.append(f'{region}/{key}: regional regression')
        # Zero local trace in both complete rigid volumes is exact for this family.
        from engine.aniso_phase1.tensor_reference import coordinates
        xx=coordinates(candidate.edges,candidate.p)[0]
        rigid=(xx<=.25+1e-12)|(xx>=.75-1e-12)
        nodefield=candidate.nodes(cr.expand(state.q)).reshape(*candidate.shape,3)
        target=np.zeros_like(nodefield[rigid]);target[xx[rigid]>=.75,:, :,0]=.005
        boundary_error=float(np.max(abs(nodefield[rigid]-target)))
        if boundary_error>1e-8:raise ValueError('candidate full rigid-volume trace failed')
        folder=run/'candidates'/item['name'];folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'space.npz',selected=selected,q=state.q,full=cr.expand(state.q),
            M=cr.original_mass,K=cr.original_stiffness,X=field['X'],x=field['x'],PK1=field['PK1'])
        record=dict(name=item['name'],solve=solve,rank=cr.audit,errors=errors,seconds=time.perf_counter()-begun,
            local_budget=144,boundary_max_abs_m=boundary_error,eligible=not reasons,promotion_reasons=reasons,
            reference_limited=True,space_sha256=candidate.signature)
        write(folder/'result.json',record);result.append(record)
        log(item['name'],'eligible',record['eligible'],'global PK1',errors['global_domain']['PK1']['relative'])
    eligible=[x for x in result if x['eligible']]
    selected=min(eligible,key=lambda x:x['errors']['global_domain']['PK1']['absolute'])['name'] if eligible else 'original144'
    decision=dict(utc=utc(),status='reference_limited' if not eligible else 'candidate_pending_dynamic_checks',
        selected=selected,original_errors=old_errors,candidates=result,spatial_certified=False,
        decision_frozen_before_heldout=True)
    write(run/'N05/decision.json',decision)
    # An unused load magnitude gives a bounded independent check of the selected
    # static space. It does not certify arbitrary material or geometry changes.
    if selected=='original144':chosen_r=r
    else:
        item=next(x for x in registered if x['name']==selected);cs,ids=select_locals(space,item['indices'],label=selected)
        ids3=(3*ids[:,None]+np.arange(3)).ravel();chosen_r=Condensation(cs,data['M'][np.ix_(ids,ids)],data['K'][np.ix_(ids3,ids3)])
    hold=.0035;cm=PracticalModel(chosen_r,order=7,device=cfg['device'],hold=hold)
    initial=np.zeros((chosen_r.parent.ndof,3));initial[:s.n]=oldfull[:s.n]*(hold/.005)
    state,stats=static_solve(cm,initial,hold=hold)
    reference_hold=PracticalModel(rr,order=7,device=cfg['device'],hold=hold)
    ref_initial=data['full']*(hold/.005);ref_state,ref_stats=static_solve(reference_hold,ref_initial,hold=hold)
    errors,_,_=compare_fields(cm,state,reference_hold,ref_state)
    write(run/'N05/heldout.json',dict(hold_m=hold,selected=selected,solve=stats,reference_solve=ref_stats,
        errors=errors,decision_preceded_evaluation=True,spatial_certified=False))
    write(run/'N05/result.json',dict(decision,heldout='N05/heldout.json',
        remaining='Any selected new space requires N06 mapping/material/time-window checks before use'))
    return decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['reference','candidates']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        (reference if a.phase=='reference' else candidates)(a.run)
