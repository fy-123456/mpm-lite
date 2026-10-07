"""Actual AVF parent with an independently posed E child; no physical coupling claim."""
import numpy as np
import warp as wp
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.research_e.flow import Grid,Darcy
from engine.aniso_phase1.research_e.poro import Skeleton,Biot
from engine.aniso_phase1.research_e.stage2.transaction import ChildAdapter
from engine.aniso_phase1.research_sequential.solver import GeneralBiot
from engine.aniso_phase1.research_sequential.condensation import CondensedModel
from engine.aniso_phase1.research_c.stage2.dynamics import StepRejected


def audit_transaction(out,metadata,v):
    from .run import PARENT,open_reduction,stepper,save_state,load_state,relative
    cfg=metadata['config'];mat=cfg['material']
    grid=Grid(tuple(cfg['shape']),cfg['lengths'])
    solid=Skeleton(grid,AnisotropicMaterialParams(mat['mu'],mat['lam'],mat['k_f'],mat['direction']),cfg['alpha'])
    if not np.array_equal(solid.fixed,cfg['fixed_u']):raise ValueError('E clamp mismatch')
    flow=Darcy(grid,np.array(cfg['K']),viscosity=cfg['viscosity'])
    solver=GeneralBiot(solid,flow,cfg['storage']);child=ChildAdapter(solver,PARENT)
    boundary={(a,b):(kind,value) for a,b,kind,value in metadata['boundary']}
    # Compare an actual new-backend step to the original E physics implementation.
    old=solver.initial();result=solver.step(old,.0005,v['load'],boundary)
    reference=Biot(solid,flow,cfg['storage']).step(old,.0005,v['load'],boundary)
    error=max(relative(result.state.u,reference.state.u),relative(result.state.p,reference.state.p))
    wp.config.kernel_cache_dir=str(out.resolve()/'warp-cache')
    model=CondensedModel(open_reduction(out),order=7,device='cuda:0')
    initial=model.rest();initial.child_states['E']=child.initial()
    initial.child_states['B']=dict(material=model.rule.signature,mass='original-q5',fixed_during_trial=True)
    parent=stepper(model,initial);before=parent.state.digest()
    def prepare(candidate,values):
        values['E']=child.prepare(values['E'],dt=.0005,load=v['load'],boundary=boundary)
        return values
    def fail(where,candidate):
        if where=='after_prepare':raise ValueError('intentional post-child failure')
    rejected=False
    try:parent.step(.0005,prepare_children=prepare,validate_trial=child.validate_parent,inject=fail)
    except StepRejected:rejected=True
    rollback=parent.state.digest()==before
    rows=[]
    for _ in range(2):
        rows.append(parent.step(.0005,prepare_children=prepare,validate_trial=child.validate_parent))
    state=parent.state;child.validate_parent(state)
    save_state(out/'joint-checkpoint.json',model,state)
    restored=load_state(out/'joint-checkpoint.json',model);child.validate_parent(restored)
    passed=bool(rejected and rollback and state.step==state.child_states['E']['step']==2
        and restored.digest()==state.digest() and error<1e-8)
    return dict(passed=passed,independent_2D_E=True,physical_3D_coupling=False,
        actual_AVF_parent=True,post_child_failure_rejected=rejected,whole_state_rollback=rollback,
        accepted_joint_steps=2,end_s=state.time,checkpoint_restore=True,
        new_backend_physics_difference=error,E_ledgers=state.child_states['E']['ledger'],C_ledgers=rows)
