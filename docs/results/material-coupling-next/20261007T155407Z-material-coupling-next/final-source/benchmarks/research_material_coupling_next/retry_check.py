import argparse,json
from pathlib import Path
import numpy as np
from .common import MaterialStateModel,write
from engine.aniso_phase1.research_material_coupling_next.budget import Budget
from engine.aniso_phase1.research_material_coupling_next.operators import Geometry
from engine.aniso_phase1.research_material_coupling_next.controller import Controller
from engine.aniso_phase1.research_formal_pressure_next.state import State,save

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();budget=Budget();m=MaterialStateModel();g=Geometry(m,budget)
    with np.load(a.run/'S1/local.npz') as d:q=d['q'].copy()
    # A short relaxation from the pre-registered non-equilibrium elastic state.
    # Fluid content is initialized consistently at its own V(q), not copied from rest.
    old=State(q,np.zeros_like(q),np.zeros(2),rule=7);digest=old.digest();h=1e-5
    c=Controller(m,g,budget,primary=2,reserve=7,journal=lambda x:write(a.run/'S2/retry-attempts.json',x));sol=c._solver(7)
    target=q.ravel()[sol.fixed].copy()
    try:
        trial,row=c.step(old,h,target,initial_guard=True)
        write(a.run/'S2/retry-row.json',row)
        direct,dr=sol.step(old,h,target)
        gaps=dict(q=float(abs(trial.q-direct.q).max()),v=float(abs(trial.velocity-direct.velocity).max()),p=float(abs(trial.p-direct.p).max()))
        assert row['fallback'] and row['active_rule']==7 and max(gaps.values())<1e-8
        assert old.digest()==digest
        failed=False
        try:c.step(old,h,target,initial_guard=True,fault=True)
        except RuntimeError as e:
            if 'injected' not in str(e):raise
            failed=True
        assert failed and old.digest()==digest
        save(trial,a.run/'S2/retry-checkpoint.npz',c.identity);write(a.run/'S2/retry-identity.json',c.identity)
        write(a.run/'S2/retry-summary.json',dict(actual_material_rejection=True,rejection_stage='initial-state preflight before nonlinear primary solve',
            full_step_recomputed=True,post_trial_actual_rejection=False,fault_rollback=failed,direct_gaps=gaps,old_digest=digest,new_digest=trial.digest(),
            dt=h,initial_state='registered non-equilibrium local elastic state, pressure zero, initial content based on V(q)',
            minJ=row['minJ'],jv_calls=row['jv_calls_step'],budget=budget.report()))
        print('real initial material rejection and whole q7 step passed',gaps,flush=True)
    except Exception as e:
        write(a.run/'S2/retry-failure.json',dict(type=type(e).__name__,reason=str(e),old_unchanged=old.digest()==digest,budget=budget.report()));raise
