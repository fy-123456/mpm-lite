"""Original AVF equations with physical gates before the sole state commit."""
from __future__ import annotations
import copy
import numpy as np
from ..research_c.stage2.dynamics import AVF, StepRejected


class ValidatedAVF(AVF):
    def __init__(self,model,config,state=None):
        self.config=copy.deepcopy(config)
        solver=config['solver']
        super().__init__(model,state,path_order=config['path_order'],
            residual_atol=solver['residual_atol'],residual_rtol=solver['residual_rtol'],
            ledger_atol=solver['ledger_atol_J'])

    def path(self,q,W,dt,direction=None):
        out=super().path(q,W,dt,direction)
        if not np.isfinite(out['min_detF']) or out['min_detF']<=self.config['acceptance']['min_detF']:
            raise ValueError('AVF path leaves the frozen valid deformation range')
        return out

    def _compute(self,state,dt,max_iters,external_force,inject):
        solver=self.config['solver']
        self.atol=min(solver['residual_atol'],dt*solver['residual_force_atol_N'])
        candidate,row=self._advance(state,dt,max_iters,external_force,inject)
        acceptance=self.config['acceptance']
        if not all(np.isfinite(v) for v in row.values() if isinstance(v,(float,int))):
            raise ValueError('nonfinite step ledger')
        if row['min_detF']<=acceptance['min_detF'] or row['true_residual']>row['residual_tolerance']:
            raise ValueError('precommit deformation/residual gate failed')
        if max(row['displacement_constraint'],row['velocity_constraint'])>acceptance['boundary_atol']:
            raise ValueError('precommit boundary gate failed')
        budget=acceptance['energy_fraction']*acceptance['energy_scale_J']
        for quantity in ('path_quadrature_error_J','solve_work_error_J'):
            key='cumulative_abs_'+quantity
            value=state.child_states.get(key,0.)+abs(row[quantity])
            if not np.isfinite(value) or value>budget:
                raise ValueError(f'{key} {value:.6g} exceeds frozen physical budget {budget:.6g}')
            candidate.child_states[key]=value;row[key]=value
        row['equivalent_force_residual_N']=row['true_residual']/dt
        candidate.child_states['last_ledger']=copy.deepcopy(row)
        return candidate,row

    def _advance(self,state,dt,max_iters,external_force,inject):
        return super()._compute(state,dt,max_iters,external_force,inject)

    def step(self,dt,*,max_iters=None,external_force=None,prepare_children=None,validate_trial=None,inject=None):
        def validate(candidate):
            # Always check physical endpoint after the child proposal, including
            # ordinary runs with no fault injection enabled.
            self.model.validate(candidate,material=True)
            out=self.model.evaluate(candidate.q)
            if out['min_detF']<=self.config['acceptance']['min_detF']:
                raise ValueError('endpoint below frozen det(F) threshold')
            if inject is not None:inject('before_commit',candidate)
            self.model.validate(candidate,material=True)
            return True if validate_trial is None else validate_trial(candidate)
        return super().step(dt,max_iters=self.config['max_iters'] if max_iters is None else max_iters,
            external_force=external_force,prepare_children=prepare_children,validate_trial=validate,inject=inject)
