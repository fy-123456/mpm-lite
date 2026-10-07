"""Owned 2D value adapter: only the caller's total transaction publishes state.

The E adapter has no committed mutable history and never commits an external
CouplingTransaction. Parent StateTransaction owns child_states['E']; its
revision tokens govern failure, retry and publication of the entire step.
"""
import copy
import hashlib
import json
import numpy as np
from ..poro import PoroState
from .evaluation import StressEvaluator, physical_diagnostics
from .residual import step_residual, valid_ledger


def digest_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def solver_identity(solver):
    s, f = solver.solid, solver.flow
    desc = dict(model='E-stage2-plane-strain-Q2-RT0-P0', shape=list(f.grid.shape),
                lengths=f.grid.lengths.tolist(), thickness=1., dtype='float64', device='cpu',
                material=dict(mu=s.material.mu, lam=s.material.lam, k_f=s.material.k_f,
                              direction=s.material.fiber_direction.tolist()),
                alpha=s.alpha, storage=solver.storage, viscosity=f.viscosity,
                K=f.K.tolist(), fixed_u=s.fixed.tolist(), gravity_rhs=f.gravity_rhs.tolist(),
                units=dict(u='m', p='Pa', flux='m^3/s', time='s'))
    return desc, digest_value(desc)


class ChildAdapter:
    key = 'E'

    def __init__(self, solver, parent_sha):
        if solver.flow.grid.dim != 2:
            raise ValueError('this child contract requires independent 2D E space')
        self.solver, self.parent_sha = solver, parent_sha
        self.config, self.signature = solver_identity(solver)
        self.evaluator = StressEvaluator(solver.solid)

    def initial(self):
        state = self.solver.initial()
        value = dict(schema_version=2, parent_bundle_sha256=self.parent_sha,
                     physical_space_sha256=self.signature, config=copy.deepcopy(self.config),
                     u=state.u.tolist(), p=state.p.tolist(), flux=np.zeros(self.solver.flow.grid.nf).tolist(),
                     time=0., step=0, ledger=[], last_step=None)
        self.validate(value)
        return value

    def validate(self, value):
        if (value.get('schema_version') != 2 or value.get('parent_bundle_sha256') != self.parent_sha
                or value.get('physical_space_sha256') != self.signature or value.get('config') != self.config
                or solver_identity(self.solver)[1] != self.signature):
            raise ValueError('wrong E schema, parent, physical space or parameters')
        u, p, flux = (np.asarray(value[k], float) for k in ('u', 'p', 'flux'))
        s = self.solver.solid
        if (u.shape != (s.ndof,) or p.shape != (s.grid.nc,) or flux.shape != (s.grid.nf,)
                or not np.isfinite(np.r_[u,p,flux,value['time']]).all()
                or np.any(u[s.fixed] != 0) or value['time'] < 0
                or type(value['step']) is not int or value['step'] < 0 or len(value['ledger']) != value['step']):
            raise ValueError('invalid E values, boundary, history or clock')
        if not physical_diagnostics(self.evaluator, self.evaluator.evaluate(u,p))['passed']:
            raise ValueError('invalid deformation/porosity diagnostic')
        if value['step']:
            last = value['last_step']
            if not isinstance(last,dict) or abs(value['time']-last['old_time']-last['dt'])>1e-12:
                raise ValueError('inconsistent E step clock')
            from ..poro import PoroStep
            old = PoroState(np.asarray(last['old_u']),np.asarray(last['old_p']),last['old_time'])
            boundary = {(r[0],r[1]):(r[2],r[3]) for r in last['boundary']}
            result = PoroStep(PoroState(u,p,value['time']),flux,{},True)
            residual = step_residual(self.solver,old,result,last['dt'],np.asarray(last['load']),boundary,last['source'])
            if residual['maximum']>1e-8 or not np.isfinite(list(residual.values())).all():
                raise ValueError('E candidate does not satisfy its true step equations')
        digest_value(value)
        return True

    def prepare(self, committed, *, dt, load, boundary, source=0., method='monolithic', **kwargs):
        self.validate(committed)
        if callable(source) or any(callable(v[1]) for v in boundary.values()):
            raise ValueError('portable child adapter requires explicit numeric sources/boundaries')
        old = PoroState(np.array(committed['u']),np.array(committed['p']),committed['time'])
        result = self.solver.step(old,dt,load,boundary,source,method=method,**kwargs)
        residual = step_residual(self.solver,old,result,dt,load,boundary,source)
        diag = physical_diagnostics(self.evaluator,self.evaluator.evaluate(result.state.u,result.state.p))
        if not result.converged or not valid_ledger(result.metrics,residual,diag):
            raise ValueError('unconverged or physically invalid E candidate')
        value = copy.deepcopy(committed)
        value.update(u=result.state.u.tolist(),p=result.state.p.tolist(),flux=result.flux.tolist(),
                     time=result.state.time,step=committed['step']+1,
                     last_step=dict(old_u=old.u.tolist(),old_p=old.p.tolist(),old_time=old.time,
                                    dt=dt,load=np.asarray(load).tolist(),source=np.asarray(source).tolist(),
                                    boundary=[[a,b,k,v] for (a,b),(k,v) in sorted(boundary.items())]))
        value['ledger'].append(dict(result.metrics, residual_blocks=residual))
        self.validate(value)
        return value

    def validate_parent(self, state):
        self.validate(state.child_states[self.key])
        child = state.child_states[self.key]
        if child['step'] != state.step or abs(child['time']-state.time)>1e-12:
            raise ValueError('parent and E child clocks differ')
        return True

    def checkpoint(self, value):
        self.validate(value)
        payload = copy.deepcopy(value)
        return dict(payload=payload,sha256=digest_value(payload))

    def restore(self, checkpoint):
        value = copy.deepcopy(checkpoint['payload'])
        if digest_value(value) != checkpoint['sha256']:
            raise ValueError('checkpoint digest mismatch')
        self.validate(value)
        return value
