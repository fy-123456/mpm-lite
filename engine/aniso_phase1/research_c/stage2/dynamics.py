"""Full-mass AVF, exact residual stopping, endpoint impulse, and one commit."""
from __future__ import annotations

import copy
import time
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, gmres
from ...research_d.common_state import StateTransaction


class StepRejected(RuntimeError):
    pass


class AVF:
    def __init__(self, model, state=None, *, path_order=3,
                 residual_atol=1e-10, residual_rtol=1e-7, ledger_atol=1e-9):
        if (isinstance(path_order, bool) or int(path_order) != path_order or path_order < 1
                or not np.isfinite([residual_atol, residual_rtol, ledger_atol]).all()
                or residual_atol <= 0 or residual_rtol < 0 or ledger_atol <= 0):
            raise ValueError('invalid path order or tolerances')
        self.model = model
        self.path_order = int(path_order)
        self.atol, self.rtol, self.ledger_atol = residual_atol, residual_rtol, ledger_atol
        initial = model.rest() if state is None else state.clone()
        model.validate(initial, material=True)
        self._transaction = StateTransaction(initial, validator=model.validate)
        self.failures = []  # observational, never used by numerical decisions

    @property
    def state(self):
        return self._transaction.snapshot()

    def path(self, q, W, dt, direction=None):
        x, w = np.polynomial.legendre.leggauss(self.path_order)
        force = np.zeros_like(q); material = np.zeros_like(q)
        action = np.zeros_like(q) if direction is not None else None
        minimum = 1e300; calls = 0
        for a, weight in zip((x+1)/2, w/2):
            out = self.model.evaluate(q+a*dt*W, direction)
            force += weight*out['force']; material += weight*out['material_force']
            minimum = min(minimum, out['min_detF']); calls += out['material_calls']
            if direction is not None:
                action += weight*a*out['tangent_action']
        return dict(force=force, material=material, action=action,
                    min_detF=minimum, material_calls=calls)

    def step(self, dt, *, max_iters=8, external_force=None, prepare_children=None,
             validate_trial=None, inject=None):
        """prepare_children returns an owned value proposal, without external writes.

        A nonlinear failure discards the predictor, ledger, rule, and proposal.
        No B/E external object atomicity is claimed by this value-only adapter.
        """
        trial = self._transaction.begin_trial()
        try:
            result = self._compute(trial.state, dt, max_iters, external_force, inject)
            candidate, row = result
            if prepare_children is not None:
                candidate.child_states = copy.deepcopy(prepare_children(
                    candidate.clone(), copy.deepcopy(candidate.child_states)))
                if not isinstance(candidate.child_states, dict):
                    raise ValueError('child proposal must be an owned dictionary')
            if inject is not None:
                inject('after_prepare', candidate)
            self.model.validate(candidate)
            if validate_trial is not None and validate_trial(candidate.clone()) is False:
                raise ValueError('post-check rejected trial')
            # A test hook can deliberately inject malformed physical values.
            # It must be detected before the sole publication operation.
            if inject is not None:
                self.model.validate(candidate, material=True)
            trial.state.q = candidate.q
            trial.state.velocity = candidate.velocity
            trial.state.time = candidate.time
            trial.state.step = candidate.step
            trial.state.predictor = candidate.predictor
            trial.state.child_states = candidate.child_states
            self._transaction.commit(trial)
            return row
        except Exception as exc:
            try:
                self._transaction.rollback(trial)
            except ValueError:  # a failed commit has already consumed the token
                pass
            self.failures.append(dict(time=self.state.time, dt=float(dt),
                                      reason=str(exc), rolled_back=True))
            raise StepRejected(str(exc)) from exc

    def _compute(self, s, dt, max_iters, external_force, inject):
        start = time.perf_counter()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite step required')
        m = self.model; m.validate(s)
        t1 = s.time+dt
        if m.boundary.hold is None and t1 > 1.6+1e-12:
            raise ValueError('step exceeds frozen cycle endpoint')
        if inject is not None:
            inject('before_material', s)
        initial = m.evaluate(s.q); K0 = m.kinetic(s.velocity)
        Wb = (m.boundary.lift(t1)-m.boundary.lift(s.time))/dt
        W = Wb.copy()
        W[m.free] = s.velocity[m.free] if s.predictor is None else s.predictor[m.free]
        ext = np.zeros_like(W) if external_force is None else np.array(external_force, dtype=float)
        if ext.shape != W.shape or not np.isfinite(ext).all():
            raise ValueError('finite interval generalized force required')
        scale = max(la.norm((m.M@s.velocity)[m.free]),
                    dt*la.norm((initial['force']-ext)[m.free]), 1e-8)
        tolerance = self.atol+self.rtol*scale
        H = 2*m.M3ff
        if m.rest_K is not None:
            H = H+.5*dt**2*m.rest_K[np.ix_(m.ids, m.ids)]
        factor = la.lu_factor(H)  # search only; no SPD assumption or physical shift
        calls = initial['material_calls']; backs = 0; krylov = 0

        def residual(velocity):
            nonlocal calls
            out = self.path(s.q, velocity, dt)
            calls += out['material_calls']
            impulse = 2*m.M@(velocity-s.velocity)+dt*(out['force']-ext)
            return impulse[m.free].ravel(), out, impulse

        r, out, midpoint = residual(W)
        for iteration in range(max_iters+1):
            norm = float(la.norm(r))
            if norm <= tolerance:
                break
            if iteration == max_iters:
                raise ValueError(f'true AVF residual {norm:g} exceeds {tolerance:g}')
            if inject is not None:
                inject('before_search', s)
            if iteration < 4 and m.rest_K is not None:
                update = la.lu_solve(factor, -r)
            else:
                def action(vector):
                    nonlocal calls
                    d = np.zeros_like(W); d[m.free] = vector.reshape(-1, 3)
                    p = self.path(s.q, W, dt, d); calls += p['material_calls']
                    return (2*m.M@d+dt**2*p['action'])[m.free].ravel()
                def count(_):
                    nonlocal krylov
                    krylov += 1
                A = LinearOperator(H.shape, matvec=action, dtype=float)
                P = LinearOperator(H.shape, matvec=lambda x: la.lu_solve(factor, x), dtype=float)
                update, info = gmres(A, -r, M=P, atol=0., rtol=1e-5,
                    restart=20, maxiter=3, callback=count, callback_type='pr_norm')
                if info:
                    raise ValueError(f'exact tangent GMRES failed: {info}')
            for ls in range(12):
                candidate = W.copy(); candidate[m.free] += (2.**-ls)*update.reshape(-1, 3)
                try:
                    newr, newout, newmid = residual(candidate)
                except ValueError:
                    continue
                if la.norm(newr) < norm:
                    W, r, out, midpoint = candidate, newr, newout, newmid
                    backs += ls
                    break
            else:
                raise ValueError('residual line search failed')

        q1 = s.q+dt*W
        velocity, endpoint, end = m.endpoint(2*W-s.velocity, t1)
        candidate = s.clone()
        candidate.q, candidate.velocity = q1, velocity
        candidate.time, candidate.step = t1, s.step+1
        candidate.predictor = W.copy()
        m.validate(candidate)
        final = m.evaluate(q1); calls += final['material_calls']
        K1 = m.kinetic(velocity)
        midpoint_work = float(np.sum(midpoint*Wb))
        external_work = dt*float(np.sum(ext*W))
        path_error = final['U']-initial['U']-dt*float(np.sum(out['force']*W))
        residual_work = float(np.sum(midpoint*(W-Wb)))
        delta = K1+final['U']-K0-initial['U']
        balance = delta-midpoint_work-end['endpoint_boundary_work_J']-external_work+end['constraint_kinetic_loss_J']
        closure = balance-path_error-residual_work
        if not np.isfinite(closure) or abs(closure) > self.ledger_atol:
            raise ValueError('step energy ledger failed')
        if abs(end['endpoint_identity_error_J']) > self.ledger_atol or end['endpoint_free_impulse'] > 1e-9:
            raise ValueError('endpoint impulse identity failed')
        boundary = m.boundary.unit
        reaction = lambda f: float(np.sum(f*boundary))
        P0, _ = m.wrench(m.M@s.velocity); P1, L1 = m.wrench(m.M@velocity, q1)
        total_impulse, _ = m.wrench(midpoint+endpoint+dt*ext)
        row = dict(step=candidate.step, time=t1, dt=dt,
            reaction_N=reaction(midpoint+endpoint)/dt,
            midpoint_reaction_N=reaction(midpoint)/dt, endpoint_reaction_N=reaction(endpoint)/dt,
            material_reaction_N=reaction(out['material']),
            stabilization_reaction_N=reaction(out['force']-out['material']),
            inertia_reaction_N=reaction(2*m.M@(W-s.velocity))/dt,
            external_reaction_N=-reaction(ext), material_J=final['material_U'],
            stabilization_J=final['stabilization_U'], kinetic_J=K1, total_J=K1+final['U'],
            delta_total_J=delta, midpoint_boundary_work_J=midpoint_work,
            boundary_work_J=midpoint_work+end['endpoint_boundary_work_J'], external_work_J=external_work,
            path_quadrature_error_J=path_error, solve_work_error_J=residual_work,
            energy_balance_J=balance, budget_defect_J=closure, **end,
            true_residual=norm, residual_tolerance=tolerance, newton_iterations=iteration,
            line_search_backtracks=backs, krylov_iterations=krylov, material_point_calls=calls,
            min_detF=min(final['min_detF'], out['min_detF']),
            linear_momentum=P1.tolist(), angular_momentum=L1.tolist(),
            linear_momentum_error=float(la.norm(P1-P0-total_impulse)),
            displacement_constraint=float(m.boundary.validate(candidate)[0]),
            velocity_constraint=float(m.boundary.validate(candidate)[1]),
            material_rule=m.rule.signature, path_order=self.path_order,
            accepted=True, wall_seconds=time.perf_counter()-start)
        candidate.child_states['last_ledger'] = copy.deepcopy(row)
        candidate.child_states['cumulative_abs_closure_J'] = (
            s.child_states.get('cumulative_abs_closure_J', 0.)+abs(closure))
        return candidate, row
