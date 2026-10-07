"""Full-mass AVF with hard-grip impulse and atomic accepted-state updates.

For this affine reference basis M is constant even when particles move.
No derivative of an Eulerian APIC metric is silently omitted. The chord
matrix is only a search direction; stopping uses the actual AVF residual.
"""
import copy
import time

import numpy as np
import scipy.linalg as la

from ..carrier_driven import displacement
from ..endpoint_boundary import prescribed_speed
from .model import DynamicState
from .transfer import roundtrip


class StepRejected(RuntimeError):
    pass


class AVF:
    def __init__(self, model, state=None, *, moving_grid=False, path_order=3,
                 residual_atol=1e-11, residual_rtol=1e-8, grid_origin=(.003, .002, .001)):
        if path_order < 1 or residual_atol <= 0 or residual_rtol < 0:
            raise ValueError('invalid integration tolerances')
        self.model = model; self.state = (state or model.rest()).clone()
        self.moving_grid = moving_grid; self.grid_origin = tuple(grid_origin)
        self.path_order = path_order; self.atol = residual_atol; self.rtol = residual_rtol
        self.previous = None; self.packet = None; self.child_states = {}
        self.last = None; self.failures = []
        self._validate(self.state)
        self._Kref = model.evaluate(self.state.q, tangent=True)['K']
        self._factors = {}

    def _validate(self, state):
        if not np.isfinite(state.time) or state.time < 0:
            raise ValueError('finite nonnegative time required')
        self.model.space.kinematics(self.model.X, state.q, state.velocity)
        fixed = self.model.space.fixed; lift = self.model.space.lift
        if max(np.max(abs((state.q-displacement(state.time)*lift)[fixed])),
               np.max(abs((state.velocity-prescribed_speed(state.time)*lift)[fixed]))) > 1e-9:
            raise ValueError('incompatible hard-grip history; no silent projection')

    def path(self, q, W, dt, tangent=False):
        sites, weights = np.polynomial.legendre.leggauss(self.path_order)
        force = np.zeros_like(q); K = np.zeros_like(self.model.M3)
        for a, w in zip((sites+1)/2, weights/2):
            out = self.model.evaluate(q+a*dt*W, tangent=tangent)
            force += w*out['force']
            if tangent:
                K += w*a*out['K']
        return force, K

    def step(self, dt, max_iters=12, *, external_force=None, prepare_children=None,
             validate_trial=None):
        """Callbacks receive owned copies; only the returned child dict commits.

        external_force is a frozen generalized force for this interval. A
        state-dependent force/tangent or independent APIC history is rejected
        by omission from this interface, pending D/E integration.
        """
        try:
            return self._trial(dt, max_iters, external_force, prepare_children, validate_trial)
        except Exception as exc:
            self.failures.append(dict(time=self.state.time, dt=float(dt), reason=str(exc), rolled_back=True))
            raise StepRejected(str(exc)) from exc

    def _trial(self, dt, max_iters, external_force, prepare_children, validate_trial):
        start = time.perf_counter()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite dt required')
        m = self.model; s = self.state; space = m.space; ids = m.free
        self._validate(s)
        t1 = s.time+dt
        initial = m.evaluate(s.q); K0 = m.kinetic(s.velocity)
        Wb = space.lift*((displacement(t1)-displacement(s.time))/dt)
        W = Wb.copy()
        W.ravel()[ids] = s.velocity.ravel()[ids] if self.previous is None else self.previous.ravel()[ids]
        ext = np.zeros_like(W) if external_force is None else np.array(external_force, dtype=float, copy=True)
        if ext.shape != W.shape or not np.isfinite(ext).all():
            raise ValueError('finite frozen generalized external force required')
        factor = self._factors.get(dt)
        if factor is None:
            H = 2*m.Mff+.5*dt*dt*self._Kref[np.ix_(ids, ids)]
            factor = la.cho_factor(H)
        scale = max(la.norm((m.M@s.velocity).ravel()[ids]), dt*la.norm((initial['force']-ext).ravel()[ids]), 1e-8)
        tolerance = self.atol+self.rtol*scale
        backs = 0
        def residual(velocity):
            f, _ = self.path(s.q, velocity, dt)
            impulse = 2*m.M@(velocity-s.velocity)+dt*(f-ext)
            return impulse.ravel()[ids], f, impulse
        for it in range(max_iters+1):
            r, force, impulse_mid = residual(W); norm = float(la.norm(r))
            if norm <= tolerance:
                break
            if it == max_iters:
                raise ValueError(f'AVF true residual {norm:g} exceeds {tolerance:g}')
            if it < 4:
                du = la.cho_solve(factor, -r)
            else:
                _, K = self.path(s.q, W, dt, tangent=True)
                du = la.solve(2*m.Mff+dt*dt*K[np.ix_(ids, ids)], -r, assume_a='sym')
            for ls in range(16):
                trial = W.copy(); trial.ravel()[ids] += 2.**(-ls)*du
                try:
                    newr, _, _ = residual(trial)
                except ValueError:
                    continue
                if la.norm(newr) < norm:
                    W = trial; backs += ls; break
            else:
                raise ValueError('AVF residual line search failed')
        q1 = s.q+dt*W; pre = 2*W-s.velocity
        delta = np.zeros_like(pre)
        delta[space.fixed] = (space.lift*prescribed_speed(t1)-pre)[space.fixed]
        rhs = -(m.M@delta).ravel()[ids]
        delta.ravel()[ids] = la.cho_solve(m.mass_factor, rhs)
        endpoint = m.M@delta; velocity = pre+delta
        endpoint_work = float(np.sum(endpoint*space.lift)*prescribed_speed(t1))
        loss = m.kinetic(delta)
        end_identity = m.kinetic(velocity)-m.kinetic(pre)-endpoint_work+loss
        if abs(end_identity) > 1e-10 or la.norm(endpoint.ravel()[ids]) > 1e-9:
            raise ValueError('endpoint impulse is not boundary-dual')
        candidate = DynamicState(q1, velocity, t1, s.step+1)
        self._validate(candidate)
        transfer = dict(transfer_error=0., transfer_energy_jump_J=0., crossed_particles=0,
                        residual_velocity_norm=0., residual_affine_norm=0., grid_nodes=0)
        packet = None
        if self.moving_grid:
            candidate.velocity, packet, transfer = roundtrip(m, candidate,
                None if self.packet is None else self.packet.cells, origin=self.grid_origin)
            self._validate(candidate)
        children = copy.deepcopy(self.child_states)
        if prepare_children is not None:
            children = prepare_children(candidate.clone(), children)
            if not isinstance(children, dict):
                raise ValueError('child transaction must return an owned dictionary')
            children = copy.deepcopy(children)
        if validate_trial is not None:
            validate_trial(candidate.clone(), copy.deepcopy(children))
        final = m.evaluate(candidate.q); K1 = m.kinetic(candidate.velocity)
        midpoint_work = float(np.sum(impulse_mid*Wb))
        external_work = dt*float(np.sum(ext*W))
        path_work = dt*float(np.sum(force*W))
        path_error = final['U']-initial['U']-path_work
        delta_energy = K1+final['U']-K0-initial['U']
        budget = delta_energy-midpoint_work-endpoint_work-external_work+loss
        residual_work = float(np.sum(impulse_mid*(W-Wb)))
        closure = budget-path_error-residual_work-transfer['transfer_energy_jump_J']
        if not np.isfinite(budget) or abs(closure) > 1e-9:
            raise ValueError('step energy ledger does not close')
        P0, L0 = m.momentum(s); P1, L1 = m.momentum(candidate)
        total_impulse = impulse_mid+endpoint
        momentum_external = np.sum((total_impulse+dt*ext)[:space.carriers], axis=0)
        stress = np.sqrt(np.sum(m.V*np.sum(final['P']**2, axis=(1, 2)))/m.V.sum())
        row = dict(step=candidate.step, time=t1, dt=dt, reaction_N=float(np.sum(total_impulse*space.lift)/dt),
            midpoint_reaction_N=float(np.sum(impulse_mid*space.lift)/dt),
            endpoint_reaction_N=float(np.sum(endpoint*space.lift)/dt),
            material_reaction_N=float(np.sum(force*space.lift)), stabilization_reaction_N=0.,
            inertia_reaction_N=float(np.sum((2*m.M@(W-s.velocity))*space.lift)/dt),
            external_reaction_N=-float(np.sum(ext*space.lift)), material_J=final['Um'], stabilization_J=0.,
            kinetic_J=K1, total_J=K1+final['U'], delta_total_J=delta_energy,
            boundary_work_J=midpoint_work+endpoint_work, midpoint_boundary_work_J=midpoint_work,
            endpoint_boundary_work_J=endpoint_work, external_work_J=external_work,
            constraint_kinetic_loss_J=loss, path_quadrature_error_J=path_error,
            solve_work_error_J=residual_work, budget_defect_J=closure, energy_balance_J=budget,
            metric_change_J=0., space_migration_jump_J=0., rule_migration_jump_J=0.,
            newton_iterations=it, true_residual=norm, residual_tolerance=tolerance,
            line_search_backtracks=backs, krylov_iterations=0, negative_curvature_fallback=False,
            min_det_F=float(np.min(np.linalg.det(final['F']))), stress_rms_Pa=float(stress),
            displacement_constraint=float(np.max(abs((q1-space.lift*displacement(t1))[space.fixed]))),
            velocity_constraint=float(np.max(abs((candidate.velocity-space.lift*prescribed_speed(t1))[space.fixed]))),
            endpoint_free_impulse=float(la.norm(endpoint.ravel()[ids])),
            linear_momentum_error=float(la.norm(P1-P0-momentum_external)),
            angular_momentum_change=(L1-L0).tolist(), linear_momentum=P1.tolist(), angular_momentum=L1.tolist(),
            space_version=space.version, rule_version=m.rule_version, accepted=True, rolled_back=False,
            wall_seconds=time.perf_counter()-start, **transfer)
        # Commit is last; failed children/validation/transfer leave all caches,
        # predictors, packets, q/v, child rules and the clock untouched.
        self.state = candidate; self.previous = W.copy(); self.packet = packet
        self.child_states = children; self.last = dict(row=row, P=final['P'], F=final['F'])
        self._factors[dt] = factor
        return row
