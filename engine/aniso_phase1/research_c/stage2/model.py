"""Explicit dynamic lift and the unchanged frozen potential and point inertia."""
from __future__ import annotations

import copy
import numpy as np
import scipy.linalg as la

from ...carrier_driven import displacement
from ...endpoint_boundary import prescribed_speed
from ...history_increment import material_tangent
from ...tensor_reference import apply_axis
from ...consistent_transfer import material_response
from ...research_b.tensor import TensorMaterialOperator, TensorRule
from ...research_d.common_kinetic import PointInertia
from ...research_d.common_state import CommonState
from ...research_d.identity import digest


class Boundary:
    def __init__(self, space, *, hold=None):
        self.space, self.hold = space, hold
        if hold is not None and not np.isfinite(hold):
            raise ValueError('finite fixed displacement required')
        self.unit = np.zeros((space.ndof, 3))
        self.unit[:space.n, 0] = space.carrier_X[:, 0] >= space.boundary['right_x']
        self.signature = digest(dict(space=space.signature, hold=hold,
            breaks=[0., .5, .6, 1.1, 1.6], peak=.005, rule='parent cosine v1'))

    def lift(self, t):
        if not np.isfinite(t) or t < 0:
            raise ValueError('finite nonnegative physical time required')
        return self.unit * (displacement(t) if self.hold is None else self.hold)

    def speed(self, t):
        self.lift(t)
        return self.unit * (prescribed_speed(t) if self.hold is None else 0.)

    def expand(self, q_free, v_free, t):
        return (self.space.expand(q_free, self.lift(t)),
                self.space.expand(v_free, self.speed(t)))

    def validate(self, state, tol=1e-9):
        s = self.space
        q = s._check(state.q, (s.ndof, 3), 'full dynamic displacement')
        v = s._check(state.velocity, (s.ndof, 3), 'full dynamic velocity')
        ids = s.fixed_scalar_ids
        errors = (np.max(abs((q-self.lift(state.time))[ids])),
                  np.max(abs((v-self.speed(state.time))[ids])))
        if max(errors) > tol:
            raise ValueError('incompatible physical boundary phase')
        return errors


def recover_mass(inertia, progress=None):
    """Three independent scalar columns per existing vector action, bounded RAM."""
    n = inertia.space.ndof
    M = np.empty((n, n))
    for start in range(0, n, 3):
        stop = min(n, start+3)
        v = np.zeros((n, 3))
        v[np.arange(start, stop), np.arange(stop-start)] = 1.
        M[:, start:stop] = inertia.apply(v)[:, :stop-start]
        if progress is not None and start % 90 == 0:
            progress(start, n)
    return M


def rest_stiffness(space, progress=None):
    """Exact constant tangent at F=I via six scalar derivative Gram blocks.

    No dense nodal basis or material Hessian is formed. This is a rest modal
    operator/search matrix only; nonlinear residuals always use full material.
    """
    from benchmarks.aniso_local_q3 import axis
    axes = [axis(e, space.p, order=6)[1] for e in space.edges]
    H = np.empty((9, 9))
    for k in range(9):
        d = np.zeros((1, 3, 3)); d.ravel()[k] = 1.
        H[:, k] = material_tangent(np.eye(3)[None], space.A[None], d, space.params).ravel()
    H = H.reshape(3, 3, 3, 3)
    n = space.ndof
    K = np.zeros((n, 3, n, 3))
    for i in range(3):
        for j in range(i, 3):
            G = np.empty((n, n))
            for start in range(0, n, 3):
                stop = min(n, start+3)
                v = np.zeros((n, 3)); v[np.arange(start, stop), np.arange(stop-start)] = 1.
                field = space.nodes(v).reshape(*space.shape, 3)
                for k, grams in enumerate(axes):
                    A = grams[1] if i == j == k else (grams[2] if k == i and i != j
                        else grams[2].T if k == j and i != j else grams[0])
                    field = apply_axis(A, field, k)
                G[:, start:stop] = space.adjoint(field.reshape(-1, 3))[:, :stop-start]
            K += np.einsum('rs,ab->rasb', G, H[:, i, :, j])
            if i != j:
                K += np.einsum('sr,ab->rasb', G, H[:, j, :, i])
            if progress is not None:
                progress(i, j)
    for a in range(3):
        K[:space.n, a, :space.n, a] += space.Ks
    return K.reshape(3*n, 3*n)


class MassRankError(ValueError):
    """The frozen free mass cannot support the requested ordinary ODE model."""


class DynamicModel:
    def __init__(self, space, mass, *, order=6, boundary=None, rest_K=None):
        self.space = space
        self.boundary = boundary or Boundary(space)
        self.M = np.array(mass, dtype=float, copy=True)
        if self.M.shape != (space.ndof, space.ndof) or not np.isfinite(self.M).all():
            raise ValueError('full scalar mass required')
        if not np.allclose(self.M, self.M.T, atol=1e-14, rtol=1e-10):
            raise ValueError('mass is not symmetric')
        self.free = space.free_scalar_ids
        self.fixed = space.fixed_scalar_ids
        self.ids = (3*self.free[:, None]+np.arange(3)).ravel()
        self.Mff = self.M[np.ix_(self.free, self.free)]
        diagonal = np.diag(self.Mff)
        if np.any(diagonal <= 0):
            raise MassRankError('free mass contains a nonpositive diagonal; dynamics blocked')
        scale = 1/np.sqrt(diagonal)
        spectrum = la.eigvalsh(scale[:, None]*self.Mff*scale[None, :])
        threshold = 64*len(self.free)*np.finfo(float).eps*max(spectrum[-1], 1.)
        rank = int(np.sum(spectrum > threshold))
        if rank != len(self.free):
            raise MassRankError(f'free scalar mass rank {rank}/{len(self.free)}; '
                                'no artificial mass, elimination, or pseudoinverse permitted')
        self.mass_factor = la.cho_factor(self.Mff)
        self.M3ff = np.kron(self.Mff, np.eye(3))
        self.rule = TensorRule.uniform(space.edges, order)
        self.operator = TensorMaterialOperator(space, self.rule)
        self.rest_K = None if rest_K is None else np.array(rest_K, copy=True)
        if self.rest_K is not None and (self.rest_K.shape != (3*space.ndof,)*2 or
                                       not np.isfinite(self.rest_K).all()):
            raise ValueError('invalid rest search matrix')
        self.identity = dict(space=space.signature, mass=digest(self.M),
            material=self.operator.signature, boundary=self.boundary.signature,
            dtype='float64', device='cpu')
        self.signature = digest(self.identity)
        self.M.setflags(write=False)

    def evaluate(self, q, direction=None):
        # Full displacement, no implicit frozen .005 m lift.
        s = self.space
        s._check(q, (s.ndof, 3), 'full dynamic displacement')
        if direction is not None:
            s._check(direction, (s.ndof, 3), 'full direction')
        return self.operator.evaluate(q, direction)

    def kinetic(self, v):
        return .5*float(np.sum(v*(self.M@v)))

    def rest(self):
        z = np.zeros((self.space.ndof, 3))
        state = CommonState(z.copy(), z.copy(), child_states={'identity': copy.deepcopy(self.identity)})
        self.boundary.validate(state)
        return state

    def validate(self, state, *, material=False):
        self.boundary.validate(state)
        if state.child_states.get('identity') != self.identity:
            raise ValueError('wrong space, mass, material, or boundary identity')
        if material:
            self.evaluate(state.q)  # tests all integration points, not only probes

    def endpoint(self, pre, time):
        delta = np.zeros_like(pre)
        delta[self.fixed] = (self.boundary.speed(time)-pre)[self.fixed]
        delta[self.free] = la.cho_solve(self.mass_factor, -(self.M@delta)[self.free])
        impulse = self.M@delta
        velocity = pre+delta
        work = float(np.sum(impulse[self.fixed]*velocity[self.fixed]))
        loss = self.kinetic(delta)
        error = self.kinetic(velocity)-self.kinetic(pre)-work+loss
        return velocity, impulse, dict(endpoint_boundary_work_J=work,
            constraint_kinetic_loss_J=loss, endpoint_identity_error_J=error,
            endpoint_free_impulse=float(la.norm(impulse[self.free])))

    def wrench(self, generalized, q=None):
        """Rigid virtual motion dual; local coefficients are never summed as forces."""
        s = self.space
        total = np.zeros(3); moment = np.zeros(3)
        positions = s.reference.copy() if q is None else s.reference+q
        for a in range(3):
            translation = np.zeros_like(generalized); translation[:s.n, a] = 1.
            total[a] = np.sum(generalized*translation)
            rotation = np.cross(np.eye(3)[a], positions)
            moment[a] = np.sum(generalized*rotation)
        return total, moment

    def fields(self, state, points):
        s = self.space
        u, gu = s._sample(s.nodes(state.q), points)
        v, gv = s._sample(s.nodes(state.velocity), points)
        X = np.stack(np.meshgrid(*points, indexing='ij'), axis=-1)
        F = np.eye(3)+gu
        if np.any(np.linalg.det(F) <= 0):
            raise ValueError('invalid probe deformation')
        _, P = material_response(F.reshape(-1, 3, 3),
            np.broadcast_to(s.A, F.reshape(-1, 3, 3).shape), s.params)
        return dict(x=X+u, F=F, v=v, C=gv@np.linalg.inv(F), PK1=P.reshape(F.shape))


class FrozenPotential:
    """Mass-independent audits remain valid when the dynamic mass gate fails."""
    def __init__(self, space, order=6):
        self.space = space
        self.rule = TensorRule.uniform(space.edges, order)
        self.operator = TensorMaterialOperator(space, self.rule)

    evaluate = DynamicModel.evaluate
    fields = DynamicModel.fields
