"""Eliminate only the physical nullspace by its original Ks equilibrium.

The material field, consistent mass and original stabilization are unchanged.
Full displacement is P w + offset. The last coordinates are original prescribed
rows; the first coordinates span the independent free material motion.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la

from ..research_b.tensor import TensorMaterialOperator, TensorRule
from ..research_c.stage2.model import Boundary, DynamicModel
from ..research_d.common_state import CommonState
from ..research_d.identity import digest
from ..research_d.stage2.contracts import array_digest
from ..consistent_transfer import material_response


class Condensation:
    def __init__(self, space, mass, stiffness):
        self.parent = space
        self.original_mass = np.array(mass, dtype=np.float64, copy=True)
        self.original_stiffness = np.array(stiffness, dtype=np.float64, copy=True)
        n = space.ndof
        if self.original_mass.shape != (n, n) or self.original_stiffness.shape != (3*n, 3*n):
            raise ValueError('complete parent mass and rest stiffness required')
        if not np.isfinite(self.original_mass).all() or not np.isfinite(self.original_stiffness).all():
            raise ValueError('finite parent operators required')
        carriers = space.free_scalar_ids[space.free_scalar_ids < space.n]
        _, sv, vt = la.svd(space.oldA[:, carriers], full_matrices=False)
        cutoff = max(space.oldA.shape)*np.finfo(float).eps*sv[0]
        null = vt[sv <= cutoff].T
        if null.shape[1] == 0:
            raise ValueError('this adapter expects the diagnosed carrier nullspace')
        self.N = np.zeros((n, null.shape[1]))
        self.N[carriers] = null
        self.R = la.null_space(self.N[space.free_scalar_ids].T)
        nr = self.R.shape[1]
        self.free = np.arange(nr)
        self.fixed = np.arange(nr, nr+len(space.fixed_scalar_ids))
        base = np.zeros((n, len(self.free)+len(self.fixed)))
        base[np.ix_(space.free_scalar_ids, self.free)] = self.R
        base[space.fixed_scalar_ids, self.fixed] = 1.
        self.G = self.N[:space.n].T @ space.Ks @ self.N[:space.n]
        if la.eigvalsh(self.G)[0] <= 1e-10*max(la.norm(self.G), 1.):
            raise ValueError('nullspace stabilization is not safely invertible')
        right = self.N[:space.n].T @ space.Ks
        self.P = base-self.N @ la.solve(self.G, right @ base[:space.n], assume_a='pos')
        self.offset = -self.N @ la.solve(self.G, right @ space.reference[:space.n], assume_a='pos')
        self.M = self.P.T @ self.original_mass @ self.P
        self.M = .5*(self.M+self.M.T)
        self.P3 = np.kron(self.P, np.eye(3))
        self.K = self.P3.T @ self.original_stiffness @ self.P3
        self.K = .5*(self.K+self.K.T)
        self.ids = (3*self.free[:, None]+np.arange(3)).ravel()
        Mff = self.M[np.ix_(self.free, self.free)]
        scale = 1/np.sqrt(np.diag(Mff))
        spectrum = la.eigvalsh(scale[:, None]*Mff*scale[None, :])
        threshold = 64*len(self.free)*np.finfo(float).eps*max(spectrum[-1], 1.)
        if np.any(spectrum <= threshold):
            raise ValueError('condensation did not yield a full-rank motion mass')
        self.signature = digest(dict(parent=space.signature, P=array_digest(self.P),
            offset=array_digest(self.offset), mass=array_digest(self.M),
            method='original-Ks-stationary-nullspace-v1'))
        self.audit = dict(parent_free_scalar=len(space.free_scalar_ids),
            nullity=null.shape[1], free_scalar=nr, free_vector=3*nr,
            carrier_rank=len(carriers)-null.shape[1],
            carrier_null_relative=float(la.norm(space.oldA @ self.N[:space.n])/max(la.norm(space.oldA), 1.)),
            mass_null_relative=float(la.norm(self.original_mass @ self.N)/la.norm(self.original_mass)),
            G_eigenvalues=la.eigvalsh(self.G).tolist(),
            independent_mass_min_scaled_eigenvalue=float(spectrum[0]),
            independent_mass_condition=float(spectrum[-1]/spectrum[0]),
            rank_threshold=float(threshold), artificial_mass=False,
            original_stabilization_preserved=True, physical_space_changed=False)
        if max(self.audit['carrier_null_relative'], self.audit['mass_null_relative']) > 1e-8:
            raise ValueError('diagnosed directions are not actual physical/mass null directions')
        for a in (self.P, self.offset, self.N, self.R, self.M, self.K):
            a.setflags(write=False)

    def expand(self, w):
        w = np.asarray(w, dtype=np.float64)
        if w.shape != (self.P.shape[1], 3) or not np.isfinite(w).all():
            raise ValueError('finite independent coordinates required')
        return self.P @ w+self.offset

    def velocity(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (self.P.shape[1], 3) or not np.isfinite(value).all():
            raise ValueError('finite independent direction required')
        return self.P @ value

    def project(self, full):
        full = self.parent._check(full, (self.parent.ndof, 3), 'parent displacement')
        return np.vstack((self.R.T @ full[self.parent.free_scalar_ids],
                          full[self.parent.fixed_scalar_ids]))

    def algebraic_residual(self, full):
        s = self.parent
        return self.N[:s.n].T @ s.Ks @ (s.reference[:s.n]+full[:s.n])


class ReducedBoundary:
    def __init__(self, reduction, *, hold=None):
        self.reduction = reduction
        self.parent = Boundary(reduction.parent, hold=hold)
        self.hold = hold
        self.unit = np.zeros((reduction.P.shape[1], 3))
        self.unit[reduction.fixed] = self.parent.unit[reduction.parent.fixed_scalar_ids]
        self.signature = digest(dict(parent=self.parent.signature, reduction=reduction.signature))

    def lift(self, time):
        result = np.zeros_like(self.unit)
        result[self.reduction.fixed] = self.parent.lift(time)[self.reduction.parent.fixed_scalar_ids]
        return result

    def speed(self, time):
        result = np.zeros_like(self.unit)
        result[self.reduction.fixed] = self.parent.speed(time)[self.reduction.parent.fixed_scalar_ids]
        return result

    def validate(self, state, tol=1e-8):
        for value in (state.q, state.velocity):
            if value.shape != self.unit.shape or not np.isfinite(value).all():
                raise ValueError('invalid independent physical state')
        ids = self.reduction.fixed
        errors = (float(np.max(abs((state.q-self.lift(state.time))[ids]))),
                  float(np.max(abs((state.velocity-self.speed(state.time))[ids]))))
        if max(errors) > tol:
            raise ValueError('incompatible prescribed boundary phase')
        return errors


class FullCoordinates:
    """Read-only view giving the existing GPU operator an explicit full input.

    No mutation of the parent lift, no monkey patch, and no lost reactions.
    """
    def __init__(self, parent):
        self.parent = parent
        self.q_shape = (parent.ndof, 3)
        self.free_scalar_ids = np.arange(parent.ndof)
        self.fixed_scalar_ids = np.array([], dtype=int)
        self.lift = np.zeros(self.q_shape)
        self.signature = digest(dict(parent=parent.signature, coordinates='full-displacement-v1'))

    def __getattr__(self, name):
        return getattr(self.parent, name)

    def expand(self, q, lift=None):
        if lift is not None and np.any(lift):
            raise ValueError('full input must already contain the prescribed lift')
        return self.parent._check(q, self.q_shape, 'full displacement').copy()

    def direction_coefficients(self, q):
        return self.expand(q)

    def restrict(self, force):
        return self.expand(force)


class CondensedModel:
    def __init__(self, reduction, *, order=7, device='cpu', hold=None):
        self.reduction = reduction
        self.parent = reduction.parent
        self.space = SimpleNamespace(ndof=reduction.P.shape[1])
        self.boundary = ReducedBoundary(reduction, hold=hold)
        self.M = reduction.M
        self.free, self.fixed, self.ids = reduction.free, reduction.fixed, reduction.ids
        self.Mff = self.M[np.ix_(self.free, self.free)]
        self.M3ff = np.kron(self.Mff, np.eye(3))
        self.mass_factor = la.cho_factor(self.Mff)
        self.rest_K = reduction.K
        self.rule = TensorRule.uniform(self.parent.edges, order)
        self.device = device
        if device == 'cpu':
            self.operator = TensorMaterialOperator(self.parent, self.rule)
        else:
            from ..research_d.stage2.gpu_operator import GPUOperator
            self.operator = GPUOperator(FullCoordinates(self.parent), order=order, device=device)
        self.identity = dict(model=reduction.signature, material=self.rule.signature,
            boundary=self.boundary.signature, device=device, dtype='float64')
        self.signature = digest(self.identity)
        self._cached_q = None
        self._cached_response = None

    def evaluate(self, q, direction=None):
        if direction is None and self._cached_q is not None and np.array_equal(q, self._cached_q):
            out = dict(self._cached_response)
            out['material_calls'] = 0
            return out
        full = self.reduction.expand(q)
        d = None if direction is None else self.reduction.velocity(direction)
        raw = self.operator.evaluate(full, d)
        force = raw.get('full_force', raw['force'])
        out = dict(raw, force=self.reduction.P.T @ force,
                   material_force=self.reduction.P.T @ raw['material_force'],
                   original_full_force=force, algebraic_force=self.reduction.N.T @ force)
        if direction is not None:
            out['tangent_action'] = self.reduction.P.T @ raw.get('full_tangent_action', raw['tangent_action'])
        else:
            self._cached_q = np.array(q, copy=True)
            self._cached_response = dict(out)
        return out

    def rest(self):
        zero = np.zeros((self.space.ndof, 3))
        state = CommonState(zero.copy(), zero.copy(), child_states={'identity': copy.deepcopy(self.identity)})
        self.boundary.validate(state)
        return state

    def validate(self, state, *, material=False):
        self.boundary.validate(state)
        if state.child_states.get('identity') != self.identity:
            raise ValueError('foreign model/material/boundary/device state')
        if la.norm(self.reduction.algebraic_residual(self.reduction.expand(state.q))) > 1e-8:
            raise ValueError('eliminated algebraic constraint violated')
        if material:
            out = self.evaluate(state.q)
            if out['min_detF'] <= .1:
                raise ValueError('state outside practical physical Jacobian range')

    def kinetic(self, v):
        return .5*float(np.sum(v*(self.M @ v)))

    endpoint = DynamicModel.endpoint

    def wrench(self, generalized, q=None):
        s = self.parent
        full = np.zeros((s.ndof, 3)) if q is None else self.reduction.expand(q)
        positions = s.reference+full
        force, moment = np.zeros(3), np.zeros(3)
        for axis in range(3):
            translation = np.zeros_like(full); translation[:s.n, axis] = 1.
            force[axis] = np.sum(generalized*self.reduction.project(translation))
            rotation = np.cross(np.eye(3)[axis], positions)
            moment[axis] = np.sum(generalized*self.reduction.project(rotation))
        return force, moment

    def fields(self, state, axes):
        s = self.parent
        u, gu = s._sample(s.nodes(self.reduction.expand(state.q)), axes)
        v, gv = s._sample(s.nodes(self.reduction.velocity(state.velocity)), axes)
        X = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
        F = np.eye(3)+gu
        _, P = material_response(F.reshape(-1, 3, 3),
            np.broadcast_to(s.A, F.reshape(-1, 3, 3).shape), s.params)
        return dict(X=X, x=X+u, F=F, velocity=v, velocity_gradient=gv @ np.linalg.inv(F), PK1=P.reshape(F.shape))
