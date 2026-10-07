"""Fixed-reference history plus Q1 increments; diagnostic CPU/float64 only.

The complete compatible position field is retained. No projection of history,
sample reduction, multiplicative spatial update, or stiffness regularization.
"""
from dataclasses import dataclass
import copy

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .consistent_transfer import MaterialQ1, Samples, material_response, minimize_with_backtracking
from .template_remap import RemappedQ1


def frozen(a):
    a = np.array(a, dtype=float, copy=True)
    a.setflags(write=False)
    return a


def material_tangent(F, A, dF, params):
    """Exact Hencky + fiber dP, via the spectral derivative on C=F.T F.

    log divided differences have their continuous repeated-eigenvalue limit.
    This is the unmodified Hessian, including any negative curvature.
    """
    if not np.isfinite(F).all() or np.any(np.linalg.det(F) <= 0):
        raise ValueError('non-positive or non-finite material Jacobian')
    c, Q = np.linalg.eigh(F.transpose(0, 2, 1) @ F)
    logs = np.log(c)
    g = params.mu*logs + .5*params.lam*logs.sum(axis=1)[:, None]
    s = g/c
    delta = c[:, :, None]-c[:, None, :]
    ratio = delta/c[:, None, :]
    logdiv = np.ones_like(delta)
    np.divide(np.log1p(ratio), ratio, out=logdiv, where=ratio != 0)
    logdiv /= c[:, None, :]
    divided = (params.mu*logdiv-s[:, None, :])/c[:, :, None]
    dC = F.transpose(0, 2, 1) @ dF + dF.transpose(0, 2, 1) @ F
    local = Q.transpose(0, 2, 1) @ dC @ Q
    dS = divided*local
    trace = (np.diagonal(local, axis1=1, axis2=2)/c).sum(axis=1)
    idx = np.arange(3)
    dS[:, idx, idx] += .5*params.lam*trace[:, None]/c
    S = (Q*s[:, None, :]) @ Q.transpose(0, 2, 1)
    # S here is twice d(psi_iso)/dC, so P_iso = F S.
    out = dF @ S + F @ Q @ dS @ Q.transpose(0, 2, 1)
    FA = F @ A
    strain = np.einsum('qij,qij->q', FA, F)-1
    contraction = np.einsum('qij,qij->q', FA, dF)
    return out + 2*params.k_f*strain[:, None, None]*(dF @ A) + 4*params.k_f*contraction[:, None, None]*FA


class ReferenceBasis:
    """Owned geometry snapshot, with no solver/history backreferences."""
    def __init__(self, source):
        self.X, self.lo = frozen(source.X), frozen(source.lo)
        self.counts = np.array(source.counts, copy=True)
        self.counts.setflags(write=False)
        self.shape, self.h, self.field = source.shape, source.h, source.field
        self.axes = tuple(frozen(a) for a in source.axes) if hasattr(source, 'axes') else None

    def sample(self, X, weights):
        method = MaterialQ1.sample if self.axes is None else RemappedQ1.sample
        return method(self, X, weights)


@dataclass(frozen=True)
class HistoryField:
    # Each term is (reference basis, position/displacement coefficients).
    # Diagnostic analytic bases evaluate value and reference gradient together.
    terms: tuple

    def evaluate(self, X):
        x = np.zeros((len(X), 3))
        F = np.zeros((len(X), 3, 3))
        for basis, coefficients in self.terms:
            if hasattr(basis, 'evaluate'):
                value, gradient = basis.evaluate(X, coefficients)
                x += value
                F += gradient
                continue
            samples = basis.sample(X, np.ones(len(X)))
            x += samples.N @ coefficients
            F += MaterialQ1.gradient(samples, coefficients)
        return x, F

    def append(self, basis, displacement):
        # Aggregate increments within one template; retain each old template.
        if self.terms[-1][0] is basis:
            return HistoryField(self.terms[:-1] + ((basis, frozen(self.terms[-1][1]+displacement)),))
        return HistoryField(self.terms + ((basis, frozen(displacement)),))

    @property
    def coefficient_bytes(self):
        return sum(c.nbytes for _, c in self.terms)


@dataclass(frozen=True)
class MaterialSites:
    X: np.ndarray
    weight: np.ndarray
    A: np.ndarray

    @classmethod
    def from_samples(cls, samples):
        return cls(frozen(samples.X), frozen(samples.weight), frozen(samples.A))


@dataclass(frozen=True)
class HistoryState:
    particles: MaterialSites
    quadrature: MaterialSites
    mass: np.ndarray
    field: HistoryField
    xp: np.ndarray
    Fp: np.ndarray
    xq: np.ndarray
    Fq: np.ndarray

    @classmethod
    def from_nodal(cls, source, positions):
        if np.shape(positions) != source.X.shape or not np.isfinite(positions).all():
            raise ValueError('finite nodal position field required')
        field = HistoryField(((ReferenceBasis(source), frozen(positions)),))
        xp, Fp = field.evaluate(source.particles.X)
        xq, Fq = field.evaluate(source.quadrature.X)
        material_response(Fp, source.particles.A, source.params)
        material_response(Fq, source.quadrature.A, source.params)
        return cls(MaterialSites.from_samples(source.particles), MaterialSites.from_samples(source.quadrature),
                   frozen(source.mass), field, frozen(xp), frozen(Fp), frozen(xq), frozen(Fq))


class HistoryIncrementalQ1:
    """One frozen motion template bound to unchanged authoritative material sites."""
    def __init__(self, source, state, axes=None):
        basis = ReferenceBasis(source)
        if axes is not None:
            axes = tuple(frozen(a) for a in axes)
            if len(axes) != 3 or any(a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
                                    or np.any(np.diff(a) <= 0) for a in axes):
                raise ValueError('three finite strictly increasing knot arrays required')
            low, high = source.X.min(axis=0), source.X.max(axis=0)
            if any(abs(a[0]-lo) > 1e-12 or abs(a[-1]-hi) > 1e-12 for a, lo, hi in zip(axes, low, high)):
                raise ValueError('template must preserve the material reference domain')
            import itertools
            basis.axes = axes
            basis.X = frozen(list(itertools.product(*axes)))
            basis.counts = np.array([len(a)-1 for a in axes])
            basis.counts.setflags(write=False)
            basis.shape = tuple(basis.counts+1)
        self.basis, self.X = basis, basis.X
        self.params = copy.deepcopy(source.params)
        self.sites = (state.particles, state.quadrature, state.mass)
        self.particles = self._map(state.particles)
        self.quadrature = self._map(state.quadrature)
        self.mass = state.mass
        self.fixed = np.isclose(self.X[:, 0], source.lo[0], atol=1e-14, rtol=0)
        self.free = ~self.fixed
        N = self.particles.N
        self.M = (N.T @ N.multiply(self.mass[:, None])).tocsc()
        self.lumped = np.asarray(N.T @ self.mass).ravel()
        # Dense eigensolve is intentional for this small diagnostic prototype.
        self.mass_eigenvalues = np.linalg.eigvalsh(self.M.toarray())
        if self.mass_eigenvalues[0] <= 1e-12*self.mass_eigenvalues[-1]:
            raise ValueError('new template mass is rank deficient; no regularization')
        self.factor = splu(self.M)
        self.free_factor = splu(self.M[self.free][:, self.free].tocsc())

    def _map(self, sites):
        samples = self.basis.sample(sites.X, sites.weight)
        return Samples(sites.X, sites.weight, samples.N, samples.D, sites.A)

    def _check_state(self, state):
        if any(a is not b for a, b in zip(self.sites, (state.particles, state.quadrature, state.mass))):
            raise ValueError('state must retain the same authoritative material sites')

    gradient = staticmethod(MaterialQ1.gradient)
    project = MaterialQ1.project

    def force(self, P):
        return sum(D.T @ (self.quadrature.weight[:, None]*P[:, :, d])
                   for d, D in enumerate(self.quadrature.D))

    def elastic(self, state, displacement=None):
        self._check_state(state)
        F = state.Fq if displacement is None else state.Fq+self.gradient(self.quadrature, displacement)
        psi, P = material_response(F, self.quadrature.A, self.params)
        return float(self.quadrature.weight @ psi), self.force(P)

    def potential(self, v, state, predictor, dt):
        self._check_dt(dt)
        U, force = self.elastic(state, dt*v)
        dv = v-predictor
        return U+.5*float(np.sum(dv*(self.M @ dv))), self.M @ dv+dt*force

    def tangent(self, v, state, direction, dt, include_mass=True):
        self._check_state(state)
        self._check_dt(dt)
        F = state.Fq+dt*self.gradient(self.quadrature, v)
        dP = material_tangent(F, self.quadrature.A, self.gradient(self.quadrature, direction), self.params)
        return (self.M @ direction if include_mass else 0)+dt**2*self.force(dP)

    def stiffness(self, state):
        """Sparse exact elastic Hessian at the history state, with no mass."""
        self._check_state(state)
        nq, nn = len(state.Fq), len(self.X)
        H = np.stack([material_tangent(state.Fq, self.quadrature.A,
                                      np.broadcast_to(e, state.Fq.shape), self.params).reshape(nq, 9)
                      for e in np.eye(9).reshape(9, 3, 3)], axis=2)
        rows, cols, data = [], [], []
        for j, D in enumerate(self.quadrature.D):
            coo = D.tocoo()
            for i in range(3):
                rows.append(coo.row*9+i*3+j)
                cols.append(coo.col*3+i)
                data.append(coo.data)
        B = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nq*9, nn*3)).tocsr()
        WH = sp.bsr_matrix((H*self.quadrature.weight[:, None, None], np.arange(nq), np.arange(nq+1)),
                           shape=(nq*9, nq*9))
        return (B.T @ WH @ B).tocsr()

    @staticmethod
    def _check_dt(dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')

    def commit(self, state, v, dt):
        self._check_state(state)
        self._check_dt(dt)
        if v.shape != self.X.shape or not np.isfinite(v).all() or np.any(v[self.fixed] != 0):
            raise ValueError('finite nodal velocity with zero clamp increment required')
        du = dt*v
        xp = state.xp+self.particles.N @ du
        Fp = state.Fp+self.gradient(self.particles, du)
        xq = state.xq+self.quadrature.N @ du
        Fq = state.Fq+self.gradient(self.quadrature, du)
        # Reject atomically before creating a new committed state.
        material_response(Fp, self.particles.A, self.params)
        material_response(Fq, self.quadrature.A, self.params)
        return HistoryState(state.particles, state.quadrature, state.mass, state.field.append(self.basis, du),
                            frozen(xp), frozen(Fp), frozen(xq), frozen(Fq))

    def step(self, state, vp, dt):
        self._check_state(state)
        self._check_dt(dt)
        if vp.shape != state.xp.shape or not np.isfinite(vp).all():
            raise ValueError('finite particle velocities required')
        predictor = self.project(vp, clamped=True)
        Uold = self.elastic(state)[0]
        Kold = .5*float(np.sum(self.mass[:, None]*vp**2))
        scale = np.sqrt(self.lumped[self.free, None])
        def objective(y):
            v = np.zeros_like(self.X)
            v[self.free] = y.reshape(-1, 3)/scale
            try:
                value, grad = self.potential(v, state, predictor, dt)
            except ValueError:
                return np.inf, np.zeros_like(y)
            return value, (grad[self.free]/scale).ravel()
        solution, iterations, backtracks = minimize_with_backtracking(objective, (predictor[self.free]*scale).ravel())
        value, grad = objective(solution)
        residual = float(np.linalg.norm(grad, ord=np.inf))
        if not np.isfinite(value) or residual > 1e-7:
            raise RuntimeError(f'history increment solve failed: residual={residual:g}')
        v = np.zeros_like(self.X)
        v[self.free] = solution.reshape(-1, 3)/scale
        next_state = self.commit(state, v, dt)
        velocity = self.particles.N @ v
        U, force = self.elastic(next_state)
        K = .5*float(np.sum(self.mass[:, None]*velocity**2))
        dv = v-predictor
        budget = dict(projection_delta=.5*float(np.sum(predictor*(self.M @ predictor)))-Kold,
                      inertia_remainder=-.5*float(np.sum(dv*(self.M @ dv))),
                      elastic_remainder=U-Uold-dt*float(np.sum(force*v)),
                      solver_work=float(np.sum(v*(self.M @ dv+dt*force))),
                      kinetic_readback_delta=K-.5*float(np.sum(v*(self.M @ v))))
        budget['energy_budget_residual'] = U+K-Uold-Kold-sum(budget.values())
        info = dict(elastic=U, kinetic=K, mechanical=U+K, iterations=iterations,
                    backtracks=backtracks, scaled_residual_inf=residual,
                    min_det=float(min(np.linalg.det(next_state.Fp).min(), np.linalg.det(next_state.Fq).min())),
                    clamp_speed=float(np.max(np.abs(v[self.fixed]))),
                    history_terms=len(next_state.field.terms), history_coefficient_bytes=next_state.field.coefficient_bytes,
                    **budget)
        return next_state, velocity, info
