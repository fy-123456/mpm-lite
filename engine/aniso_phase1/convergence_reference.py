"""Fixed-space diagnostics for time convergence and common-state references.

No production defaults change. The anchored residual space remains fixed for
the complete post-switch trajectory; commits add its field to current history.
"""
import itertools
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular, cho_solve

from .consistent_transfer import MaterialQ1, material_response, minimize_with_backtracking
from .history_increment import HistoryIncrementalQ1, HistoryField, HistoryState, MaterialSites, ReferenceBasis, frozen
from .residual_enrichment import ResidualEnrichedQ1
from .types import AnisotropicMaterialParams


def geometry(grid):
    """Only reference geometry; mass is built later on the shared sample set."""
    if grid < 9 or (grid-1) % 8:
        raise ValueError('grid must be 8*k+1, at least 9')
    m = MaterialQ1.__new__(MaterialQ1)
    m.h = 1/(grid-1)
    m.lo = np.array([.25, .4375, .4375])
    m.counts = np.array([(grid-1)//2, (grid-1)//8, (grid-1)//8])
    m.shape = tuple(m.counts+1)
    m.X = np.array(list(itertools.product(*[m.lo[d]+m.h*np.arange(m.shape[d]) for d in range(3)])))
    m.fixed = m.X[:, 0] == m.lo[0]
    m.free = ~m.fixed
    m.field = 'smooth'
    m.params = AnisotropicMaterialParams(10., 20., 200.)
    return m


def knots(source, shifted=False):
    axes = [np.unique(source.X[:, d]) for d in range(3)]
    if shifted:
        axes[0][len(axes[0])//2] += .2*source.h
    return axes


def union_knots(*groups):
    return [np.unique(np.round(np.concatenate([g[d] for g in groups]), 14)) for d in range(3)]


def tensor_rule(axes, order):
    if order < 2:
        raise ValueError('full integration requires order >= 2')
    z, w = np.polynomial.legendre.leggauss(order)
    q = np.array(list(itertools.product(z, repeat=3)))
    ww = np.prod(np.array(list(itertools.product(w, repeat=3))), axis=1)/8
    cells = np.array(list(itertools.product(*[range(len(a)-1) for a in axes])))
    lo = np.column_stack([a[cells[:, d]] for d, a in enumerate(axes)])
    hi = np.column_stack([a[cells[:, d]+1] for d, a in enumerate(axes)])
    X = ((lo[:, None]+hi[:, None])/2+(hi-lo)[:, None]*q[None]/2).reshape(-1, 3)
    weights = (np.prod(hi-lo, axis=1)[:, None]*ww).ravel()
    return X, weights


def directions(X):
    angle = (np.asarray(X)[:, 1]-.4375)/.125*np.pi/2
    a = np.column_stack([np.cos(angle), np.sin(angle), np.zeros(len(X))])
    return np.einsum('qi,qj->qij', a, a)


def sites(X, weight):
    return MaterialSites(frozen(X), frozen(weight), frozen(directions(X)))


def state_from_field(field, particles, quadrature):
    xp, Fp = field.evaluate(particles.X)
    xq, Fq = field.evaluate(quadrature.X)
    return HistoryState(particles, quadrature, particles.weight, field, frozen(xp), frozen(Fp), frozen(xq), frozen(Fq))


def sum_fields(*fields):
    terms = []
    for field in fields:
        for basis, coefficients in field.terms:
            for i, (old, values) in enumerate(terms):
                same = (old.same_basis(basis) if hasattr(old, 'same_basis') else
                        type(old) is type(basis) and np.array_equal(old.X, basis.X))
                if same:
                    terms[i] = old, values+coefficients
                    break
            else:
                terms.append((basis, coefficients))
    return HistoryField(tuple((b, frozen(c)) for b, c in terms))


class FrozenResidualQ1(ResidualEnrichedQ1):
    """Anchor e at the switch; keep mass, basis and boundary nullspace fixed."""
    def _check_state(self, state):
        HistoryIncrementalQ1._check_state(self, state)

    def mass_solve(self, rhs):
        return cho_solve(self.reduced_factor, rhs)

    def mass_cholesky(self):
        if not hasattr(self, '_mass_lower'):
            self._mass_lower = np.tril(self.reduced_factor[0])
        return self._mass_lower

    def mass_to_y(self, z):
        return self.mass_cholesky().T @ z

    def mass_from_y(self, y):
        return solve_triangular(self.mass_cholesky().T, y, lower=False, check_finite=False)

    def mass_force_to_y(self, force):
        return solve_triangular(self.mass_cholesky(), force, lower=True, check_finite=False)

    def coefficient_field(self, coefficients):
        C = (self.transform @ coefficients[self.nnode:]).T
        correction = coefficients[:self.nnode]-self.xbar @ C.T
        anchor_terms = HistoryField(tuple((b, frozen(c @ C.T)) for b, c in self.state.field.terms))
        return sum_fields(anchor_terms, HistoryField(((self.basis, frozen(correction)),)))

    def commit(self, state, v, dt):
        self._check_state(state)
        self._check_dt(dt)
        if v.shape != (self.ncoeff, 3) or not np.isfinite(v).all():
            raise ValueError('finite frozen-space coefficients required')
        if self.clamped and np.max(abs(self.boundary_N @ v)) > 1e-9:
            raise ValueError('total clamp increment must vanish')
        xp = state.xp+dt*(self.particles.N @ v)
        Fp = state.Fp+dt*self.gradient(self.particles, v)
        xq = state.xq+dt*(self.quadrature.N @ v)
        Fq = state.Fq+dt*self.gradient(self.quadrature, v)
        material_response(Fp, self.particles.A, self.params)
        material_response(Fq, self.quadrature.A, self.params)
        field = sum_fields(state.field, self.coefficient_field(dt*v))
        return HistoryState(state.particles, state.quadrature, state.mass, field,
                            frozen(xp), frozen(Fp), frozen(xq), frozen(Fq))

    def step(self, state, vp, dt, tolerance=1e-9):
        """Same L-BFGS/Armijo, with an exact mass change of coordinates.

        Mr=L L.T, y=L.T z. This removes constant mass conditioning, without
        modifying mass, potential, stiffness, or the time integration method.
        """
        self._check_state(state)
        self._check_dt(dt)
        if vp.shape != state.xp.shape or not np.isfinite(vp).all():
            raise ValueError('finite particle velocities required')
        rhs = self.R.T @ (self.particles.N.T @ (self.mass[:, None]*vp))
        zhat = self.mass_solve(rhs)
        predictor = self.R @ zhat
        yhat = self.mass_to_y(zhat)
        Uold = self.elastic(state)[0]
        Kold = .5*float(np.sum(self.mass[:, None]*vp**2))
        def objective(y):
            z = self.mass_from_y(y.reshape(-1, 3))
            v = self.R @ z
            try:
                U, force = self.elastic(state, dt*v)
            except ValueError:
                return np.inf, np.zeros_like(y)
            dy = y.reshape(-1, 3)-yhat
            grad = dy+dt*self.mass_force_to_y(self.R.T @ force)
            return U+.5*float(np.sum(dy**2)), grad.ravel()
        solution, iterations, backtracks = minimize_with_backtracking(objective, yhat.ravel(), tolerance=tolerance)
        value, grad = objective(solution)
        residual = float(np.max(abs(grad)))
        if not np.isfinite(value) or residual > tolerance*1.01:
            raise RuntimeError('frozen-space solve did not converge')
        z = self.mass_from_y(solution.reshape(-1, 3))
        v = self.R @ z
        next_state = self.commit(state, v, dt)
        velocity = self.particles.N @ v
        self.last_velocity = self.coefficient_field(v)
        U, force = self.elastic(state, dt*v)
        K = .5*float(np.sum(self.mass[:, None]*velocity**2))
        dv = v-predictor
        budget = dict(projection_delta=.5*float(np.sum(predictor*(self.M @ predictor)))-Kold,
                      inertia_remainder=-.5*float(np.sum(dv*(self.M @ dv))),
                      elastic_remainder=U-Uold-dt*float(np.sum(force*v)),
                      solver_work=float(np.sum(v*(self.M @ dv+dt*force))),
                      kinetic_readback_delta=K-.5*float(np.sum(v*(self.M @ v))))
        budget['energy_budget_residual'] = U+K-Uold-Kold-sum(budget.values())
        info = dict(elastic=U, kinetic=K, mechanical=U+K, iterations=iterations, backtracks=backtracks,
                    scaled_residual_inf=residual,
                    min_det=float(min(np.linalg.det(next_state.Fp).min(), np.linalg.det(next_state.Fq).min())),
                    clamp_speed=float(np.max(abs(self.boundary_N @ v))) if self.clamped else 0.,
                    history_terms=len(next_state.field.terms), **budget)
        return next_state, velocity, info


def save_fields(path, **fields):
    """Small, pickle-free restart/evaluation artifact; no sampled point clouds."""
    arrays = {}
    for name, field in fields.items():
        arrays[name+'_count'] = len(field.terms)
        for i, (b, c) in enumerate(field.terms):
            prefix = f'{name}_{i}_'
            arrays[prefix+'coefficients'] = c
            if getattr(b, 'kind', '') == 'global-polynomial-v1':
                arrays[prefix+'kind'] = b.kind
                arrays[prefix+'degrees'] = b.degrees
                arrays[prefix+'origin'] = b.origin
                arrays[prefix+'scale'] = b.scale
                continue
            arrays[prefix+'h'] = b.h
            arrays[prefix+'regular'] = b.axes is None
            for d in range(3):
                arrays[prefix+f'axis{d}'] = np.unique(b.X[:, d])
    np.savez_compressed(Path(path), **arrays)


def load_fields(path, names=('position', 'velocity')):
    out = {}
    with np.load(path, allow_pickle=False) as arrays:
        for name in names:
            terms = []
            for i in range(int(arrays[name+'_count'])):
                prefix = f'{name}_{i}_'
                if prefix+'kind' in arrays:
                    from .smooth_history import PolynomialHistoryBasis
                    if str(arrays[prefix+'kind']) != PolynomialHistoryBasis.kind:
                        raise ValueError('unknown analytic history basis')
                    b = PolynomialHistoryBasis(arrays[prefix+'degrees'], arrays[prefix+'origin'], arrays[prefix+'scale'])
                    terms.append((b, frozen(arrays[prefix+'coefficients'])))
                    continue
                axes = tuple(frozen(arrays[prefix+f'axis{d}']) for d in range(3))
                b = ReferenceBasis.__new__(ReferenceBasis)
                b.X = frozen(list(itertools.product(*axes)))
                b.lo = frozen([a[0] for a in axes])
                b.shape = tuple(len(a) for a in axes)
                b.counts = np.array(b.shape)-1
                b.h, b.field = float(arrays[prefix+'h']), 'smooth'
                b.axes = None if bool(arrays[prefix+'regular']) else axes
                terms.append((b, frozen(arrays[prefix+'coefficients'])))
            out[name] = HistoryField(tuple(terms))
    return out


def rms(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    return float(np.sqrt(np.sum(weights*np.sum(values.reshape(len(weights), -1)**2, axis=1))/weights.sum()))


def compare_fields(position, velocity, ref_position, ref_velocity, X, weights, params):
    x, F = position.evaluate(X)
    xr, Fr = ref_position.evaluate(X)
    v, _ = velocity.evaluate(X)
    vr, _ = ref_velocity.evaluate(X)
    A = directions(X)
    P = material_response(F, A, params)[1]
    Pr = material_response(Fr, A, params)[1]
    errors = dict(x=x-xr, F=F-Fr, P=P-Pr, v=v-vr)
    masks = dict(all=np.ones(len(X), dtype=bool), clamp=X[:, 0] < .3125,
                 switch=(X[:, 0] >= .4375) & (X[:, 0] <= .5625))
    masks['bulk'] = ~(masks['clamp'] | masks['switch'])
    report = {}
    for name, mask in masks.items():
        w = weights[mask]
        row = dict(volume_fraction=float(w.sum()/weights.sum()))
        for key, error in errors.items():
            total = float(np.sum(weights*np.sum(error.reshape(len(X), -1)**2, axis=1)))
            squared = float(np.sum(w*np.sum(error[mask].reshape(mask.sum(), -1)**2, axis=1)))
            row[key+'_rms'] = rms(error[mask], w)
            row[key+'_squared_error_share'] = squared/total if total > 0 else 0.
        report[name] = row
    report['normalizers'] = dict(displacement_rms=rms(xr-X, weights), F_minus_I_rms=rms(Fr-np.eye(3), weights),
                                 P_rms=rms(Pr, weights), v_rms=rms(vr, weights))
    # Spatial profiles are per-bin RMS values, not unweighted point averages.
    edges = np.linspace(.25, .75, 33)
    bins = np.clip(np.searchsorted(edges, X[:, 0], side='right')-1, 0, 31)
    wsum = np.bincount(bins, weights=weights, minlength=32)
    profile = dict(x=((edges[:-1]+edges[1:])/2).tolist())
    for key, error in errors.items():
        squared = np.sum(error.reshape(len(X), -1)**2, axis=1)
        profile[key+'_rms'] = np.sqrt(np.bincount(bins, weights=weights*squared, minlength=32)/wsum).tolist()
    report['profile'] = profile
    return report
