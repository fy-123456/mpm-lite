"""Shape-residual motion enrichment on frozen material sites (CPU reference).

Scalar residual functions span the rows of C e. A mass-weighted SVD removes
dependent functions; each retained scalar function has three vector DOFs.
Modes are frozen for one step. History commits are exact field algebra.
"""
import itertools

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, null_space

from .consistent_transfer import Samples, material_response, minimize_with_backtracking
from .history_increment import HistoryIncrementalQ1, HistoryState, HistoryField, frozen, material_tangent


class ResidualEnrichedQ1(HistoryIncrementalQ1):
    def __init__(self, source, state, axes=None, *, enrich=True, clamped=True,
                 rank_rtol=1e-10, rank_atol=1e-12):
        super().__init__(source, state, axes)
        if not 0 < rank_rtol < 1 or not 0 < rank_atol < 1:
            raise ValueError('rank tolerances must be between zero and one')
        self.state, self.clamped = state, clamped
        self.nodal_particles, self.nodal_quadrature = self.particles, self.quadrature
        self.nodal_mass_eigenvalues = self.mass_eigenvalues
        self.nnode = len(self.X)
        self.xbar = frozen(super().project(state.xp))
        ep = state.xp-self.particles.N @ self.xbar
        weighted = np.sqrt(self.mass[:, None])*ep
        _, singular, Vt = np.linalg.svd(weighted, full_matrices=False)
        length = np.linalg.norm(np.ptp(self.X, axis=0))
        cutoff = max(rank_rtol*singular[0], rank_atol*np.sqrt(self.mass.sum())*length)
        keep = singular > cutoff if enrich else np.zeros(3, dtype=bool)
        # Unit mass-averaged norm gives dimensionless scalar basis functions.
        self.transform = frozen(Vt[keep].T/ singular[keep]*np.sqrt(self.mass.sum()))
        self.rank = int(keep.sum())
        self.ncoeff = self.nnode+self.rank
        self.rank_info = dict(singular_values=singular.tolist(), cutoff=float(cutoff),
                              scalar_modes=self.rank, vector_dofs=3*self.rank)
        self.particles = self.map_sites(state.particles)
        self.quadrature = self.map_sites(state.quadrature)
        # A particle-null residual may still have observable gradients. Reject
        # it rather than silently declaring sampled rank to be physical rank.
        discarded = np.eye(3)-Vt[keep].T @ Vt[keep]
        dq = state.Fq-self.gradient(self.nodal_quadrature, self.xbar)
        loss = np.einsum('ij,qjk->qik', discarded, dq)
        self.rank_info['discarded_gradient_relative'] = float(np.linalg.norm(loss)/np.linalg.norm(state.Fq))
        if enrich and self.rank_info['discarded_gradient_relative'] > 1e-9:
            raise ValueError('particle rank cannot resolve material history gradients')
        N = self.particles.N
        self.M = (N.T @ N.multiply(self.mass[:, None])).tocsc()
        self.mass_eigenvalues = np.linalg.eigvalsh(self.M.toarray())
        self.full_factor = cho_factor(self.M.toarray(), lower=True)
        self.boundary_X = self._boundary_points()
        self.boundary_N = self.map_points(self.boundary_X).N
        self.R = self._constraint_transform()
        self.Mr = (self.R.T @ self.M @ self.R).tocsc()
        self.reduced_factor = cho_factor(self.Mr.toarray(), lower=True)
        # Scaled rank check avoids basis-unit dependence; no diagonal shift.
        diag = np.sqrt(self.Mr.diagonal())
        scaled = self.Mr.toarray()/diag[:, None]/diag[None, :]
        eigen = np.linalg.eigvalsh(scaled)
        if eigen[0] <= 1e-12*eigen[-1]:
            raise ValueError('constrained enriched mass is rank deficient')
        self.rank_info.update(reduced_scalar_dofs=self.R.shape[1], min_scaled_mass_eigenvalue=float(eigen[0]),
                              boundary_null_residual=float(np.max(abs((self.boundary_N @ self.R).toarray()))) if clamped else 0.)

    def _check_state(self, state):
        super()._check_state(state)
        if state is not self.state:
            raise ValueError('rebuild frozen residual modes for the new history state')

    def map_points(self, X, weights=None, A=None):
        X = np.asarray(X)
        weights = np.ones(len(X)) if weights is None else weights
        base = self.basis.sample(X, weights)
        x, F = self.state.field.evaluate(X)
        e = (x-base.N @ self.xbar) @ self.transform
        de = np.einsum('qij,ik->qkj', F-self.gradient(base, self.xbar), self.transform)
        N = sp.hstack((base.N, sp.csr_matrix(e)), format='csr')
        D = tuple(sp.hstack((d, sp.csr_matrix(de[:, :, j])), format='csr') for j, d in enumerate(base.D))
        return Samples(X, weights, N, D, base.A if A is None else A)

    def map_sites(self, sites):
        return self.map_points(sites.X, sites.weight, sites.A)

    def _boundary_points(self):
        # All old/new trace knots: values at these union-grid vertices determine
        # the entire piecewise bilinear boundary, not just new boundary nodes.
        bases = [self.basis]+[b for b, _ in self.state.field.terms]
        axes = [np.unique(np.concatenate([np.unique(b.X[:, d]) for b in bases])) for d in (1, 2)]
        return np.array([[self.X[:, 0].min(), y, z] for y, z in itertools.product(*axes)])

    def _constraint_transform(self):
        if not self.clamped:
            return sp.eye(self.ncoeff, format='csr')
        free = np.flatnonzero(self.free)
        fixed = np.flatnonzero(self.fixed)
        Ef = self.map_points(self.X[fixed]).N[:, self.nnode:].toarray()
        # Eliminate nodal boundary values against the enrichment contribution.
        # Remaining trace constraints involve at most three scalar modes.
        trace = self.boundary_N[:, self.nnode:].toarray()-self.boundary_N[:, fixed] @ Ef
        if self.rank == 0 or np.linalg.norm(trace) < 1e-10:
            Z = np.eye(self.rank)
        else:
            Z = null_space(trace, rcond=1e-10)
        R = np.zeros((self.ncoeff, len(free)+Z.shape[1]))
        R[free, np.arange(len(free))] = 1
        R[fixed, len(free):] = -Ef @ Z
        R[self.nnode:, len(free):] = Z
        if np.max(abs(self.boundary_N @ R), initial=0.) > 1e-9:
            raise ValueError('total boundary trace constraint failed')
        return sp.csr_matrix(R)

    def project(self, values, clamped=None):
        clamped = self.clamped if clamped is None else clamped
        rhs = self.particles.N.T @ (self.mass[:, None]*values)
        if clamped:
            if not self.clamped:
                raise ValueError('clamped projection requires a constrained model')
            return self.R @ cho_solve(self.reduced_factor, self.R.T @ rhs)
        return cho_solve(self.full_factor, rhs)

    def stiffness(self, state):
        self._check_state(state)
        nq = len(state.Fq)
        H = np.stack([material_tangent(state.Fq, self.quadrature.A, np.broadcast_to(e, state.Fq.shape),
                                      self.params).reshape(nq, 9) for e in np.eye(9).reshape(9, 3, 3)], axis=2)
        rows, cols, data = [], [], []
        for j, D in enumerate(self.quadrature.D):
            coo = D.tocoo()
            for i in range(3):
                rows.append(coo.row*9+3*i+j); cols.append(coo.col*3+i); data.append(coo.data)
        B = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nq*9, self.ncoeff*3)).tocsr()
        WH = sp.bsr_matrix((H*self.quadrature.weight[:, None, None], np.arange(nq), np.arange(nq+1)), shape=(nq*9, nq*9))
        return (B.T @ WH @ B).tocsr()

    def commit(self, state, v, dt):
        self._check_state(state)
        self._check_dt(dt)
        if v.shape != (self.ncoeff, 3) or not np.isfinite(v).all():
            raise ValueError('finite enriched coefficients required')
        if self.clamped and np.max(abs(self.boundary_N @ v)) > 1e-9:
            raise ValueError('total clamp increment must vanish')
        du = dt*v
        xp, xq = state.xp+self.particles.N @ du, state.xq+self.quadrature.N @ du
        Fp, Fq = state.Fp+self.gradient(self.particles, du), state.Fq+self.gradient(self.quadrature, du)
        material_response(Fp, self.particles.A, self.params)
        material_response(Fq, self.quadrature.A, self.params)
        C = (self.transform @ du[self.nnode:]).T
        terms = [(basis, coefficients @ (np.eye(3)+C).T) for basis, coefficients in state.field.terms]
        correction = du[:self.nnode]-self.xbar @ C.T
        for i, (basis, coefficients) in enumerate(terms):
            if np.array_equal(basis.X, self.basis.X):
                terms[i] = (basis, coefficients+correction)
                break
        else:
            terms.append((self.basis, correction))
        field = HistoryField(tuple((b, frozen(c)) for b, c in terms))
        return HistoryState(state.particles, state.quadrature, state.mass, field,
                            frozen(xp), frozen(Fp), frozen(xq), frozen(Fq))

    def step(self, state, vp, dt, tolerance=1e-9):
        self._check_state(state)
        self._check_dt(dt)
        if vp.shape != state.xp.shape or not np.isfinite(vp).all():
            raise ValueError('finite particle velocities required')
        rhs = self.R.T @ (self.particles.N.T @ (self.mass[:, None]*vp))
        reduced_predictor = cho_solve(self.reduced_factor, rhs)
        predictor = self.R @ reduced_predictor
        Uold = self.elastic(state)[0]
        Kold = .5*float(np.sum(self.mass[:, None]*vp**2))
        scale = np.sqrt(self.Mr.diagonal())[:, None]
        def objective(y):
            v = self.R @ (y.reshape(-1, 3)/scale)
            try:
                value, grad = self.potential(v, state, predictor, dt)
            except ValueError:
                return np.inf, np.zeros_like(y)
            return value, ((self.R.T @ grad)/scale).ravel()
        solution, iterations, backtracks = minimize_with_backtracking(objective,
                (reduced_predictor*scale).ravel(), tolerance=tolerance)
        value, grad = objective(solution)
        residual = float(np.linalg.norm(grad, ord=np.inf))
        if not np.isfinite(value) or residual > max(1e-9, tolerance*2):
            raise RuntimeError(f'enriched solve failed: residual={residual:g}')
        v = self.R @ (solution.reshape(-1, 3)/scale)
        next_state = self.commit(state, v, dt)
        velocity = self.particles.N @ v
        # Evaluate the trial through the frozen field, not a rebuilt next-step basis.
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
                    scaled_residual_inf=residual, min_det=float(min(np.linalg.det(next_state.Fp).min(), np.linalg.det(next_state.Fq).min())),
                    clamp_speed=float(np.max(abs(self.boundary_N @ v))) if self.clamped else 0.,
                    history_terms=len(next_state.field.terms), history_coefficient_bytes=next_state.field.coefficient_bytes,
                    residual_scalar_modes=self.rank, **budget)
        return next_state, velocity, info
