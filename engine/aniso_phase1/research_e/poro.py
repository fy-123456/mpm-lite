"""Small-strain quasistatic Biot, full Q2 elasticity + RT0 flux + P0 pressure.

A u - G.T p = load
G (u-u_old) + C (p-p_old) + dt B flux = dt source_volume
H flux - B.T p = pressure/gravity boundary RHS

No pressure penalty/stabilization is added to the monolithic equations.
Mechanical DOFs are total displacements. Only homogeneous mechanical clamps
and pressure / zero-flux fluid boundaries are supported in this prototype.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .flow import quadrature, solve_checked
from .material import linear_stress


def q2_basis(s, h):
    N = [np.array([2*x*x-3*x+1, 4*x-4*x*x, 2*x*x-x]) for x in s]
    dN = [np.array([4*x-3, 4-8*x, 4*x-1])/h[d] for d,x in enumerate(s)]
    ids = list(product(range(3), repeat=len(s)))
    values = np.array([np.prod([N[d][i[d]] for d in range(len(s))]) for i in ids])
    grad = np.array([[np.prod([dN[k][i[k]] if k == d else N[k][i[k]] for k in range(len(s))])
                      for d in range(len(s))] for i in ids])
    return values, grad


class Skeleton:
    def __init__(self, grid, material, alpha=1., clamp=None):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1: raise ValueError('0<=Biot alpha<=1 required')
        if material.mu <= 0 or material.lam+2*material.mu/3 <= 0: raise ValueError('stable elastic matrix required')
        self.grid, self.material, self.alpha = grid, material, float(alpha)
        d = grid.dim; self.node_shape = tuple(2*n+1 for n in grid.shape)
        self.nodes = np.array(list(np.ndindex(self.node_shape)))*grid.h/2
        self.ndof = len(self.nodes)*d
        loc = np.array(list(product(range(3), repeat=d)))
        cn = np.array([[np.ravel_multi_index(tuple(2*ix+j), self.node_shape) for j in loc] for ix in grid.indices])
        self.cell_nodes = cn
        self.cell_dofs = (cn[:, :, None]*d+np.arange(d)).reshape(grid.nc, -1)
        nloc = self.cell_dofs.shape[1]
        local = np.zeros((nloc, nloc)); div = np.zeros(nloc)
        self.integration_points = grid.nc*3**d
        for s, w in quadrature(d, 3):
            _, grad = q2_basis(s, grid.h)
            modes = np.zeros((nloc, d, d))
            for i in range(len(loc)):
                for component in range(d): modes[i*d+component, component] = grad[i]
            stress = linear_stress(modes, material)
            local += grid.volume*w*np.einsum('aij,bij->ab', modes, stress)
            div += alpha*grid.volume*w*np.trace(modes, axis1=1, axis2=2)
        rows = np.broadcast_to(self.cell_dofs[:, :, None], (grid.nc, nloc, nloc)).ravel()
        cols = np.broadcast_to(self.cell_dofs[:, None, :], (grid.nc, nloc, nloc)).ravel()
        self.A = sp.coo_matrix((np.tile(local.ravel(), grid.nc), (rows, cols)), shape=(self.ndof, self.ndof)).tocsc()
        self.G = sp.coo_matrix((np.tile(div, grid.nc),
                  (np.repeat(np.arange(grid.nc), nloc), self.cell_dofs.ravel())), shape=(grid.nc, self.ndof)).tocsr()
        clamp = (lambda x, component: abs(x[0]) < 1e-12) if clamp is None else clamp
        self.fixed = np.array([i*d+c for i,x in enumerate(self.nodes) for c in range(d) if clamp(x, c)], int)
        self.free = np.setdiff1d(np.arange(self.ndof), self.fixed)
        if not len(self.fixed): raise ValueError('mechanical rigid modes must be constrained')
        self.Af = self.A[self.free][:, self.free].tocsc(); self.Gf = self.G[:, self.free]
        self.factor = splu(self.Af)

    def traction(self, axis, side, value):
        """Integrate a constant vector traction using the same Q2 trace."""
        g = self.grid; value = np.asarray(value, float)
        if value.shape != (g.dim,) or not np.isfinite(value).all(): raise ValueError('finite traction vector required')
        rhs = np.zeros(self.ndof)
        for c, ix in enumerate(g.indices):
            if ix[axis] != (g.shape[axis]-1 if side else 0): continue
            face_q = [(np.array([]), 1.)] if g.dim == 1 else quadrature(g.dim-1, 3)
            for tangential, weight in face_q:
                s = np.empty(g.dim); s[axis] = side; s[np.arange(g.dim) != axis] = tangential
                N, _ = q2_basis(s, g.h)
                np.add.at(rhs, self.cell_dofs[c], (N[:, None]*value).ravel()*g.areas[axis]*weight)
        return rhs

    def body_force(self, function):
        rhs = np.zeros(self.ndof)
        for c, ix in enumerate(self.grid.indices):
            for point, weight in quadrature(self.grid.dim, 4):
                value = np.asarray(function((ix+point)*self.grid.h), float)
                if value.shape != (self.grid.dim,) or not np.isfinite(value).all():
                    raise ValueError('finite body force vector required')
                N, _ = q2_basis(point, self.grid.h)
                np.add.at(rhs, self.cell_dofs[c], (N[:, None]*value).ravel()*self.grid.volume*weight)
        return rhs

    def equilibrate(self, pressure, load):
        u = np.zeros(self.ndof)
        u[self.free] = self.factor.solve((np.asarray(load)+self.G.T @ pressure)[self.free])
        return u

    def cell_gradients(self, u):
        _, grad = q2_basis(np.full(self.grid.dim, .5), self.grid.h)
        return np.asarray(u)[self.cell_dofs].reshape(self.grid.nc, -1, self.grid.dim).transpose(0, 2, 1) @ grad

    def cell_stress(self, u, p):
        return linear_stress(self.cell_gradients(u), self.material)-self.alpha*np.asarray(p)[:, None, None]*np.eye(self.grid.dim)


@dataclass
class PoroState:
    u: np.ndarray
    p: np.ndarray
    time: float = 0.

    def copy(self): return PoroState(self.u.copy(), self.p.copy(), float(self.time))


@dataclass
class PoroStep:
    state: PoroState
    flux: np.ndarray
    metrics: dict
    converged: bool


class Biot:
    def __init__(self, skeleton, flow, storage=0.):
        if skeleton.grid is not flow.grid: raise ValueError('same fixed physical grid required')
        if not np.isfinite(storage) or storage < 0: raise ValueError('nonnegative physical storage required')
        self.solid, self.flow, self.storage = skeleton, flow, float(storage)
        self.C = sp.eye(flow.grid.nc, format='csc')*storage*flow.grid.volume
        self._factor_cache = {}

    def initial(self, p=None, load=None):
        p = np.zeros(self.flow.grid.nc) if p is None else np.asarray(p, float)
        load = np.zeros(self.solid.ndof) if load is None else np.asarray(load, float)
        return PoroState(self.solid.equilibrate(p, load), p.copy())

    def energy(self, state):
        return .5*float(state.u @ (self.solid.A @ state.u)+state.p @ (self.C @ state.p))

    def _system(self, old, dt, load, boundary, source):
        g = self.flow.grid; s = self.solid
        if not np.isfinite(dt) or dt <= 0: raise ValueError('positive time step required')
        if old.u.shape != (s.ndof,) or old.p.shape != (g.nc,) or not np.isfinite(np.r_[old.u, old.p, old.time]).all():
            raise ValueError('invalid old state')
        if np.any(old.u[s.fixed] != 0): raise ValueError('old state violates homogeneous clamps')
        load = np.asarray(load, float)
        if load.shape != (s.ndof,) or not np.isfinite(load).all(): raise ValueError('finite mechanical load required')
        src = g.average(source) if callable(source) else np.broadcast_to(source, (g.nc,)).astype(float)
        if not np.isfinite(src).all(): raise ValueError('finite fluid source required')
        free, q0, frhs, _ = self.flow.boundary_data(boundary)
        if np.any(q0): raise ValueError('nonzero prescribed flux not supported by Biot boundary work ledger')
        A, G = s.Af, s.Gf; B = self.flow.B[:, free]; H = self.flow.H[free][:, free]
        M = sp.bmat([[A, -G.T, None], [G, self.C, dt*B], [None, -B.T, H]], format='csc')
        b = np.r_[load[s.free], self.solid.G @ old.u+self.C @ old.p+dt*g.volume*src, frhs[free]]
        return M, b, free, frhs, src, load

    def step(self, old, dt, load, boundary, source=0., method='monolithic', maxiter=200, tolerance=1e-8):
        """Return a trial only. Nonconverged trials are never commit eligible."""
        M, b, free, frhs, src, load = self._system(old, dt, load, boundary, source)
        s, g = self.solid, self.flow.grid; nu = len(s.free); np_ = g.nc
        if method == 'monolithic':
            key = (float(dt), tuple(free))
            if key not in self._factor_cache:
                if len(self._factor_cache) >= 8: self._factor_cache.clear()
                self._factor_cache[key] = splu(M)
            x = self._factor_cache[key].solve(b); iterations = 1
        elif method in ('once', 'fixed_stress'):
            # L >= alpha^2 / drained bulk modulus: iteration aid only, vanishes
            # at convergence and is not counted as physical storage.
            L = s.alpha**2/(s.material.lam+2*s.material.mu/3)*g.volume
            B = self.flow.B[:, free]; H = self.flow.H[free][:, free]
            block = sp.bmat([[self.C+sp.eye(np_)*L, dt*B], [-B.T, H]], format='csc')
            fact = splu(block); p = old.p.copy(); u = s.equilibrate(p, load)[s.free]
            count = 1 if method == 'once' else maxiter
            if count < 1: raise ValueError('positive iteration limit required')
            for iterations in range(1, count+1):
                pq = fact.solve(np.r_[b[nu:nu+np_]+L*p-s.Gf @ u, b[nu+np_:]])
                p = pq[:np_]; u = s.factor.solve(load[s.free]+s.Gf.T @ p)
                x = np.r_[u, pq]
                if np.linalg.norm(M @ x-b, np.inf) <= tolerance*max(1., np.linalg.norm(b, np.inf)): break
        else:
            raise ValueError('unknown coupling method')
        true = float(np.linalg.norm(M @ x-b, np.inf)/max(1., np.linalg.norm(b, np.inf)))
        converged = bool(np.isfinite(x).all() and np.isfinite(true) and true <= tolerance)
        u = np.zeros(s.ndof); u[s.free] = x[:nu]
        p = x[nu:nu+np_]; q = np.zeros(g.nf); q[free] = x[nu+np_:]
        state = PoroState(u, p.copy(), old.time+dt)
        du, dp = u-old.u, p-old.p
        mass = s.G @ du+self.C @ dp+dt*self.flow.B @ q-dt*g.volume*src
        dissipation = dt*float(q @ (self.flow.H @ q))
        numeric = .5*float(du @ (s.A @ du)+dp @ (self.C @ dp))
        work = float(load @ du+dt*(p @ (src*g.volume)+q @ frhs))
        energy_residual = self.energy(state)-self.energy(old)+dissipation+numeric-work
        mech = s.A @ u-s.G.T @ p-load
        # Uniform displacement virtual work gives global reaction balance.
        reactions = np.zeros(s.ndof); reactions[s.fixed] = mech[s.fixed]
        momentum = (load+reactions).reshape(-1, g.dim).sum(axis=0)
        if dissipation < -1e-10 or not np.isfinite(energy_residual): converged = False
        metrics = dict(true_residual=true, mass_defect=float(abs(mass.sum())),
            local_mass_defect=float(np.max(np.abs(mass))), energy_residual=float(energy_residual),
            physical_dissipation=dissipation, numerical_loss=numeric, external_work=work,
            elastic_energy=.5*float(u @ (s.A @ u)), storage_energy=.5*float(p @ (self.C @ p)),
            kinetic_energy=0., momentum_balance=float(np.max(np.abs(momentum))), iterations=iterations,
            boundary_outflow_volume=float(dt*np.sum(self.flow.B @ q)),
            fluid_content_change=float(np.sum(s.G @ du+self.C @ dp)))
        return PoroStep(state, q, metrics, converged)

    def pressure_mechanical_schur(self):
        G = self.solid.Gf
        return G @ self.solid.factor.solve(G.T.toarray())


class CouplingTransaction:
    """Private C-compatible trial/accept/rollback boundary for the Biot substep."""
    def __init__(self, solver, initial):
        self.solver = solver; self._committed = initial.copy(); self._trial = None; self.generation = 0

    @property
    def committed(self): return self._committed.copy()

    def begin_trial(self, **kwargs):
        self._trial = None
        candidate = self.solver.step(self._committed, **kwargs)
        if not candidate.converged: raise RuntimeError('coupling residual not converged; rollback required')
        self._trial = candidate
        return PoroStep(candidate.state.copy(), candidate.flux.copy(), candidate.metrics.copy(), True)

    def commit(self):
        if self._trial is None: raise RuntimeError('no converged trial')
        self._committed = self._trial.state.copy(); self._trial = None; self.generation += 1

    def rollback(self): self._trial = None
