"""Cost reductions for the unchanged fixed-space diagnostic equations."""
import copy
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, solve_triangular

from .convergence_reference import FrozenResidualQ1, sum_fields
from .history_increment import HistoryState, HistoryField, ReferenceBasis, frozen
from .consistent_transfer import Samples, material_response


class CachedFrozenQ1(FrozenResidualQ1):
    """Reuse identical accepted constitutive evaluations; same mass and solver."""
    def elastic(self, state, displacement=None):
        self._check_state(state)
        cache = getattr(self, '_elastic_cache', None)
        if cache is not None:
            old_state, old_du, result = cache
            if old_state is state and ((old_du is None and displacement is None) or
                (old_du is not None and displacement is not None and np.array_equal(old_du, displacement))):
                return result
        result = self.evaluate_elastic(state, displacement)
        self._elastic_cache = (state, None if displacement is None else displacement.copy(), result)
        return result

    def evaluate_elastic(self,state,displacement):
        return super().elastic(state,displacement)

    def commit(self, state, v, dt):
        self._check_state(state)
        self._check_dt(dt)
        if v.shape != (self.ncoeff, 3) or not np.isfinite(v).all():
            raise ValueError('finite frozen-space coefficients required')
        if self.clamped and np.max(abs(self.boundary_N @ v)) > 1e-9:
            raise ValueError('total clamp increment must vanish')
        du = dt*v
        xp = state.xp+self.particles.N @ du
        Fp = state.Fp+self.gradient(self.particles, du)
        xq = state.xq+self.quadrature.N @ du
        Fq = state.Fq+self.gradient(self.quadrature, du)
        # These are exactly material_response's admissibility conditions; no
        # stress is needed during commit. The accepted force was already used.
        for F in (Fp, Fq):
            if not np.isfinite(F).all() or np.any(np.linalg.det(F) <= 0):
                raise ValueError('non-positive or non-finite material Jacobian')
        field = sum_fields(state.field, self.coefficient_field(dt*v))
        return HistoryState(state.particles, state.quadrature, state.mass, field,
                            frozen(xp), frozen(Fp), frozen(xq), frozen(Fq))

    def step(self, state, vp, dt, tolerance=1e-9):
        result = super().step(state, vp, dt, tolerance)
        # commit and elastic now form F with the identical arithmetic order.
        self._elastic_cache = (result[0], None, self._elastic_cache[2])
        return result


class TensorReferenceQ1(CachedFrozenQ1):
    """Unenriched Cartesian reference with exact separable consistent mass.

    Only the left x face is clamped. This gives M_r=Mx_free (x) My (x) Mz;
    the Kronecker Cholesky is the same mass whitening without dense 3-D factors.
    Material quadrature and the nonlinear solver remain unchanged.
    """
    def __init__(self, source, state, workers=4, device=None):
        self.state, self.clamped = state, True
        self.basis = ReferenceBasis(source)
        self.X = self.basis.X
        self.params = copy.deepcopy(source.params)
        self.sites = (state.particles,state.quadrature,state.mass)
        self.mass = state.mass
        self.particles = self.map_points(state.particles.X,state.particles.weight,state.particles.A)
        self.quadrature = self.map_points(state.quadrature.X,state.quadrature.weight,state.quadrature.A)
        self.fixed = np.isclose(self.X[:,0],source.lo[0],atol=1e-14,rtol=0)
        self.free = ~self.fixed
        self.nnode = self.ncoeff = len(self.X)
        self.rank = 0
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.device_material = None
        if device is not None:
            from .refinement_gpu import DeviceMaterial
            self.device_material = DeviceMaterial(self.quadrature.A,self.params,device)
        axes = [np.unique(self.X[:,d]) for d in range(3)]
        masses = []
        for a in axes:
            h = np.diff(a)
            diag = np.r_[h[0],h[:-1]+h[1:],h[-1]]/3
            masses.append(sp.diags([h/6,diag,h/6],[-1,0,1],format='csc'))
        self.M = sp.kron(sp.kron(masses[0],masses[1]),masses[2],format='csc')
        sampled = self.particles.N.T @ self.particles.N.multiply(self.mass[:,None])
        defect = sampled-self.M
        if np.linalg.norm(defect.data) > 1e-11*np.linalg.norm(self.M.data):
            raise ValueError('material mass samples do not exactly integrate tensor mass')
        ids = np.flatnonzero(self.free)
        self.R = sp.csc_matrix((np.ones(len(ids)),(ids,np.arange(len(ids)))),shape=(self.nnode,len(ids)))
        self.Mr = self.M[self.free][:,self.free].tocsc()
        reduced = [masses[0][1:,1:].toarray(),masses[1].toarray(),masses[2].toarray()]
        self.tensor_shape = tuple(len(a) for a in reduced)
        self.cholesky_axes = tuple(cholesky(a,lower=True) for a in reduced)
        self.boundary_X = self.X[self.fixed]
        self.boundary_N = self.map_points(self.boundary_X).N
        eigen = [np.linalg.eigvalsh(a) for a in reduced]
        self.rank_info = dict(scalar_modes=0,vector_dofs=0,reduced_scalar_dofs=len(ids),
                              min_mass_eigenvalue=float(np.prod([e[0] for e in eigen])),
                              mass_sample_relative=float(np.linalg.norm(defect.data)/np.linalg.norm(self.M.data)))

    def map_points(self,X,weights=None,A=None):
        weights = np.ones(len(X)) if weights is None else weights
        base = self.basis.sample(X,weights)
        return Samples(X,weights,base.N,base.D,base.A if A is None else A)

    def evaluate_elastic(self,state,displacement):
        F = state.Fq if displacement is None else state.Fq+self.gradient(self.quadrature,displacement)
        if self.device_material is not None:
            psi,P = self.device_material(F)
            return float(self.quadrature.weight @ psi),self.force(P)
        A = self.quadrature.A
        slices = [slice(a,min(a+16000,len(F))) for a in range(0,len(F),16000)]
        chunks = list(self.pool.map(lambda s: material_response(F[s],A[s],self.params),slices))
        psi = np.concatenate([c[0] for c in chunks])
        P = np.concatenate([c[1] for c in chunks])
        return float(self.quadrature.weight @ psi), self.force(P)

    def close(self):
        self.pool.shutdown()

    def coefficient_field(self,coefficients):
        return HistoryField(((self.basis,frozen(coefficients)),))

    def _tensor_apply(self,values,transpose=False,solve=False):
        data = values.reshape(self.tensor_shape+(values.shape[-1],))
        for d,L in enumerate(self.cholesky_axes):
            matrix = L.T if transpose else L
            moved = np.moveaxis(data,d,0)
            flat = moved.reshape(len(L),-1)
            new = (solve_triangular(matrix,flat,lower=not transpose,check_finite=False)
                   if solve else matrix @ flat)
            data = np.moveaxis(new.reshape(moved.shape),0,d)
        return data.reshape(values.shape)

    def mass_solve(self,rhs):
        return self.mass_from_y(self.mass_force_to_y(rhs))

    def mass_to_y(self,z):
        return self._tensor_apply(z,transpose=True)

    def mass_from_y(self,y):
        return self._tensor_apply(y,transpose=True,solve=True)

    def mass_force_to_y(self,force):
        return self._tensor_apply(force,solve=True)

    def project(self,values,clamped=True):
        if not clamped:
            raise ValueError('reference projection is clamped on the left face')
        rhs = self.R.T @ (self.particles.N.T @ (self.mass[:,None]*values))
        return self.R @ self.mass_solve(rhs)
