"""Conservative Cartesian RT0/P0 mixed Darcy in 1D and 2D.

Flux DOFs are integrated volume flow through faces in the positive axis
orientation. Full anisotropic inverse permeability is integrated (no TPFA
approximation for oblique tensors). Pressure is cellwise constant.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu, MatrixRankWarning, LinearOperator


def positive_tensor(K, dim):
    K = np.asarray(K, float)
    if K.shape[-2:] != (dim, dim) or not np.isfinite(K).all():
        raise ValueError('finite permeability tensor with matching dimension required')
    scale = max(float(np.max(np.abs(K))), np.finfo(float).tiny)
    if np.max(np.abs(K-np.swapaxes(K, -1, -2))) > 1e-12*scale or np.min(np.linalg.eigvalsh(K)) <= 0:
        raise ValueError('permeability must be symmetric positive definite')
    return .5*(K+np.swapaxes(K, -1, -2))


def oriented_permeability(parallel=0.01, transverse=0.001, angle=0.):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return positive_tensor(R @ np.diag([parallel, transverse]) @ R.T, 2)


def quadrature(dim, order=3):
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = (x+1)/2, w/2
    for ix in product(range(order), repeat=dim):
        yield np.array([x[i] for i in ix]), float(np.prod([w[i] for i in ix]))


class Grid:
    def __init__(self, shape, lengths=None):
        self.shape = tuple(int(n) for n in shape)
        self.dim = len(self.shape)
        if self.dim not in (1, 2) or min(self.shape) < 1 or tuple(shape) != self.shape:
            raise ValueError('positive integer 1D/2D grid required')
        self.lengths = np.ones(self.dim) if lengths is None else np.asarray(lengths, float)
        if self.lengths.shape != (self.dim,) or not np.isfinite(self.lengths).all() or np.any(self.lengths <= 0):
            raise ValueError('positive finite lengths required')
        self.h = self.lengths/np.array(self.shape)
        self.volume = float(np.prod(self.h))
        self.areas = self.volume/self.h
        self.indices = np.array(list(np.ndindex(self.shape)))
        self.centers = (self.indices+.5)*self.h
        self.nc = len(self.indices)
        self.face_shapes, offsets = [], [0]
        for d in range(self.dim):
            sh = list(self.shape); sh[d] += 1
            self.face_shapes.append(tuple(sh)); offsets.append(offsets[-1]+int(np.prod(sh)))
        self.nf = offsets[-1]
        self.cell_faces = np.empty((self.nc, 2*self.dim), int)
        self.face_centers = np.empty((self.nf, self.dim))
        self.face_axis = np.empty(self.nf, int)
        self.boundary = {}  # id -> (axis, side, outward sign)
        for d, sh in enumerate(self.face_shapes):
            for ix in np.ndindex(sh):
                f = offsets[d]+np.ravel_multi_index(ix, sh)
                point = (np.array(ix)+.5)*self.h; point[d] = ix[d]*self.h[d]
                self.face_centers[f] = point; self.face_axis[f] = d
                if ix[d] in (0, self.shape[d]):
                    side = int(ix[d] > 0); self.boundary[f] = (d, side, 2*side-1)
        for c, ix in enumerate(self.indices):
            for d, sh in enumerate(self.face_shapes):
                for side in (0, 1):
                    ind = ix.copy(); ind[d] += side
                    self.cell_faces[c, 2*d+side] = offsets[d]+np.ravel_multi_index(ind, sh)
        rows = np.repeat(np.arange(self.nc), 2*self.dim)
        signs = np.tile([-1., 1.], self.dim*self.nc)
        self.B = sp.coo_matrix((signs, (rows, self.cell_faces.ravel())), shape=(self.nc, self.nf)).tocsr()

    def average(self, function, order=4):
        out = 0.
        for s, w in quadrature(self.dim, order):
            out = out + w*np.asarray([function((ix+s)*self.h) for ix in self.indices])
        return np.asarray(out)

    def face_average(self, f, function):
        if not callable(function):
            return float(function)
        d = self.face_axis[f]
        if self.dim == 1:
            return float(function(self.face_centers[f]))
        result = 0.
        for s, w in quadrature(1, 4):
            x = self.face_centers[f].copy(); x[1-d] += (s[0]-.5)*self.h[1-d]
            result += w*function(x)
        return float(result)

    def flux_basis(self, s):
        phi = np.zeros((2*self.dim, self.dim))
        for d in range(self.dim):
            phi[2*d, d] = (1-s[d])/self.areas[d]
            phi[2*d+1, d] = s[d]/self.areas[d]
        return phi

    def cell_flux(self, faces):
        return np.asarray(faces)[self.cell_faces] @ self.flux_basis(np.full(self.dim, .5))


@dataclass
class FlowResult:
    pressure: np.ndarray
    face_flux: np.ndarray
    cell_flux: np.ndarray
    mass_residual: np.ndarray
    true_residual: float
    dissipation_rate: float


class Darcy:
    def __init__(self, grid, permeability, viscosity=1., density=1000., gravity=None):
        self.grid = grid
        if not np.isfinite([viscosity, density]).all() or viscosity <= 0 or density <= 0:
            raise ValueError('positive viscosity and density required')
        K = positive_tensor(permeability, grid.dim)
        self.K = np.broadcast_to(K, (grid.nc, grid.dim, grid.dim)).copy()
        self.viscosity = float(viscosity)
        inv = viscosity*np.linalg.inv(self.K)
        local = np.zeros((grid.nc, 2*grid.dim, 2*grid.dim))
        for s, w in quadrature(grid.dim, 2):
            phi = grid.flux_basis(s)
            local += grid.volume*w*np.einsum('ai,cij,bj->cab', phi, inv, phi)
        faces = grid.cell_faces
        rows = np.broadcast_to(faces[:, :, None], local.shape).ravel()
        cols = np.broadcast_to(faces[:, None, :], local.shape).ravel()
        self.H = sp.coo_matrix((local.ravel(), (rows, cols)), shape=(grid.nf, grid.nf)).tocsc()
        self.B = grid.B
        self.gravity_rhs = np.zeros(grid.nf)
        gravity = np.zeros(grid.dim) if gravity is None else np.asarray(gravity, float)
        if gravity.shape != (grid.dim,) or not np.isfinite(gravity).all():
            raise ValueError('finite gravity vector required')
        for c in range(grid.nc):
            for d in range(grid.dim):
                np.add.at(self.gravity_rhs, faces[c, 2*d:2*d+2], .5*grid.h[d]*density*gravity[d])

    def boundary_data(self, boundary):
        """Every boundary side must declare ('pressure', p) or ('flux', q.n)."""
        expected = {(d, side) for d in range(self.grid.dim) for side in (0, 1)}
        if set(boundary) != expected:
            raise ValueError('declare all boundary sides exactly once')
        rhs = self.gravity_rhs.copy(); fixed = {}; pressure_count = 0
        for f, (d, side, normal) in self.grid.boundary.items():
            kind, value = boundary[d, side]
            value = self.grid.face_average(f, value)
            if not np.isfinite(value):
                raise ValueError('finite boundary data required')
            if kind == 'pressure':
                rhs[f] -= normal*value; pressure_count += 1
            elif kind == 'flux':
                fixed[f] = normal*self.grid.areas[d]*value
            else:
                raise ValueError('boundary kind must be pressure or flux')
        known = np.array(sorted(fixed), int)
        q0 = np.zeros(self.grid.nf)
        for f, value in fixed.items(): q0[f] = value
        free = np.setdiff1d(np.arange(self.grid.nf), known)
        return free, q0, rhs, pressure_count

    def solve(self, boundary, source=0., mean_pressure=0.):
        g = self.grid
        src = g.average(source) if callable(source) else np.broadcast_to(source, (g.nc,)).astype(float)
        if not np.isfinite(src).all() or not np.isfinite(mean_pressure):
            raise ValueError('finite source and pressure gauge required')
        free, q0, rhs, n_dir = self.boundary_data(boundary)
        H = self.H[free][:, free]; B = self.B[:, free]
        forcing = rhs[free]-(self.H @ q0)[free]
        mass = src*g.volume-self.B @ q0
        if n_dir:
            block = sp.bmat([[H, -B.T], [B, None]], format='csc')
            b = np.r_[forcing, mass]
        else:
            compatibility = float(mass.sum())
            if abs(compatibility) > 1e-10*max(1., np.sum(np.abs(mass))):
                raise ValueError('incompatible pure Neumann source and boundary flux')
            v = sp.csc_matrix(np.full((g.nc, 1), g.volume))
            block = sp.bmat([[H, -B.T, None], [B, None, v], [None, v.T, None]], format='csc')
            b = np.r_[forcing, mass, mean_pressure*g.nc*g.volume]
        x, residual = solve_checked(block, b)
        q = q0.copy(); q[free] = x[:len(free)]
        p = x[len(free):len(free)+g.nc]
        return FlowResult(p, q, g.cell_flux(q), self.B @ q-src*g.volume,
                          residual, float(q @ (self.H @ q)))

    def schur(self, boundary):
        """Pressure Schur action; never asserts the mixed block is SPD."""
        free, _, _, _ = self.boundary_data(boundary)
        B = self.B[:, free]; lu = splu(self.H[free][:, free])
        return LinearOperator((self.grid.nc, self.grid.nc), matvec=lambda x: B @ lu.solve(B.T @ x), dtype=float)


def solve_checked(A, b, tolerance=1e-8):
    with warnings.catch_warnings():
        warnings.simplefilter('error', MatrixRankWarning)
        x = spsolve(A, b)
    residual = float(np.linalg.norm(A @ x-b, np.inf)/max(1., np.linalg.norm(b, np.inf)))
    if not np.isfinite(x).all() or not np.isfinite(residual) or residual > tolerance:
        raise RuntimeError(f'failed true residual check: {residual}')
    return x, residual
