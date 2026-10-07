"""A component-closed Q1 carrier + three Q2/Q3/Q4 bubble reference.

Coordinates are DISPLACEMENTS, in metres. All components share each scalar
basis. This small conforming model has no carrier hourglass nullspace, so its
stabilization is zero; it is not the archived 225+432 DOF candidate or v20.
Inertia is continuum point inertia, rho*dV, not the legacy APIC microinertia.
C is the derived spatial velocity gradient and has no independent mass.
"""
from dataclasses import dataclass
import hashlib
import itertools

import numpy as np
import scipy.linalg as la

from ..consistent_transfer import material_response
from ..history_increment import material_tangent
from ..types import AnisotropicMaterialParams


EDGES = (np.array([.125, .25, .5, .75, .875]),
         np.array([.375, .625]), np.array([.375, .625]))


def quadrature(orders=(5, 3, 3)):
    axes, weights = [], []
    for edges, order in zip(EDGES, orders):
        if order < 1:
            raise ValueError('positive quadrature orders required')
        t, w = np.polynomial.legendre.leggauss(order)
        axes.append(np.concatenate([(a+b)/2+(b-a)*t/2 for a, b in zip(edges[:-1], edges[1:])]))
        weights.append(np.concatenate([(b-a)*w/2 for a, b in zip(edges[:-1], edges[1:])]))
    X = np.array(list(itertools.product(*axes)))
    V = np.prod(np.array(list(itertools.product(*weights))), axis=1)
    return X, V


class EnrichedSpace:
    def __init__(self, local_modes=3):
        if local_modes not in range(4):
            raise ValueError('choose 0..3 nested local modes')
        self.local_modes = local_modes
        self.Y0 = np.array(list(itertools.product(*EDGES)))
        self.carriers = len(self.Y0)
        self.n = self.carriers + local_modes
        fixed_carriers = (self.Y0[:, 0] <= .25) | (self.Y0[:, 0] >= .75)
        self.fixed = np.r_[fixed_carriers, np.zeros(local_modes, dtype=bool)]
        self.free = np.flatnonzero(~self.fixed)
        self.lift = np.zeros((self.n, 3))
        self.lift[:self.carriers, 0] = self.Y0[:, 0] >= .75
        self.version = f'c-small-q1-q4-m{local_modes}-v1'

    def basis(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
            raise ValueError('finite (points,3) coordinates required')
        values, derivatives = [], []
        for j, edges in enumerate(EDGES):
            if np.any(X[:, j] < edges[0]-1e-12) or np.any(X[:, j] > edges[-1]+1e-12):
                raise ValueError('points outside declared physical support')
            cell = np.clip(np.searchsorted(edges, X[:, j], side='right')-1, 0, len(edges)-2)
            h = edges[cell+1]-edges[cell]
            t = (X[:, j]-edges[cell])/h
            N = np.zeros((len(X), len(edges))); D = N.copy()
            ii = np.arange(len(X))
            N[ii, cell] = 1-t; N[ii, cell+1] = t
            D[ii, cell] = -1/h; D[ii, cell+1] = 1/h
            values.append(N); derivatives.append(D)
        def tensor(factors):
            return np.einsum('pi,pj,pk->pijk', *factors).reshape(len(X), -1)
        N = tensor(values)
        D = np.stack([tensor([derivatives[k] if k == j else values[k] for k in range(3)])
                      for j in range(3)], axis=2)
        t = (X[:, 0]-.25)/.5
        inside = (t > 0) & (t < 1)
        b = 4*t*(1-t); db = 8*(1-2*t)
        for k in range(self.local_modes):
            s = 2*t-1
            v = np.where(inside, b*s**k, 0.)
            d = np.where(inside, db*s**k + (4*k*b*s**(k-1) if k else 0.), 0.)
            N = np.column_stack((N, v))
            col = np.zeros((len(X), 1, 3)); col[:, 0, 0] = d
            D = np.concatenate((D, col), axis=1)
        return N, D

    def kinematics(self, X, q, velocity):
        if np.shape(q) != (self.n, 3) or np.shape(velocity) != (self.n, 3):
            raise ValueError('generalized state shape mismatch')
        if not np.isfinite(q).all() or not np.isfinite(velocity).all():
            raise ValueError('nonfinite generalized state')
        N, D = self.basis(X)
        F = np.eye(3) + np.einsum('pnj,ni->pij', D, q)
        if np.min(np.linalg.det(F)) <= 0:
            raise ValueError('inverted material state')
        L = np.einsum('pnk,pkj->pnj', D, np.linalg.inv(F))
        return dict(x=X+N@q, F=F, v=N@velocity,
                    C=np.einsum('pnj,ni->pij', L, velocity), T=N, L=L)


@dataclass
class DynamicState:
    q: np.ndarray
    velocity: np.ndarray
    time: float = 0.
    step: int = 0

    def clone(self):
        return DynamicState(self.q.copy(), self.velocity.copy(), self.time, self.step)


class Model:
    def __init__(self, local_modes=3, orders=(5, 3, 3), density=1.):
        if not np.isfinite(density) or density <= 0:
            raise ValueError('positive finite density required')
        self.space = EnrichedSpace(local_modes)
        self.X, self.V = quadrature(orders)
        self.orders = tuple(orders); self.density = density
        self.N, self.D = self.space.basis(self.X)
        self.mass = density*self.V
        self.M = self.N.T@(self.mass[:, None]*self.N)
        self.params = AnisotropicMaterialParams(10., 20., 200., np.array([1., 1., 0.]))
        self.A = np.broadcast_to(self.params.A0, (len(self.X), 3, 3))
        self.B = np.zeros((len(self.X), 9, self.space.n*3))
        for i in range(3):
            for j in range(3):
                self.B[:, 3*i+j, i::3] = self.D[:, :, j]
        self.M3 = np.kron(self.M, np.eye(3))
        eig = la.eigvalsh(self.M)
        if eig[0] <= 1e-12*eig[-1]:
            raise ValueError('unobservable inertia; refine quadrature, no mass shifts')
        self.free = np.array([3*i+j for i in self.space.free for j in range(3)])
        self.Mff = self.M3[np.ix_(self.free, self.free)]
        self.mass_factor = la.cho_factor(self.Mff)
        self.rule_version = 'full-gauss-' + '-'.join(map(str, orders))
        self.signature = hashlib.sha256(self.N.tobytes()+self.D.tobytes()+self.V.tobytes()).hexdigest()

    def rest(self):
        return DynamicState(np.zeros((self.space.n, 3)), np.zeros((self.space.n, 3)))

    def evaluate(self, q, tangent=False):
        q = np.asarray(q)
        if q.shape != (self.space.n, 3) or not np.isfinite(q).all():
            raise ValueError('invalid displacement coefficients')
        F = np.eye(3) + np.einsum('pnj,ni->pij', self.D, q)
        psi, P = material_response(F, self.A, self.params)
        force = np.einsum('p,pnj,pij->ni', self.V, self.D, P, optimize=True)
        out = dict(U=float(self.V@psi), Um=float(self.V@psi), Us=0., F=F, P=P, force=force)
        if tangent:
            H = np.stack([material_tangent(F, self.A, np.broadcast_to(e, F.shape), self.params).reshape(-1, 9)
                          for e in np.eye(9).reshape(9, 3, 3)], axis=2)
            out['K'] = np.einsum('pai,pab,pbj,p->ij', self.B, H, self.B, self.V, optimize=True)
        return out

    def tangent_action(self, q, dq):
        F = np.eye(3)+np.einsum('pnj,ni->pij', self.D, q)
        dF = np.einsum('pnj,ni->pij', self.D, dq)
        dP = material_tangent(F, self.A, dF, self.params)
        return np.einsum('p,pnj,pij->ni', self.V, self.D, dP, optimize=True)

    def kinetic(self, velocity):
        return .5*float(np.sum(velocity*(self.M@velocity)))

    def momentum(self, state):
        x = self.X+self.N@state.q; v = self.N@state.velocity
        return (np.sum(self.mass[:, None]*v, axis=0),
                np.sum(self.mass[:, None]*np.cross(x, v), axis=0))
