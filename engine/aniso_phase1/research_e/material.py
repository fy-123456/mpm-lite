"""Complete, paired material integration and analytic baseline Hencky tangent."""
from __future__ import annotations
import numpy as np
from ..constitutive import energy, pk1, a0_to_A0, dP_fiber_apply
from ..types import AnisotropicMaterialParams


def tangent(F, A, dF, params):
    """Exact directional derivative, including repeated spectral eigenvalues."""
    F, dF = np.asarray(F, float), np.asarray(dF, float)
    # Validate with the actual baseline law, rather than repairing invalid F.
    pk1(F, A, params)
    if dF.shape != (3, 3) or not np.isfinite(dF).all():
        raise ValueError('dF must be finite (3,3)')
    C = F.T @ F
    eig, U = np.linalg.eigh(C)
    logs = np.log(eig)
    L = (U * logs) @ U.T
    divided = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            if abs(eig[i]-eig[j]) < 1e-8*max(eig[i], eig[j]):
                divided[i, j] = 2/(eig[i]+eig[j])
            else:
                divided[i, j] = (logs[i]-logs[j])/(eig[i]-eig[j])
    dC = dF.T @ F + F.T @ dF
    dL = U @ (divided*(U.T @ dC @ U)) @ U.T
    inv = np.linalg.inv(C)
    T = params.mu*L + .5*params.lam*np.trace(L)*np.eye(3)
    S = T @ inv
    dS = (params.mu*dL + .5*params.lam*np.trace(dL)*np.eye(3)) @ inv - S @ dC @ inv
    return dF @ S + F @ dS + dP_fiber_apply(F, A, dF, params)


class FullMaterial:
    """No direction/F averaging, compression, or independent fluid history."""
    def __init__(self, directions, volumes, params):
        self.A = np.stack([a0_to_A0(a) for a in directions])
        self.volumes = np.asarray(volumes, float).copy()
        if self.volumes.shape != (len(self.A),) or not np.isfinite(self.volumes).all() or np.any(self.volumes <= 0):
            raise ValueError('positive finite reference volumes required')
        if params.mu <= 0 or params.lam + 2*params.mu/3 <= 0 or params.k_f < 0:
            raise ValueError('stable matrix and nonnegative fiber modulus required')
        self.params = params

    def evaluate(self, F):
        F = np.asarray(F, float)
        if F.shape != self.A.shape:
            raise ValueError('one deformation per material point required')
        W = np.array([energy(f, a, self.params) for f, a in zip(F, self.A)])
        P = np.array([pk1(f, a, self.params) for f, a in zip(F, self.A)])
        return float(self.volumes @ W), P

    def tangent_apply(self, F, dF):
        if np.shape(F) != self.A.shape or np.shape(dF) != self.A.shape:
            raise ValueError('paired deformation/tangent shapes required')
        return np.array([tangent(f, a, df, self.params) for f, a, df in zip(F, self.A, dF)])


def linear_stress(grad, params):
    """Derivative of the existing law at F=I; 2D is plane strain in 3D."""
    dim = grad.shape[-1]
    E = .5*(grad+np.swapaxes(grad, -1, -2))
    a = params.fiber_direction[:dim]
    A = np.outer(a, a)
    return (2*params.mu*E + params.lam*np.trace(E, axis1=-2, axis2=-1)[..., None, None]*np.eye(dim)
            + 4*params.k_f*np.einsum('...ij,ij->...', E, A)[..., None, None]*A)
