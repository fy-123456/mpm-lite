"""Reference-coordinate fiber fields and exact original Hencky derivatives."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from ..rules import readonly

PARENT_SHA = '55682a7b8e90b62c1306818cdf9c1f060b4286a174da069ce2217aa5b53b3c7c'
SPACE_SHA = '7422b2b099127bebe9c144ea6cac94409bde861aadb94693b7fc650b397c0332'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value, dtype='<f8')
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


@dataclass(frozen=True)
class MaterialSource:
    """One immutable physical material, independent of its integration rule.

    Multiple families add weighted *fiber energies*. The isotropic potential
    is included once. Fields and discontinuities are in reference meters.
    """
    field: str = 'F45'
    mu: float = 10.
    lam: float = 20.
    k_f: float = 200.
    angle: float = float(np.pi/4)
    turn: float = float(np.pi/6)
    split_x: float = .5
    parent_sha256: str = PARENT_SHA
    space_sha256: str = SPACE_SHA

    def __post_init__(self):
        if self.field not in ('F45', 'partition', 'turning', 'two_family_3d'):
            raise ValueError('unknown reference fiber field')
        if not np.isfinite([self.mu, self.lam, self.k_f, self.angle, self.turn, self.split_x]).all() or min(self.mu, self.k_f) < 0 or self.lam < 0:
            raise ValueError('invalid material parameters')
        if any(len(x) != 64 or any(c not in '0123456789abcdef' for c in x) for x in (self.parent_sha256, self.space_sha256)):
            raise ValueError('material requires exact parent and space hashes')

    def manifest(self):
        return dict(schema='B-stage2-material-v1', **self.__dict__,
                    units={'position': 'm', 'moduli': 'Pa'},
                    direction_coordinates='reference', law='Hencky + bilateral quadratic fiber',
                    family_weights=[.6, .4] if self.field == 'two_family_3d' else [1.],
                    field_formula_version=1)

    @property
    def signature(self):
        return digest(self.manifest())

    @property
    def breaks(self):
        return (self.split_x,) if self.field == 'partition' else ()

    def fibers(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all():
            raise ValueError('finite reference positions (points,3) required')
        theta = np.full(len(X), self.angle)
        if self.field == 'partition':
            theta += np.where(X[:, 0] < self.split_x, -self.turn, self.turn)
        elif self.field in ('turning', 'two_family_3d'):
            theta += self.turn*np.sin(2*np.pi*(X[:, 0]-.125)/.75)
        phi = np.zeros(len(X))
        if self.field == 'two_family_3d':
            phi = (np.pi/9)*np.cos(2*np.pi*(X[:, 0]-.125)/.75)
        a = np.column_stack((np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)))
        if self.field != 'two_family_3d':
            return ((1., a),)
        b = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(len(X))))
        return ((.6, a), (.4, b))


class ConstitutiveState:
    """Compute spectral data once per slab, reuse for all tangent directions.

    This is the exact Hessian, including negative curvature. No SPD shift or
    approximate constitutive model is used. Eigenvalue collisions use the
    continuous divided-difference limit, as in the parent material_tangent.
    """
    def __init__(self, F, X, source, *, min_detF=.15, tangent=False):
        self.F = F
        det = np.linalg.det(F)
        if not np.isfinite(F).all() or not np.isfinite(det).all() or det.min() <= min_detF:
            raise ValueError('material state outside detF domain')
        self.minimum = float(det.min())
        c, Q = np.linalg.eigh(F.transpose(0, 2, 1) @ F)
        if not np.isfinite(c).all() or np.min(c) <= 1e-16:
            raise ValueError('singular material state')
        logs = np.log(c)
        tr = logs.sum(axis=1)
        g = source.mu*logs + .5*source.lam*tr[:, None]
        s = g/c
        self.S = (Q*s[:, None, :]) @ Q.transpose(0, 2, 1)
        self.energy = .25*source.mu*(logs*logs).sum(axis=1) + .125*source.lam*tr**2
        self.P = F @ self.S
        self.families = []
        for weight, a in source.fibers(X):
            A = a[:, :, None]*a[:, None, :]
            FA = F @ A
            strain = np.einsum('pij,pij->p', FA, F)-1
            k = source.k_f*weight
            self.energy += .5*k*strain**2
            self.P += 2*k*strain[:, None, None]*FA
            self.families.append((k, A, FA, strain))
        if tangent:
            delta = c[:, :, None]-c[:, None, :]
            ratio = delta/c[:, None, :]
            logdiv = np.ones_like(delta)
            np.divide(np.log1p(ratio), ratio, out=logdiv, where=ratio != 0)
            logdiv /= c[:, None, :]
            self.divided = (source.mu*logdiv-s[:, None, :])/c[:, :, None]
            self.c, self.Q, self.lam = c, Q, source.lam
        if not np.isfinite(self.energy).all() or not np.isfinite(self.P).all():
            raise ValueError('non-finite constitutive response')

    def action(self, dF):
        F, Q, c = self.F, self.Q, self.c
        dC = F.transpose(0, 2, 1) @ dF + dF.transpose(0, 2, 1) @ F
        local = Q.transpose(0, 2, 1) @ dC @ Q
        dS = self.divided*local
        trace = (np.diagonal(local, axis1=1, axis2=2)/c).sum(axis=1)
        idx = np.arange(3)
        dS[:, idx, idx] += .5*self.lam*trace[:, None]/c
        out = dF @ self.S + F @ Q @ dS @ Q.transpose(0, 2, 1)
        for k, A, FA, strain in self.families:
            contraction = np.einsum('pij,pij->p', FA, dF)
            out += 2*k*strain[:, None, None]*(dF @ A) + 4*k*contraction[:, None, None]*FA
        if not np.isfinite(out).all():
            raise ValueError('non-finite exact tangent')
        return out
