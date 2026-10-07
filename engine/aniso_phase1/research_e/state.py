"""Finite-deformation saturated substate transaction, separate from C's step.

Incompressible grains AND fluid in this adapter: a proposed deformation must
carry exactly the associated pore-volume inflow. It is not a nonlinear Biot
solver. Permeability eigenvalues are fixed, directions rotate by polar R.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from .flow import positive_tensor


class SaturatedState:
    schema_version = 1

    def __init__(self, volumes, porosity0, permeability0, directions, rho_s=2000., rho_f=1000.):
        V = np.asarray(volumes, float)
        if V.ndim != 1 or not len(V) or not np.isfinite(V).all() or np.any(V <= 0):
            raise ValueError('positive reference volumes required')
        if not np.isfinite([porosity0, rho_s, rho_f]).all() or not 0 < porosity0 < 1 or min(rho_s, rho_f) <= 0:
            raise ValueError('valid porosity and densities required')
        self.V, self.n0, self.rho_s, self.rho_f = V.copy(), porosity0, rho_s, rho_f
        N = len(V)
        self.K0 = np.broadcast_to(positive_tensor(permeability0, 3), (N, 3, 3)).copy()
        a = np.asarray(directions, float)
        if a.shape != (N, 3) or not np.isfinite(a).all() or np.any(np.linalg.norm(a, axis=1) == 0):
            raise ValueError('one finite nonzero direction per point required')
        self.a0 = a/np.linalg.norm(a, axis=1)[:, None]
        self._committed = dict(F=np.tile(np.eye(3), (N, 1, 1)), n=np.full(N, porosity0),
            K=self.K0.copy(), direction=self.a0.copy(), solid_mass=rho_s*(1-porosity0)*V,
            fluid_mass=rho_f*porosity0*V, pressure=np.zeros(N), solid_velocity=np.zeros((N, 3)),
            fluid_velocity=np.zeros((N, 3)))
        self._trial = None
        self.generation = 0

    @property
    def committed(self): return deepcopy(self._committed)

    @property
    def trial(self): return deepcopy(self._trial)

    def begin_trial(self, F, fluid_volume_increment, pressure=None, solid_velocity=None, fluid_velocity=None):
        self._trial = None
        old = self._committed; N = len(self.V)
        F = np.asarray(F, float)
        inflow = np.asarray(fluid_volume_increment, float)
        if F.shape != (N, 3, 3) or not np.isfinite(F).all() or inflow.shape != (N,) or not np.isfinite(inflow).all():
            raise ValueError('finite paired F and net pore inflow required')
        J = np.linalg.det(F)
        if np.any(J <= 0): raise ValueError('det(F) must be positive')
        n = 1-(1-self.n0)/J
        if np.any((n <= 0) | (n >= 1)): raise ValueError('porosity outside (0,1); no clipping allowed')
        mf = old['fluid_mass']+self.rho_f*inflow
        target = self.rho_f*n*J*self.V
        if not np.allclose(mf, target, rtol=1e-9, atol=1e-10):
            raise ValueError('fluid mass does not match saturated pore volume')
        U, _, Vt = np.linalg.svd(F); R = U @ Vt
        K = R @ self.K0 @ np.swapaxes(R, 1, 2)
        positive_tensor(K, 3)
        direction = np.einsum('nij,nj->ni', F, self.a0)
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        new = deepcopy(old); new.update(F=F.copy(), n=n, K=K, direction=direction, fluid_mass=mf)
        for key, value, shape in [('pressure', pressure, (N,)), ('solid_velocity', solid_velocity, (N, 3)), ('fluid_velocity', fluid_velocity, (N, 3))]:
            if value is not None:
                value = np.asarray(value, float)
                if value.shape != shape or not np.isfinite(value).all(): raise ValueError(f'invalid {key}')
                new[key] = value.copy()
        self._trial = new
        return self.trial

    def commit(self):
        if self._trial is None: raise RuntimeError('no valid trial to commit')
        self._committed = self._trial; self._trial = None; self.generation += 1

    def rollback(self): self._trial = None

    def config(self):
        return dict(volumes=self.V.tolist(), porosity0=self.n0, permeability0=self.K0.tolist(),
                    directions=self.a0.tolist(), rho_s=self.rho_s, rho_f=self.rho_f)

    def checkpoint(self, path):
        if self._trial is not None: raise RuntimeError('checkpoint only committed state')
        payload = dict(schema_version=1, model='incompressible_saturated_polar_K', config=self.config(),
                       generation=self.generation, state={k:v.tolist() for k,v in self._committed.items()})
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        Path(path).write_text(json.dumps(dict(payload=payload, sha256=hashlib.sha256(encoded.encode()).hexdigest()), indent=2))

    @classmethod
    def restart(cls, path):
        data = json.loads(Path(path).read_text()); p = data['payload']
        digest = hashlib.sha256(json.dumps(p, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if p.get('schema_version') != 1 or p.get('model') != 'incompressible_saturated_polar_K' or data['sha256'] != digest:
            raise ValueError('unsupported or corrupt checkpoint')
        obj = cls(**p['config']); s = {k:np.array(v, float) for k,v in p['state'].items()}
        obj.begin_trial(s['F'], (s['fluid_mass']-obj._committed['fluid_mass'])/obj.rho_f,
                        s['pressure'], s['solid_velocity'], s['fluid_velocity'])
        if set(s) != set(obj._trial) or any(not np.allclose(s[k], obj._trial[k], rtol=1e-10, atol=1e-12) for k in s):
            raise ValueError('checkpoint violates model invariants')
        obj.commit(); obj.generation = int(p['generation'])
        return obj
