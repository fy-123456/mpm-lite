"""Experimental common carrier/particle deformation history.

Keep fixed reference Lite gradients G0 on material carriers and a per-particle
right factor R containing initial local history: F_p = (G0 Y)_p R_p.
F_p(v) = F_p^n + dt (G0 N_n v)_p R_p. Differentiate both the original particle
quadrature and patch energy through this SAME map. No reset, energy offset,
damping or stiffness coefficient change. Fresh-reference stiffness is unchanged.
x/v/C retain the moving Lite transfer; this closes F/Y, not x/Y compatibility.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.types import vec3
from .material_patch import MaterialPatchLiteSolver, MaterialPatchEnhancements
from .projected_history import pack_maps


def carrier_gradient(G, Y):
    return np.stack([g @ Y for g in G], axis=2)


def pack_reference(G, R):
    state = {'compatible_R': np.array(R, copy=True)}
    for k, g in enumerate(G):
        g = g.tocsr()
        for name in ('data', 'indices', 'indptr'):
            state[f'compatible_G{k}_{name}'] = getattr(g, name).copy()
        state[f'compatible_G{k}_shape'] = np.array(g.shape)
    return state


def unpack_reference(state):
    G = tuple(sp.csr_matrix((state[f'compatible_G{k}_data'], state[f'compatible_G{k}_indices'],
                            state[f'compatible_G{k}_indptr']), shape=tuple(state[f'compatible_G{k}_shape']))
              for k in range(3))
    R = np.array(state['compatible_R'], copy=True)
    if not np.isfinite(R).all() or np.any(np.linalg.det(R) <= 0):
        raise ValueError('compatible history requires finite positive-Jacobian right factors')
    return G, R


class CompatiblePatchEnhancements(MaterialPatchEnhancements):
    def prepare(self):
        super().prepare()
        # Both material and stabilization tangents factor through N. If the
        # active grid has more scalar DOFs than carriers, ker(N) necessarily
        # contains unsupported modes. Do not hide them behind grid inertia.
        if self.reference_valid and self.N.shape[1] > self.N.shape[0]:
            self.reference_valid = False
            self.reconstruction_stats.update(
                patch_failure='compatible_patch does not support grid expansion beyond carrier capacity',
                compatible_support_rejected=True,
                active_grid_nodes=self.N.shape[1], carrier_capacity=self.N.shape[0])

    def commit(self):
        super().commit()
        # Accepted trial owns F. No energy read occurs after the legacy transfer
        # writes its F and before this replacement.
        wp.copy(self.s.ptc_F, self.s.local_trial)

    def state(self):
        result = super().state()
        if not self.s.compatible_initialized:
            raise ValueError('prepare material map before saving compatible history')
        result.update(pack_reference(self.s.compatible_G, self.s.compatible_R))
        return result


class CompatiblePatchLiteSolver(MaterialPatchLiteSolver):
    """Opt-in common F/Y history; forces derive from the trial energy."""
    def __init__(self, *args, **kwargs):
        if kwargs.pop('stabilization', 'compatible_patch') != 'compatible_patch':
            raise ValueError('compatible_patch required')
        super().__init__(*args, stabilization='material_patch', **kwargs)
        self.stabilization = 'compatible_patch'
        self.enhancements = CompatiblePatchEnhancements(self)
        self.compatible_initialized = False

    def _prepare_material_map(self):
        if self._mapped_ready:
            return
        super()._prepare_material_map()
        e = self.enhancements
        e.prepare()
        if not e.reference_valid:
            raise ValueError(e.reconstruction_stats)
        if not self.compatible_initialized:
            if e.restart_state is not None:
                if 'compatible_R' not in e.restart_state:
                    self._mapped_ready = False
                    raise ValueError('restart needs complete compatible G/R history; implicit reanchoring is forbidden')
                self.compatible_G, self.compatible_R = unpack_reference(e.restart_state)
            else:
                if self.sim_steps != 0:
                    raise ValueError('new compatible history must start at initial state')
                self.compatible_G = tuple((self.mapped_S @ d).tocsr() for d in self.mapped_D)
                H = carrier_gradient(self.compatible_G, e.origin.numpy())
                self.compatible_R = np.linalg.solve(H, self.mapped_old_F)
            self.compatible_initialized = True
        if any(g.shape != (self.n_ptc, len(e.origin)) for g in self.compatible_G):
            raise ValueError('compatible reference shape does not match particle/carrier labels')
        expected = carrier_gradient(self.compatible_G, e.origin.numpy()) @ self.compatible_R
        error = float(np.max(abs(expected - self.mapped_old_F)))
        if not np.isfinite(error) or error > 1e-10:
            self._mapped_ready = False
            raise ValueError(f'particle/carrier history mismatch: {error}')
        G = tuple(g @ e.N for g in self.compatible_G)
        R = self.compatible_R
        self.local_host_maps = tuple(sum(g.multiply(R[:, j, k, None]) for j, g in enumerate(G)).tocsr()
                                     for k in range(3))
        offsets, ids, gradients = pack_maps(self.local_host_maps)
        self.local_offsets = wp.array(offsets, dtype=int, device=self.device)
        self.local_ids = wp.array(ids, dtype=int, device=self.device)
        self.local_gradients = wp.array(gradients, dtype=vec3, device=self.device)
        self._mapped_stats.update(compatible_history_error=error,
            material_history_update='common_carrier_gradient', local_map_entries=len(ids))
