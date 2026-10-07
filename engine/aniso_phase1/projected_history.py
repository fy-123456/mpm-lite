"""Opt-in center material map consistent with the frozen particle F update.

F_c(v) = W_cp [F_p^n + dt (S_pc D_ci v_i) F_p^n].
W, S, D and F^n are frozen once per step. Only compact center maps are
accessed by Newton/PCG. Moving-weight remapping remains a measured error.
"""
from __future__ import annotations

import itertools
import time
import numpy as np
import scipy.sparse as sp
import warp as wp
from warp._src.types import type_size_in_bytes

from engine.types import real, vec3, mat33
from engine.sp_grid import B
from engine.kernel.constant import s_min
from .solver import AnisotropicLiteImplicitSolver
from .particle_quadrature import node_inertia, node_tangent_mass, ParticleQuadratureImplicitSolver
from .kernels import aniso_pk1_wp, aniso_dpk1_wp
from .potential import particle_potential


def frozen_maps(x, volume, F, centers, nodes, dx):
    """Sparse, complete-support W/S/D construction, independently auditable.

    Return scalar sparse maps B[k] such that delta F[:, a, k] = B[k] v[:, a].
    Reference-volume weights are used for history, not mass weights.
    """
    x, volume, F = np.asarray(x), np.asarray(volume), np.asarray(F)
    if dx <= 0 or not np.isfinite(dx):
        raise ValueError('positive finite dx required')
    if not np.isfinite(x).all() or not np.isfinite(F).all() or np.any(volume <= 0):
        raise ValueError('finite state and positive volume required')
    if np.any(np.linalg.det(F) <= 0):
        raise ValueError('positive particle Jacobian required')
    corners = np.array(list(itertools.product((0, 1), repeat=3)))
    c_lookup = {tuple(c): i for i, c in enumerate(centers)}
    n_lookup = {tuple(n): i for i, n in enumerate(nodes)}
    q = x / dx - .5
    base = np.floor(q).astype(int)
    fraction = q - base
    ids = np.array([[c_lookup.get(tuple(c), -1) for c in base + offset] for offset in corners]).T
    weights = np.prod(np.where(corners[None, :, :], fraction[:, None, :], 1-fraction[:, None, :]), axis=2)
    valid = weights > 0
    if np.any(ids[valid] < 0):
        raise ValueError('complete occupied center support required')
    rows = np.broadcast_to(np.arange(len(x))[:, None], ids.shape)
    S = sp.coo_matrix((weights[valid], (rows[valid], ids[valid])), shape=(len(x), len(centers))).tocsr()
    V = np.asarray(S.T @ volume).ravel()
    if np.any(V <= 0):
        raise ValueError('strictly occupied centers required')
    W = sp.diags(1/V) @ S.T @ sp.diags(volume)
    ni = np.array([[n_lookup.get(tuple(c+o), -1) for o in corners] for c in centers])
    if np.any(ni < 0):
        raise ValueError('complete grid support required')
    ci = np.repeat(np.arange(len(centers)), 8)
    D = tuple(sp.coo_matrix((np.tile((2*corners[:, k]-1)/(4*dx), len(centers)),
                            (ci, ni.ravel())), shape=(len(centers), len(nodes))).tocsr() for k in range(3))
    G = tuple(S @ d for d in D)
    maps = tuple((W @ sum(g.multiply(F[:, a, k, None]) for a, g in enumerate(G))).tocsr() for k in range(3))
    F0 = np.asarray(W @ F.reshape(len(F), 9)).reshape(-1, 3, 3)
    return maps, F0, V, W.tocsr(), S, D


def pack_maps(maps):
    """Pack the union of three CSR patterns without a dense C-by-N allocation."""
    columns = maps[0].shape[1]
    coo = [m.tocoo() for m in maps]
    keys = [m.row.astype(np.int64)*columns + m.col for m in coo]
    unique = np.unique(np.concatenate(keys))
    values = np.zeros((len(unique), 3))
    for k, (key, m) in enumerate(zip(keys, coo)):
        values[np.searchsorted(unique, key), k] = m.data
    keep = np.any(values != 0, axis=1)
    unique, values = unique[keep], values[keep]
    row = unique // columns
    offsets = np.r_[0, np.cumsum(np.bincount(row, minlength=maps[0].shape[0]))].astype(np.int32)
    return offsets, (unique % columns).astype(np.int32), values


@wp.kernel
def mapped_residual(offsets: wp.array(dtype=int), ids: wp.array(dtype=int), gradients: wp.array(dtype=vec3),
                    velocity: wp.array(dtype=vec3), F0: wp.array(dtype=mat33), A: wp.array(dtype=mat33),
                    volume: wp.array(dtype=real), trial: wp.array(dtype=mat33), out: wp.array(dtype=vec3),
                    invalid: wp.array(dtype=int), dt: real, mu: real, lam: real, kf: real):
    c = wp.tid()
    F = F0[c]
    for j in range(offsets[c], offsets[c+1]):
        F += dt * wp.outer(velocity[ids[j]], gradients[j])
    trial[c] = F
    U, sigma, V = wp.svd3(F)
    if not wp.isfinite(wp.determinant(F)) or wp.determinant(F) <= real(0) or wp.min(sigma) <= s_min:
        wp.atomic_max(invalid, 0, 1)
        return
    P = aniso_pk1_wp(F, A[c], mu, lam, kf)
    for j in range(offsets[c], offsets[c+1]):
        wp.atomic_add(out, ids[j], dt*volume[c]*(P@gradients[j]))


@wp.kernel
def mapped_tangent(offsets: wp.array(dtype=int), ids: wp.array(dtype=int), gradients: wp.array(dtype=vec3),
                   direction: wp.array(dtype=vec3), trial: wp.array(dtype=mat33), A: wp.array(dtype=mat33),
                   volume: wp.array(dtype=real), out: wp.array(dtype=vec3),
                   dt: real, mu: real, lam: real, kf: real, project_pd: int):
    c = wp.tid()
    dF = mat33(real(0))
    for j in range(offsets[c], offsets[c+1]):
        dF += dt*wp.outer(direction[ids[j]], gradients[j])
    dP = aniso_dpk1_wp(trial[c], A[c], dF, mu, lam, kf, project_pd)
    for j in range(offsets[c], offsets[c+1]):
        wp.atomic_add(out, ids[j], dt*volume[c]*(dP@gradients[j]))


@wp.kernel
def mapped_force(offsets: wp.array(dtype=int), ids: wp.array(dtype=int), gradients: wp.array(dtype=vec3),
                 trial: wp.array(dtype=mat33), A: wp.array(dtype=mat33), volume: wp.array(dtype=real),
                 out: wp.array(dtype=vec3), mu: real, lam: real, kf: real):
    c = wp.tid()
    P = aniso_pk1_wp(trial[c], A[c], mu, lam, kf)
    for j in range(offsets[c], offsets[c+1]):
        wp.atomic_add(out, ids[j], volume[c]*(P@gradients[j]))


@wp.kernel
def scatter_trial(address: wp.array(dtype=wp.vec2i), source: wp.array(dtype=mat33),
                  target: wp.array(dtype=mat33, ndim=4)):
    c = wp.tid()
    block, local = address[c][0], address[c][1]
    target[0, block, local//(B*B), local % (B*B)] = source[c]


class ProjectedHistoryLiteSolver(AnisotropicLiteImplicitSolver):
    """Center energy with the exact frozen average of particle trial F.

    This is an experimental spatial discretization, not a promise of improved
    continuum accuracy. Velocity/APIC transfers and particle F updates are retained.
    """
    _project = ParticleQuadratureImplicitSolver._project
    history_consistency = 'projected_center'

    def __init__(self, *args, **kwargs):
        if kwargs.get('force_discretization', 'variational') != 'variational' or kwargs.get('history_mode', 'particle_resample') != 'particle_resample':
            raise ValueError('projected history requires variational forces and particle_resample')
        if kwargs.get('direction_model', 'mean_tensor') != 'mean_tensor' or kwargs.get('stabilization', 'none') != 'none':
            raise ValueError('projected history currently supports mean_tensor and no stabilization')
        super().__init__(*args, **kwargs)
        self._mapped_ready = False
        self._mapped_stats = {}

    def _transfer_to_centers(self):
        self._mapped_ready = False
        return super()._transfer_to_centers()

    def _prepare_material_map(self):
        if self._mapped_ready:
            return
        start = time.perf_counter()
        n, nc = int(self.n_active_nodes.numpy()[0]), int(self.n_active_centers.numpy()[0])
        ndof, cdof = self.ndof2bijk[:n].numpy(), self.cdof2bijk[:nc].numpy()
        blocks = self.block_xyz_by_id[:int(self.bcn)].numpy()
        def coords(address):
            local = address[:, 1]
            return blocks[address[:, 0]]*B + np.column_stack((local//(B*B), (local//B) % B, local % B))
        self.mapped_centers, self.mapped_nodes = coords(cdof), coords(ndof)
        self.mapped_old_x = self.ptc_x.numpy().copy()
        self.mapped_old_F = self.ptc_F.numpy().copy()
        maps, F0, V, W, S, D = frozen_maps(self.mapped_old_x, self.ptc_vol0.numpy(), self.mapped_old_F,
                                         self.mapped_centers, self.mapped_nodes, self.dx)
        self.mapped_host_maps, self.mapped_W = maps, W
        self.mapped_S, self.mapped_D = S, D
        offsets, ids, gradients = pack_maps(maps)
        address = (cdof[:, 0], cdof[:, 1]//(B*B), cdof[:, 1] % (B*B))
        A = self.aniso_A0[:,:int(self.bcn)].numpy()[0][address]
        for name, values, dtype in [('offsets', offsets, int), ('ids', ids, int), ('gradients', gradients, vec3),
                                    ('F0', F0, mat33), ('A', A, mat33), ('V', V, real), ('address', cdof, wp.vec2i)]:
            setattr(self, 'mapped_'+name, wp.array(values, dtype=dtype, device=self.device))
        self.mapped_trial = wp.clone(self.mapped_F0)
        self.mapped_values = wp.zeros_like(self.node_residual)
        self.mapped_force_buffer = wp.zeros_like(self.node_residual)
        self._mapped_stats = dict(material_map_seconds=time.perf_counter()-start,
                                  material_map_entries=len(ids), material_samples=nc)
        self._mapped_ready = True

    def evaluate_residual(self):
        self._prepare_material_map()
        self.node_residual.zero_(); self.node_Hii_inv.zero_(); self.aniso_invalid_trial.zero_()
        wp.launch(node_inertia, dim=int(self.n_active_nodes.numpy()[0]), inputs=[self.ndof2bijk, self.grid_m,
                  self.grid_v, self.grid_v_it, self.mapped_values, self.node_residual, self.node_Hii_inv,
                  self.gravity, self.dt], device=self.device)
        p = self.aniso_params
        wp.launch(mapped_residual, dim=len(self.mapped_V), inputs=[self.mapped_offsets, self.mapped_ids,
                  self.mapped_gradients, self.mapped_values, self.mapped_F0, self.mapped_A, self.mapped_V,
                  self.mapped_trial, self.node_residual, self.aniso_invalid_trial, self.dt, p.mu, p.lam, p.k_f], device=self.device)
        wp.launch(scatter_trial, dim=len(self.mapped_V), inputs=[self.mapped_address, self.mapped_trial,
                  self.aniso_trial_F], device=self.device)
        self._project(self.node_residual)
        return int(self.aniso_invalid_trial.numpy()[0]) == 0

    def _elastic_potential(self):
        p = self.aniso_params
        wp.launch(particle_potential, dim=len(self.mapped_V), inputs=[self.mapped_V, self.mapped_trial,
                  self.mapped_A, p.mu, p.lam, p.k_f, self._potential_sum], device=self.device)

    def apply_tangent(self, p, Ap, project_pd=False):
        p = self.project_direction(p)
        Ap.zero_()
        wp.launch(node_tangent_mass, dim=int(self.n_active_nodes.numpy()[0]), inputs=[self.ndof2bijk,
                  self.grid_m, p, Ap], device=self.device)
        a = self.aniso_params
        wp.launch(mapped_tangent, dim=len(self.mapped_V), inputs=[self.mapped_offsets, self.mapped_ids,
                  self.mapped_gradients, p, self.mapped_trial, self.mapped_A, self.mapped_V, Ap,
                  self.dt, a.mu, a.lam, a.k_f, int(project_pd)], device=self.device)
        self._project(Ap)

    def projected_internal_force(self):
        self.mapped_force_buffer.zero_()
        p = self.aniso_params
        wp.launch(mapped_force, dim=len(self.mapped_V), inputs=[self.mapped_offsets, self.mapped_ids,
                  self.mapped_gradients, self.mapped_trial, self.mapped_A, self.mapped_V,
                  self.mapped_force_buffer, p.mu, p.lam, p.k_f], device=self.device)
        return self.mapped_force_buffer[:int(self.n_active_nodes.numpy()[0])].numpy().copy()

    def _aniso_implicit_step(self, **kwargs):
        if kwargs.get('damping', 1.) != 1.:
            raise ValueError('projected history requires damping=1')
        success = super()._aniso_implicit_step(**kwargs)
        if success:
            # The base transfer updates particles with the unchanged physical L.
            # Commit the center state from the SAME trial used for energy/forces.
            wp.launch(scatter_trial, dim=len(self.mapped_V), inputs=[self.mapped_address, self.mapped_trial,
                      self.aniso_committed_F], device=self.device)
        self.last_step_stats.update(self._mapped_stats, history_consistency=self.history_consistency)
        return success

    def _aniso_memory_bytes(self):
        return super()._aniso_memory_bytes() + sum(a.size*type_size_in_bytes(a.dtype)
            for k, a in vars(self).items() if k.startswith('mapped_') and isinstance(a, wp.array))
