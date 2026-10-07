"""Opt-in, synchronized energy accounting for the actual two-level transfers.

Host diagnostics are deliberately excluded from performance measurements.
"""
from __future__ import annotations

import csv
import io
import itertools
import numpy as np
from engine.sp_grid import B


def particle_kinetic(x, v, G, mass, dx, enable_apic=True):
    """K of v_p + G_p (x_i-x_p), with w_pi = sum_c w_pc / 8.

    For complete trilinear center support, D_p is diagonal with entries
    dx² [f_a(1-f_a)+1/4]. The 1/4 is the center-to-node second moment.
    This is the energy before either mass-weighted mixing operation.
    """
    translation = float(.5 * np.sum(mass[:, None] * v * v))
    affine = 0.
    if enable_apic:
        f = x / dx - .5
        f -= np.floor(f)
        D = dx * dx * (f * (1-f) + .25)
        affine = float(.5 * np.sum(mass[:, None, None] * G * G * D[:, None, :]))
    return translation, affine


def energy_density(F, A, params):
    if len(F) == 0:
        return np.zeros(0)
    if np.any(np.linalg.det(F) <= 0) or not np.isfinite(F).all():
        raise ValueError("energy diagnostic requires finite positive-J states")
    logs = np.log(np.maximum(np.linalg.svd(F, compute_uv=False), 1e-12))
    i4 = np.einsum('pij,pki,pkj->p', A, F, F)
    return params.mu * np.sum(logs**2, axis=1) + .5 * params.lam * logs.sum(axis=1)**2 + .5 * params.k_f * (i4 - 1)**2


def center_snapshot(s):
    n = int(s.bcn)
    vol = s.center_vol[:, :n].numpy()[0].reshape(-1)
    active = vol > 0
    indices = np.flatnonzero(active)
    local = np.stack(np.unravel_index(indices % B**3, (B, B, B)), axis=1)
    coords = s.block_xyz_by_id[:n].numpy()[indices // B**3] * B + local
    F = s.aniso_committed_F[:, :n].numpy()[0].reshape(-1, 3, 3)[active]
    A = s.aniso_A0[:, :n].numpy()[0].reshape(-1, 3, 3)[active]
    psi = energy_density(F, A, s.aniso_params)
    if getattr(s,'direction_model','mean_tensor')=='fourth_moment':
        M=s.enhancements.M.numpy()
        C=np.einsum('pji,pjk->pik',F,F).reshape(-1,9)
        av=A.reshape(-1,9)
        psi+=.5*s.aniso_params.k_f*(np.einsum('pi,pij,pj->p',C,M,C)-np.einsum('pi,pi->p',C,av)**2)
    return coords, vol[active], psi, active


def grid_kinetic(s, field):
    m = s.grid_m[:int(s.bcn)].numpy()
    v = field[:int(s.bcn)].numpy()
    return float(.5 * np.sum(m[..., None] * v*v))


class EnergyLedger:
    """A telescoping budget, with no assumption of energy conservation.

    Terms are signed changes. 'solve' includes integration, force formulation,
    and nonlinear/linear errors; it must be interpreted using parameter sweeps.
    """
    def __init__(self):
        self.rows = []
        self.history = {}  # Last energy density at each global center, including inactive ones.
        self.previous_elastic = 0.
        self.previous_volumes = {}
        self.current = {}

    def kinetic(self, s):
        x = s.ptc_x.numpy()
        # Formula assumes full support; reject clipped stencils explicitly.
        if np.any(x < .5*s.dx) or np.any(x > (np.asarray(tuple(s.center_size))- .5)*s.dx):
            raise ValueError("APIC energy requires complete particle-to-center support")
        return particle_kinetic(x, s.ptc_v.numpy(), s.ptc_G.numpy(), s.ptc_m.numpy(), s.dx, s.enable_apic)

    def elastic_energy(self, s, snapshot=None):
        _, volumes, psi, _ = center_snapshot(s) if snapshot is None else snapshot
        return float(np.dot(volumes, psi))

    def begin(self, s):
        self.particle_momentum_start = np.sum(s.ptc_m.numpy()[:, None]*s.ptc_v.numpy(), axis=0)
        kt, ka = self.kinetic(s)
        self.k0 = kt + ka
        self.e0 = self.k0 + self.previous_elastic
        self.current = {}
        self.reactivations0 = int(s.aniso_reactivation_count.numpy()[0])
        if not self.rows:
            self.rows.append(dict(step=0, time=float(s.sim_time), kinetic_translation=kt,
                                  kinetic_affine=ka, kinetic=kt+ka, elastic=0., mechanical=kt+ka,
                                  delta_kinetic=0., delta_elastic=0., delta_mechanical=0., cumulative_delta=0.))

    def initialize_centers(self, s):
        """Include seeded prestress in the initial energy, not as a reset injection."""
        coords, volumes, psi, _ = center_snapshot(s)
        self.previous_elastic = self.elastic_energy(s, (coords, volumes, psi, None))
        self.history = {tuple(c):float(e) for c,e in zip(coords,psi)}
        self.previous_volumes = {tuple(c):float(v) for c,v in zip(coords,volumes)}
        self.e0 = self.k0 + self.previous_elastic
        self.rows[0].update(elastic=self.previous_elastic, mechanical=self.e0)

    def p2c(self, s):
        coords, volumes, psi, active = center_snapshot(s)
        keys = [tuple(c) for c in coords]
        oldpsi = np.array([self.history.get(k, 0.) for k in keys])
        reset = float(np.dot(volumes, psi-oldpsi))
        self.u_start = float(np.dot(volumes, psi))
        transporting = getattr(s, 'history_mode', 'grid_locked') == 'particle_resample'
        self.current['state_reset_delta'] = 0. if transporting else reset
        self.current['state_transport_delta'] = reset if transporting else 0.
        self.current['volume_remap_delta'] = float(np.dot(volumes, oldpsi)) - self.previous_elastic
        self.current['reactivated_centers'] = sum(k in self.history and k not in self.previous_volumes for k in keys)
        self.current['new_centers'] = sum(k not in self.history for k in keys)
        n = int(s.bcn)
        m = s.center_m[:n].numpy().reshape(-1)[active]
        v = s.center_v[:n].numpy().reshape(-1, 3)[active]
        G = s.center_G[:n].numpy().reshape(-1, 3, 3)[active]
        kc = float(.5*np.sum(m[:, None]*v*v))
        if s.enable_apic:
            kc += float(s.dx**2/8 * np.sum(m[:, None, None]*G*G))
        self.current['p2c_delta'] = kc-self.k0
        # Reconstruct the exact pre-projection C2G field using the same 8 weights.
        corners = np.asarray(list(itertools.product((0, 1), repeat=3)))
        nodes = coords[:, None, :] + corners[None, :, :]
        size = tuple(s.grid_size)
        ids = np.ravel_multi_index(nodes.reshape(-1, 3).T, size)
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        masses = np.broadcast_to(m[:, None]/8, (len(m), 8)).reshape(-1)
        velocities = np.broadcast_to(v[:, None, :], (len(m), 8, 3)).copy()
        if s.enable_apic:
            velocities += np.einsum('pij,cj->pci', G, (corners-.5)*s.dx)
        nodem = np.bincount(inverse, weights=masses)
        momentum = np.stack([np.bincount(inverse, weights=masses*velocities[:, :, d].reshape(-1)) for d in range(3)], axis=1)
        self.k_raw = float(.5*np.sum(momentum*momentum / nodem[:, None]))
        self.current['c2g_delta'] = self.k_raw-kc
        self.nodes = np.stack(np.unravel_index(unique_ids, size), axis=1)
        self.node_m, self.raw_v = nodem, momentum/nodem[:, None]
        self.inverse, self.corners = inverse, corners

    def grid(self, s):
        self.k_projected = grid_kinetic(s, s.grid_v)
        self.current['boundary_projection_delta'] = self.k_projected-self.k_raw

    def solved(self, s):
        self.k_solved = grid_kinetic(s, s.grid_v_it)

    def transferred_grid(self, s):
        self.k_final_grid = grid_kinetic(s, s.grid_v_new)
        self.current['final_projection_damping_delta'] = self.k_final_grid-self.k_solved

    def finish(self, s):
        self.particle_momentum_end = np.sum(s.ptc_m.numpy()[:, None]*s.ptc_v.numpy(), axis=0)
        n = int(s.bcn)
        grid_momentum = np.sum(s.grid_m[:n].numpy()[..., None]*s.grid_v_new[:n].numpy(), axis=(0, 1, 2, 3))
        gap = self.particle_momentum_end-grid_momentum
        self.current["particle_grid_momentum_gap_norm"] = float(np.linalg.norm(gap))
        for axis, value in zip("xyz", gap):
            self.current["particle_grid_momentum_gap_"+axis] = float(value)
        kt, ka = self.kinetic(s)
        coords, volumes, psi, _ = center_snapshot(s)
        elastic = self.elastic_energy(s, (coords, volumes, psi, None))
        if getattr(s,'enhancements',None) is not None:
            hg=float(s.enhancements.hg_energy.numpy()[0])
            elastic+=hg
            self.current['stabilization_energy']=hg
            self.current['stabilization_solve_delta']=hg-self.current.get('stabilization_start_energy',hg)
        self.current['solve_delta'] = self.k_solved-self.k_projected + elastic-self.u_start
        self.current['g2p_delta'] = kt+ka-self.k_final_grid
        mechanical = kt+ka+elastic
        delta = mechanical-self.e0
        terms = ('p2c_delta', 'c2g_delta', 'boundary_projection_delta',
                 'state_reset_delta', 'state_transport_delta', 'volume_remap_delta', 'solve_delta',
                 'final_projection_damping_delta', 'g2p_delta')
        row = dict(step=s.sim_steps, time=float(s.sim_time), kinetic_translation=kt,
                   kinetic_affine=ka, kinetic=kt+ka, elastic=elastic, mechanical=mechanical,
                   delta_kinetic=kt+ka-self.k0, delta_elastic=elastic-self.previous_elastic,
                   delta_mechanical=delta, cumulative_delta=mechanical-self.rows[0]['mechanical'],
                   budget_closure=delta-sum(self.current[k] for k in terms),
                   reactivation_events=int(s.aniso_reactivation_count.numpy()[0])-self.reactivations0,
                   **self.current)
        for k in ('newton_iterations', 'cg_iterations', 'last_residual_norm', 'last_update'):
            row[k] = s.last_step_stats[k]
        self.rows.append(row)
        self.previous_elastic = elastic
        self.previous_volumes = {tuple(c): V for c, V in zip(coords, volumes)}
        self.history.update({tuple(c): e for c, e in zip(coords, psi)})

    def csv(self):
        out = io.StringIO()
        fields = list(dict.fromkeys(k for row in self.rows for k in row))
        writer = csv.DictWriter(out, fields, restval=0.)
        writer.writeheader()
        writer.writerows(self.rows)
        return out.getvalue()


class ParticleEnergyLedger(EnergyLedger):
    """Same transfer kinetic budget, but elastic energy is particle quadrature.

    No center-F remapping term is attributed to particle elastic energy. The
    optional nodal stabilization history is still rebuilt by the shared solver.
    """
    def elastic_energy(self, s, snapshot=None):
        return float(np.dot(s.ptc_vol0.numpy(), energy_density(s.ptc_F.numpy(),s.ptc_A0.numpy(),s.aniso_params)))

    def p2c(self, s):
        super().p2c(s)
        self.u_start=self.elastic_energy(s)
        self.current['state_reset_delta']=0.
        self.current['state_transport_delta']=0.
        # Includes minus the previous stabilization energy, which the shared
        # step subsequently combines with the newly reconstructed contribution.
        self.current['volume_remap_delta']=self.u_start-self.previous_elastic

    def solved(self, s):
        super().solved(s)
        self.step_start_particle_F=s.ptc_F.numpy().copy()
