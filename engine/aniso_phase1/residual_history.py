"""Opt-in local-history integration with a consistent derived center history.

E_local = sum_cp Vc Wcp psi(Fc + (Fp-Fc), Ap) = sum_p Vp psi(Fp).
The actual Lite particle trial and commit map are identical. Optional existing
corotated stabilization supplies an objective, energy-derived hourglass term.
The center transfer path is retained to preserve incremental APIC/BC impulses;
the material quadrature is explicitly particle based and is reported as such.
"""
import numpy as np
import warp as wp
from warp._src.types import type_size_in_bytes
from engine.types import real, mat33, vec3
from .projected_history import (ProjectedHistoryLiteSolver, pack_maps, mapped_residual,
    mapped_tangent, mapped_force, scatter_trial)
from .particle_quadrature import node_inertia, node_tangent_mass
from .potential import particle_potential
from .diagnostics import ParticleEnergyLedger
from .enhancements import CenterEnhancements, add_correction, add_energy


@wp.kernel
def average_trial(offsets: wp.array(dtype=int), particles: wp.array(dtype=int), weights: wp.array(dtype=real),
                  Fp: wp.array(dtype=mat33), Fc: wp.array(dtype=mat33)):
    c = wp.tid()
    value = mat33(real(0))
    for k in range(offsets[c], offsets[c+1]):
        value += weights[k]*Fp[particles[k]]
    Fc[c] = value


class ResidualHistoryLiteSolver(ProjectedHistoryLiteSolver):
    history_consistency = 'residual_center'
    material_quadrature = 'individual_particles'

    def __init__(self, *args, **kwargs):
        kind = kwargs.pop('stabilization', 'none')
        if kind not in ('none', 'corotated'):
            raise ValueError('local history currently supports none or corotated stabilization')
        super().__init__(*args, stabilization='none', **kwargs)
        self.stabilization = kind
        if kind == 'corotated':
            if self.aniso_params.mu <= 0 or self.aniso_params.lam+2*self.aniso_params.mu/3 <= 0:
                raise ValueError('corotated stabilization requires positive shear and bulk modulus')
            self.enhancements = CenterEnhancements(self)
        if kwargs.get('energy_diagnostics', False):
            self.energy_ledger = ParticleEnergyLedger()

    def _prepare_material_map(self):
        if self._mapped_ready:
            return
        super()._prepare_material_map()
        F = self.mapped_old_F
        G = tuple(self.mapped_S @ d for d in self.mapped_D)
        maps = tuple(sum(g.multiply(F[:, j, k, None]) for j, g in enumerate(G)).tocsr() for k in range(3))
        self.local_host_maps = maps
        offsets, ids, gradients = pack_maps(maps)
        for name, values, dtype in (
            ('offsets', offsets, int), ('ids', ids, int), ('gradients', gradients, vec3),
            ('F0', F, mat33), ('A', self.ptc_A0.numpy(), mat33), ('V', self.ptc_vol0.numpy(), real),
            ('W_offsets', self.mapped_W.indptr.astype(np.int32), int),
            ('W_ids', self.mapped_W.indices.astype(np.int32), int), ('W_weights', self.mapped_W.data, real)):
            setattr(self, 'local_'+name, wp.array(values, dtype=dtype, device=self.device))
        self.local_trial = wp.clone(self.local_F0)
        self._mapped_stats.update(material_quadrature=self.material_quadrature,
                                  material_samples=self.n_ptc, local_map_entries=len(ids))

    def evaluate_residual(self):
        self._prepare_material_map()
        self.node_residual.zero_(); self.node_Hii_inv.zero_(); self.aniso_invalid_trial.zero_()
        wp.launch(node_inertia, dim=int(self.n_active_nodes.numpy()[0]), inputs=[self.ndof2bijk,
            self.grid_m, self.grid_v, self.grid_v_it, self.mapped_values, self.node_residual,
            self.node_Hii_inv, self.gravity, self.dt], device=self.device)
        a = self.aniso_params
        wp.launch(mapped_residual, dim=self.n_ptc, inputs=[self.local_offsets, self.local_ids,
            self.local_gradients, self.mapped_values, self.local_F0, self.local_A, self.local_V,
            self.local_trial, self.node_residual, self.aniso_invalid_trial, self.dt, a.mu, a.lam, a.k_f], device=self.device)
        wp.launch(average_trial, dim=len(self.mapped_V), inputs=[self.local_W_offsets, self.local_W_ids,
            self.local_W_weights, self.local_trial, self.mapped_trial], device=self.device)
        wp.launch(scatter_trial, dim=len(self.mapped_V), inputs=[self.mapped_address, self.mapped_trial,
            self.aniso_trial_F], device=self.device)
        if self.enhancements is not None:
            self.enhancements.correction()
            wp.launch(add_correction, dim=int(self.n_active_nodes.numpy()[0]),
                      inputs=[self.enhancements.extra, self.node_residual], device=self.device)
        self._project(self.node_residual)
        return int(self.aniso_invalid_trial.numpy()[0]) == 0

    def _elastic_potential(self):
        a = self.aniso_params
        wp.launch(particle_potential, dim=self.n_ptc, inputs=[self.local_V, self.local_trial,
            self.local_A, a.mu, a.lam, a.k_f, self._potential_sum], device=self.device)
        if self.enhancements is not None:
            wp.launch(add_energy, dim=2, inputs=[self.enhancements.energy,
                self.enhancements.hg_energy, self._potential_sum], device=self.device)

    def incremental_potential(self):
        value = super().incremental_potential()
        # Material samples are particles despite the retained center transfer path.
        count = int(self.n_active_nodes.numpy()[0])+self.n_ptc
        if self.enhancements is not None:
            count += 8*len(self.mapped_V)
        _, scale = self._potential_sum.numpy()
        self._last_potential_roundoff = np.finfo(float).eps*(64*scale+(count+16)*abs(value))
        return value

    def apply_tangent(self, p, Ap, project_pd=False):
        p = self.project_direction(p)
        Ap.zero_()
        wp.launch(node_tangent_mass, dim=int(self.n_active_nodes.numpy()[0]),
                  inputs=[self.ndof2bijk, self.grid_m, p, Ap], device=self.device)
        a = self.aniso_params
        wp.launch(mapped_tangent, dim=self.n_ptc, inputs=[self.local_offsets, self.local_ids,
            self.local_gradients, p, self.local_trial, self.local_A, self.local_V, Ap,
            self.dt, a.mu, a.lam, a.k_f, int(project_pd)], device=self.device)
        if self.enhancements is not None:
            self.enhancements.tangent(p, Ap, project_pd)
        self._project(Ap)

    def projected_internal_force(self):
        # grip_reactions adds the stabilizer's unprojected force separately.
        self.mapped_force_buffer.zero_()
        a = self.aniso_params
        wp.launch(mapped_force, dim=self.n_ptc, inputs=[self.local_offsets, self.local_ids,
            self.local_gradients, self.local_trial, self.local_A, self.local_V,
            self.mapped_force_buffer, a.mu, a.lam, a.k_f], device=self.device)
        return self.mapped_force_buffer[:int(self.n_active_nodes.numpy()[0])].numpy().copy()

    def _aniso_memory_bytes(self):
        return super()._aniso_memory_bytes()+sum(a.size*type_size_in_bytes(a.dtype)
            for k, a in vars(self).items() if k.startswith('local_') and isinstance(a, wp.array))
