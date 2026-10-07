"""Experimental sparse-grid implicit solver using the stage-one kernels."""

from __future__ import annotations

import numpy as np
import time
import warp as wp
from warp._src.types import type_size_in_bytes

from engine.solver3d import MPMSolver
from engine.types import Material, mat33, real, vec3
from engine.sp_grid import B
from engine.kernel.d3.kernel_lite import lite_c2g, lite_c2p_kernel, lite_g2c_kernel
from engine.kernel.d3.kernel_lite_misc import (
    lite_initialize_v_iterate_kernel,
    lite_center_grad_from_grid_velocity_kernel,
    lite_reduce_active_nodes_and_centers_kernel,
)
from engine.kernel.d3.helper_misc import update_vnew_with_iterate_kernel
from engine.kernel.d3.kernel_lite_implicit import lite_implicit_precond_kernel_Hii, lite_implicit_grad_v_dv_kernel
from engine.math.conjugate_gradient import matrix_free_cg

from .kernels import commit_center_trial_kernel
from .sparse_kernels import (
    aniso_commit_kernel,
    aniso_center_tau_kernel,
    aniso_df_kernel,
    aniso_p2c,
    aniso_residual_kernel,
    aniso_rollback_trial_kernel,
    aniso_stress_scratch_kernel,
    aniso_update_trial_kernel,
    remap_aniso_int_kernel,
    remap_aniso_mat33_kernel,
    remap_aniso_u8_kernel,
    scatter_aniso_int_kernel,
    scatter_aniso_mat33_kernel,
    scatter_aniso_u8_kernel,
)
from .types import AnisotropicMaterialParams
from .constitutive import structure_tensor_mixing
from .linear import nonsymmetric_solve, guarded_pcg
from engine.math.conjugate_gradient import dot_array
from engine.boundary_utils import boundary_projection_kernel
from .potential import inertia_potential, center_potential


class AnisotropicLiteImplicitSolver(MPMSolver):
    """A one-material isolated sparse-grid solver for phase-one anisotropy.

    The legacy ``MPMSolver`` remains unchanged at the API level.  This class
    reuses its sparse block/grid transfer and boundary storage, but owns its
    anisotropic center arrays and never calls the stress-to-stretch inversion.
    Phase one intentionally supports one material family (``n_psi == 1``).
    """

    def __init__(
        self,
        grid_size,
        params: AnisotropicMaterialParams,
        dx: float | None = None,
        device: str = "cpu",
        gravity: float = -9.81,
        ppc: float = 4,
        enable_apic: bool = True,
        flip_ratio: float = 0.9,
        energy_diagnostics: bool = False,
        force_discretization: str = "variational",
        history_mode: str = "particle_resample",
        direction_model: str = "mean_tensor",
        stabilization: str = "none",
        stabilization_strength: float = 1.,
        boundary_impulse_transfer: bool = False,
        apic_transfer: str = "overwrite",
        affine_flip_ratio: float | None = None,
        velocity_dissipation: str = "none",
    ):
        if stabilization in ('corotated','quadratic','material_quadratic') and params.lam + 2*params.mu/3 < 0:
            raise ValueError('corotated stabilization requires nonnegative reference bulk modulus')
        super().__init__(
            grid_size=grid_size,
            dx=dx,
            device=device,
            gravity=gravity,
            n_psi=1,
            ppc=ppc,
            solver_type="lite_implicit",
            enable_apic=enable_apic,
            flip_ratio=flip_ratio,
        )
        if force_discretization not in ("variational", "legacy_kirchhoff"):
            raise ValueError("force_discretization must be variational or legacy_kirchhoff")
        if history_mode not in ("particle_resample", "grid_locked"):
            raise ValueError("history_mode must be particle_resample or grid_locked")
        if apic_transfer not in ("overwrite", "incremental"):
            raise ValueError("unknown APIC transfer mode")
        if apic_transfer == "incremental" and (not enable_apic or not boundary_impulse_transfer):
            raise ValueError("incremental APIC requires APIC and boundary impulse transfer")
        self.apic_transfer = apic_transfer
        if affine_flip_ratio is not None:
            if apic_transfer != "incremental":
                raise ValueError("affine_flip_ratio requires incremental APIC")
            if not np.isfinite(affine_flip_ratio) or not 0 <= affine_flip_ratio <= 1:
                raise ValueError("affine_flip_ratio must be in [0,1]")
        self.affine_flip_ratio = affine_flip_ratio
        self.velocity_dissipation = velocity_dissipation
        self.dissipation_stats = {}
        self._dissipation_unfiltered = None
        if velocity_dissipation not in ("none","null","weak"):
            raise ValueError("unknown velocity dissipation")
        self.ptc_L = wp.zeros(0, dtype=mat33, device=device)
        self._apic_difference = None
        self.boundary_impulse_transfer = bool(boundary_impulse_transfer)
        self.grid_v_raw = None
        self.history_mode = history_mode
        if direction_model not in ('mean_tensor','fourth_moment') or stabilization not in ('none','supplemental','hourglass','corotated','quadratic','material_quadratic'):
            raise ValueError('unknown direction model or stabilization')
        if not np.isfinite(stabilization_strength) or stabilization_strength < 0:
            raise ValueError('stabilization strength must be finite and nonnegative')
        if (direction_model != 'mean_tensor' or stabilization != 'none') and (force_discretization != 'variational' or history_mode != 'particle_resample'):
            raise ValueError('enhancements require variational forces and particle_resample history')
        self.direction_model=direction_model
        self.stabilization=stabilization
        self.stabilization_strength=stabilization_strength
        self.enhancements=None
        self.force_discretization = force_discretization
        self._potential_sum = wp.zeros(2, dtype=real, device=device)
        self.last_newton_trace = []
        from .diagnostics import EnergyLedger
        self.energy_ledger = EnergyLedger() if energy_diagnostics else None
        self.aniso_params = params
        self._allocate_aniso_state()
        self._aniso_has_active_blocks = False
        self._aniso_seen_block_coords: set[tuple[int, int, int]] = set()
        self.ptc_A0 = wp.zeros(0, dtype=mat33, device=device)
        self.ptc_reference_x = wp.zeros(0,dtype=vec3,device=device)
        if direction_model != 'mean_tensor' or stabilization != 'none':
            from .enhancements import CenterEnhancements
            self.enhancements=CenterEnhancements(self)
        self.last_step_stats: dict[str, float | int | bool] = {}
        # The legacy solver allocates sparse boundary storage only after
        # ``paint_boundary``.  The isolated solver also supports free-boundary
        # affine probes, so install an explicit empty boundary field up front.
        self.bc_block2bid = wp.full(self.block_size, -1, dtype=wp.int32, device=device)
        self.bc_type = wp.zeros((1, B, B, B), dtype=wp.int32, device=device)
        self.bc_norm = wp.zeros((1, B, B, B), dtype=vec3, device=device)
        self.bc_velo = wp.zeros((1, B, B, B), dtype=vec3, device=device)

        # Keep the inherited particle update path supplied with a valid
        # isotropic parameter record.  Center forces use only aniso arrays.
        lame_sum = params.lam + params.mu
        if lame_sum <= 0.0:
            raise ValueError("lambda + mu must be positive for the isotropic particle fallback")
        E = params.mu * (3.0 * params.lam + 2.0 * params.mu) / lame_sum
        nu = params.lam / (2.0 * lame_sum)
        super().add_material(0, Material.elastic, E=E, nu=nu)

    @property
    def ptc_C(self):
        """APIC affine coefficient; ptc_G is the backward-compatible storage name."""
        return self.ptc_G

    def _allocate_aniso_state(self) -> None:
        shape = (1, self.MAX_BLOCKS, B, B * B)
        self.aniso_committed_F = wp.zeros(shape, dtype=mat33, device=self.device)
        self.aniso_trial_F = wp.zeros(shape, dtype=mat33, device=self.device)
        self.aniso_A0 = wp.zeros(shape, dtype=mat33, device=self.device)
        self.aniso_A0_sum = wp.zeros(shape, dtype=mat33, device=self.device)
        self.aniso_state_valid = wp.zeros(shape, dtype=wp.int32, device=self.device)
        self.aniso_occupied_prev = wp.zeros(shape, dtype=wp.uint8, device=self.device)
        self.aniso_reactivation_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.aniso_invalid_trial = wp.zeros(1, dtype=wp.int32, device=self.device)

    def _resize_aniso_state_if_needed(self) -> None:
        if self.aniso_committed_F.shape[1] == self.MAX_BLOCKS:
            return
        old_count = self.aniso_committed_F.shape[1]
        old_reactivation_count = self.aniso_reactivation_count.numpy()
        old = [a.numpy() for a in (self.aniso_committed_F, self.aniso_trial_F, self.aniso_A0, self.aniso_A0_sum, self.aniso_state_valid, self.aniso_occupied_prev)]
        self._allocate_aniso_state()
        for target, source in zip((self.aniso_committed_F, self.aniso_trial_F, self.aniso_A0, self.aniso_A0_sum, self.aniso_state_valid, self.aniso_occupied_prev), old):
            host = target.numpy()
            host[:, :old_count] = source
            target.assign(wp.array(host, dtype=target.dtype, device=self.device))
        self.aniso_reactivation_count.assign(wp.array(old_reactivation_count, dtype=wp.int32, device=self.device))

    def activate_sparse_grid(self):
        old_count = 0
        old_coords = np.empty((0, 3), dtype=np.int32)
        old_arrays = None
        if self._aniso_has_active_blocks:
            old_count = int(self.block_count.numpy()[0])
            old_coords = self.block_xyz_by_id.numpy()[:old_count].copy()
            old_arrays = (
                self.aniso_committed_F,
                self.aniso_trial_F,
                self.aniso_A0,
                self.aniso_state_valid,
                self.aniso_occupied_prev,
            )
        super().activate_sparse_grid()
        new_count = int(self.bcn)
        old_coord_set = {tuple(int(v) for v in coord) for coord in old_coords}
        new_coords_for_history = self.block_xyz_by_id.numpy()[:new_count]
        reactivated_blocks = [
            tuple(int(v) for v in coord)
            for coord in new_coords_for_history
            if tuple(int(v) for v in coord) in self._aniso_seen_block_coords
            and tuple(int(v) for v in coord) not in old_coord_set
        ]
        if reactivated_blocks:
            count = int(self.aniso_reactivation_count.numpy()[0]) + len(reactivated_blocks)
            self.aniso_reactivation_count.assign(wp.array([count], dtype=wp.int32, device=self.device))
        self._aniso_seen_block_coords.update(tuple(int(v) for v in coord) for coord in new_coords_for_history)
        state_resized = self.aniso_committed_F.shape[1] != self.MAX_BLOCKS
        if state_resized:
            old_reactivation_count = self.aniso_reactivation_count.numpy()
            self._allocate_aniso_state()
            self.aniso_reactivation_count.assign(
                wp.array(old_reactivation_count, dtype=wp.int32, device=self.device)
            )
        if old_arrays is not None and old_count > 0 and new_count > 0:
            new_coords = self.block_xyz_by_id.numpy()[:new_count]
            old_lookup = {tuple(int(v) for v in coord): i for i, coord in enumerate(old_coords)}
            old_to_new = np.full(old_count, -1, dtype=np.int32)
            identity = old_count == new_count and not state_resized
            for new_bid, coord in enumerate(new_coords):
                old_bid = old_lookup.get(tuple(int(v) for v in coord), -1)
                if old_bid >= 0:
                    old_to_new[old_bid] = new_bid
                    identity = identity and old_bid == new_bid
            if not identity:
                self._remap_aniso_state(old_arrays, old_to_new, new_count)
        elif old_arrays is not None:
            # No active blocks remain.  Keep the arrays valid and let the next
            # activation initialize centers from the incoming particle data.
            self.aniso_occupied_prev.zero_()
            self.aniso_state_valid.zero_()
            self.aniso_trial_F.zero_()
        self._aniso_has_active_blocks = True

    def _remap_aniso_state(self, old_arrays, old_to_new: np.ndarray, new_count: int) -> None:
        """Preserve center history when prefix-sum block IDs change."""
        old_committed, old_trial, old_A0, old_valid, old_occupied = old_arrays
        old_count = len(old_to_new)
        map_wp = wp.array(old_to_new, dtype=wp.int32, device=self.device)
        compact_committed = wp.zeros((1, new_count, B, B * B), dtype=mat33, device=self.device)
        compact_trial = wp.zeros_like(compact_committed)
        compact_A0 = wp.zeros_like(compact_committed)
        compact_valid = wp.zeros((1, new_count, B, B * B), dtype=wp.int32, device=self.device)
        compact_occupied = wp.zeros((1, new_count, B, B * B), dtype=wp.uint8, device=self.device)
        dim_old = (old_count, B, B, B)
        wp.launch(remap_aniso_mat33_kernel, dim=dim_old, inputs=[map_wp, old_committed, compact_committed], device=self.device)
        wp.launch(remap_aniso_mat33_kernel, dim=dim_old, inputs=[map_wp, old_trial, compact_trial], device=self.device)
        wp.launch(remap_aniso_mat33_kernel, dim=dim_old, inputs=[map_wp, old_A0, compact_A0], device=self.device)
        wp.launch(remap_aniso_int_kernel, dim=dim_old, inputs=[map_wp, old_valid, compact_valid], device=self.device)
        wp.launch(remap_aniso_u8_kernel, dim=dim_old, inputs=[map_wp, old_occupied, compact_occupied], device=self.device)

        # If capacity grew, allocate fresh arrays before scattering.  The
        # reactivation counter is scalar and is preserved explicitly.
        if self.aniso_committed_F.shape[1] != self.MAX_BLOCKS:
            raise RuntimeError("anisotropic state capacity is inconsistent")
        self.aniso_committed_F.zero_()
        self.aniso_trial_F.zero_()
        self.aniso_A0.zero_()
        self.aniso_state_valid.zero_()
        self.aniso_occupied_prev.zero_()
        dim_new = (new_count, B, B, B)
        wp.launch(scatter_aniso_mat33_kernel, dim=dim_new, inputs=[compact_committed, self.aniso_committed_F], device=self.device)
        wp.launch(scatter_aniso_mat33_kernel, dim=dim_new, inputs=[compact_trial, self.aniso_trial_F], device=self.device)
        wp.launch(scatter_aniso_mat33_kernel, dim=dim_new, inputs=[compact_A0, self.aniso_A0], device=self.device)
        wp.launch(scatter_aniso_int_kernel, dim=dim_new, inputs=[compact_valid, self.aniso_state_valid], device=self.device)
        wp.launch(scatter_aniso_u8_kernel, dim=dim_new, inputs=[compact_occupied, self.aniso_occupied_prev], device=self.device)

    def reset_grid(self):
        super().reset_grid()
        self.aniso_A0_sum.zero_()
        self.aniso_trial_F.zero_()

    def add_material(self, *args, **kwargs):
        raise ValueError("anisotropic solver supports only its constructor's single elastic material")

    def seed_particles(
        self,
        x,
        use_material_k=0,
        density=1.0,
        vol0=1.0,
        velocity=None,
        fiber_directions=None,
        velocity_gradient=None,
        deformation_gradient=None,
        reference_positions=None,
        **kwargs,
    ):
        # Validate all anisotropic inputs before the inherited append mutates state.
        if use_material_k != 0:
            raise ValueError("anisotropic solver supports only material index 0")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1:] != (3,) or not len(x) or not np.isfinite(x).all():
            raise ValueError("x must be a nonempty finite (N, 3) array")
        if not np.isfinite([density, vol0]).all() or density <= 0 or vol0 <= 0:
            raise ValueError("density and vol0 must be finite and positive")
        directions = (np.repeat(self.aniso_params.fiber_direction[None, :], len(x), axis=0)
                      if fiber_directions is None else np.asarray(fiber_directions, dtype=np.float64))
        if directions.shape != (len(x), 3):
            raise ValueError("fiber_directions must have shape (N, 3)")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all() or not np.isfinite(directions).all():
            raise ValueError("fiber_directions must be finite and non-zero")
        if velocity is not None and not np.isfinite(velocity).all():
            raise ValueError("velocity must be finite")
        if velocity_gradient is not None:
            gradient_array = np.asarray(velocity_gradient, dtype=np.float64)
            if gradient_array.shape == (3, 3):
                gradient_array = np.repeat(gradient_array[None, :, :], len(x), axis=0)
            if gradient_array.shape != (len(x), 3, 3) or not np.isfinite(gradient_array).all():
                raise ValueError("velocity_gradient must be finite with shape (3, 3) or (N, 3, 3)")
        if deformation_gradient is not None:
            F = np.asarray(deformation_gradient, dtype=np.float64)
            if F.shape == (3, 3):
                F = np.broadcast_to(F, (len(x), 3, 3)).copy()
            if F.shape != (len(x), 3, 3) or not np.isfinite(F).all() or np.any(np.linalg.det(F) <= 0):
                raise ValueError("deformation_gradient must contain finite positive-J matrices")
        reference=x if reference_positions is None else np.asarray(reference_positions,dtype=float)
        if reference.shape!=x.shape or not np.isfinite(reference).all():
            raise ValueError('reference_positions must match particle positions')
        if self.stabilization!='none' and reference_positions is None and deformation_gradient is not None and not np.allclose(deformation_gradient,np.eye(3)):
            raise ValueError('predeformed stabilization requires material reference_positions')
        old_n = self.n_ptc
        per_particle_velocity = None
        if velocity is not None:
            velocity_array = np.asarray(velocity, dtype=np.float64)
            if velocity_array.shape == (len(x), 3):
                per_particle_velocity = velocity_array
                seed_velocity = np.zeros(3, dtype=np.float64)
            elif velocity_array.shape == (3,):
                seed_velocity = velocity_array
            else:
                raise ValueError("velocity must have shape (3,) or (N, 3)")
        else:
            seed_velocity = None
        super().seed_particles(x, use_material_k, density, vol0, velocity=seed_velocity, **kwargs)
        if per_particle_velocity is not None:
            old_v = self.ptc_v.numpy()[:-len(x)] if old_n else np.empty((0, 3), dtype=np.float64)
            self.ptc_v = wp.from_numpy(
                np.concatenate([old_v, per_particle_velocity], axis=0),
                dtype=vec3,
                device=self.device,
            )
        if velocity_gradient is not None:
            old_G = self.ptc_G.numpy()[:-len(x)] if old_n else np.empty((0, 3, 3), dtype=np.float64)
            self.ptc_G = wp.from_numpy(
                np.concatenate([old_G, gradient_array], axis=0),
                dtype=mat33,
                device=self.device,
            )
        # L is independent storage, initialized from the supplied physical gradient.
        initial_L = gradient_array if velocity_gradient is not None else np.zeros((len(x),3,3))
        self.ptc_L = wp.array(np.concatenate((self.ptc_L.numpy(), initial_L)), dtype=mat33, device=self.device)
        directions = directions / norms[:, None]
        A0 = np.einsum("pi,pj->pij", directions, directions)
        old = self.ptc_A0.numpy()
        all_A0 = np.concatenate([old, A0], axis=0) if old_n else A0
        self.ptc_A0 = wp.array(all_A0, dtype=mat33, device=self.device)
        self.ptc_reference_x=wp.array(np.concatenate((self.ptc_reference_x.numpy(),reference)),dtype=vec3,device=self.device)
        if deformation_gradient is not None:
            old_F = self.ptc_F[:old_n].numpy().copy() if old_n else np.empty((0,3,3))
            self.ptc_F = wp.array(np.concatenate((old_F,F)), dtype=mat33, device=self.device)

    def _validate_velocity_dissipation(self):
        if self.velocity_dissipation not in ('none','null','weak'):
            raise ValueError('unknown velocity dissipation')
        if self.velocity_dissipation != 'none':
            if (str(self.device) != 'cpu' or self.apic_transfer != 'incremental' or
                    getattr(self,'history_consistency','standard') != 'residual_center'):
                raise ValueError('velocity dissipation prototype requires CPU incremental APIC and residual_center history')
            if not np.isfinite(self.aniso_params.mu) or self.aniso_params.mu <= 0:
                raise ValueError('velocity dissipation requires positive matrix shear modulus')

    def step(self, **kwargs):
        self._validate_velocity_dissipation()
        self.dissipation_stats = {}
        self._dissipation_unfiltered = None
        if self.affine_flip_ratio is not None and (self.apic_transfer != "incremental" or
                not np.isfinite(self.affine_flip_ratio) or not 0 <= self.affine_flip_ratio <= 1):
            raise ValueError("affine_flip_ratio requires incremental APIC and a value in [0,1]")
        if self.apic_transfer == "incremental":
            if (getattr(self, "quadrature_kind", "center") != "center" or
                    not self.enable_apic or not self.boundary_impulse_transfer):
                raise ValueError("incremental APIC requires center quadrature, APIC and boundary impulse transfer")
            if not np.isfinite(self.flip_ratio) or not 0 <= self.flip_ratio <= 1:
                raise ValueError("incremental APIC requires flip_ratio in [0,1]")
        if self.boundary_impulse_transfer and getattr(self, "quadrature_kind", "center") != "center":
            raise ValueError("boundary impulse transfer currently supports center quadrature only")
        reaction_atol = kwargs.get("reaction_force_atol")
        if reaction_atol is not None:
            if not np.isfinite(reaction_atol) or reaction_atol <= 0:
                raise ValueError("reaction_force_atol must be finite and positive")
            if self.force_discretization != "variational" or kwargs.get("damping", 1.) != 1.:
                raise ValueError("reaction force stopping requires variational forces and damping=1")
            if not np.isfinite(self.dt) or self.dt <= 0:
                raise ValueError("reaction force stopping requires a positive finite dt")
        if self.enhancements is not None and kwargs.get('damping',1.)!=1.:
            raise ValueError('enhanced energy accounting currently requires damping=1')
        if self.energy_ledger is not None:
            if self.gravity != 0:
                raise ValueError("mechanical energy ledger currently requires gravity=0")
            self.energy_ledger.begin(self)
        self.activate_sparse_grid()
        self.reset_grid()
        success = self._aniso_implicit_step(**kwargs)
        if success:
            self.sim_steps += 1
            self.sim_time += self.dt
            if self.energy_ledger is not None:
                self.energy_ledger.finish(self)
        return success

    def center_particle_F_error(self) -> dict[str, float | int]:
        """Report the derived particle-F versus committed center-F mismatch."""
        if self.n_ptc == 0:
            return {"count": 0, "mean_frobenius": 0.0, "max_frobenius": 0.0}
        x = self.ptc_x.numpy()
        particle_F = self.ptc_F.numpy()
        center_F = self.aniso_committed_F.numpy()[0]
        valid = self.aniso_state_valid.numpy()[0] > 0
        block2bid = self.block2bid.numpy()
        errors = []
        center_extent = np.array([self.center_size.x, self.center_size.y, self.center_size.z], dtype=np.int64)
        for p in range(self.n_ptc):
            pos = x[p] / self.dx - 0.5
            base = np.floor(pos).astype(np.int64)
            frac = pos - base
            interpolated = np.zeros((3, 3), dtype=np.float64)
            total = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        c = base + np.array([i, j, k], dtype=np.int64)
                        if np.any(c < 0) or np.any(c >= center_extent):
                            continue
                        bc = c // B
                        bid = int(block2bid[bc[0], bc[1], bc[2]])
                        if bid < 0:
                            continue
                        local = c % B
                        lci, lcj, lck = local
                        if not valid[bid, lci, lcj * B + lck]:
                            continue
                        w = (frac[0] if i else 1.0 - frac[0]) * (frac[1] if j else 1.0 - frac[1]) * (frac[2] if k else 1.0 - frac[2])
                        interpolated += w * center_F[bid, lci, lcj * B + lck]
                        total += w
            if total > 0.0:
                errors.append(float(np.linalg.norm(particle_F[p] - interpolated / total)))
        if not errors:
            return {"count": 0, "mean_frobenius": 0.0, "max_frobenius": 0.0}
        return {"count": len(errors), "mean_frobenius": float(np.mean(errors)), "max_frobenius": float(np.max(errors))}

    def center_structure_tensor_mixing(self) -> dict[str, float | int]:
        """Report multi-direction content of occupied center structure tensors."""
        tensors = self.aniso_A0.numpy()[0]
        valid = self.aniso_state_valid.numpy()[0] > 0
        if not np.any(valid):
            return {"count": 0, "mean": 0.0, "max": 0.0}
        values = []
        for tensor in tensors[valid]:
            values.append(structure_tensor_mixing(tensor))
        values = np.asarray(values, dtype=np.float64)
        return {"count": len(values), "mean": float(np.mean(values)), "max": float(np.max(values))}

    def prepare_centers(self):
        """Prepare a frozen quadrature snapshot without advancing time or particles."""
        self.activate_sparse_grid()
        self.reset_grid()
        if not self._transfer_to_centers():
            raise ValueError("history resampling produced a nonpositive/nonfinite center F")

    def _transfer_to_centers(self):
        aniso_p2c(
            self.block_count, self.block2bid, self.block_xyz_by_id,
            self.ptc_x, self.ptc_v, self.ptc_G, self.ptc_A0,
            self.ptc_vol0, self.ptc_m,
            self.center_m, self.center_v, self.center_G, self.center_vol,
            self.aniso_A0_sum, self.aniso_A0, self.aniso_committed_F,
            self.aniso_state_valid, self.aniso_occupied_prev, self.aniso_reactivation_count,
            self.center_size, self.dx, self.n_ptc,
            self.device, self.enable_apic,
        )
        if self.history_mode == "particle_resample":
            from .history import resample_history
            valid=resample_history(self)
            if self.enhancements is not None:self.enhancements.resample()
            return valid
        return True

    def _aniso_implicit_step(self, **kwargs):
        step_start = time.perf_counter()
        stats = {
            "direction_model": self.direction_model,
            "stabilization": self.stabilization,
            "stabilization_strength": self.stabilization_strength,
            "history_mode": self.history_mode,
            "p2c_seconds": 0.0,
            "cg_seconds": 0.0,
            "material_seconds": 0.0,
            "transfer_seconds": 0.0,
            "matvec_calls": 0,
            "cg_iterations": 0,
            "newton_iterations": 0,
            "last_residual_norm": 0.0,
            "last_update": 0.0,
        }
        p2c_start = time.perf_counter()
        if not self._transfer_to_centers():
            stats.update(converged=False, history_failure=True, history_mode=self.history_mode,
                         last_residual_norm=float("inf"), step_seconds=time.perf_counter()-step_start,
                         linear_iterations=0)
            self.last_step_stats=stats
            return False
        if self.energy_ledger is not None and self.sim_steps == 0:
            self.energy_ledger.initialize_centers(self)
        if self.energy_ledger is not None:
            self.energy_ledger.p2c(self)
        stats["p2c_seconds"] = time.perf_counter() - p2c_start
        stats["reactivations"] = int(self.aniso_reactivation_count.numpy()[0])
        wp.launch(
            aniso_center_tau_kernel,
            dim=(self.bcn, B, B, B),
            inputs=[
                self.block_count, self.block_xyz_by_id, self.center_vol,
                self.aniso_committed_F, self.aniso_A0, self.center_tau,
                self.center_size, self.aniso_params.mu, self.aniso_params.lam,
                self.aniso_params.k_f,
            ],
            device=self.device,
        )
        # Preserve the unprojected C2G velocity for the FLIP increment.
        # Allocate only active-block capacity and refresh after sparse remapping.
        if self.boundary_impulse_transfer:
            if self.grid_v_raw is None or self.grid_v_raw.shape[0] < self.bcn:
                self.grid_v_raw = wp.zeros((self.bcn, B, B, B), dtype=vec3, device=self.device)
            self.grid_v_raw.zero_()
        lite_c2g(
            self.block_count, self.block2bid, self.block_xyz_by_id,
            self.bc_block2bid, self.bc_type, self.bc_norm, self.bc_velo,
            self.hf_bc_p, self.hf_bc_n, self.hf_bc_v, self.hf_bc_type, self.num_hf,
            self.center_m, self.center_v, self.center_G, self.center_vol,
            self.center_tau, self.grid_m, self.grid_v, self.grid_v_new,
            self.grid_size, self.center_size, self.gravity, self.dx, self.dt,
            self.n_psi, self.device, explicit_force=False, enable_apic=self.enable_apic,
            grid_v_raw=self.grid_v_raw if self.boundary_impulse_transfer else None,
        )
        if self.energy_ledger is not None:
            self.energy_ledger.grid(self)
        self.reduce_sparse_grid()
        if self.enhancements is not None:
            self.enhancements.prepare()
            stats.update(getattr(self.enhancements,'reconstruction_stats',{}))
            if not self.enhancements.reference_valid:
                stats.update(converged=False,reference_mapping_failure=True,
                             last_residual_norm=float('inf'),linear_iterations=0,
                             step_seconds=time.perf_counter()-step_start)
                self.last_step_stats=stats
                return False
            self.enhancements.correction(trial=False)
            if self.energy_ledger is not None:
                hg=float(self.enhancements.hg_energy.numpy()[0])
                self.energy_ledger.current['stabilization_start_energy']=hg
                previous_hg=hg if self.sim_steps==0 else self.energy_ledger.rows[-1].get('stabilization_energy',0.)
                self.energy_ledger.current['stabilization_rebuild_delta']=hg-previous_hg
                self.energy_ledger.u_start+=hg
                self.energy_ledger.current['state_transport_delta']+=hg
                if self.sim_steps==0:
                    self.energy_ledger.e0+=hg
                    self.energy_ledger.previous_elastic+=hg
                    self.energy_ledger.rows[0]['elastic']+=hg
                    self.energy_ledger.rows[0]['mechanical']+=hg
                    self.energy_ledger.rows[0]['stabilization_energy']=hg
                    self.energy_ledger.current['volume_remap_delta']-=hg
        n_active_nodes = int(self.n_active_nodes.numpy()[0])
        n_active_centers = int(self.n_active_centers.numpy()[0])
        wp.launch(
            lite_initialize_v_iterate_kernel,
            dim=n_active_nodes,
            inputs=[
                self.n_active_nodes, self.ndof2bijk, self.block_xyz_by_id,
                self.bc_block2bid, self.bc_type, self.bc_norm, self.bc_velo,
                self.hf_bc_p, self.hf_bc_n, self.hf_bc_v, self.hf_bc_type, self.num_hf,
                self.grid_m, self.grid_v, self.grid_v_it, self.grid_size,
                self.gravity, self.dt, self.dx,
            ],
            device=self.device,
        )

        max_iters = kwargs.get("max_iters", 4)
        v_tol = kwargs.get("v_tol", 1.0e-5)
        print_every = kwargs.get("print_every", 1)
        converged = False
        variational = self.force_discretization == "variational"
        linear_solver = kwargs.get("linear_solver", "pcg" if variational else "bicgstab")
        if linear_solver not in ("bicgstab", "gmres", "pcg", "pcg_projected"):
            raise ValueError("unknown linear_solver")
        if not variational and linear_solver.startswith("pcg"):
            raise ValueError("PCG requires the variational symmetric discretization")
        stats["force_discretization"] = self.force_discretization
        stats["projected_tangent_solves"] = 0
        stats["curvature_failures"] = 0
        self.last_newton_trace = []
        stats["linear_solver"] = linear_solver
        stats["line_search_backtracks"] = 0
        initial_residual = None
        reaction_atol = kwargs.get("reaction_force_atol")
        # N_active >= N_free: a conservative bound for the resultant of the
        # projected force residual, valid without assuming a specific clamp.
        force_factor = np.sqrt(max(1, n_active_nodes)) / self.dt if reaction_atol is not None else None
        force_target = .5 * reaction_atol / force_factor if reaction_atol is not None else None
        if reaction_atol is not None:
            stats.update(reaction_force_atol=float(reaction_atol),
                         reaction_bound_nodes=n_active_nodes,
                         reaction_momentum_target=float(force_target))
        for newton in range(max_iters):
            stats["newton_iterations"] = newton + 1
            if not self.evaluate_residual():
                stats["last_residual_norm"] = float("inf")
                break
            r0 = self.compute_residual_norm(self.node_residual)
            stats["last_residual_norm"] = r0
            if initial_residual is None:
                initial_residual = r0
            target = max(kwargs.get("newton_atol", 1e-10), kwargs.get("cg_atol", 1e-9)*1.01, kwargs.get("newton_rtol", 1e-8)*initial_residual)
            if force_target is not None:
                target = min(target, force_target)
                stats["reaction_force_residual_bound"] = float(force_factor * r0)
            stats["newton_residual_target"] = float(target)
            if r0 <= target:
                converged = True
                break
            self.dv_guess.zero_()

            project_pd = linear_solver == "pcg_projected"

            def A_matvec(p, Ap):
                material_start = time.perf_counter()
                self.apply_tangent(p, Ap, project_pd=project_pd)
                stats["matvec_calls"] += 1
                stats["material_seconds"] += time.perf_counter() - material_start

            def M_inv_matvec(p, Mp):
                wp.launch(
                    lite_implicit_precond_kernel_Hii,
                    dim=self.MAX_DOF,
                    inputs=[
                        self.n_active_nodes, self.ndof2bijk, self.block_xyz_by_id,
                        self.node_Hii_inv, p, Mp, self.bc_block2bid, self.bc_type,
                        self.bc_norm, self.bc_velo, self.hf_bc_p, self.hf_bc_n,
                        self.hf_bc_v, self.hf_bc_type, self.num_hf, self.dx,
                    ],
                    device=self.device,
                )

            cg_start = time.perf_counter()
            solve_options = dict(A_matvec=A_matvec, b=-self.node_residual, x=self.dv_guess,
                                 maxiter=kwargs.get("max_cg_iters", max(10, n_active_nodes*3)),
                                 tol=kwargs.get("cg_tol", 1e-3), atol=kwargs.get("cg_atol", 1e-9),
                                 M_inv_matvec=M_inv_matvec)
            if force_target is not None:
                # Cap BOTH linear tolerances; a loose relative CG tolerance or
                # absolute floor must not override the requested force scale.
                linear_cap = .1 * target
                solve_options["atol"] = min(solve_options["atol"], linear_cap)
                solve_options["tol"] = min(solve_options["tol"], linear_cap / r0)
            if linear_solver.startswith("pcg"):
                n_cg, error, tol, reason = guarded_pcg(**solve_options)
                slope = dot_array(self.node_residual, self.dv_guess)
                if not project_pd and (reason != "converged" or not np.isfinite(slope) or slope >= 0):
                    stats["curvature_failures"] += int(reason == "nonpositive_curvature")
                    stats["pcg_fallback_reason"] = reason if slope < 0 else "non_descent_or_"+reason
                    project_pd = True
                    extra, error, tol, reason = guarded_pcg(**solve_options)
                    n_cg += extra
                stats["pcg_status"] = reason
                stats["projected_tangent_solves"] += int(project_pd)
            else:
                n_cg, error, tol = nonsymmetric_solve(**solve_options, method=linear_solver)
                reason = "converged" if np.isfinite(error) and error <= tol*1.01 else "linear_failure"
            stats["linear_residual_norm"] = float(error)
            stats["linear_residual_target"] = float(tol)
            stats["cg_seconds"] += time.perf_counter() - cg_start
            stats["cg_iterations"] += int(n_cg)
            if reason != "converged" or not np.isfinite(error) or error > max(tol*1.01, 1e-15):
                stats["linear_failure"] = True
                break
            phi0 = self.incremental_potential() if variational else 0.
            roundoff0 = self._last_potential_roundoff if variational else 0.
            slope = dot_array(self.node_residual, self.dv_guess)
            if variational and (not np.isfinite(slope) or slope >= 0):
                stats["non_descent_failure"] = True
                break
            saved_velocity = wp.clone(self.grid_v_it[:self.bcn])
            accepted = False
            for backtrack in range(11):
                alpha = .5**backtrack
                wp.copy(self.grid_v_it, saved_velocity, count=saved_velocity.size)
                self.max_update.zero_()
                wp.launch(
                    self._commit_dv_kernel,
                    dim=n_active_nodes,
                    inputs=[
                        self.n_active_nodes, self.ndof2bijk, self.block_xyz_by_id,
                        self.bc_block2bid, self.bc_type, self.bc_norm, self.bc_velo,
                        self.hf_bc_p, self.hf_bc_n, self.hf_bc_v, self.hf_bc_type,
                        self.num_hf, self.dv_guess, self.grid_v_it, self.max_update,
                        alpha, self.dx,
                    ],
                    device=self.device,
                )
                valid_trial = self.evaluate_residual()
                r_trial = self.compute_residual_norm(self.node_residual) if valid_trial else float("inf")
                phi_trial = self.incremental_potential() if variational and valid_trial else float("inf")
                # Near rest, relative-to-Phi tolerance underestimates SVD/log
                # cancellation. A roundoff-only acceptance must also reduce r.
                slack = roundoff0 + self._last_potential_roundoff if variational and valid_trial else 0.
                armijo = phi_trial <= phi0 + 1e-4*alpha*slope
                acceptable = (armijo or (phi_trial <= phi0 + 1e-4*alpha*slope + slack and r_trial < r0)) if variational else (r_trial <= (1-1e-4*alpha)*r0 or r_trial <= target)
                if np.isfinite(r_trial) and acceptable:
                    accepted = True
                    if variational:
                        self.last_newton_trace.append(dict(potential_before=phi0, potential_after=phi_trial,
                            slope=slope, alpha=alpha, armijo_slack=slack, roundoff_acceptance=not armijo,
                            residual_before=r0, residual_after=r_trial, projected_tangent=project_pd))
                    stats["last_residual_norm"] = r_trial
                    stats["line_search_backtracks"] += backtrack
                    break
            if not accepted:
                wp.copy(self.grid_v_it, saved_velocity, count=saved_velocity.size)
                stats["line_search_failure"] = True
                break
            update = float(self.max_update.numpy()[0])
            stats["last_update"] = update
            if print_every and (newton % print_every == 0):
                print(f"[Aniso solver] sim_step={self.sim_steps:04d} newton={newton:02d} cg={n_cg:03d} residual={r0:.3e} update={update:.3e}")
            if force_target is not None:
                stats["reaction_force_residual_bound"] = float(force_factor * r_trial)
            update_target = target if force_target is not None else max(target, kwargs.get("cg_atol", 1e-9))
            if r_trial <= target or (update < v_tol and r_trial <= update_target):
                converged = True
                break

        stats["linear_iterations"] = stats["cg_iterations"]

        # A failed step leaves particle and committed-center history untouched.
        if not converged:
            wp.launch(
                aniso_rollback_trial_kernel,
                dim=self.aniso_trial_F.shape,
                inputs=[
                    self.aniso_committed_F, self.aniso_trial_F,
                    self.aniso_state_valid,
                ],
                device=self.device,
            )
            stats.update({
                "converged": False,
                "active_centers": n_active_centers,
                "active_nodes": n_active_nodes,
                "step_seconds": time.perf_counter() - step_start,
                "memory_bytes": self._aniso_memory_bytes(),
            })
            self.last_step_stats = stats
            return False

        # Transfer the converged grid state and commit the center deformation
        # only after the Newton loop has declared success.
        if self.energy_ledger is not None:
            self.energy_ledger.solved(self)
            self.energy_ledger.step_start_F = self.aniso_committed_F[:, :int(self.bcn)].numpy().copy()
        if self.enhancements is not None:
            self.enhancements.correction()
        transfer_start = time.perf_counter()
        wp.launch(
            update_vnew_with_iterate_kernel,
            dim=(self.bcn, B, B, B),
            inputs=[
                self.block_count, self.block_xyz_by_id, self.bc_block2bid,
                self.bc_type, self.bc_norm, self.bc_velo, self.hf_bc_p,
                self.hf_bc_n, self.hf_bc_v, self.hf_bc_type, self.num_hf,
                self.grid_m, self.grid_v_it, self.grid_v_new, self.grid_size,
                self.dx, kwargs.get("damping", 1.0),
            ],
            device=self.device,
        )
        if self.energy_ledger is not None:
            self.energy_ledger.transferred_grid(self)
        from .affine_transfer import prepare_incremental, finish_transfer
        if self.apic_transfer == "incremental":
            prepare_incremental(self)
        wp.launch(
            lite_g2c_kernel,
            dim=(self.bcn, B, B, B),
            inputs=[
                self.block_count, self.block2bid, self.block_xyz_by_id,
                self.grid_v_raw if self.boundary_impulse_transfer else self.grid_v,
                self.grid_v_new, self.center_v, self.center_dv,
                self.center_G, self.center_size, self.dx,
            ],
            device=self.device,
        )
        dissipation_plan = None
        if self.velocity_dissipation != 'none':
            try:
                from .unresolved_velocity import prepare
                dissipation_plan = prepare(self)
            except (ValueError, np.linalg.LinAlgError) as error:
                wp.launch(aniso_rollback_trial_kernel, dim=self.aniso_trial_F.shape,
                    inputs=[self.aniso_committed_F,self.aniso_trial_F,self.aniso_state_valid],device=self.device)
                stats.update(converged=False,velocity_dissipation_failure=str(error),
                    active_centers=n_active_centers,active_nodes=n_active_nodes,
                    step_seconds=time.perf_counter()-step_start,memory_bytes=self._aniso_memory_bytes())
                self.last_step_stats=stats
                return False
        wp.launch(
            lite_c2p_kernel,
            dim=self.n_ptc,
            inputs=[
                self.block2bid, self.ptc_x, self.ptc_v, self.ptc_k, self.ptc_F,
                self.ptc_G, self.ptc_dlogJ, self.center_m, self.center_v,
                self.center_dv, self.center_G, self.psi_params, self.center_size,
                self.dx, self.dt, self.flip_ratio,
            ],
            device=self.device,
        )
        finish_transfer(self)
        if dissipation_plan is not None:
            from .unresolved_velocity import commit
            commit(self,dissipation_plan)
            stats.update(self.dissipation_stats)
        if self.enhancements is not None and hasattr(self.enhancements,"commit"):
            self.enhancements.commit()
        wp.launch(
            aniso_commit_kernel,
            dim=n_active_centers,
            inputs=[
                self.n_active_centers, self.cdof2bijk, self.block_xyz_by_id,
                self.center_G, self.center_vol, self.aniso_committed_F,
                self.center_size, self.dt, int(converged),
            ],
            device=self.device,
        )
        stats["transfer_seconds"] = time.perf_counter() - transfer_start
        stats.update({
            "boundary_impulse_transfer": self.boundary_impulse_transfer,
            "apic_transfer": self.apic_transfer,
            "affine_flip_ratio": self.flip_ratio if self.affine_flip_ratio is None else self.affine_flip_ratio,
            "velocity_dissipation": self.velocity_dissipation,
            "converged": True,
            "active_centers": n_active_centers,
            "active_nodes": n_active_nodes,
            "step_seconds": time.perf_counter() - step_start,
            "memory_bytes": self._aniso_memory_bytes(),
        })
        stats["linear_iterations"] = stats["cg_iterations"]
        self.last_step_stats = stats
        return True

    def evaluate_residual(self):
        """Evaluate actual sparse trial/residual kernels at grid_v_it; freeze committed state."""
        n_active_nodes = int(self.n_active_nodes.numpy()[0])
        n_active_centers = int(self.n_active_centers.numpy()[0])
        wp.launch(
            lite_center_grad_from_grid_velocity_kernel,
            dim=n_active_centers,
            inputs=[
                self.n_active_centers, self.cdof2bijk, self.block2bid,
                self.block_xyz_by_id, self.grid_m, self.grid_v_it,
                self.center_G, self.grid_size, self.dx,
            ],
            device=self.device,
        )
        self.aniso_invalid_trial.zero_()
        wp.launch(
            aniso_update_trial_kernel,
            dim=n_active_centers,
            inputs=[
                self.n_active_centers, self.cdof2bijk, self.block_xyz_by_id,
                self.center_G, self.center_vol, self.aniso_committed_F,
                self.aniso_trial_F, self.aniso_invalid_trial, self.center_size, self.dt,
            ],
            device=self.device,
        )
        if int(self.aniso_invalid_trial.numpy()[0]) != 0:
            return False
        wp.launch(
            aniso_residual_kernel,
            dim=self.MAX_DOF,
            inputs=[
                self.n_active_nodes, self.ndof2bijk, self.center2dof,
                self.block2bid, self.block_xyz_by_id, self.node_residual,
                self.node_Hii_inv, self.bc_block2bid, self.bc_type,
                self.bc_norm, self.bc_velo, self.hf_bc_p, self.hf_bc_n,
                self.hf_bc_v, self.hf_bc_type, self.num_hf, self.center_vol,
                self.aniso_trial_F, self.aniso_committed_F, self.aniso_A0, self.grid_m, self.grid_v,
                self.grid_v_it, self.grid_size, self.gravity, self.dx, self.dt,
                self.aniso_params.mu, self.aniso_params.lam, self.aniso_params.k_f,
                int(self.force_discretization == "variational"),
            ],
            device=self.device,
        )
        if self.enhancements is not None:
            from .enhancements import add_correction
            self.enhancements.correction()
            wp.launch(add_correction,dim=n_active_nodes,inputs=[self.enhancements.extra,self.node_residual],device=self.device)
            self.node_residual.assign(self.project_direction(self.node_residual))
        return int(self.aniso_invalid_trial.numpy()[0])==0

    def project_direction(self, p):
        if not hasattr(self, "_projected_direction") or len(self._projected_direction) != len(p):
            self._projected_direction = wp.zeros_like(p)
        wp.copy(self._projected_direction, p)
        wp.launch(boundary_projection_kernel, dim=len(p), inputs=[
            self.n_active_nodes, self.ndof2bijk, self.block_xyz_by_id, self._projected_direction,
            self.bc_block2bid, self.bc_type, self.bc_norm, self.bc_velo,
            self.hf_bc_p, self.hf_bc_n, self.hf_bc_v, self.hf_bc_type, self.num_hf,
            self.grid_size, self.dx], device=self.device)
        return self._projected_direction

    def incremental_potential(self):
        """Phi at the last evaluated trial, with fixed geometry, mass, V0 and F^n."""
        self._potential_sum.zero_()
        wp.launch(inertia_potential, dim=int(self.n_active_nodes.numpy()[0]), inputs=[
            self.ndof2bijk, self.grid_m, self.grid_v, self.grid_v_it,
            self.gravity, self.dt, self._potential_sum], device=self.device)
        self._elastic_potential()
        value, scale = self._potential_sum.numpy()
        count = int(self.n_active_nodes.numpy()[0]) + (self.n_ptc if getattr(self, 'quadrature_kind', 'center') == 'particle' else int(self.n_active_centers.numpy()[0]))
        if getattr(self,'quadrature_kind','center')=='group4x8':
            count=int(self.n_active_nodes.numpy()[0])+len(self.group_V)
        self._last_potential_roundoff = np.finfo(float).eps*(64*scale+(count+16)*abs(value))
        return float(value)

    def _elastic_potential(self):
        wp.launch(center_potential, dim=int(self.n_active_centers.numpy()[0]), inputs=[
            self.cdof2bijk, self.center_vol, self.aniso_trial_F, self.aniso_A0,
            self.aniso_params.mu, self.aniso_params.lam, self.aniso_params.k_f,
            self._potential_sum], device=self.device)
        if self.enhancements is not None:
            # Correction reductions were evaluated with the same trial residual.
            from .enhancements import add_energy
            wp.launch(add_energy,dim=2,inputs=[self.enhancements.energy,self.enhancements.hg_energy,self._potential_sum],device=self.device)

    def apply_tangent(self, p, Ap, project_pd=False):
        """Apply Q J Q; optional PSD material approximation is NOT the exact J."""
        if project_pd and self.force_discretization != "variational":
            raise ValueError("projected tangent requires variational forces")
        p = self.project_direction(p)
        wp.launch(
            self._aniso_grad_kernel,
            dim=self.MAX_DOF,
            inputs=[
                self.n_active_centers, self.cdof2bijk, self.node2dof,
                self.block2bid, self.block_xyz_by_id, p,
                self.center_grad_dv_tmp, self.grid_size, self.dx,
            ],
            device=self.device,
        )
        wp.launch(
            aniso_stress_scratch_kernel,
            dim=(1, self.MAX_DOF),
            inputs=[
                self.n_active_centers, self.cdof2bijk, self.center_stress_scratch,
                self.center_grad_dv_tmp, self.center_vol, self.aniso_committed_F,
                self.aniso_trial_F, self.aniso_A0, self.aniso_params.mu, self.aniso_params.lam,
                self.aniso_params.k_f, self.dt, int(self.force_discretization == "variational"), int(project_pd),
            ],
            device=self.device,
        )
        wp.launch(
            aniso_df_kernel,
            dim=self.MAX_DOF,
            inputs=[
                self.n_active_nodes, self.ndof2bijk, self.center2dof,
                self.block2bid, self.block_xyz_by_id, p, Ap,
                self.center_stress_scratch, self.center_vol,
                self.bc_block2bid, self.bc_type, self.bc_norm, self.bc_velo,
                self.hf_bc_p, self.hf_bc_n, self.hf_bc_v, self.hf_bc_type,
                self.num_hf, self.grid_m, self.grid_size, self.dx, self.dt,
            ],
            device=self.device,
        )
        if self.enhancements is not None:
            self.enhancements.tangent(p,Ap,project_pd)
            Ap.assign(self.project_direction(Ap))

    def _aniso_memory_bytes(self) -> int:
        arrays = (
            self.aniso_committed_F, self.aniso_trial_F, self.aniso_A0,
            self.aniso_A0_sum, self.aniso_state_valid, self.aniso_occupied_prev, self.ptc_A0,
        )
        arrays += (self.ptc_L,)
        arrays += (self._apic_difference,) if self._apic_difference is not None else ()
        arrays += (self.grid_v_raw,) if self.grid_v_raw is not None else ()
        return int(sum(int(a.size) * type_size_in_bytes(a.dtype) for a in arrays))+self.ptc_reference_x.size*type_size_in_bytes(vec3)+(self.enhancements.memory_bytes() if self.enhancements is not None else 0)

    @property
    def _aniso_grad_kernel(self):
        return lite_implicit_grad_v_dv_kernel

    @property
    def _commit_dv_kernel(self):
        from engine.kernel.d3.helper_misc import commit_dv_to_iterate_kernel

        return commit_dv_to_iterate_kernel
