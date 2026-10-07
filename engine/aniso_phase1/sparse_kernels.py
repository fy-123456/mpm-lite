"""Sparse-grid kernels for the isolated stage-one solver adapter.

These kernels use the same block/cell addressing as ``engine.kernel.d3`` but
keep the anisotropic state in this package.  The legacy kernels are untouched.
"""

import warp as wp

from engine.types import mat33, real, vec3
from engine.kernel.constant import _0, _1, _05, _n1, _14, dPdF_eps_pd, dPdF_eps_small, s_min
from engine.kernel.d3.helper_constitutive_model import dPdF_StVK_Hencky_3D_analytic
from engine.kernel.d3.kernel_lite import lite_c2g
from engine.kernel.d3.kernel_lite_misc import lite_center_grad_from_grid_velocity_kernel
from engine.kernel.d3.helper_misc import in_region
from engine.kernel.d3.kernel_lite_implicit import lite_implicit_precond_kernel_mass
from engine.kernel.d3.helper_constitutive_model import StVK_Hencky_PK1_3D
from engine.sp_grid import (
    B,
    lin_I_JK,
    lin_JK,
    lin_IJK,
    unlin_I_JK,
    unlin_JK,
    unlin_IJK,
    block_coords_from_node,
    local_coords_in_block,
    local_linear,
    local2global,
)
from engine.boundary_utils import query_bc_sp, proj_boundary_delta_vel
from .kernels import aniso_dpk1_wp, aniso_pk1_wp, fiber_dpk1_wp, mat_to_vec9_wp, vec9_to_mat_wp


@wp.kernel
def remap_aniso_mat33_kernel(
    old_to_new: wp.array(dtype=wp.int32),
    source: wp.array(dtype=mat33, ndim=4),
    compact: wp.array(dtype=mat33, ndim=4),
):
    old_bid, li, lj, lk = wp.tid()
    new_bid = old_to_new[old_bid]
    if new_bid >= 0:
        compact[0, new_bid, li, lj * B + lk] = source[0, old_bid, li, lj * B + lk]


@wp.kernel
def remap_aniso_int_kernel(
    old_to_new: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.int32, ndim=4),
    compact: wp.array(dtype=wp.int32, ndim=4),
):
    old_bid, li, lj, lk = wp.tid()
    new_bid = old_to_new[old_bid]
    if new_bid >= 0:
        compact[0, new_bid, li, lj * B + lk] = source[0, old_bid, li, lj * B + lk]


@wp.kernel
def remap_aniso_u8_kernel(
    old_to_new: wp.array(dtype=wp.int32),
    source: wp.array(dtype=wp.uint8, ndim=4),
    compact: wp.array(dtype=wp.uint8, ndim=4),
):
    old_bid, li, lj, lk = wp.tid()
    new_bid = old_to_new[old_bid]
    if new_bid >= 0:
        compact[0, new_bid, li, lj * B + lk] = source[0, old_bid, li, lj * B + lk]


@wp.kernel
def scatter_aniso_mat33_kernel(
    compact: wp.array(dtype=mat33, ndim=4),
    target: wp.array(dtype=mat33, ndim=4),
):
    bid, li, lj, lk = wp.tid()
    target[0, bid, li, lj * B + lk] = compact[0, bid, li, lj * B + lk]


@wp.kernel
def scatter_aniso_int_kernel(
    compact: wp.array(dtype=wp.int32, ndim=4),
    target: wp.array(dtype=wp.int32, ndim=4),
):
    bid, li, lj, lk = wp.tid()
    target[0, bid, li, lj * B + lk] = compact[0, bid, li, lj * B + lk]


@wp.kernel
def scatter_aniso_u8_kernel(
    compact: wp.array(dtype=wp.uint8, ndim=4),
    target: wp.array(dtype=wp.uint8, ndim=4),
):
    bid, li, lj, lk = wp.tid()
    target[0, bid, li, lj * B + lk] = compact[0, bid, li, lj * B + lk]


@wp.kernel
def aniso_p2c_kernel_1(
    block2bid: wp.array(dtype=int, ndim=3),
    ptc_x: wp.array(dtype=vec3),
    ptc_v: wp.array(dtype=vec3),
    ptc_G: wp.array(dtype=mat33),
    ptc_A0: wp.array(dtype=mat33),
    ptc_vol0: wp.array(dtype=real),
    ptc_m: wp.array(dtype=real),
    center_m: wp.array(dtype=real, ndim=4),
    center_v: wp.array(dtype=vec3, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    center_A0_sum: wp.array(dtype=mat33, ndim=4),
    center_size: wp.vec3i,
    dx: real,
    enable_apic: bool,
):
    p = wp.tid()
    xp = ptc_x[p]
    pos_c = xp / dx - vec3(_05)
    base = wp.vec3i(int(wp.floor(pos_c[0])), int(wp.floor(pos_c[1])), int(wp.floor(pos_c[2])))
    fx = pos_c - vec3(real(base.x), real(base.y), real(base.z))
    wx0 = _1 - fx[0]
    wx1 = fx[0]
    wy0 = _1 - fx[1]
    wy1 = fx[1]
    wz0 = _1 - fx[2]
    wz1 = fx[2]
    Gp = ptc_G[p]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                center = base + wp.vec3i(i, j, k)
                if not in_region(center, center_size):
                    continue
                w = (wx0 if i == 0 else wx1) * (wy0 if j == 0 else wy1) * (wz0 if k == 0 else wz1)
                bc = block_coords_from_node(center.x, center.y, center.z)
                bid = block2bid[bc.x, bc.y, bc.z]
                if bid < 0:
                    continue
                local = local_coords_in_block(center.x, center.y, center.z)
                li, lj, lk = local.x, local.y, local.z
                dm = w * ptc_m[p]
                vel = ptc_v[p]
                if enable_apic:
                    xc = dx * vec3(real(center.x) + _05, real(center.y) + _05, real(center.z) + _05)
                    vel = vel + Gp @ (xc - xp)
                lcjk = lin_JK(lj, lk)
                wp.atomic_add(center_m, bid, li, lj, lk, dm)
                wp.atomic_add(center_v, bid, li, lj, lk, dm * vel)
                wp.atomic_add(center_G, bid, li, lj, lk, dm * Gp)
                wp.atomic_add(center_vol, 0, bid, li, lcjk, w * ptc_vol0[p])
                wp.atomic_add(center_A0_sum, 0, bid, li, lcjk, w * ptc_vol0[p] * ptc_A0[p])


@wp.kernel
def aniso_p2c_kernel_2(
    block_count: wp.array(dtype=int),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),
    center_m: wp.array(dtype=real, ndim=4),
    center_v: wp.array(dtype=vec3, ndim=4),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    center_A0_sum: wp.array(dtype=mat33, ndim=4),
    center_A0: wp.array(dtype=mat33, ndim=4),
    committed_F: wp.array(dtype=mat33, ndim=4),
    state_valid: wp.array(dtype=wp.int32, ndim=4),
    occupied_prev: wp.array(dtype=wp.uint8, ndim=4),
    reactivation_count: wp.array(dtype=wp.int32),
    center_size: wp.vec3i,
):
    bid, lci, lcj, lck = wp.tid()
    if bid >= block_count[0]:
        return
    center = local2global(block_xyz_by_id[bid], lin_IJK(lci, lcj, lck))
    if not in_region(center, center_size):
        return
    lcjk = lin_JK(lcj, lck)
    if center_m[bid, lci, lcj, lck] <= _0:
        occupied_prev[0, bid, lci, lcjk] = wp.uint8(0)
        return
    invm = _1 / center_m[bid, lci, lcj, lck]
    center_v[bid, lci, lcj, lck] *= invm
    center_G[bid, lci, lcj, lck] *= invm
    volume = center_vol[0, bid, lci, lcjk]
    if volume <= _0:
        occupied_prev[0, bid, lci, lcjk] = wp.uint8(0)
        return
    was_occupied = occupied_prev[0, bid, lci, lcjk]
    if state_valid[0, bid, lci, lcjk] == 0 or was_occupied == 0:
        center_A0[0, bid, lci, lcjk] = center_A0_sum[0, bid, lci, lcjk] / volume
        committed_F[0, bid, lci, lcjk] = wp.identity(3, dtype=real)
        if state_valid[0, bid, lci, lcjk] != 0 and was_occupied == 0:
            wp.atomic_add(reactivation_count, 0, 1)
        state_valid[0, bid, lci, lcjk] = 1
    occupied_prev[0, bid, lci, lcjk] = wp.uint8(1)


@wp.kernel
def aniso_center_tau_kernel(
    block_count: wp.array(dtype=int),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),
    center_vol: wp.array(dtype=real, ndim=4),
    committed_F: wp.array(dtype=mat33, ndim=4),
    center_A0: wp.array(dtype=mat33, ndim=4),
    center_tau: wp.array(dtype=mat33, ndim=4),
    center_size: wp.vec3i,
    mu: real,
    lam: real,
    k_f: real,
):
    bid, lci, lcj, lck = wp.tid()
    if bid >= block_count[0]:
        return
    center = local2global(block_xyz_by_id[bid], lin_IJK(lci, lcj, lck))
    if not in_region(center, center_size):
        return
    lcjk = lin_JK(lcj, lck)
    if center_vol[0, bid, lci, lcjk] <= _0:
        return
    F = committed_F[0, bid, lci, lcjk]
    center_tau[0, bid, lci, lcjk] = aniso_pk1_wp(F, center_A0[0, bid, lci, lcjk], mu, lam, k_f) @ wp.transpose(F)


@wp.kernel
def aniso_update_trial_kernel(
    n_active_centers: wp.array(dtype=int),
    cdof2bijk: wp.array(dtype=wp.vec2i),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    committed_F: wp.array(dtype=mat33, ndim=4),
    trial_F: wp.array(dtype=mat33, ndim=4),
    invalid_trial: wp.array(dtype=wp.int32),
    center_size: wp.vec3i,
    dt: real,
):
    cdof = wp.tid()
    if cdof >= n_active_centers[0]:
        return
    bid, local_c1d = cdof2bijk[cdof].x, cdof2bijk[cdof].y
    cijk = local2global(block_xyz_by_id[bid], local_c1d)
    if not in_region(cijk, center_size):
        return
    lci, lcj, lck = unlin_IJK(local_c1d)
    lcjk = lin_JK(lcj, lck)
    if center_vol[0, bid, lci, lcjk] <= _0:
        return
    F = committed_F[0, bid, lci, lcjk]
    trial = (wp.identity(3, dtype=real) + dt * center_G[bid, lci, lcj, lck]) @ F
    trial_F[0, bid, lci, lcjk] = trial
    U, sigma, V = wp.svd3(trial)
    if not wp.isfinite(wp.determinant(trial)) or wp.determinant(trial) <= _0 or wp.min(sigma) <= s_min:
        wp.atomic_max(invalid_trial, 0, 1)


@wp.kernel
def aniso_residual_kernel(
    n_active_nodes: wp.array(dtype=int),
    ndof2bijk: wp.array(dtype=wp.vec2i),
    center2dof: wp.array(dtype=int, ndim=4),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),
    node_residual: wp.array(dtype=vec3),
    node_Hii_inv: wp.array(dtype=mat33),
    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),
    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,
    center_vol: wp.array(dtype=real, ndim=4),
    trial_F: wp.array(dtype=mat33, ndim=4),
    committed_F: wp.array(dtype=mat33, ndim=4),
    center_A0: wp.array(dtype=mat33, ndim=4),
    grid_m: wp.array(dtype=real, ndim=4),
    grid_v: wp.array(dtype=vec3, ndim=4),
    grid_v_it: wp.array(dtype=vec3, ndim=4),
    grid_size: wp.vec3i,
    gravity: real,
    dx: real,
    dt: real,
    mu: real,
    lam: real,
    k_f: real,
    variational: int,
):
    dof = wp.tid()
    node_residual[dof] = vec3(_0)
    if dof >= n_active_nodes[0]:
        node_Hii_inv[dof] = mat33(_0)
        return
    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)
    if not in_region(node, grid_size):
        node_Hii_inv[dof] = mat33(_0)
        return
    li, lj, lk = unlin_IJK(local_n1d)
    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
    mi = grid_m[bid, li, lj, lk]
    if bct == 1:
        node_Hii_inv[dof] = mat33(_0)
        return
    node_residual[dof] = mi * (grid_v_it[bid, li, lj, lk] - grid_v[bid, li, lj, lk]) + vec3(_0, _0, -dt * mi * gravity)
    Hii = mi * wp.identity(3, dtype=real)
    inv_4dx = _14 / dx
    center_size = wp.vec3i(grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                ci, cj, ck = node.x - di, node.y - dj, node.z - dk
                center = wp.vec3i(ci, cj, ck)
                if not in_region(center, center_size):
                    continue
                bc = block_coords_from_node(ci, cj, ck)
                cbid = block2bid[bc.x, bc.y, bc.z]
                if cbid < 0:
                    continue
                local_c1d = local_linear(local_coords_in_block(ci, cj, ck))
                clci, clcjk = unlin_I_JK(local_c1d)
                clcj, clck = unlin_JK(clcjk)
                cdof = center2dof[cbid, clci, clcj, clck]
                if cdof < 0 or center_vol[0, cbid, clci, clcjk] <= _0:
                    continue
                dw = inv_4dx * vec3(_n1 if di == 0 else _1, _n1 if dj == 0 else _1, _n1 if dk == 0 else _1)
                F = trial_F[0, cbid, clci, clcjk]
                pullback = F
                if variational != 0:
                    pullback = committed_F[0, cbid, clci, clcjk]
                tau = aniso_pk1_wp(F, center_A0[0, cbid, clci, clcjk], mu, lam, k_f) @ wp.transpose(pullback)
                V = center_vol[0, cbid, clci, clcjk]
                node_residual[dof] += dt * V * (tau @ dw)
    node_Hii_inv[dof] = wp.inverse(Hii)
    if bct > 0:
        node_residual[dof] = proj_boundary_delta_vel(node_residual[dof], bct, bcn)


@wp.kernel
def aniso_stress_scratch_kernel(
    n_active_centers: wp.array(dtype=int),
    cdof2bijk: wp.array(dtype=wp.vec2i),
    center_stress_scratch: wp.array(dtype=mat33, ndim=2),
    center_grad_dv: wp.array(dtype=mat33),
    center_vol: wp.array(dtype=real, ndim=4),
    committed_F: wp.array(dtype=mat33, ndim=4),
    trial_F: wp.array(dtype=mat33, ndim=4),
    center_A0: wp.array(dtype=mat33, ndim=4),
    mu: real,
    lam: real,
    k_f: real,
    dt: real,
    variational: int,
    project_pd: int,
):
    psi_k, cdof = wp.tid()
    if psi_k != 0 or cdof >= n_active_centers[0]:
        return
    bid, local_c1d = cdof2bijk[cdof].x, cdof2bijk[cdof].y
    lci, lcjk = unlin_I_JK(local_c1d)
    V = center_vol[0, bid, lci, lcjk]
    if V <= _0:
        center_stress_scratch[0, cdof] = mat33(_0)
        return
    F = trial_F[0, bid, lci, lcjk]
    dF = dt * center_grad_dv[cdof] @ committed_F[0, bid, lci, lcjk]
    dP = aniso_dpk1_wp(F, center_A0[0, bid, lci, lcjk], dF, mu, lam, k_f, project_pd)
    P = aniso_pk1_wp(F, center_A0[0, bid, lci, lcjk], mu, lam, k_f)
    if variational != 0:
        center_stress_scratch[0, cdof] = V * (dP @ wp.transpose(committed_F[0, bid, lci, lcjk]))
    else:
        center_stress_scratch[0, cdof] = V * (dP @ wp.transpose(F) + P @ wp.transpose(dF))


@wp.kernel
def aniso_df_kernel(
    n_active_nodes: wp.array(dtype=int),
    ndof2bijk: wp.array(dtype=wp.vec2i),
    center2dof: wp.array(dtype=int, ndim=4),
    block2bid: wp.array(dtype=int, ndim=3),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),
    dv: wp.array(dtype=vec3),
    df: wp.array(dtype=vec3),
    center_stress_scratch: wp.array(dtype=mat33, ndim=2),
    center_vol: wp.array(dtype=real, ndim=4),
    bc_block2bid: wp.array(dtype=int, ndim=3),
    bc_type: wp.array(dtype=int, ndim=4),
    bc_norm: wp.array(dtype=vec3, ndim=4),
    bc_velo: wp.array(dtype=vec3, ndim=4),
    hf_bc_p: wp.array(dtype=vec3),
    hf_bc_n: wp.array(dtype=vec3),
    hf_bc_v: wp.array(dtype=vec3),
    hf_bc_type: wp.array(dtype=int),
    num_hf: wp.int32,
    grid_m: wp.array(dtype=real, ndim=4),
    grid_size: wp.vec3i,
    dx: real,
    dt: real,
):
    dof = wp.tid()
    if dof >= n_active_nodes[0]:
        df[dof] = vec3(_0)
        return
    bid, local_n1d = ndof2bijk[dof].x, ndof2bijk[dof].y
    node = local2global(block_xyz_by_id[bid], local_n1d)
    li, lj, lk = unlin_IJK(local_n1d)
    bct, bcn, bcv = query_bc_sp(node, bc_block2bid, bc_type, bc_norm, bc_velo, hf_bc_p, hf_bc_n, hf_bc_v, hf_bc_type, num_hf, dx)
    if bct == 1:
        df[dof] = vec3(_0)
        return
    df[dof] = grid_m[bid, li, lj, lk] * dv[dof]
    inv_4dx = _14 / dx
    center_size = wp.vec3i(grid_size[0] - 1, grid_size[1] - 1, grid_size[2] - 1)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                center = wp.vec3i(node.x - di, node.y - dj, node.z - dk)
                if not in_region(center, center_size):
                    continue
                bc = block_coords_from_node(center.x, center.y, center.z)
                cbid = block2bid[bc.x, bc.y, bc.z]
                if cbid < 0:
                    continue
                local_c1d = local_linear(local_coords_in_block(center.x, center.y, center.z))
                clci, clcj, clck = unlin_IJK(local_c1d)
                cdof = center2dof[cbid, clci, clcj, clck]
                if cdof < 0 or center_vol[0, cbid, clci, lin_JK(clcj, clck)] <= _0:
                    continue
                dw = inv_4dx * vec3(_n1 if di == 0 else _1, _n1 if dj == 0 else _1, _n1 if dk == 0 else _1)
                df[dof] += dt * (center_stress_scratch[0, cdof] @ dw)
    if bct > 0:
        df[dof] = proj_boundary_delta_vel(df[dof], bct, bcn)


@wp.kernel
def aniso_commit_kernel(
    n_active_centers: wp.array(dtype=int),
    cdof2bijk: wp.array(dtype=wp.vec2i),
    block_xyz_by_id: wp.array(dtype=wp.vec3i),
    center_G: wp.array(dtype=mat33, ndim=4),
    center_vol: wp.array(dtype=real, ndim=4),
    committed_F: wp.array(dtype=mat33, ndim=4),
    center_size: wp.vec3i,
    dt: real,
    commit: wp.int32,
):
    cdof = wp.tid()
    if commit == 0 or cdof >= n_active_centers[0]:
        return
    bid, local_c1d = cdof2bijk[cdof].x, cdof2bijk[cdof].y
    cijk = local2global(block_xyz_by_id[bid], local_c1d)
    if not in_region(cijk, center_size):
        return
    lci, lcj, lck = unlin_IJK(local_c1d)
    lcjk = lin_JK(lcj, lck)
    if center_vol[0, bid, lci, lcjk] > _0:
        committed_F[0, bid, lci, lcjk] = (wp.identity(3, dtype=real) + dt * center_G[bid, lci, lcj, lck]) @ committed_F[0, bid, lci, lcjk]


@wp.kernel
def aniso_rollback_trial_kernel(
    committed_F: wp.array(dtype=mat33, ndim=4),
    trial_F: wp.array(dtype=mat33, ndim=4),
    state_valid: wp.array(dtype=wp.int32, ndim=4),
):
    psi, bid, lci, lcjk = wp.tid()
    if psi != 0 or state_valid[0, bid, lci, lcjk] == 0:
        return
    trial_F[0, bid, lci, lcjk] = committed_F[0, bid, lci, lcjk]


def aniso_p2c(
    block_count,
    block2bid,
    block_xyz_by_id,
    ptc_x,
    ptc_v,
    ptc_G,
    ptc_A0,
    ptc_vol0,
    ptc_m,
    center_m,
    center_v,
    center_G,
    center_vol,
    center_A0_sum,
    center_A0,
    committed_F,
    state_valid,
    occupied_prev,
    reactivation_count,
    center_size,
    dx,
    n_ptc,
    device,
    enable_apic=True,
):
    center_A0_sum.zero_()
    wp.launch(aniso_p2c_kernel_1, n_ptc, [block2bid, ptc_x, ptc_v, ptc_G, ptc_A0, ptc_vol0, ptc_m, center_m, center_v, center_G, center_vol, center_A0_sum, center_size, dx, enable_apic], device=device)
    wp.launch(aniso_p2c_kernel_2, dim=center_m.shape, inputs=[block_count, block_xyz_by_id, center_m, center_v, center_G, center_vol, center_A0_sum, center_A0, committed_F, state_valid, occupied_prev, reactivation_count, center_size], device=device)
