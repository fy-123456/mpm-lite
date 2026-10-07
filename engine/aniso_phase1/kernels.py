"""Warp helpers for fixed-reference fiber stress and center-state commit."""

from __future__ import annotations

import warp as wp

from engine.types import mat33, mat99, real, vec3, vec9
from engine.kernel.constant import _0, _1, _2, _4, dPdF_eps_pd, dPdF_eps_small, s_min
from engine.kernel.d3.helper_constitutive_model import (
    StVK_Hencky_PK1_3D,
    dPdF_StVK_Hencky_3D_analytic,
)


@wp.func
def fiber_invariant_wp(F: mat33, A0: mat33) -> real:
    C = wp.transpose(F) @ F
    return wp.trace(A0 @ C)


@wp.func
def fiber_pk1_wp(F: mat33, A0: mat33, k_f: real) -> mat33:
    I4 = fiber_invariant_wp(F, A0)
    return (_2 * k_f * (I4 - _1)) * (F @ A0)


@wp.func
def fiber_dpk1_wp(F: mat33, A0: mat33, dF: mat33, k_f: real) -> mat33:
    FA = F @ A0
    I4 = fiber_invariant_wp(F, A0)
    contraction = wp.trace(wp.transpose(FA) @ dF)
    return (_2 * k_f * (I4 - _1)) * (dF @ A0) + (_4 * k_f * contraction) * FA


@wp.func
def aniso_pk1_wp(F: mat33, A0: mat33, mu: real, lam: real, k_f: real) -> mat33:
    return StVK_Hencky_PK1_3D(F, mu, lam, s_min) + fiber_pk1_wp(F, A0, k_f)


@wp.func
def mat_to_vec9_wp(M: mat33) -> vec9:
    return vec9(
        M[0, 0], M[0, 1], M[0, 2],
        M[1, 0], M[1, 1], M[1, 2],
        M[2, 0], M[2, 1], M[2, 2],
    )


@wp.func
def vec9_to_mat_wp(v: vec9) -> mat33:
    return mat33(
        v[0], v[1], v[2],
        v[3], v[4], v[5],
        v[6], v[7], v[8],
    )


@wp.kernel
def evaluate_aniso_stress_kernel(
    F: wp.array(dtype=mat33),
    A0: wp.array(dtype=mat33),
    tau: wp.array(dtype=mat33),
    mu: real,
    lam: real,
    k_f: real,
):
    p = wp.tid()
    P = aniso_pk1_wp(F[p], A0[p], mu, lam, k_f)
    tau[p] = P @ wp.transpose(F[p])


@wp.kernel
def commit_center_trial_kernel(
    committed_F: wp.array(dtype=mat33),
    trial_F: wp.array(dtype=mat33),
    valid: wp.array(dtype=wp.int32),
    commit: wp.int32,
):
    c = wp.tid()
    if commit != 0:
        committed_F[c] = trial_F[c]
        valid[c] = 1


@wp.kernel
def center_trial_F_kernel(
    committed_F: wp.array(dtype=mat33),
    center_grad_v: wp.array(dtype=mat33),
    trial_F: wp.array(dtype=mat33),
    dt: real,
):
    c = wp.tid()
    trial_F[c] = (wp.identity(3, dtype=real) + dt * center_grad_v[c]) @ committed_F[c]


@wp.kernel
def center_stress_kernel(
    trial_F: wp.array(dtype=mat33),
    A0: wp.array(dtype=mat33),
    tau: wp.array(dtype=mat33),
    mu: real,
    lam: real,
    k_f: real,
):
    c = wp.tid()
    F = trial_F[c]
    tau[c] = aniso_pk1_wp(F, A0[c], mu, lam, k_f) @ wp.transpose(F)


@wp.kernel
def accumulate_structure_tensor_kernel(
    particle_A0: wp.array(dtype=mat33),
    particle_center: wp.array(dtype=wp.int32),
    particle_weight: wp.array(dtype=real),
    center_A0_sum: wp.array(dtype=mat33),
    center_weight_sum: wp.array(dtype=real),
):
    p = wp.tid()
    c = particle_center[p]
    w = particle_weight[p]
    if w > _0:
        wp.atomic_add(center_A0_sum, c, w * particle_A0[p])
        wp.atomic_add(center_weight_sum, c, w)


@wp.kernel
def normalize_structure_tensor_kernel(
    center_A0_sum: wp.array(dtype=mat33),
    center_weight_sum: wp.array(dtype=real),
    center_A0: wp.array(dtype=mat33),
):
    c = wp.tid()
    w = center_weight_sum[c]
    if w > _0:
        center_A0[c] = center_A0_sum[c] / w


@wp.kernel
def center_gradient_from_nodes_kernel(
    node_velocity: wp.array(dtype=vec3),
    grad_weights: wp.array(dtype=real, ndim=3),
    center_grad: wp.array(dtype=mat33),
):
    c = wp.tid()
    G = mat33(_0)
    for n in range(node_velocity.shape[0]):
        dw = vec3(grad_weights[c, n, 0], grad_weights[c, n, 1], grad_weights[c, n, 2])
        G += wp.outer(node_velocity[n], dw)
    center_grad[c] = G


@wp.kernel
def center_residual_kernel(
    node_velocity: wp.array(dtype=vec3),
    node_velocity_old: wp.array(dtype=vec3),
    node_mass: wp.array(dtype=real),
    gravity: vec3,
    dt: real,
    trial_F: wp.array(dtype=mat33),
    A0: wp.array(dtype=mat33),
    center_volume: wp.array(dtype=real),
    grad_weights: wp.array(dtype=real, ndim=3),
    residual: wp.array(dtype=vec3),
    mu: real,
    lam: real,
    k_f: real,
):
    n = wp.tid()
    r = node_mass[n] * (node_velocity[n] - node_velocity_old[n] - dt * gravity)
    for c in range(trial_F.shape[0]):
        dw = vec3(grad_weights[c, n, 0], grad_weights[c, n, 1], grad_weights[c, n, 2])
        tau = aniso_pk1_wp(trial_F[c], A0[c], mu, lam, k_f) @ wp.transpose(trial_F[c])
        r += dt * center_volume[c] * (tau @ dw)
    residual[n] = r


@wp.kernel
def center_matvec_kernel(
    direction: wp.array(dtype=vec3),
    node_mass: wp.array(dtype=real),
    dt: real,
    committed_F: wp.array(dtype=mat33),
    trial_F: wp.array(dtype=mat33),
    A0: wp.array(dtype=mat33),
    center_volume: wp.array(dtype=real),
    grad_weights: wp.array(dtype=real, ndim=3),
    center_grad_direction: wp.array(dtype=mat33),
    matvec: wp.array(dtype=vec3),
    mu: real,
    lam: real,
    k_f: real,
):
    n = wp.tid()
    out = node_mass[n] * direction[n]
    for c in range(committed_F.shape[0]):
        F = trial_F[c]
        dF = dt * center_grad_direction[c] @ committed_F[c]
        H = dPdF_StVK_Hencky_3D_analytic(
            F, mu, lam, wp.int32(0), s_min, dPdF_eps_small, dPdF_eps_pd
        )
        dP_iso = vec9_to_mat_wp(H @ mat_to_vec9_wp(dF))
        dP = dP_iso + fiber_dpk1_wp(F, A0[c], dF, k_f)
        P = aniso_pk1_wp(F, A0[c], mu, lam, k_f)
        dTau = dP @ wp.transpose(F) + P @ wp.transpose(dF)
        dw = vec3(grad_weights[c, n, 0], grad_weights[c, n, 1], grad_weights[c, n, 2])
        out += dt * center_volume[c] * (dTau @ dw)
    matvec[n] = out


@wp.func
def aniso_energy_wp(F: mat33, A0: mat33, mu: real, lam: real, k_f: real) -> real:
    U, sigma, V = wp.svd3(F)
    logs = vec3(wp.log(sigma[0]), wp.log(sigma[1]), wp.log(sigma[2]))
    tr = logs[0] + logs[1] + logs[2]
    strain = fiber_invariant_wp(F, A0) - real(1)
    return mu * wp.dot(logs, logs) + real(.5)*lam*tr*tr + real(.5)*k_f*strain*strain


@wp.func
def aniso_dpk1_wp(F: mat33, A0: mat33, dF: mat33, mu: real, lam: real,
                  k_f: real, project_pd: int) -> mat33:
    # Exact material Hessian, or a PSD approximation used ONLY for search directions.
    H = dPdF_StVK_Hencky_3D_analytic(F, mu, lam, project_pd, s_min, dPdF_eps_small, dPdF_eps_pd)
    fiber = fiber_dpk1_wp(F, A0, dF, k_f)
    if project_pd != 0:
        FA = F @ A0
        coefficient = wp.max(real(0), real(2)*k_f*(fiber_invariant_wp(F, A0)-real(1)))
        fiber = coefficient*(dF @ A0) + real(4)*k_f*wp.trace(wp.transpose(FA) @ dF)*FA
    return vec9_to_mat_wp(H @ mat_to_vec9_wp(dF)) + fiber
