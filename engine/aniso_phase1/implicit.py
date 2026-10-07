"""Small reference implementation of the center-only implicit equations.

This module is intentionally NumPy based.  It is a correctness oracle for the
Warp matrix-free kernels: each center is evaluated once, while CG directions
only use center data and never particle constitutive calls.
"""

from __future__ import annotations

import numpy as np

from .constitutive import dP_apply_finite_difference, kirchhoff, pk1
from .types import AnisotropicMaterialParams


def matrix_free_pcg(
    matvec,
    rhs: np.ndarray,
    preconditioner=None,
    maxiter: int = 200,
    tol: float = 1.0e-8,
) -> tuple[np.ndarray, dict]:
    """Small deterministic PCG reference driver for center operators."""
    b = np.asarray(rhs, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError("rhs must have shape (N, 3)")
    if maxiter <= 0 or tol <= 0.0:
        raise ValueError("maxiter and tol must be positive")
    apply_M = (lambda x: x) if preconditioner is None else preconditioner
    x = np.zeros_like(b)
    r = b.copy()
    z = apply_M(r)
    p = z.copy()
    rz = float(np.sum(r * z))
    initial = float(np.sqrt(max(np.sum(r * r), 0.0)))
    target = max(tol * max(initial, 1.0), tol)
    if initial <= target:
        return x, {"iterations": 0, "residual_norm": initial, "converged": True}
    residual_norm = initial
    for iteration in range(1, maxiter + 1):
        Ap = np.asarray(matvec(p), dtype=np.float64)
        denom = float(np.sum(p * Ap))
        if not np.isfinite(denom) or denom <= 0.0:
            return x, {"iterations": iteration - 1, "residual_norm": residual_norm, "converged": False, "breakdown": True}
        alpha = rz / denom
        x += alpha * p
        r -= alpha * Ap
        residual_norm = float(np.sqrt(max(np.sum(r * r), 0.0)))
        if residual_norm <= target:
            return x, {"iterations": iteration, "residual_norm": residual_norm, "converged": True}
        z = apply_M(r)
        rz_new = float(np.sum(r * z))
        if not np.isfinite(rz_new) or rz == 0.0:
            return x, {"iterations": iteration, "residual_norm": residual_norm, "converged": False, "breakdown": True}
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, {"iterations": maxiter, "residual_norm": residual_norm, "converged": False}


def center_trial_F(committed_F: np.ndarray, center_grad_v: np.ndarray, dt: float) -> np.ndarray:
    """Compute ``F_c(v) = (I + dt G_c(v)) F_c^n`` for all centers."""
    F0 = np.asarray(committed_F, dtype=np.float64)
    G = np.asarray(center_grad_v, dtype=np.float64)
    if F0.ndim != 3 or F0.shape[1:] != (3, 3) or G.shape != F0.shape:
        raise ValueError("committed_F and center_grad_v must have shape (C, 3, 3)")
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError("dt must be finite and non-negative")
    return np.einsum("cij,cjk->cik", np.eye(3)[None, :, :] + dt * G, F0)


def center_velocity_gradient(node_velocity: np.ndarray, grad_weights: np.ndarray) -> np.ndarray:
    """Accumulate ``G_c = Σ_i v_i ⊗ ∇w_ic``.

    ``grad_weights`` uses shape ``(centers, nodes, 3)`` and shares the node
    axis with ``node_velocity``.
    """
    v = np.asarray(node_velocity, dtype=np.float64)
    dw = np.asarray(grad_weights, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or dw.ndim != 3 or dw.shape[1:] != (v.shape[0], 3):
        raise ValueError("node_velocity must be (N,3), grad_weights must be (C,N,3)")
    return np.einsum("ni,cnj->cij", v, dw)


def center_residual(
    node_velocity: np.ndarray,
    node_velocity_old: np.ndarray,
    node_mass: np.ndarray,
    gravity: np.ndarray,
    dt: float,
    committed_F: np.ndarray,
    A0: np.ndarray,
    center_volume: np.ndarray,
    grad_weights: np.ndarray,
    params: AnisotropicMaterialParams,
) -> np.ndarray:
    """Evaluate the center-only implicit residual from the plan."""
    v = np.asarray(node_velocity, dtype=np.float64)
    v_old = np.asarray(node_velocity_old, dtype=np.float64)
    mass = np.asarray(node_mass, dtype=np.float64)
    g = np.asarray(gravity, dtype=np.float64)
    volume = np.asarray(center_volume, dtype=np.float64)
    if v.shape != v_old.shape or v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("node velocities must have shape (N, 3)")
    if mass.shape != (v.shape[0],) or g.shape != (3,) or volume.ndim != 1:
        raise ValueError("mass, gravity, or center volume has an invalid shape")
    G = center_velocity_gradient(v, grad_weights)
    F = center_trial_F(committed_F, G, dt)
    taus = np.stack([kirchhoff(F[c], A0[c], params) for c in range(F.shape[0])])
    force = dt * np.einsum("c,cij,cnj->ni", volume, taus, grad_weights)
    return mass[:, None] * (v - v_old) - dt * mass[:, None] * g[None, :] + force


def center_matvec(
    direction: np.ndarray,
    node_mass: np.ndarray,
    dt: float,
    committed_F: np.ndarray,
    A0: np.ndarray,
    center_volume: np.ndarray,
    grad_weights: np.ndarray,
    params: AnisotropicMaterialParams,
    finite_difference_eps: float = 1.0e-6,
    trial_F: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the residual Jacobian to a node-velocity direction.

    Differentiate PK1 at the frozen Newton ``trial_F``, while the kinematic
    increment remains ``dt * grad(direction) @ committed_F``. Omitting trial_F
    selects the zero-velocity linearization for backwards compatibility.
    This NumPy oracle uses finite differences; device kernels use analytic dP.
    """
    p = np.asarray(direction, dtype=np.float64)
    mass = np.asarray(node_mass, dtype=np.float64)
    dw = np.asarray(grad_weights, dtype=np.float64)
    Gp = center_velocity_gradient(p, dw)
    F0 = np.asarray(committed_F, dtype=np.float64)
    A = np.asarray(A0, dtype=np.float64)
    volume = np.asarray(center_volume, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or mass.shape != (p.shape[0],):
        raise ValueError("direction and mass have incompatible shapes")
    if F0.shape != A.shape or F0.ndim != 3 or F0.shape[1:] != (3, 3):
        raise ValueError("committed_F and A0 must have shape (C, 3, 3)")
    F = F0 if trial_F is None else np.asarray(trial_F, dtype=np.float64)
    if F.shape != F0.shape:
        raise ValueError("trial_F must match committed_F shape")
    out = mass[:, None] * p
    for c in range(F0.shape[0]):
        dF = dt * Gp[c] @ F0[c]
        dP = dP_apply_finite_difference(F[c], A[c], dF, params, eps=finite_difference_eps)
        # tau = P F^T, so d tau = dP F^T + P dF^T.
        dTau = dP @ F[c].T + pk1(F[c], A[c], params) @ dF.T
        out += dt * volume[c] * (dTau @ dw[c].T).T
    return out
