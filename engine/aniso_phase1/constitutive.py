"""Reference constitutive equations for stage-one anisotropy."""

from __future__ import annotations

import numpy as np

from .types import AnisotropicMaterialParams


def a0_to_A0(direction: np.ndarray) -> np.ndarray:
    """Return the unoriented structure tensor ``a ⊗ a``."""
    a = np.asarray(direction, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("direction must be a finite 3-vector")
    norm = np.linalg.norm(a)
    if norm <= 0.0:
        raise ValueError("direction must have non-zero length")
    a = a / norm
    return np.outer(a, a)


def average_structure_tensors(A0: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted center average of particle structure tensors.

    The result is normalized by total weight and remains symmetric positive
    semi-definite.  An empty or zero-weight center is rejected explicitly.
    """
    tensors = np.asarray(A0, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if tensors.ndim != 3 or tensors.shape[1:] != (3, 3):
        raise ValueError("A0 must have shape (N, 3, 3)")
    if not np.isfinite(tensors).all():
        raise ValueError("A0 must contain finite values")
    if np.max(np.abs(tensors - np.swapaxes(tensors, 1, 2))) > 1.0e-10:
        raise ValueError("A0 tensors must be symmetric")
    if np.min(np.linalg.eigvalsh(tensors)) < -1.0e-10:
        raise ValueError("A0 tensors must be positive semi-definite")
    if w.shape != (tensors.shape[0],) or not np.isfinite(w).all() or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("at least one positive center weight is required")
    out = np.tensordot(w, tensors, axes=(0, 0)) / total
    return 0.5 * (out + out.T)


def structure_tensor_mixing(A0: np.ndarray) -> float:
    """Return a scalar measure of multi-direction content in ``A0``.

    A pure unoriented fiber has eigenvalues ``(1, 0, 0)``.  A weighted
    mixture has a smaller largest eigenvalue, so ``1 - lambda_max`` is zero
    for a pure direction and positive when several directions are present.
    """
    tensor = np.asarray(A0, dtype=np.float64)
    if tensor.shape != (3, 3) or not np.isfinite(tensor).all():
        raise ValueError("A0 must be a finite 3x3 matrix")
    if np.max(np.abs(tensor - tensor.T)) > 1.0e-10:
        raise ValueError("A0 must be symmetric")
    if np.linalg.eigvalsh(tensor).min() < -1.0e-10:
        raise ValueError("A0 must be positive semi-definite")
    trace = float(np.trace(tensor))
    if trace <= 0.0:
        raise ValueError("A0 must have positive trace")
    normalized = tensor / trace
    return float(max(0.0, 1.0 - np.linalg.eigvalsh(normalized)[-1]))


def _check_F(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    if F.shape != (3, 3) or not np.isfinite(F).all():
        raise ValueError("F must be a finite 3x3 matrix")
    if np.linalg.det(F) <= 0.0:
        raise ValueError("F must have positive determinant")
    return F


def fiber_invariant(F: np.ndarray, A0: np.ndarray) -> float:
    F = _check_F(F)
    A0 = np.asarray(A0, dtype=np.float64)
    if A0.shape != (3, 3):
        raise ValueError("A0 must have shape (3, 3)")
    C = F.T @ F
    return float(np.einsum("ij,ij", A0, C))


def _iso_energy_and_pk1(F: np.ndarray, mu: float, lam: float) -> tuple[float, np.ndarray]:
    U, sigma, Vt = np.linalg.svd(F)
    sigma = np.maximum(sigma, 1.0e-12)
    log_sigma = np.log(sigma)
    tr = float(log_sigma.sum())
    psi = float(mu * np.dot(log_sigma, log_sigma) + 0.5 * lam * tr * tr)
    m = (2.0 * mu * log_sigma + lam * tr) / sigma
    return psi, U @ np.diag(m) @ Vt


def energy(F: np.ndarray, A0: np.ndarray, params: AnisotropicMaterialParams) -> float:
    F = _check_F(F)
    iso, _ = _iso_energy_and_pk1(F, params.mu, params.lam)
    I4 = fiber_invariant(F, A0)
    return iso + 0.5 * params.k_f * (I4 - 1.0) ** 2


def pk1(F: np.ndarray, A0: np.ndarray, params: AnisotropicMaterialParams) -> np.ndarray:
    """First Piola stress ``P_iso + 2 ψ4 F A0``."""
    F = _check_F(F)
    _, P_iso = _iso_energy_and_pk1(F, params.mu, params.lam)
    I4 = fiber_invariant(F, A0)
    psi4 = params.k_f * (I4 - 1.0)
    return P_iso + 2.0 * psi4 * F @ np.asarray(A0, dtype=np.float64)


def kirchhoff(F: np.ndarray, A0: np.ndarray, params: AnisotropicMaterialParams) -> np.ndarray:
    F = _check_F(F)
    return pk1(F, A0, params) @ F.T


def dP_fiber_apply(
    F: np.ndarray,
    A0: np.ndarray,
    dF: np.ndarray,
    params: AnisotropicMaterialParams,
) -> np.ndarray:
    """Analytical fiber directional derivative.

    This is the matrix-free expression used by the implicit operator:
    ``2 ψ4 dF A0 + 4 ψ44 ((F A0):dF) F A0``.
    """
    F = _check_F(F)
    dF = np.asarray(dF, dtype=np.float64)
    A0 = np.asarray(A0, dtype=np.float64)
    if dF.shape != (3, 3) or not np.isfinite(dF).all():
        raise ValueError("dF must be a finite 3x3 matrix")
    if A0.shape != (3, 3):
        raise ValueError("A0 must have shape (3, 3)")
    FA = F @ A0
    I4 = fiber_invariant(F, A0)
    psi4 = params.k_f * (I4 - 1.0)
    psi44 = params.k_f
    contraction = float(np.einsum("ij,ij", FA, dF))
    return 2.0 * psi4 * (dF @ A0) + 4.0 * psi44 * contraction * FA


def dP_apply_finite_difference(
    F: np.ndarray,
    A0: np.ndarray,
    dF: np.ndarray,
    params: AnisotropicMaterialParams,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Reference total directional derivative for Jacobian checks."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return (pk1(F + eps * dF, A0, params) - pk1(F - eps * dF, A0, params)) / (2.0 * eps)
