"""One Hencky + bilateral quadratic-fiber potential, including exact A4.

Conditional moments are exact at a common F only. No stress reconstruction,
positive-definite Hessian replacement or geometry/history update is performed.
"""
from dataclasses import replace
import numpy as np
from ..consistent_transfer import material_response
from ..history_increment import material_tangent


def check_state(F):
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 3 or F.shape[1:] != (3, 3) or not np.isfinite(F).all():
        raise ValueError("F must be finite with shape (samples, 3, 3)")
    det = np.linalg.det(F)
    if not np.isfinite(det).all() or np.any(det <= 0):
        raise ValueError("non-positive or non-finite material Jacobian")
    if np.any(np.linalg.svd(F, compute_uv=False)[:, -1] < 1e-9):
        raise ValueError("singular material state")
    return F


def response(F, A2, A4, params, direction=None):
    """Energy density, sample PK1 and optional exact dPK1 from this energy."""
    F = check_state(F)
    if A2.shape != F.shape or A4.shape != (len(F), 9, 9):
        raise ValueError("moment/sample shape mismatch")
    iso = replace(params, k_f=0.)
    energy, P = material_response(F, A2, iso)
    strain = (F.transpose(0, 2, 1) @ F - np.eye(3)).reshape(-1, 9)
    S = np.einsum('pij,pj->pi', A4, strain).reshape(-1, 3, 3)
    energy += .5 * params.k_f * np.einsum('pi,pi->p', strain, S.reshape(-1, 9))
    P += 2 * params.k_f * (F @ S)
    if not np.isfinite(energy).all() or not np.isfinite(P).all():
        raise ValueError("non-finite constitutive response")
    if direction is None:
        return energy, P
    dF = np.asarray(direction, dtype=float)
    if dF.shape != F.shape or not np.isfinite(dF).all():
        raise ValueError("finite tangent direction with same shape as F required")
    dC = dF.transpose(0, 2, 1) @ F + F.transpose(0, 2, 1) @ dF
    dS = np.einsum('pij,pj->pi', A4, dC.reshape(-1, 9)).reshape(-1, 3, 3)
    dP = material_tangent(F, A2, dF, iso) + 2 * params.k_f * (dF @ S + F @ dS)
    if not all(np.isfinite(a).all() for a in (energy, P, dP)):
        raise ValueError("non-finite constitutive response")
    return energy, P, dP
