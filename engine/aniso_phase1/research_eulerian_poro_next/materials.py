"""Bounded material rules; a director fit is NOT a distribution closure.

The new rule fits the local projective log of rank-one particle directions.
It may extrapolate a direction near boundaries but constructs positive moments
from that direction, rather than extrapolating tensors with negative weights.
The old positive tensor rule remains available for compatible structured input.
No analytic direction field or particle arrays enter the resulting operator.
"""
import numpy as np
from scipy.spatial import cKDTree
from engine.aniso_phase1.research_unified_lite_poro.model import FrozenOperator


def validate_moments(A2, A4):
    A2, A4 = np.asarray(A2), np.asarray(A4)
    if (A2.ndim != 3 or A2.shape[1:] != (3, 3) or
            A4.shape != (len(A2), 9, 9) or
            not np.isfinite(A2).all() or not np.isfinite(A4).all()):
        raise ValueError('finite matched direction moment arrays required')
    contraction = (A4 @ np.eye(3).ravel()).reshape(-1, 3, 3)
    if not np.allclose(contraction, A2, atol=1e-9, rtol=1e-8):
        raise ValueError('A4 contraction must equal A2')
    if (np.max(abs(A2-A2.swapaxes(1,2))) > 1e-9 or
            np.max(abs(A4-A4.swapaxes(1,2))) > 1e-9 or
            np.max(abs(np.trace(A2, axis1=1, axis2=2)-1)) > 1e-8 or
            np.linalg.eigvalsh(A2).min() < -1e-9 or
            np.linalg.eigvalsh(A4).min() < -1e-9):
        raise ValueError('normalized symmetric positive direction moments required')


def director_moments(space, particles, points, neighbors=24):
    """Prepare-only fit, restricted to a smooth single director per sample."""
    p = particles
    if (len(p.X) < 4 or not np.isfinite(p.X).all() or
            not np.isfinite(p.A).all() or not np.isfinite(p.weight).all() or
            np.any(p.weight <= 0)):
        raise ValueError('finite positive particle samples required')
    if not np.allclose(p.A, p.A.swapaxes(1, 2), atol=1e-9):
        raise ValueError('symmetric rank-one particle directions required')
    lam, vec = np.linalg.eigh(p.A)
    if not np.allclose(lam, [0., 0., 1.], atol=1e-7, rtol=0):
        raise ValueError('director rule only supports rank-one directions, not mixtures')
    # Canonical order makes nearest-neighbor ties independent of particle order.
    order = np.lexsort((p.X[:,2], p.X[:,1], p.X[:,0]))
    X, directors, weights = p.X[order], vec[order,:,-1], p.weight[order]
    if len(np.unique(X, axis=0)) != len(X):
        raise ValueError('duplicate reference samples require a distribution rule')
    scale = np.asarray(space.h)
    _, indices = cKDTree(X/scale).query(points/scale, k=min(neighbors, len(X)))
    result = []; max_residual = 0.; max_condition = 0.
    for point, ids in zip(points, indices):
        a = directors[ids].copy(); anchor = a[0]
        a *= np.where(a@anchor < 0., -1., 1.)[:,None]
        cosine = np.clip(a@anchor, -1., 1.)
        theta = np.arccos(cosine)
        if theta.max() >= np.pi/3:
            raise ValueError('ambiguous or rapidly varying local director chart')
        tangent = a-cosine[:,None]*anchor
        factor = np.ones_like(theta)
        mask = theta > 1e-8
        factor[mask] = theta[mask]/np.sin(theta[mask])
        log = tangent*factor[:,None]
        delta = (X[ids]-point)/scale
        design = np.c_[np.ones(len(ids)), delta]
        w = np.sqrt(weights[ids]/weights[ids].mean()/(1+np.sum(delta**2,axis=1)))
        coeff, _, rank, sv = np.linalg.lstsq(design*w[:,None], log*w[:,None], rcond=None)
        if rank != 4 or sv[0]/sv[-1] > 1e6:
            raise ValueError('local spatial director fit is rank deficient')
        residual = float(np.max(np.linalg.norm(design@coeff-log,axis=1)))
        if residual > .035:
            raise ValueError('local director fit exceeds smooth-field allowance')
        v = coeff[0]-np.dot(coeff[0],anchor)*anchor
        angle = np.linalg.norm(v)
        if angle >= np.pi/3:
            raise ValueError('director extrapolation exceeds local chart')
        reconstructed = np.cos(angle)*anchor + np.sinc(angle/np.pi)*v
        reconstructed /= np.linalg.norm(reconstructed)
        result.append(reconstructed)
        max_residual=max(max_residual,residual);max_condition=max(max_condition,float(sv[0]/sv[-1]))
    a = np.asarray(result); A2=a[:,:,None]*a[:,None,:]; flat=A2.reshape(-1,9)
    A4=flat[:,:,None]*flat[:,None,:]
    validate_moments(A2,A4)
    return A2,A4,dict(max_log_fit_residual_rad=max_residual,max_fit_condition=max_condition,
                       scope='local smooth single director; no distribution closure')


def make_operator(space, particles, rule='director-log', order=4):
    # A alone cannot specify the fourth moment of a mixture at one particle.
    # Mixtures across individually rank-one particles are supported by fallback.
    if (not np.isfinite(particles.A).all() or
            not np.allclose(particles.A,particles.A.swapaxes(1,2),atol=1e-9) or
            not np.allclose(np.linalg.eigvalsh(particles.A),[0.,0.,1.],atol=1e-7,rtol=0)):
        raise ValueError('rank-one particle A required; mixed particle state needs explicit A4')
    if rule == 'fixed-positive':
        op=FrozenOperator.fixed_positive(space,particles,order)
        info={'selected':'fixed-positive','fallback':False}
    elif rule in ('director-log','director-or-positive'):
        X,w,_=space.rule(order)
        try:
            A2,A4,info=director_moments(space,particles,X)
            op=FrozenOperator(space,X,w,A2,A4,order=order)
            info.update(selected='director-log',fallback=False)
        except ValueError as error:
            if rule != 'director-or-positive':raise
            op=FrozenOperator.fixed_positive(space,particles,order)
            info={'selected':'fixed-positive','fallback':True,'reason':str(error)}
    else:raise ValueError('unknown material rule')
    validate_moments(op.A2,op.A4)
    return op,info
