"""Spatial Q1 / quadratic B-spline samples and explicit affine extension.

All choices are frozen for one step. Extension changes the approximation
space through a documented constraint, never by altering the mass matrix.
"""
from itertools import product
import numpy as np
from scipy import sparse


def sample_basis(x,h,origin,kernel):
    q=(x-origin)/h
    if kernel=='q1':
        offsets=np.asarray(list(product(range(2),repeat=3)))
        cells=np.floor(q).astype(int);d=q[:,None,:]-(cells[:,None,:]+offsets)
        weights=1-np.abs(d);derivatives=np.broadcast_to((2*offsets-1)/h,d.shape)
    elif kernel=='quadratic':
        offsets=np.asarray(list(product(range(3),repeat=3)))
        cells=np.floor(q-.5).astype(int);d=q[:,None,:]-(cells[:,None,:]+offsets)
        a=np.abs(d)
        weights=np.where(a<.5,.75-d*d,np.where(a<1.5,.5*(1.5-a)**2,0.))
        derivatives=np.where(a<.5,-2*d,np.where(a<1.5,-np.sign(d)*(1.5-a),0.))/h
    else:raise ValueError('kernel must be q1 or quadratic')
    nodes=cells[:,None,:]+offsets
    coordinates,ids=np.unique(nodes.reshape(-1,3),axis=0,return_inverse=True)
    rows=np.repeat(np.arange(len(x)),len(offsets));shape=(len(x),len(coordinates))
    N=sparse.coo_matrix((weights.prod(2).ravel(),(rows,ids)),shape=shape).tocsr()
    D=tuple(sparse.coo_matrix(((derivatives[:,:,j]*weights[:,:,[k for k in range(3) if k!=j]].prod(2)).ravel(),(rows,ids)),shape=shape).tocsr() for j in range(3))
    N.eliminate_zeros()
    return coordinates,N,D


def affine_weights(nodes,target,h):
    """Minimum-norm coefficients reproducing constants and physical position."""
    A=np.vstack([np.ones(len(nodes)),((nodes-target)/h).T])
    if np.linalg.matrix_rank(A)<4:raise ValueError('affine support is not three-dimensional')
    return A.T@np.linalg.solve(A@A.T,np.array([1.,0.,0.,0.]))


def extend_weak_nodes(N,D,coordinates,h,origin,mass,threshold):
    if not 0<threshold<1:raise ValueError('support threshold must lie in (0,1)')
    diagonal=np.asarray(N.power(2).T@mass).ravel()
    strong=np.flatnonzero(diagonal>=threshold*diagonal.max())
    nodes=origin+h*coordinates
    T=np.zeros((len(nodes),len(strong)));T[strong,np.arange(len(strong))]=1.
    weak=np.setdiff1d(np.arange(len(nodes)),strong)
    for node in weak:
        order=np.argsort(np.sum((nodes[strong]-nodes[node])**2,axis=1),kind='stable')
        # Local affine extrapolation; expand only if the closest stencil is flat.
        for count in (8,16,len(strong)):
            selected=order[:min(count,len(strong))]
            try:w=affine_weights(nodes[strong[selected]],nodes[node],h)
            except ValueError:continue
            T[node,selected]=w;break
        else:raise ValueError('insufficient well-supported nodes for affine extension')
    T=sparse.csr_matrix(T)
    return (N@T).tocsr(),tuple((d@T).tocsr() for d in D),coordinates[strong],len(weak)


def residual_modes(x,mass,residual):
    """Analytic compact radial fields matching up to three velocity residuals.

The compact kernel is phi(s)=(1-s)^4(4s+1), s<1. Its gradient is
-20(1-s)^3 (x-center)/radius^2. The interpolation matrix must be SPD;
there is no diagonal regularization. This is a small-scene experiment.
"""
    from scipy.linalg import cho_factor,cho_solve
    from scipy.spatial.distance import cdist
    weights=mass/mass.sum();U,s,_=np.linalg.svd(np.sqrt(weights)[:,None]*residual,full_matrices=False)
    keep=s>max(1e-10,1e-9*s[0])
    if not np.any(keep):return np.empty((len(x),0)),tuple(np.empty((len(x),0)) for _ in range(3)),s,0.
    Q=U[:,keep]/np.sqrt(weights)[:,None]
    distance=cdist(x,x);nearest=distance.copy();np.fill_diagonal(nearest,np.inf)
    radius=2.5*float(np.median(nearest.min(axis=1)))
    if not np.isfinite(radius) or radius<=0:raise ValueError('distinct residual-mode sites required')
    scaled=distance/radius;t=np.maximum(1-scaled,0.)
    K=t**4*(4*scaled+1)
    coefficients=cho_solve(cho_factor(K,lower=True),Q)
    D=tuple((-20*t**3*(x[:,j,None]-x[None,:,j])/radius**2)@coefficients for j in range(3))
    return K@coefficients,D,s,radius
