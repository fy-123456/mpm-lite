"""Affine projected coupling, with full pressure and explicit reference limits."""
import numpy as np
import scipy.linalg as la

def mass_basis(M,columns,rank=12):
    """Ordered twice-reorthogonalized basis; no changes to production DOFs."""
    L=la.cholesky(M,lower=True);Q=[]
    for v in np.asarray(columns).T:
        w=L.T@v;initial=la.norm(w)
        if initial==0:continue
        w=w/initial
        for _ in range(2):
            for q in Q:w-=q*(q@w)
        n=la.norm(w)
        if n<1e-9:continue
        Q.append(w/n)
        if len(Q)==rank:break
    if len(Q)<rank:raise ValueError('insufficient independent diagnostic directions')
    U=la.solve_triangular(L.T,np.array(Q).T,lower=False)
    if la.norm(U.T@M@U-np.eye(rank))>1e-8:raise ValueError('mass orthogonality failed')
    return U

def affine_operator(K,G,C,L,D,accel,pdot,alpha=.8):
    """Mass-orthonormal solid basis, zero-velocity anchor; D=d(Bz)/dq."""
    K=np.asarray(K);r=len(K);c=len(C)
    if K.shape!=(r,r) or G.shape!=(c,r) or D.shape!=(c,r) or np.any(C<=0):raise ValueError('invalid coupled blocks')
    A=np.zeros((2*r+c,2*r+c));A[:r,r:2*r]=np.eye(r)
    A[r:2*r,:r]=-K;A[r:2*r,2*r:]=alpha*G.T
    A[2*r:,:r]=-D/C[:,None];A[2*r:,r:2*r]=-alpha*G/C[:,None];A[2*r:,2*r:]=-L/C[:,None]
    drift=np.r_[np.zeros(r),accel,pdot]
    if not np.isfinite(A).all() or not np.isfinite(drift).all():raise ValueError('nonfinite affine system')
    return A,drift

def exact(A,drift,times,initial=None,scale=None):
    """Diagonal similarity balances units without changing eigenvalues or drift."""
    n=len(A);initial=np.zeros(n) if initial is None else np.asarray(initial)
    scale=np.ones(n) if scale is None else np.asarray(scale)
    if scale.shape!=(n,) or np.any(scale<=0) or not np.isfinite(scale).all():raise ValueError('positive state scales required')
    aug=np.zeros((n+1,n+1));aug[:n,:n]=A*scale[None,:]/scale[:,None];aug[:n,n]=drift/scale
    return np.array([(la.expm(float(t)*aug)@np.r_[initial/scale,1.])[:n]*scale for t in times])

def interval_average(cumulative,times):
    t=np.asarray(times)
    if np.any(np.diff(t)<=0):raise ValueError('increasing physical times required')
    return np.diff(cumulative,axis=0)/np.diff(t)[:,None]
