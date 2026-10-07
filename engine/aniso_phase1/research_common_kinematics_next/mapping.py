"""Updated SH modal carrier plus material bubbles, with exact chain derivative.

The virtual grid has a complete halo and origin -h/2; its center knots are i*h.
H averages nodal modal profiles at eight corners. The quadratic nodal profile
x^2-h_x^2/4 averages to center x^2. S interpolates these center values.
This is a 10-scalar research subspace, NOT the formal 144-function space.
"""
import itertools
import numpy as np
CORNERS=np.array(list(itertools.product((0,1),repeat=3)))
SCHEMA='updated-SH-modal-plus-material-bubbles-v1'


def carrier(x, h, lengths):
    x=np.asarray(x,float);h=np.asarray(h,float);lengths=np.asarray(lengths,float)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all():
        raise ValueError('finite current positions required')
    z=x/h;base=np.floor(z).astype(int);f=z-base
    N=np.zeros((len(x),8));D=np.zeros((len(x),8,3))
    for corner in CORNERS:
        c=(base+corner)*h;t=c[:,0]/lengths[0]
        y=c[:,1]/lengths[1]-.5;zz=c[:,2]/lengths[2]-.5
        profile=np.stack([t,t*y,t*zz,t*y*zz,t*t,t*t*y,t*t*zz,t*t*y*zz],axis=1)
        factors=np.where(corner,f,1-f)
        N+=np.prod(factors,axis=1)[:,None]*profile
        for a in range(3):
            dw=(2*corner[a]-1)*np.prod(np.delete(factors,a,axis=1),axis=1)/h[a]
            D[:,:,a]+=dw[:,None]*profile
    return N,D


def basis(space,X,x,F):
    X,x,F=map(np.asarray,(X,x,F))
    if x.shape!=X.shape or F.shape!=(len(X),3,3) or not np.isfinite(F).all() or np.any(np.linalg.det(F)<=.1):
        raise ValueError('matched admissible step-initial x/F required')
    N,Dx=carrier(x,space.h,space.lengths)
    DX=np.einsum('qni,qij->qnj',Dx,F)
    Nb,Db=space.basis(X,free=False)
    return np.c_[N,Nb[:,space.ncarrier:]],np.concatenate([DX,Db[:,space.ncarrier:]],axis=1)


def push(space,X,x,F,d):
    N,D=basis(space,X,x,F)
    return x+N@d,F+np.einsum('qnd,ni->qid',D,d)
