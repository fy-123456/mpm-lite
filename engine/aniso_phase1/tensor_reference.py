"""v21 tensor-product Qp elasticity with an exact separable preconditioner.

Only the physical free span is solved; rigid grip volumes are restored on
output. The operator is the same unshifted material energy as independent
assembled FEM. The preconditioner changes convergence speed, not stiffness.
"""
import itertools
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator,cg
from benchmarks.aniso_local_q3 import axis,basis,gradient,chunks


def apply_axis(A,u,k):
    v=np.moveaxis(u,k,0);shape=v.shape
    out=A@v.reshape(shape[0],-1)
    return np.moveaxis(out.reshape(A.shape[0],*shape[1:]),0,k)


def coordinates(edges,degree):
    return [axis(e,degree)[0] for e in edges]


def interpolation_axis(source,degree,target):
    e=np.asarray(source);x=np.asarray(target);i=np.clip(np.searchsorted(e,x,side='right')-1,0,len(e)-2)
    N,_=basis((x-e[i])/(e[i+1]-e[i]),degree);ids=degree*i[:,None]+np.arange(degree+1)
    return sp.csr_matrix((N.ravel(),(np.repeat(np.arange(len(x)),degree+1),ids.ravel())),shape=(len(x),degree*(len(e)-1)+1))


def interpolate(source,p,values,target,q):
    old=tuple(p*(len(e)-1)+1 for e in source);u=np.asarray(values).reshape(*old,-1)
    for k,x in enumerate(coordinates(target,q)):u=apply_axis(interpolation_axis(source[k],p,x),u,k)
    return u.reshape(-1,values.shape[-1])


class TensorElastic:
    def __init__(self,edges,p,H):
        self.edges=[np.asarray(e) for e in edges];self.p=p;self.H=np.asarray(H).reshape(3,3,3,3)
        self.axes=[axis(e,p) for e in edges];self.shape=tuple(len(a[0]) for a in self.axes);self.n=int(np.prod(self.shape));self.free_shape=(self.shape[0]-2,*self.shape[1:])
        eig=[];vectors=[]
        for k,(_,grams) in enumerate(self.axes):
            M,K=grams[:2];sl=slice(1,-1) if k==0 else slice(None)
            w,V=la.eigh(K[sl,sl].toarray(),M[sl,sl].toarray());w[np.abs(w)<1e-8]=0.;eig.append(w);vectors.append(V)
        self.vectors=vectors
        weights=np.array([[self.H[a,k,a,k] for k in range(3)] for a in range(3)])
        self.den=np.stack([sum(weights[a,k]*eig[k].reshape(tuple(len(eig[k]) if j==k else 1 for j in range(3))) for k in range(3)) for a in range(3)],axis=-1)
        assert self.den.min()>0
    def apply(self,v):
        u=np.asarray(v).reshape(3,*self.shape).transpose(1,2,3,0);out=np.zeros_like(u)
        for i in range(3):
            for j in range(3):
                H=self.H[:,i,:,j]
                if not np.any(H):continue
                w=u
                for k,(_,grams) in enumerate(self.axes):
                    A=grams[1] if i==j==k else (grams[2] if k==i and i!=j else (grams[2].T if k==j and i!=j else grams[0]))
                    w=apply_axis(A,w,k)
                out+=w@H.T
        return out.transpose(3,0,1,2).ravel()
    def free_apply(self,v):
        u=np.zeros((3,*self.shape));u[:,1:-1]=v.reshape(3,*self.free_shape)
        return self.apply(u).reshape(3,*self.shape)[:,1:-1].ravel()
    def precondition(self,v):
        u=v.reshape(3,*self.free_shape).transpose(1,2,3,0)
        for k,V in enumerate(self.vectors):u=apply_axis(V.T,u,k)
        u/=self.den
        for k,V in enumerate(self.vectors):u=apply_axis(V,u,k)
        return u.transpose(3,0,1,2).ravel()


def solve(edges,p,H,initial=None,rtol=2e-11,callback=None):
    start=time.monotonic();edges=[np.asarray(e) for e in edges];active=[edges[0][(edges[0]>=.25-1e-12)&(edges[0]<=.75+1e-12)],*edges[1:]]
    if len(active[0])<2 or abs(active[0][0]-.25)>1e-12 or abs(active[0][-1]-.75)>1e-12:raise ValueError('mesh must include both grip transitions')
    op=TensorElastic(active,p,H);shape=op.shape;u=np.zeros((3,*shape));u[0,-1]=.005
    rhs=-op.apply(u).reshape(3,*shape)[:,1:-1].ravel();n=len(rhs);A=LinearOperator((n,n),matvec=op.free_apply,dtype=float);M=LinearOperator((n,n),matvec=op.precondition,dtype=float);count=0
    x0=None
    if initial is not None:
        es,ps,us=initial;ini=interpolate(es,ps,us,active,p).reshape(*shape,3).transpose(3,0,1,2);x0=ini[:,1:-1].ravel()
    def cb(v):
        nonlocal count
        count+=1
        if callback is not None and count%50==0:callback(count)
    v,info=cg(A,rhs,x0=x0,M=M,rtol=rtol,atol=1e-14,maxiter=3000,callback=cb);u[:,1:-1]=v.reshape(3,*op.free_shape);f=op.apply(u).reshape(3,*shape)
    residual=float(la.norm(f[:,1:-1])/la.norm(rhs));R=float(f[0,-1].sum());energy=float(.5*np.sum(u*f));axes=coordinates(edges,p);full=np.zeros((*map(len,axes),3));full[axes[0]>=.75-1e-12,...,0]=.005;where=(axes[0]>=.25-1e-12)&(axes[0]<=.75+1e-12);full[where]=u.transpose(1,2,3,0)
    result=dict(degree=p,cells=[len(e)-1 for e in edges],nodes=int(np.prod(full.shape[:3])),free_dofs=n,iterations=count,linear_info=int(info),relative_residual=residual,reaction_N=R,energy_J=energy,work_identity_relative=abs(energy-.5*.005*R)/energy,passed=bool(info==0 and residual<1e-8),mass_included=False,stiffness_shift=0.,seconds=time.monotonic()-start,preconditioner='separable generalized-eigenvalue Laplacian; operator unchanged')
    return full.reshape(-1,3),result
