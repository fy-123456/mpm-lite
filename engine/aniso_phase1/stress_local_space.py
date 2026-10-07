"""v21 stress-selected, component-closed local variational correction spaces.

A patch correction solves the ORIGINAL material residual on a bounded support.
Its component functions enter one scalar space shared by all vector components,
so rigid rotations of any deformed state remain representable. Patch selection
may use a coarse independent stress comparison; reference displacements are
never inserted as basis vectors. Original carrier stabilization is retained.
"""
import itertools
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from . import local_reference as ref
from .tensor_reference import apply_axis,interpolate
from .compatible_carrier import shape_matrices
from .history_increment import material_response,material_tangent


def patches():
    centers=list(itertools.product([.28125,.34375,.4375,.5625,.65625,.71875],[.375,.5,.625],[.375,.5,.625]))
    return [dict(center=list(c),lo=np.maximum(np.array(c)-.09375,[.25,.375,.375]).tolist(),hi=np.minimum(np.array(c)+.09375,[.75,.625,.625]).tolist()) for c in centers]


def patch_mask(X,patch):
    lo,hi=np.asarray(patch['lo']),np.asarray(patch['hi']);m=np.ones(len(X),bool)
    for k in range(3):
        left=X[:,k]>lo[k]+1e-12 if k==0 or lo[k]>.375+1e-12 else X[:,k]>=lo[k]-1e-12
        right=X[:,k]<hi[k]-1e-12 if k==0 or hi[k]<.625-1e-12 else X[:,k]<=hi[k]+1e-12
        m&=left&right
    return m


def geometric_order(patch_list):
    centers=np.array([p['center'] for p in patch_list]);v=(centers-[.25,.375,.375])/[.5,.25,.25];first=int(np.argmin(np.sum((v-.5)**2,axis=1)));order=[first]
    while len(order)<len(v):
        distance=np.min(np.sum((v[:,None]-v[order][None,:])**2,axis=2),axis=1);distance[order]=-1;order.append(int(np.argmax(distance)))
    return order


def patch_corrections(nodes,K,u,patch_list,callback=None):
    n=len(nodes);force=K@u.T.ravel();columns=[];records=[]
    for number,patch in enumerate(patch_list):
        idx=np.flatnonzero(patch_mask(nodes,patch));ids=np.concatenate([idx+a*n for a in range(3)]);A=K[ids][:,ids].tocsc();rhs=-force[ids];fac=splu(A);v=fac.solve(rhs);err=float(la.norm(A@v-rhs)/max(la.norm(rhs),1e-30));assert err<1e-8
        # Keep the scalar span of all components, then close it under rotations.
        w=v.reshape(3,-1).T;U,s,_=la.svd(w,full_matrices=False);rank=int(np.sum(s>max(s[0]*1e-10,1e-18)));begin=len(columns)
        for k in range(rank):
            z=np.zeros(n);z[idx]=U[:,k];columns.append(z)
        rec=dict(patch=number,local_vector_dofs=len(ids),scalar_rank=rank,columns=list(range(begin,len(columns))),relative_residual=err,residual_energy_reduction=float(.5*rhs@v),support=patch);records.append(rec)
        if callback is not None:callback(rec)
    return np.column_stack(columns),records


def scalar_grams(edges,T):
    axes=[ref.axis(e,2) for e in edges];shape=tuple(len(a[0]) for a in axes);field=T.reshape(*shape,T.shape[1]);G={}
    for i in range(3):
        for j in range(i,3):
            w=field
            for k,(_,g) in enumerate(axes):
                A=g[1] if i==j==k else (g[2] if k==i and i!=j else (g[2].T if k==j and i!=j else g[0]))
                w=apply_axis(A,w,k)
            G[i,j]=T.T@w.reshape(T.shape);G[j,i]=G[i,j].T if i!=j else G[i,j]
    return G


def reduced_solve(G,H,patch_matrix,patch_lift,patch_lift_energy,nf,ids,Wgram):
    """Solve original quadratic potential in a selected scalar subspace."""
    ids=np.asarray(ids,dtype=int);sel=np.r_[np.arange(nf),nf+ids];m=len(sel);nz=len(ids);T=np.eye(m);local_gram_min=None
    if nz:
        gram=Wgram[np.ix_(ids,ids)];local_gram_min=float(la.eigvalsh(gram,subset_by_index=[0,0])[0]);R=la.cholesky(gram,lower=False);T[nf:,nf:]=la.solve_triangular(R,np.eye(nz))
    h=H.reshape(3,3,3,3);K=np.zeros((3*m,3*m));g=np.zeros((3,m));constant=patch_lift_energy
    for a in range(3):
        for b in range(3):
            block=sum(h[a,i,b,j]*(T.T@G[i,j][np.ix_(sel,sel)]@T) for i in range(3) for j in range(3) if h[a,i,b,j])
            if np.isscalar(block):block=np.zeros((m,m))
            if a==b:block[:nf,:nf]+=patch_matrix
            K[a*m:(a+1)*m,b*m:(b+1)*m]=block
        g[a]=T.T@sum(h[a,i,0,j]*G[i,j][sel,-1] for i in range(3) for j in range(3));g[a,:nf]+=patch_lift[:,a]
    constant+=.5*sum(h[0,i,0,j]*G[i,j][-1,-1] for i in range(3) for j in range(3));K=(K+K.T)/2;v=la.solve(K,-g.ravel(),assume_a='pos');coef=(T@v.reshape(3,m).T);U=float(constant+g.ravel()@v+.5*v@K@v);R=float((2*constant+g.ravel()@v)/.005);eig=la.eigvalsh(K);tol=max(abs(eig))*1e-9;res=float(la.norm(K@v+g.ravel()));out=dict(scalar_local_dofs=nz,free_vector_dofs=3*m,reaction_N=R,energy_J=U,work_identity_relative=abs(U-.5*.005*R)/U,free_residual=res,static_min=float(eig[0]),static_tolerance=float(tol),negative_modes=int(np.sum(eig<-tol)),zero_modes=int(np.sum(abs(eig)<=tol)),static_passed=bool(eig[0]>tol),local_scalar_gram_min=local_gram_min,mass_included=False,stiffness_shift=0.)
    if nz:
        ci=np.concatenate([np.arange(a*m,a*m+nf) for a in range(3)]);ai=np.setdiff1d(np.arange(3*m),ci);Kaa=K[np.ix_(ai,ai)];Kya=K[np.ix_(ci,ai)];S=K[np.ix_(ci,ci)]-Kya@la.solve(Kaa,Kya.T,assume_a='pos');out.update(local_stiffness_min=float(la.eigvalsh(Kaa,subset_by_index=[0,0])[0]),schur_min=float(la.eigvalsh(S,subset_by_index=[0,0])[0]))
    return coef,T,out,K


class LocalPotential:
    """Same material+patch potential, differentiated on native FE quadrature."""
    def __init__(self,edges,A,Z,Ks,params,fiber_tensor):
        self.edges=edges;self.T=np.column_stack((A,Z));self.n=A.shape[1];self.Ks=Ks;self.params=params;self.fiber_tensor=fiber_tensor
    def evaluate(self,Y,direction=None,order=3):
        u=self.T@Y;du=None if direction is None else self.T@direction;force=np.zeros_like(u);action=np.zeros_like(u);energy=0.
        for X,V in ref.chunks(self.edges,order=order,size=32):
            _,*B=shape_matrices(X,self.edges);F=np.stack([b@u for b in B],axis=-1);psi,P=material_response(F,np.broadcast_to(self.fiber_tensor,F.shape),self.params);energy+=float(V@psi)
            for j,b in enumerate(B):force+=b.T@(V[:,None]*P[:,:,j])
            if direction is not None:
                dF=np.stack([b@du for b in B],axis=-1);dP=material_tangent(F,np.broadcast_to(self.fiber_tensor,F.shape),dF,self.params)
                for j,b in enumerate(B):action+=b.T@(V[:,None]*dP[:,:,j])
        out=dict(U=energy+.5*float(np.sum(Y[:self.n]*(self.Ks@Y[:self.n]))),force=self.T.T@force);out['force'][:self.n]+=self.Ks@Y[:self.n]
        if direction is not None:out['tangent_action']=self.T.T@action;out['tangent_action'][:self.n]+=self.Ks@direction[:self.n]
        return out
