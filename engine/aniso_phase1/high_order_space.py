"""Higher-order, component-closed local spaces in the original potential.

Cartesian local corrections impose zero trace only on artificial patch faces
and hard grips. Physical side surfaces retain their natural traction boundary.
No mass or stiffness regularization is used. All degrees share one scalar basis
for the three vector components, preserving finite rigid-rotation closure.
"""
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator,cg
from benchmarks.aniso_local_q3 import axis
from .tensor_reference import TensorElastic,apply_axis,coordinates
from .tensor_metrics import sampling,quadrature_axis,evaluate_gradient
from .history_increment import material_response,material_tangent


class BoxElastic(TensorElastic):
    def __init__(self,edges,p,H,dirichlet=((True,True),(False,False),(False,False))):
        self.edges=[np.asarray(e) for e in edges];self.p=p;self.H=np.asarray(H).reshape(3,3,3,3)
        self.axes=[axis(e,p) for e in edges];self.shape=tuple(len(a[0]) for a in self.axes);self.n=int(np.prod(self.shape));self.slices=tuple(slice(int(lo),-1 if hi else None) for lo,hi in dirichlet);self.free_shape=tuple(len(np.arange(n)[sl]) for n,sl in zip(self.shape,self.slices));eig=[];self.vectors=[]
        for (_,grams),sl in zip(self.axes,self.slices):
            M,K=grams[:2];w,V=la.eigh(K[sl,sl].toarray(),M[sl,sl].toarray());eig.append(w);self.vectors.append(V)
        weights=np.array([[self.H[a,k,a,k] for k in range(3)] for a in range(3)])
        self.den=np.stack([sum(weights[a,k]*eig[k].reshape(tuple(len(eig[k]) if j==k else 1 for j in range(3))) for k in range(3)) for a in range(3)],axis=-1)
        if self.den.min()<=0:raise ValueError('Patch requires sufficient Dirichlet support; no artificial shift is applied.')
    def free_apply(self,v):
        u=np.zeros((3,*self.shape));u[(slice(None),*self.slices)]=v.reshape(3,*self.free_shape)
        return self.apply(u).reshape(3,*self.shape)[(slice(None),*self.slices)].ravel()
    def correction(self,rhs,rtol=2e-10):
        n=rhs.size;A=LinearOperator((n,n),matvec=self.free_apply,dtype=float);M=LinearOperator((n,n),matvec=self.precondition,dtype=float);its=0
        def cb(_):
            nonlocal its
            its+=1
        if np.linalg.norm(rhs)==0:return np.zeros_like(rhs),dict(iterations=0,relative_residual=0.)
        w,info=cg(A,rhs,M=M,rtol=rtol,atol=0.,maxiter=1500,callback=cb);res=float(la.norm(A@w-rhs)/la.norm(rhs))
        if info or res>max(1e-8,5*rtol):raise RuntimeError(('local correction did not converge',info,res))
        return w,dict(iterations=its,relative_residual=res)


def local_box(edges,p,patch,H):
    starts=[];stops=[];local=[];faces=[];global_shape=tuple(p*(len(e)-1)+1 for e in edges)
    for k,e in enumerate(edges):
        e=np.asarray(e);i=max(0,np.searchsorted(e,patch['lo'][k]+1e-12,side='right')-1);j=min(len(e)-1,np.searchsorted(e,patch['hi'][k]-1e-12,side='left'))
        if k==0:i=max(i,np.searchsorted(e,.25));j=min(j,np.searchsorted(e,.75))
        lo=k==0 or i>0;hi=k==0 or j<len(e)-1;local.append(e[i:j+1]);faces.append((lo,hi));starts.append(p*i+int(lo));stops.append(p*j+1-int(hi))
    ids=np.ravel_multi_index(np.array(np.meshgrid(*(np.arange(i,j) for i,j in zip(starts,stops)),indexing='ij')).reshape(3,-1),global_shape)
    op=BoxElastic(local,p,H,faces);assert len(ids)==np.prod(op.free_shape)
    return ids,op,dict(edges=[e.tolist() for e in local],dirichlet_faces=faces)


def scalar_grams(edges,p,T):
    axes=[axis(e,p) for e in edges];shape=tuple(len(a[0]) for a in axes);field=T.reshape(*shape,T.shape[1]);G={}
    for i in range(3):
        for j in range(i,3):
            w=field
            for k,(_,g) in enumerate(axes):
                A=g[1] if i==j==k else (g[2] if k==i and i!=j else (g[2].T if k==j and i!=j else g[0]))
                w=apply_axis(A,w,k)
            G[i,j]=T.T@w.reshape(T.shape);G[j,i]=G[i,j].T if i!=j else G[i,j]
    return G


def transpose_gradient(dual,edges,p,points,weights):
    nshape=tuple(p*(len(e)-1)+1 for e in edges);force=np.zeros((*nshape,3));B=[(sampling(e,p,x),sampling(e,p,x,True)) for e,x in zip(edges,points)]
    V=weights[0][:,None,None]*weights[1][None,:,None]*weights[2][None,None,:]
    for j in range(3):
        w=dual[...,j]*V[...,None]
        for k in (2,1,0):w=apply_axis(B[k][k==j].T,w,k)
        force+=w
    return force.reshape(-1,3)


def reference_stress_load(edges,p,reference,H):
    er,pr,_=reference;common=[np.union1d(e,r) for e,r in zip(edges,er)];order=max(p,pr)+1;qs=[quadrature_axis(e,order) for e in common];force=np.zeros((np.prod([p*(len(e)-1)+1 for e in edges]),3));norm=0.
    for start in range(0,len(qs[0][0]),2*order):
        sl=slice(start,start+2*order);points=[qs[0][0][sl],qs[1][0],qs[2][0]];weights=[qs[0][1][sl],qs[1][1],qs[2][1]];L=evaluate_gradient(reference,points);P=(L.reshape(-1,9)@H.T).reshape(L.shape);dual=(P.reshape(-1,9)@H).reshape(L.shape);V=weights[0][:,None,None]*weights[1][None,:,None]*weights[2][None,None,:];norm+=float(np.sum(V*np.sum(P*P,axis=(-1,-2))));force+=transpose_gradient(dual,edges,p,points,weights)
    return force.T.ravel(),norm


class HighOrderPotential:
    def __init__(self,edges,p,A,Z,Ks,params,fiber_tensor):
        self.edges=edges;self.p=p;self.T=np.column_stack((A,Z));self.n=A.shape[1];self.Ks=Ks;self.params=params;self.fiber_tensor=fiber_tensor
    def evaluate(self,Y,direction=None,order=None):
        order=order or self.p+1;u=self.T@Y;du=None if direction is None else self.T@direction;force=np.zeros_like(u);action=np.zeros_like(u);energy=0.;qs=[quadrature_axis(e,order) for e in self.edges]
        for start in range(0,len(qs[0][0]),2*order):
            sl=slice(start,start+2*order);points=[qs[0][0][sl],qs[1][0],qs[2][0]];weights=[qs[0][1][sl],qs[1][1],qs[2][1]];F=evaluate_gradient((self.edges,self.p,u),points);shape=F.shape;flat=F.reshape(-1,3,3);psi,P=material_response(flat,np.broadcast_to(self.fiber_tensor,flat.shape),self.params);V=weights[0][:,None,None]*weights[1][None,:,None]*weights[2][None,None,:];energy+=float(V.ravel()@psi);force+=transpose_gradient(P.reshape(shape),self.edges,self.p,points,weights)
            if direction is not None:
                dF=evaluate_gradient((self.edges,self.p,du),points).reshape(-1,3,3);dP=material_tangent(flat,np.broadcast_to(self.fiber_tensor,flat.shape),dF,self.params);action+=transpose_gradient(dP.reshape(shape),self.edges,self.p,points,weights)
        result=dict(U=energy+.5*float(np.sum(Y[:self.n]*(self.Ks@Y[:self.n]))),force=self.T.T@force);result['force'][:self.n]+=self.Ks@Y[:self.n]
        if direction is not None:result['tangent_action']=self.T.T@action;result['tangent_action'][:self.n]+=self.Ks@direction[:self.n]
        return result
