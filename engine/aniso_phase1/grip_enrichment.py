"""Variational local Q2 enrichment near physical grip transitions (v20).

The scalar continuous FE field is u=A y+Z a. Original carrier patch energy
acts only on y. Local amplitudes a minimize the SAME physical material energy;
no penalty reduction or artificial mass is added. This is a spatial prototype.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from . import local_reference as ref
from .compatible_carrier import shape_matrices,CompatibleReconstruction
from .carrier_joint import gradient
from .history_increment import material_response,material_tangent


def hat_matrix(x,edges):
    edges=np.asarray(edges);p=np.clip(np.searchsorted(edges,x,side='right')-1,0,len(edges)-2);q=(x-edges[p])/(edges[p+1]-edges[p]);N,_=ref.basis(q,2)
    out=np.zeros((len(x),2*len(edges)-1));inside=(x>=edges[0]-1e-12)&(x<=edges[-1]+1e-12);rows=np.flatnonzero(inside)
    out[rows[:,None],2*p[rows,None]+np.arange(3)]=N[rows];return out


def local_basis(nodes,level):
    if level==0:return np.empty((len(nodes),0))
    # Six x hats vanish at both ends of the two transition slabs.
    xs=np.column_stack([hat_matrix(nodes[:,0],e)[:,1:-1] for e in ([.25,.3125,.375],[.625,.6875,.75])])
    if level==1:ys=zs=np.ones((len(nodes),1))
    elif level in (2,3):
        n=2 if level==2 else 3;ys=hat_matrix(nodes[:,1],np.linspace(.375,.625,n));zs=hat_matrix(nodes[:,2],np.linspace(.375,.625,n))
    else:raise ValueError('levels 0,1,2,3')
    return np.einsum('pi,pj,pk->pijk',xs,ys,zs).reshape(len(nodes),-1)


def static_solve(rec,Ks,H,level):
    nodes,Kfe=ref.assemble(rec.edges,2,H);n=len(rec.nodes);Z=local_basis(nodes,level);nz=Z.shape[1];free=(rec.nodes[:,0]>.25+1e-12)&(rec.nodes[:,0]<.75-1e-12);Q=np.eye(n)[:,free];nf=Q.shape[1]
    B=np.column_stack((rec.A@Q,Z));BB=sp.block_diag([sp.csr_matrix(B)]*3,format='csr');lift=np.zeros((n,3));lift[rec.nodes[:,0]>=.75-1e-12,0]=.005;u0=rec.A@lift
    K=(BB.T@(Kfe@BB)).toarray();patch=Q.T@Ks@Q
    for a in range(3):K[a*(nf+nz):a*(nf+nz)+nf,a*(nf+nz):a*(nf+nz)+nf]+=patch
    rhs=-(BB.T@(Kfe@u0.T.ravel()));patchfixed=Q.T@Ks@lift
    for a in range(3):rhs[a*(nf+nz):a*(nf+nz)+nf]-=patchfixed[:,a]
    K=(K+K.T)/2;v=la.solve(K,rhs,assume_a='pos');coef=v.reshape(3,-1).T;y=lift+Q@coef[:nf];aa=coef[nf:];u=rec.A@y+Z@aa;force=(Kfe@u.T.ravel()).reshape(3,-1).T;fc=rec.A.T@force+Ks@y;R=float(fc[rec.nodes[:,0]>=.75-1e-12,0].sum());Us=.5*float(np.sum(y*(Ks@y)));Um=.5*float(np.sum(u*force))
    eig=la.eigvalsh(K);tol=1e-9*max(abs(eig));free_res=float(la.norm(K@v-rhs));out=dict(level=level,scalar_local_dofs=nz,free_vector_dofs=len(v),reaction_N=R,energy_J=Um+Us,material_J=Um,stabilization_J=Us,free_residual=free_res,work_identity_relative=abs(Um+Us-.5*.005*R)/(Um+Us),min_eigenvalue=float(eig[0]),zero_modes=int(np.sum(abs(eig)<=tol)),negative_modes=int(np.sum(eig<-tol)),static_passed=bool(eig[0]>tol),mass_included=False,stiffness_shift=0.)
    if nz:
        ci=np.concatenate([np.arange(a*(nf+nz),a*(nf+nz)+nf) for a in range(3)]);ai=np.setdiff1d(np.arange(len(K)),ci);Kaa=K[np.ix_(ai,ai)];Kya=K[np.ix_(ci,ai)];Keff=K[np.ix_(ci,ci)]-Kya@la.solve(Kaa,Kya.T,assume_a='pos');out.update(local_min_eigenvalue=float(la.eigvalsh(Kaa)[0]),schur_min_eigenvalue=float(la.eigvalsh(Keff)[0]))
    return u,y,aa,out


class EnrichedPotential:
    """Nonlinear energy/force and exact tangent action, same scalar field space."""
    def __init__(self,rec,Ks,level,params,A):
        self.rec=rec;self.Ks=Ks;self.Z=local_basis(rec.fe_nodes,level);self.T=np.column_stack((rec.A,self.Z));self.n=len(rec.nodes);self.params=params;self.A=A
    def evaluate(self,Y,direction=None):
        out=dict(U=0.,force=np.zeros_like(Y));action=np.zeros_like(Y)
        for X,V in ref.chunks(self.rec.edges,order=3,size=16):
            B=[M@self.T for M in shape_matrices(X,self.rec.edges)[1:]];F=gradient(B,Y);psi,P=material_response(F,np.broadcast_to(self.A,F.shape),self.params);out['U']+=float(V@psi)
            out['force']+=sum(b.T@(V[:,None]*P[:,:,j]) for j,b in enumerate(B))
            if direction is not None:
                dF=gradient(B,direction);dP=material_tangent(F,np.broadcast_to(self.A,F.shape),dF,self.params);action+=sum(b.T@(V[:,None]*dP[:,:,j]) for j,b in enumerate(B))
        out['U']+=.5*float(np.sum(Y[:self.n]*(self.Ks@Y[:self.n])));out['force'][:self.n]+=self.Ks@Y[:self.n]
        if direction is not None:action[:self.n]+=self.Ks@direction[:self.n];out['tangent_action']=action
        return out
