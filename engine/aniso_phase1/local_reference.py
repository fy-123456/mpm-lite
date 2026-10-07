"""Nonuniform conforming tensor Q1/Q2 reference for original hard grips.

Only physical material is meshed. Exact Gauss integration; no MPM maps, mass,
or stabilization. Local refinement means splitting selected physical intervals;
tensor closure extends a marked interval through the corresponding slab.
"""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
from .boundary_reference import LO, HI, tensor_matrix


def basis(q,degree):
    q=np.asarray(q)
    if degree==1:return np.stack((1-q,q),axis=-1),np.stack((-np.ones_like(q),np.ones_like(q)),axis=-1)
    if degree==2:return np.stack((2*q*q-3*q+1,4*q-4*q*q,2*q*q-q),axis=-1),np.stack((4*q-3,4-8*q,4*q-1),axis=-1)
    raise ValueError('Q1 or Q2 required')


def axis(edges,degree,order=None):
    edges=np.asarray(edges);h=np.diff(edges);p=degree;n=len(h)
    z,w=np.polynomial.legendre.leggauss(order or p+1);q,w=(z+1)/2,w/2;N,D=basis(q,p)
    blocks=[h[:,None,None]*(N.T@(w[:,None]*N)),(D.T@(w[:,None]*D))/h[:,None,None],np.broadcast_to(D.T@(w[:,None]*N),(n,p+1,p+1))]
    ids=p*np.arange(n)[:,None]+np.arange(p+1);rows=np.repeat(ids,p+1,axis=1).ravel();cols=np.tile(ids,(1,p+1)).ravel()
    coords=np.r_[np.concatenate([edges[:-1]+k/p*h for k in range(p)]).reshape(p,n).T.ravel(),edges[-1]]
    return coords,[sp.csr_matrix((b.ravel(),(rows,cols)),shape=(n*p+1,)*2) for b in blocks]


def assemble(edges,degree,H):
    axes=[axis(e,degree) for e in edges];coords=[a[0] for a in axes];grams={}
    for i in range(3):
        for j in range(3):
            factors=[a[1][0] for a in axes]
            if i==j:factors[i]=axes[i][1][1]
            else:factors[i],factors[j]=axes[i][1][2],axes[j][1][2].T
            grams[i,j]=tensor_matrix(factors)
    blocks=[]
    for a in range(3):
        row=[]
        for b in range(3):
            terms=[H[3*a+i,3*b+j]*grams[i,j] for i in range(3) for j in range(3) if H[3*a+i,3*b+j]!=0]
            row.append(sum(terms) if terms else sp.csr_matrix(grams[0,0].shape))
        blocks.append(row)
    return np.array(list(itertools.product(*coords))),sp.bmat(blocks,format='csr')


def solve(edges,degree,H):
    nodes,K=assemble(edges,degree,H);n=len(nodes)
    fixed=(nodes[:,0]<=.25+1e-12)|(nodes[:,0]>=.75-1e-12);right=nodes[:,0]>=.75-1e-12
    free=np.flatnonzero(np.tile(~fixed,3));u=np.zeros(3*n);u[:n][right]=.005
    rhs=-(K@u)[free];A=K[free][:,free].tocsr();inv=1/A.diagonal();count=0
    def callback(_):
        nonlocal count
        count+=1
    u[free],info=cg(A,rhs,M=LinearOperator(A.shape,matvec=lambda v:inv*v),rtol=2e-11,atol=1e-14,maxiter=30000,callback=callback)
    force=K@u;residual=float(np.linalg.norm(force[free])/np.linalg.norm(rhs));energy=float(.5*u@force);R=float(force[:n][right].sum())
    return nodes,u.reshape(3,n).T,dict(degree=degree,cells=[len(e)-1 for e in edges],nodes=n,free_dofs=len(free),iterations=count,relative_residual=residual,linear_info=int(info),reaction_N=R,energy_J=energy,work_identity_relative=abs(energy-.5*.005*R)/energy,passed=bool(info==0 and residual<1e-8),mass_included=False,stiffness_shift=0.)


def gradient(X,edges,degree,u):
    p=degree;triples=np.array(list(itertools.product(range(p+1),repeat=3)));cell=[];q=[];hs=[]
    for k,e in enumerate(edges):
        e=np.asarray(e);idx=np.clip(np.searchsorted(e,X[:,k],side='right')-1,0,len(e)-2)
        h=e[idx+1]-e[idx];cell.append(idx);q.append((X[:,k]-e[idx])/h);hs.append(h)
    cell=np.array(cell).T;q=np.array(q).T;hs=np.array(hs).T
    ids=np.ravel_multi_index((p*cell[:,None,:]+triples).reshape(-1,3).T,tuple(p*(len(e)-1)+1 for e in edges)).reshape(-1,len(triples))
    N,D=basis(q,p);v=u[ids];out=np.empty((len(X),3,3))
    for k in range(3):
        w=np.ones((len(X),len(triples)))/hs[:,k,None]
        for j in range(3):w*=(D if j==k else N)[:,j,triples[:,j]]
        out[:,:,k]=np.einsum('pna,pn->pa',v,w)
    return out


def chunks(edges,order=3,size=512):
    q,w=np.polynomial.legendre.leggauss(order);q,w=(q+1)/2,w/2
    q=np.array(list(itertools.product(q,repeat=3)));w=np.prod(np.array(list(itertools.product(w,repeat=3))),axis=1)
    cells=np.array(list(itertools.product(*[range(len(e)-1) for e in edges])))
    for start in range(0,len(cells),size):
        c=cells[start:start+size];lo=np.column_stack([edges[k][c[:,k]] for k in range(3)]);hi=np.column_stack([edges[k][c[:,k]+1] for k in range(3)]);h=hi-lo
        yield (lo[:,None,:]+h[:,None,:]*q).reshape(-1,3),(h.prod(axis=1)[:,None]*w).ravel()


def compare(a,b,H):
    ea,pa,ua,ra=a;eb,pb,ub,rb=b;common=[np.union1d(x,y) for x,y in zip(ea,eb)]
    totals={k:np.zeros(4) for k in ('whole','near_grip','interior','deep_interior')};peaks=np.zeros(2)
    for X,V in chunks(common):
        La,Lb=gradient(X,ea,pa,ua),gradient(X,eb,pb,ub);Pa,Pb=(La.reshape(-1,9)@H.T).reshape(-1,3,3),(Lb.reshape(-1,9)@H.T).reshape(-1,3,3)
        norm=lambda x:np.sum(x*x,axis=(1,2))
        values=np.column_stack((norm(La-Lb),norm(Lb),norm(Pa-Pb),norm(Pb)))*V[:,None]
        dist=np.minimum(abs(X[:,0]-.25),abs(X[:,0]-.75));masks=dict(whole=np.ones(len(X),bool),near_grip=dist<.0625,interior=(X[:,0]>.3125)&(X[:,0]<.6875),deep_interior=(X[:,0]>.375)&(X[:,0]<.625))
        for k,m in masks.items():totals[k]+=values[m].sum(axis=0)
        peaks=np.maximum(peaks,[np.sqrt(norm(Pa)).max(),np.sqrt(norm(Pb)).max()])
    regions={k:dict(F_relative=float(np.sqrt(v[0]/v[1])),P_relative=float(np.sqrt(v[2]/v[3]))) for k,v in totals.items()}
    R=abs(ra['reaction_N']-rb['reaction_N'])/abs(rb['reaction_N'])
    return dict(reaction_relative=R,regions=regions,near_grip_share=float(totals['near_grip'][2]/totals['whole'][2]) if totals['whole'][2]>0 else 0.,sampled_peak_P=peaks.tolist(),reaction_passed=R<=.01,interior_passed=max(regions['interior'].values())<=.02,boundary_passed=max(regions['near_grip'].values())<=.02,global_passed=max(regions['whole'].values())<=.02)
