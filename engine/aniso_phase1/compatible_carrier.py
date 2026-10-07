"""v19 experimental scalar H1 reconstruction on a physical conforming Q2 mesh.

Every spatial component uses the SAME scalar operator, preserving objectivity.
Position and material gradient are value/derivative of one continuous field.
Quadratic one-sided grip traces preserve affine/rigid/quadratic fields. No
stiffness/mass shift and no coefficient tuning are used. Dense <=512 carriers.
"""
import itertools
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from . import local_reference as ref
from .boundary_reference import tensor_matrix
from .carrier_joint import gradient,CarrierEnergy,State
from .carrier_driven import CORNERS


def shape_matrices(X,edges):
    triples=np.array(list(itertools.product(range(3),repeat=3)));cell=[];q=[];hs=[]
    for k,e in enumerate(edges):
        i=np.clip(np.searchsorted(e,X[:,k],side='right')-1,0,len(e)-2);h=e[i+1]-e[i];cell.append(i);q.append((X[:,k]-e[i])/h);hs.append(h)
    cell=np.array(cell).T;q=np.array(q).T;hs=np.array(hs).T;shape=tuple(2*(len(e)-1)+1 for e in edges)
    ids=np.ravel_multi_index((2*cell[:,None,:]+triples).reshape(-1,3).T,shape).reshape(len(X),27);N,D=ref.basis(q,2)
    vals=np.prod(N[:,np.arange(3)[None,:],triples],axis=-1);der=[]
    for k in range(3):
        v=np.ones_like(vals)/hs[:,k,None]
        for j in range(3):v*=(D if j==k else N)[:,j,triples[:,j]]
        der.append(v)
    rows=np.repeat(np.arange(len(X)),27);cols=ids.ravel();n=int(np.prod(shape))
    return [sp.csr_matrix((v.ravel(),(rows,cols)),shape=(len(X),n)) for v in [vals]+der]


def lite_gradient(X,nodes,h):
    ijk=np.rint(nodes/h).astype(int);lo=ijk.min(0);shape=ijk.max(0)-lo+1;q=X/h-.5;base=np.floor(q).astype(int);f=q-base
    G=[np.zeros((len(X),len(nodes))) for _ in range(3)];rows=np.arange(len(X))
    for a in CORNERS:
        w=np.prod(np.where(a,f,1-f),axis=1)
        for b in CORNERS:
            ids=np.ravel_multi_index((base+a+b-lo).T,tuple(shape))
            for k in range(3):G[k][rows,ids]+=w*(2*b[k]-1)/(4*h)
    return G


def quadratic_trace(X,nodes,h):
    """Tensor quadratic Lagrange trace with one-sided physical grip support."""
    grid=np.rint(nodes/h).astype(int);lo=grid.min(0);shape=grid.max(0)-lo+1;loc=X/h
    starts=np.clip(np.floor(loc).astype(int)-1,lo,lo+shape-3)
    starts[X[:,0]<=.25+1e-12,0]=lo[0];starts[X[:,0]>=.75-1e-12,0]=lo[0]+shape[0]-3
    f=loc-starts;axis=np.stack((.5*(f-1)*(f-2),-f*(f-2),.5*f*(f-1)),axis=-1)
    T=np.zeros((len(X),len(nodes)));rows=np.arange(len(X))
    for a in itertools.product(range(3),repeat=3):
        ids=np.ravel_multi_index((starts+np.array(a)-lo).T,tuple(shape));T[rows,ids]+=np.prod(axis[:,np.arange(3),a],axis=1)
    return T


class CompatibleReconstruction:
    def __init__(self,nodes,h,resolution=16):
        self.nodes=np.array(nodes);self.h=h;self.resolution=resolution
        # Mesh must split all Lite gradient kinks and both clamp transitions.
        self.edges=[np.linspace(a,b,round((b-a)*resolution)+1) for a,b in zip(ref.LO,ref.HI)]
        for e,a,b in zip(self.edges,ref.LO,ref.HI):
            required=(np.arange(int(a/h)-2,int(b/h)+3)+.5)*h;required=required[(required>a)&(required<b)]
            if any(np.min(abs(e-v))>1e-12 for v in required):raise ValueError('Q2 mesh must split center gradient kinks')
        axes=[ref.axis(e,2) for e in self.edges];self.fe_nodes=np.array(list(itertools.product(*[a[0] for a in axes])))
        K=sum(tensor_matrix([axes[j][1][1 if j==k else 0] for j in range(3)]) for k in range(3)).tocsr()
        rhs=np.zeros((len(self.fe_nodes),len(nodes)))
        for X,V in ref.chunks(self.edges,order=3,size=16):
            mats=shape_matrices(X,self.edges);G=lite_gradient(X,nodes,h)
            rhs+=sum(mats[k+1].T@(V[:,None]*G[k]) for k in range(3))
        fixed=(self.fe_nodes[:,0]<=.25+1e-12)|(self.fe_nodes[:,0]>=.75-1e-12);free=~fixed
        A=np.zeros_like(rhs);A[fixed]=quadratic_trace(self.fe_nodes[fixed],nodes,h)
        b=rhs[free]-K[free][:,fixed]@A[fixed];Af=splu(K[free][:,free].tocsc()).solve(b);A[free]=Af
        self.A=A;self.info=dict(fe_nodes=len(A),free_scalar=int(free.sum()),relative_residual=float(la.norm(K[free]@A-rhs[free])/la.norm(rhs[free])))
        if self.info['relative_residual']>1e-9:raise ValueError('inexact compatible reconstruction')
    def maps(self,X):return tuple(M@self.A for M in shape_matrices(X,self.edges))


def quadrature(edges,order):
    data=list(ref.chunks(edges,order=order));return np.concatenate([x for x,v in data]),np.concatenate([v for x,v in data])


def make_case(reconstruction=None,order=3,label='F45'):
    """Real weighted material/kinetic samples; original patch potential retained."""
    from benchmarks.aniso_v17_modes import controlled_case
    from benchmarks.aniso_v18_space import gauss_sites
    from benchmarks.aniso_apic_frequency import Oracle
    from .selective_patch import scalar_matrix
    from .types import AnisotropicMaterialParams
    base,old,m,h,meta=controlled_case();nodes=meta['carrier_reference']
    X,V=gauss_sites(h,order) if reconstruction is None else quadrature(reconstruction.edges,order)
    o=Oracle(X,V,h);assert np.allclose(o.nodes*h,nodes)
    _,ids,P=scalar_matrix(nodes,o.c,o.S.T@V,h)
    if reconstruction is None:T=None;B=lite_gradient(X,nodes,h)
    else:T,*B=reconstruction.maps(X)
    a={'ISO':np.array([1.,0,0]),'F0':np.array([1.,0,0]),'F45':np.array([1.,1.,0])/np.sqrt(2),'F90':np.array([0.,1.,0])}[label]
    A=np.broadcast_to(np.outer(a,a),(len(X),3,3)).copy();params=AnisotropicMaterialParams(10.,20.,0. if label=='ISO' else 200.)
    e=CarrierEnergy(B,np.tile(np.eye(3),(len(X),1,1)),V,A,ids,P,10*(o.S.T@V)/(h*h*ids.shape[1]),params)
    e.position_basis=T;e.reference_particles=X.copy();e.reconstruction=reconstruction
    s=State(X.copy(),nodes.copy(),np.zeros((len(X),3)),np.zeros((len(X),3,3)))
    return s,e,V,h,dict(particles=len(X),order=order,label=label,reconstruction=bool(reconstruction))
