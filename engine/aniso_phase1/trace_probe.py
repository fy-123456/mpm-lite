"""Read-only static boundary probes; production boundary/transfer rules unchanged.

The center-local polynomial gradients do not define a unique global displacement.
A trilinear partition of unity supplies an explicit trace for controlled tests.
A second candidate differentiates that global trace INCLUDING partition terms.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from scipy.linalg import svd,cho_factor,cho_solve,eigvalsh
from .quadratic import polynomial,weighted_inverse,node_patch
from .beam_reference import reference_hessian
from .material_snapshot import center_support
from .joint_sampling import build_rule

CORNERS=np.array(list(itertools.product((0,1),repeat=3)))


class BlendedMLS:
    """N(x)=sum_c w_c(x) N_c(x); each N_c uses a fixed center-local fit.

    Fits may extend to unoccupied centers to retain the full trilinear partition
    near surfaces. No additional material or quadrature weights are introduced.
    """
    def __init__(self,nodes,h):
        self.nodes=np.asarray(nodes);self.h=h;self.tree=cKDTree(nodes);self.cache={}

    def patch(self,key):
        key=tuple(key)
        if key not in self.cache:
            center=(np.array(key)+.5)*self.h
            for radius in (2.1,2.6,3.5):
                ids=np.array(sorted(self.tree.query_ball_point(center,radius*self.h)),dtype=int)
                z=(self.nodes[ids]-center)/self.h;P,_=polynomial(z,True)
                try:inverse,_=weighted_inverse(P,np.exp(-np.sum(z*z,axis=1)/2))
                except ValueError:continue
                self.cache[key]=(ids,inverse);break
            else:raise ValueError('no full-rank MLS boundary extension')
        return self.cache[key]

    def evaluate(self,points):
        rows=[];cols=[];val=[];gv=[[],[],[]];h=self.h
        for q,x in enumerate(np.asarray(points)):
            z=x/h-.5;base=np.floor(z).astype(int);f=z-base
            for corner in CORNERS:
                terms=np.where(corner,f,1-f);w=np.prod(terms)
                dw=np.array([(1 if corner[d] else -1)*np.prod(np.delete(terms,d))/h for d in range(3)])
                if w==0 and np.all(dw==0):continue
                key=base+corner;ids,inverse=self.patch(key);center=(key+.5)*h
                P,D=polynomial(((x-center)/h)[None],True)
                N=(P@inverse)[0];G=np.einsum('dk,kn->nd',D[0],inverse)/h
                rows.extend([q]*len(ids));cols.extend(ids);val.extend(w*N)
                for d in range(3):gv[d].extend(w*G[:,d]+dw[d]*N)
        shape=(len(points),len(self.nodes))
        N=sparse.coo_matrix((val,(rows,cols)),shape=shape).tocsr()
        G=[sparse.coo_matrix((v,(rows,cols)),shape=shape).tocsr() for v in gv]
        return N,G

    def local_traces(self,points):
        """Each contributing patch trace separately, before blending."""
        rows=[];cols=[];values=[];row=0
        for x in points:
            z=x/self.h-.5;base=np.floor(z).astype(int);f=z-base
            for corner in CORNERS:
                if np.prod(np.where(corner,f,1-f))<=0:continue
                key=base+corner;ids,inverse=self.patch(key)
                P,_=polynomial(((x-(key+.5)*self.h)/self.h)[None],True)
                rows.extend([row]*len(ids));cols.extend(ids);values.extend((P@inverse)[0]);row+=1
        return sparse.coo_matrix((values,(rows,cols)),shape=(row,len(self.nodes))).tocsr()


def lite_values(nodes,points,h):
    lookup={tuple(np.rint(x/h).astype(int)):i for i,x in enumerate(nodes)};rows=[];cols=[];vals=[]
    for q,x in enumerate(points):
        z=x/h-.5;base=np.floor(z).astype(int);f=z-base
        for c in CORNERS:
            w=np.prod(np.where(c,f,1-f))/8
            if w==0:continue
            for nc in CORNERS:
                key=tuple(base+c+nc)
                if key not in lookup:raise ValueError('Lite trace support clipped')
                rows.append(q);cols.append(lookup[key]);vals.append(w)
    return sparse.coo_matrix((vals,(rows,cols)),shape=(len(points),len(nodes))).tocsr()


def face_points(grid,x,subdivisions=2,edges=False):
    """Uniform face quadrature, or independent denser validation including edges."""
    n=round(.125*(grid-1))*subdivisions
    axis=np.linspace(.4375,.5625,n+1) if edges else .4375+(np.arange(n)+.5)*.125/n
    return np.array([(x,y,z) for y in axis for z in axis])


def grouped_maps(snapshot,nodes,h):
    """Freeze the identical group4x8 samples/weights for both gradient candidates."""
    tree=cKDTree(nodes);rows=[];cols=[];gv=[[],[],[]];positions=[];weights=[];moments=[];count=0
    for key,ids,w in center_support(snapshot,h):
        center=(key+.5)*h;x,V,A,M=build_rule(snapshot,ids,w,h,'group4x8')
        patch,g,_,_=node_patch(nodes,center,h,tree,queries=x)
        rows.extend(np.repeat(np.arange(count,count+len(V)),len(patch)));cols.extend(np.tile(patch,len(V)))
        for d in range(3):gv[d].extend(g[:,:,d].ravel())
        positions.extend(x);weights.extend(V);moments.extend(M);count+=len(V)
    shape=(count,len(nodes));G=[sparse.coo_matrix((v,(rows,cols)),shape=shape).tocsr() for v in gv]
    return np.array(positions),np.array(weights),np.array(moments),G


def stiffness(gradients,weights,moments):
    """Same-law exact tangent at F=I, assembled without mass/stabilization."""
    n=gradients[0].shape[1];H=reference_hessian(kf=0)[None]+800*moments
    H=H.reshape(-1,3,3,3,3);blocks=[]
    for a in range(3):
        row=[]
        for b in range(3):
            K=sparse.csr_matrix((n,n))
            for i in range(3):
                for j in range(3):
                    w=weights*H[:,a,i,b,j]
                    if np.any(w):K=K+gradients[i].T@gradients[j].multiply(w[:,None])
            row.append(K)
        blocks.append(row)
    K=sparse.bmat(blocks,format='csr');order=np.arange(3*n).reshape(3,n).T.ravel()
    return K[order][:,order]


class ConstrainedStatic:
    """Exact homogeneous C u=0 via an SVD nullspace; no penalty stiffness.

    Scalar constraints apply equally to each component. Redundant rows are
    removed by rank revelation, with the residual checked in the original rows.
    """
    def __init__(self,K,C):
        self.K=K;self.C=np.asarray(C.toarray() if sparse.issparse(C) else C)
        _,s,Vh=svd(self.C,full_matrices=self.C.shape[0]<self.C.shape[1])
        self.rank=int(np.sum(s>1e-10*s[0]));self.singular=s
        self.Z=np.kron(Vh[self.rank:].T,np.eye(3));self.A=self.Z.T@(K@self.Z)
        self.symmetry=float(np.linalg.norm(self.A-self.A.T)/np.linalg.norm(self.A))
        self.A=(self.A+self.A.T)/2
        self.eigen=eigvalsh(self.A);self.soft=int(np.sum(self.eigen<=1e-9*max(1,self.eigen[-1])))
        if self.soft:raise ValueError(f'constrained static operator has {self.soft} soft modes')
        self.factor=cho_factor(self.A)

    def solve(self,f):
        u=self.Z@cho_solve(self.factor,self.Z.T@f);reaction=self.K@u-f
        return u,reaction
