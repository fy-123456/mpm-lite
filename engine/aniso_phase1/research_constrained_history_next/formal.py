"""Full formal144 static constrained carrier; no automatic dynamic promotion.

Free carrier = material C1 mask * spatial SH(old Q2 carrier). Fixed carrier
and local functions retain their reference polynomial definition. Material box
halo sampling uses explicit polynomial continuation of the endpoint Q2 element.
This is a candidate space: preserving Ks as a coefficient potential is NOT a
proof that accumulated Eulerian histories have the old absolute coordinates.
"""
from pathlib import Path
import hashlib,json,itertools
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from engine.aniso_phase1.tensor_metrics import sampling,quadrature_axis
from engine.aniso_phase1.tensor_reference import apply_axis
from engine.aniso_phase1.research_spatial_phase_next.performance import load

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'docs/results/cross-direction/20261001T170848Z-cross-direction/S1/candidates/cross-direction-snapshot6/space-package.json'
PACKAGE_SHA='885a2870a9908f5fd5a0904173a5c1244695e4b7e5a80bf953ff445fda644a30'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def mask(x,left=.25,right=.75,width=2/64):
    def smooth(t):
        z=np.clip(t,0,1);return z*z*(3-2*z),np.where((t>0)&(t<1),6*z*(1-z)/width,0.)
    a,da=smooth((x-left)/width);b,db=smooth((right-x)/width)
    return a*b,da*b-a*db

def sh_axis(edges,x,dx=1/64,derivative=False):
    x=np.asarray(x);q=x/dx-.5;base=np.floor(q).astype(int);f=q-base
    nodes=base[:,None]+np.arange(3);unique,cols=np.unique(nodes,return_inverse=True)
    weights=np.tile([-.5/dx,0.,.5/dx],(len(x),1)) if derivative else np.c_[(1-f)/2,np.full(len(x),.5),f/2]
    S=sp.csr_matrix((weights.ravel(),(np.repeat(np.arange(len(x)),3),cols.ravel())),shape=(len(x),len(unique)))
    return (S@sampling(edges,2,unique*dx)).tocsr()

def paired(mats):
    n=mats[0].shape[0];shape=tuple(m.shape[1] for m in mats);rows=[];cols=[];values=[]
    for p in range(n):
        ix=[m.indices[m.indptr[p]:m.indptr[p+1]] for m in mats]
        vv=[m.data[m.indptr[p]:m.indptr[p+1]] for m in mats]
        grids=np.meshgrid(*ix,indexing='ij');ids=np.ravel_multi_index([g.ravel() for g in grids],shape)
        weights=np.einsum('i,j,k->ijk',*vv).ravel()
        rows.extend([p]*len(ids));cols.extend(ids);values.extend(weights)
    return sp.csr_matrix((values,(rows,cols)),shape=(n,int(np.prod(shape))))

class FormalCandidate:
    def __init__(self,width=2/64):
        if not np.isfinite(width) or width<=0 or width>=.25:raise ValueError('invalid C1 mask width')
        if sha(PACKAGE)!=PACKAGE_SHA:raise ValueError('wrong formal space')
        cache=PACKAGE.parent/'qualified-array-cache/cache.json'
        entry=dict(path=str(PACKAGE),sha256=PACKAGE_SHA)
        self.reduction=load(dict(path=str(cache),sha256=sha(cache)),entry)
        self.space=self.reduction.parent;self.width=width;self.dx=1/64
        self.free_carriers=self.space.free_scalar_ids[self.space.free_scalar_ids<self.space.n]
        self.identity=dict(schema='formal144-C1-SH-static-v1',parent_sha256=PACKAGE_SHA,width=width,dx=self.dx,
            fixed_and_local='unchanged material coordinates',halo='Q2 endpoint polynomial continuation',dynamic_adopted=False)

    def carrier_points(self,X,x=None,F=None):
        s=self.space;X=np.asarray(X,float);x=X if x is None else np.asarray(x,float)
        F=np.tile(np.eye(3),(len(X),1,1)) if F is None else np.asarray(F,float)
        if X.ndim!=2 or X.shape[1]!=3 or x.shape!=X.shape or F.shape!=(len(X),3,3):raise ValueError('matched X/x/F required')
        if not all(np.isfinite(a).all() for a in (X,x,F)) or np.any(np.linalg.det(F)<=.1):raise ValueError('inadmissible point history')
        R=[sh_axis(e,x[:,a]) for a,e in enumerate(s.oldedges)]
        N=paired(R)@s.oldA
        Dx=np.stack([paired([sh_axis(s.oldedges[a],x[:,a],derivative=True) if a==j else R[a] for a in range(3)])@s.oldA for j in range(3)],axis=-1)
        D=np.einsum('qni,qij->qnj',Dx,F)
        chi,g=mask(X[:,0],width=self.width);D=chi[:,None,None]*D;D[:,:,0]+=g[:,None]*N
        N*=chi[:,None]
        # Prescribed rows use their original material field, with no moving-mask ambiguity.
        fixed=np.setdiff1d(np.arange(s.n),self.free_carriers)
        Q=[sampling(e,2,X[:,a]) for a,e in enumerate(s.oldedges)]
        original=paired(Q)@s.oldA;N[:,fixed]=original[:,fixed]
        for a in range(3):
            derivative=paired([sampling(s.oldedges[j],2,X[:,j],True) if j==a else Q[j] for j in range(3)])@s.oldA
            D[:,fixed,a]=derivative[:,fixed]
        return N,D

    def local_points(self,X):
        s=self.space;Q=[sampling(e,s.p,X[:,a]) for a,e in enumerate(s.edges)]
        N=(paired(Q)@s.raw)@s.transform
        D=np.stack([(paired([sampling(s.edges[j],s.p,X[:,j],True) if j==a else Q[j] for j in range(3)])@s.raw)@s.transform for a in range(3)],axis=-1)
        return N,D

    def basis(self,X,x=None,F=None):
        a,da=self.carrier_points(X,x,F);b,db=self.local_points(X)
        return np.c_[a,b],np.concatenate([da,db],axis=1)

    def mass(self,order=7,block=4):
        """Reassemble ALL changed rows; exact inherited unchanged local/fixed blocks.

        Tensor contractions avoid an Nquadrature x Nbasis global matrix.
        Common cell partition includes old/local knots, SH center knots and mask
        transitions. q7 integrates products of the resulting initial polynomials.
        """
        s=self.space;R=[];O=[];Q=[];W=[]
        for a,(edges,old) in enumerate(zip(s.edges,s.oldedges)):
            knots=(np.arange(int(np.floor(edges[0]/self.dx))-1,int(np.ceil(edges[-1]/self.dx))+1)+.5)*self.dx
            cuts=np.unique(np.r_[edges,old,knots[(knots>edges[0])&(knots<edges[-1])]])
            if a==0:cuts=np.unique(np.r_[cuts,.25,.25+self.width,.75-self.width,.75])
            points,w=quadrature_axis(cuts,order);r=sh_axis(old,points)
            if a==0:r=sp.diags(mask(points,width=self.width)[0])@r
            R.append(r);O.append(sampling(old,2,points));Q.append(sampling(edges,s.p,points));W.append(sp.diags(w))
        def gram(A,B):return [(aa.T@w@bb).tocsr() for aa,w,bb in zip(A,W,B)]
        def act(matrices,coeff,shape):
            z=coeff.reshape(*shape,-1)
            for a,m in enumerate(matrices):z=apply_axis(m,z,a)
            return z.reshape(-1,coeff.shape[-1])
        ids=self.free_carriers;fixed=np.setdiff1d(np.arange(s.n),ids)
        M=self.reduction.original_mass.copy()
        rr=s.oldA.T@act(gram(R,R),s.oldA,s.oldshape)
        ro=s.oldA.T@act(gram(R,O),s.oldA,s.oldshape)
        M[np.ix_(ids,ids)]=rr[np.ix_(ids,ids)]
        M[np.ix_(ids,fixed)]=ro[np.ix_(ids,fixed)];M[np.ix_(fixed,ids)]=ro[np.ix_(ids,fixed)].T
        matrices=gram(R,Q)
        for j in range(0,s.ndof-s.n,block):
            stop=min(j+block,s.ndof-s.n);nodal=s.raw@s.transform[:,j:stop]
            reduced=act(matrices,nodal,s.shape);cross=s.oldA[:,ids].T@reduced
            M[np.ix_(ids,np.arange(s.n+j,s.n+stop))]=cross
            M[np.ix_(np.arange(s.n+j,s.n+stop),ids)]=cross.T
        return (M+M.T)/2

    def stationary_coordinates(self,M):
        """Fresh stationary elimination, qualified only for static coefficients."""
        s=self.space;ids=self.free_carriers
        _,sv,vt=la.svd(s.oldA[:,ids],full_matrices=False)
        null=vt[sv<=max(s.oldA.shape)*np.finfo(float).eps*sv[0]].T
        N=np.zeros((s.ndof,null.shape[1]));N[ids]=null
        R=la.null_space(N[s.free_scalar_ids].T);nf=R.shape[1]
        B=np.zeros((s.ndof,nf+len(s.fixed_scalar_ids)));B[np.ix_(s.free_scalar_ids,np.arange(nf))]=R
        B[s.fixed_scalar_ids,nf+np.arange(len(s.fixed_scalar_ids))]=1
        G=N[:s.n].T@s.Ks@N[:s.n];right=N[:s.n].T@s.Ks
        P=B-N@la.solve(G,right@B[:s.n],assume_a='pos')
        offset=-N@la.solve(G,right@s.reference[:s.n],assume_a='pos')
        reduced=P.T@M@P;ev=la.eigvalsh(reduced[:nf,:nf]/np.sqrt(np.diag(reduced)[:nf,None]*np.diag(reduced)[None,:nf]))
        if ev[0]<=1e-10:raise ValueError('candidate independent mass is not full rank')
        return P,offset,N,dict(free_scalar=nf,nullity=null.shape[1],scaled_eigen_min=float(ev[0]),scaled_condition=float(ev[-1]/ev[0]),
            mass_null_relative=float(la.norm(M@N)/max(la.norm(M),1e-30)))

    def stabilization(self,full,direction=None):
        s=self.space;Y=s.reference[:s.n]+full[:s.n];f=np.zeros_like(full);f[:s.n]=s.Ks@Y
        return .5*float(np.sum(Y*(s.Ks@Y))),f,(None if direction is None else np.vstack([s.Ks@direction[:s.n],np.zeros_like(direction[s.n:])]))
