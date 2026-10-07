"""Two fixed MATERIAL volumes via the divergence theorem.

For a continuous piecewise Qp deformation, x.cof(F)N has tangential degree
at most 3p-1. Composite Gauss ceil(3p/2) integrates it exactly (p=6 -> q9).
G is the boundary virtual volume integral B^T cof(F)N, including prescribed
coordinates. No geometry rule changes during a material-rule retry.
"""
import numpy as np
from engine.aniso_phase1.tensor_metrics import quadrature_axis,sampling,evaluate_gradient
from engine.aniso_phase1.tensor_reference import apply_axis


def cofactor_column(F,a):
    return np.cross(F[..., :, (a+1)%3],F[..., :, (a+2)%3])

class Geometry:
    def __init__(self,model,order=9):
        self.model=model;self.order=int(order);s=model.space
        if order<1 or order!=int(order):raise ValueError('positive integer geometry order')
        self.cache={};self.faces=[];self.reference_volume=np.array([.375*.25*.25]*2)
        for c,(lo,hi) in enumerate(((.125,.5),(.5,.875))):
            edges=[np.unique(np.r_[lo,s.edges[0][(s.edges[0]>lo)&(s.edges[0]<hi)],hi]),s.edges[1],s.edges[2]]
            for a in range(3):
                for side,sign in ((0,-1),(-1,1)):
                    qs=[(np.array([e[side]]),np.ones(1)) if j==a else quadrature_axis(e,order) for j,e in enumerate(edges)]
                    pts=[q[0] for q in qs];ww=[q[1] for q in qs]
                    W=ww[0][:,None,None]*ww[1][None,:,None]*ww[2][None,None,:]
                    X=np.stack(np.meshgrid(*pts,indexing='ij'),axis=-1)
                    B=[sampling(e,s.p,x) for e,x in zip(s.edges,pts)]
                    self.faces.append((c,a,sign,pts,W,X,B))
        self.points=sum(f[4].size for f in self.faces)
    def evaluate(self,q,direction=None):
        key=q.tobytes()
        if direction is None and key in self.cache:return self.cache[key]
        out=self._evaluate(q,direction)
        if direction is None:
            if len(self.cache)>=8:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=out
        return out
    def _evaluate(self,q,direction=None):
        m=self.model;s=m.space;u=s.nodes(m.r.expand(q));du=None if direction is None else s.nodes(m.r.velocity(direction))
        nodal=[np.zeros_like(u),np.zeros_like(u)];dnodal=None if du is None else [np.zeros_like(u),np.zeros_like(u)]
        V=np.zeros(2);minimum=float('inf')
        for c,a,sign,pts,W,X,B in self.faces:
            value=u.reshape(*s.shape,3)
            for k,b in enumerate(B):value=apply_axis(b,value,k)
            F=np.eye(3)+evaluate_gradient((s.edges,s.p,u),pts)
            det=np.linalg.det(F);minimum=min(minimum,float(det.min()))
            if not np.isfinite(det).all() or minimum<=.1:raise ValueError('inadmissible pressure boundary geometry')
            normal=sign*cofactor_column(F,a)
            V[c]+=float(np.sum(W*np.sum((X+value)*normal,axis=-1)))/3
            force=W[...,None]*normal
            for k in (2,1,0):force=apply_axis(B[k].T,force,k)
            nodal[c]+=force.reshape(-1,3)
            if du is not None:
                dF=evaluate_gradient((s.edges,s.p,du),pts)
                b,z=(a+1)%3,(a+2)%3
                dn=sign*(np.cross(dF[..., :,b],F[..., :,z])+np.cross(F[..., :,b],dF[..., :,z]))
                force=W[...,None]*dn
                for k in (2,1,0):force=apply_axis(B[k].T,force,k)
                dnodal[c]+=force.reshape(-1,3)
        G=np.stack([m.r.P.T@s.adjoint(f) for f in nodal])
        out=dict(V=V,G=G,min_boundary_J=minimum)
        if du is not None:out['dG']=np.stack([m.r.P.T@s.adjoint(f) for f in dnodal])
        return out
    def chain(self,q0,q1,direction=None):
        # Simpson is exact for G along affine material-coordinate motion.
        values=[self.evaluate(q,direction if i==2 else (None if direction is None else direction*.5)) if i else self.evaluate(q)
                for i,q in enumerate((q0,(q0+q1)/2,q1))]
        out=dict(V0=values[0]['V'],V1=values[2]['V'],G=(values[0]['G']+4*values[1]['G']+values[2]['G'])/6,
                 min_boundary_J=min(x['min_boundary_J'] for x in values))
        if direction is not None:out['dG']=(4*values[1]['dG']+values[2]['dG'])/6
        return out
