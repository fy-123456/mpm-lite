"""Bounded CPU reference topology. Original GPU <=32-cell constructor unchanged.

Topology construction follows the authenticated Cartesian implementation;
only the independent CPU reference limit is 128. Full tensor assembly and
orientation are inherited without global mutation.
"""
import numpy as np
from engine.aniso_phase1.research_cross_direction_next.det_rt0 import FastTopology

class ReferenceTopology(FastTopology):
    def __init__(self,cuts):
        if len(cuts)!=3:raise ValueError('three explicit Cartesian cut arrays required')
        self.cuts=tuple(np.array(v,dtype=float,copy=True) for v in cuts)
        if any(v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() or np.any(np.diff(v)<=0) for v in self.cuts):raise ValueError('finite strictly increasing cuts required')
        self.shape=tuple(len(v)-1 for v in self.cuts);self.cells=int(np.prod(self.shape))
        if not 1<=self.cells<=128:raise ValueError('bounded research grid supports at most 128 CPU reference cells')
        self.bounds=np.array([[x[0],x[-1]] for x in self.cuts]);self.signs=np.array([-1.,1.,-1.,1.,-1.,1.]);self.cell_bounds=[]
        self.faces=[];centres=[];axes=[];areas=[];face_ids={};volumes=[]
        for idx in np.ndindex(self.shape):
            b=np.array([self.cuts[a][idx[a]:idx[a]+2] for a in range(3)]);self.cell_bounds.append(b);width=np.diff(b,axis=1).ravel();volumes.append(float(np.prod(width)));row=[]
            for a in range(3):
                for side in (0,1):
                    key=(a,idx[a]+side,*[idx[k] for k in range(3) if k!=a])
                    if key not in face_ids:
                        face_ids[key]=len(centres);c=b.mean(axis=1);c[a]=b[a,side];centres.append(c);axes.append(a);areas.append(float(np.prod(np.delete(width,a))))
                    row.append(face_ids[key])
            self.faces.append(row)
        self.faces=np.array(self.faces,dtype=int);self.nflux=len(centres);self.centres=np.array(centres);self.axes=np.array(axes);self.areas=np.array(areas);self.V0=np.array(volumes)
        self.B=np.zeros((self.cells,self.nflux))
        for c,faces in enumerate(self.faces):self.B[c,faces]=self.signs
        self.boundary=np.where(np.count_nonzero(self.B,axis=0)==1)[0];self.internal=np.where(np.count_nonzero(self.B,axis=0)==2)[0];self.boundary_sign=self.B.sum(axis=0)
        if np.any(self.B[:,self.internal].sum(axis=0)!=0):raise ValueError('internal face must have opposite ownership')
        for v in (*self.cuts,self.bounds,self.signs,self.faces,self.centres,self.axes,self.areas,self.V0,self.B,self.boundary,self.internal,self.boundary_sign):v.setflags(write=False)

    def source(self,density=.001):
        mid=self.bounds[0].mean();out=[]
        for b in self.cell_bounds:
            width=np.diff(b,axis=1).ravel();overlap=max(0.,min(b[0,1],mid)-b[0,0]);out.append(density*overlap*width[1]*width[2])
        return np.array(out)

    def locate(self,X):
        ids=[np.clip(np.searchsorted(e,X[:,a],side='right')-1,0,len(e)-2) for a,e in enumerate(self.cuts)]
        return np.ravel_multi_index(ids,self.shape)

