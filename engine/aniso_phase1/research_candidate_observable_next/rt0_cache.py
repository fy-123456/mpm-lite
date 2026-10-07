"""Optional bounded per-geometry RT0 reference metadata; always rebuild current H.

Only integration indices and scalar reference basis factors are retained. No F,
J, gradient, H, residual or trial-dependent tensor survives an assemble call.
"""
import hashlib
import time
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_cross_direction_next.det_rt0 import det3


def fingerprint(*values):
    h=hashlib.sha256()
    for value in values:
        a=np.asarray(value);h.update(str((a.shape,a.dtype.str)).encode());h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


class StaticRT0Metadata:
    def __init__(self,topology,X,weights,cell_ids,mobility,*,max_bytes,owner_identity):
        started=time.perf_counter();self.topology=topology;self.owner_identity=owner_identity
        X=np.asarray(X);weights=np.asarray(weights);cell_ids=np.asarray(cell_ids);mobility=np.asarray(mobility)
        self.key=self.binding(X,weights,cell_ids,mobility)
        # For each Cartesian axis the two face functions sum to width/V.
        # Retaining the three minus-face values plus point indices uses 32 B
        # per point, instead of 104 B for six phi and six weighted phi values.
        N=len(weights);estimate=N*(8+3*8)+72
        if estimate>max_bytes:raise MemoryError('static RT0 metadata exceeds registered byte cap')
        if X.shape!=(N,3) or cell_ids.shape!=(N,) or not np.isfinite(X).all() or not np.isfinite(weights).all() or np.any(weights<0) or not np.issubdtype(cell_ids.dtype,np.integer) or np.any((cell_ids<0)|(cell_ids>=topology.cells)):raise ValueError('invalid static integration arrays')
        if mobility.shape!=(3,3) or not np.isfinite(mobility).all() or not np.allclose(mobility,mobility.T) or la.eigvalsh(mobility)[0]<=0:raise ValueError('SPD full mobility required')
        self.invk=la.solve(mobility,np.eye(3),assume_a='pos');self.invk.setflags(write=False);self.axes=np.repeat(np.arange(3),2);self.parts=[];self.bytes=self.invk.nbytes
        for cell in range(topology.cells):
            indices=np.flatnonzero(cell_ids==cell)
            if not len(indices):raise ValueError('every pressure cell needs integration points')
            bounds=topology.cell_bounds[cell];chunks=[]
            for start in range(0,len(indices),16384):
                ix=indices[start:start+16384].copy();minus=(X[ix]-bounds[:,1])/topology.V0[cell]*(-1.)
                for a in (ix,minus):a.setflags(write=False);self.bytes+=a.nbytes
                chunks.append((ix,minus))
            self.parts.append(chunks)
        if self.bytes>max_bytes:raise MemoryError('actual static cache exceeded budget')
        self.build_seconds=time.perf_counter()-started;self.max_bytes=int(max_bytes)

    def binding(self,X,weights,cell_ids,mobility):
        t=self.topology
        return fingerprint(X,weights,cell_ids,mobility,t.faces,t.signs,t.V0,t.cell_bounds)

    def assemble(self,X,weights,cell_ids,F,mobility):
        if self.binding(X,weights,cell_ids,mobility)!=self.key:raise ValueError('foreign or changed geometry/quadrature/mobility for static cache')
        F=np.asarray(F)
        if F.shape!=(len(weights),3,3) or not np.isfinite(F).all():raise ValueError('invalid current deformation')
        H=np.zeros((self.topology.nflux,self.topology.nflux));minJ=float('inf')
        for cell,chunks in enumerate(self.parts):
            local=np.zeros((6,6))
            pair_sum=np.diff(self.topology.cell_bounds[cell],axis=1).ravel()/self.topology.V0[cell]
            for ix,minus in chunks:
                f=F[ix];J=det3(f);minJ=min(minJ,float(J.min()))
                if np.any(J<=.1):raise ValueError('RT0 geometry leaves positive-J range')
                invK=(np.swapaxes(f,1,2)@(self.invk@f))/J[:,None,None]
                phi=np.empty((len(ix),6));phi[:,::2]=minus;phi[:,1::2]=pair_sum-minus
                for a in range(6):
                    weighted=weights[ix]*phi[:,a]
                    for b in range(a,6):
                        value=float(np.dot(weighted*phi[:,b],invK[:,self.axes[a],self.axes[b]]));local[a,b]+=value
                        if a!=b:local[b,a]+=value
            faces=self.topology.faces[cell];H[np.ix_(faces,faces)]+=local
        return H,minJ


def install(geometry,*,max_bytes):
    cache=StaticRT0Metadata(geometry.topology,geometry.X,geometry.total_weights,geometry.cell_ids,geometry.mobility,max_bytes=max_bytes,owner_identity=geometry.identity)
    geometry.topology.assemble=cache.assemble
    geometry.cache.clear()
    geometry.static_rt0_metadata=cache
    return cache
