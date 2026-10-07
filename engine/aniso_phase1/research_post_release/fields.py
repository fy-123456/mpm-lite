"""Bounded immutable probe maps; identical physical fields without ambient arrays."""
import time
import numpy as np
import scipy.sparse as sp
from engine.aniso_phase1.tensor_metrics import sampling
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.stage2.contracts import array_digest


def kron3(matrices):return sp.kron(sp.kron(matrices[0],matrices[1],format='csr'),matrices[2],format='csr')


class CachedProbes:
    def __init__(self,model,shape=(33,7,7),*,max_bytes=256<<20):
        start=time.perf_counter();self.model=model;s=model.parent;r=model.reduction
        self.shape=tuple(shape)
        if len(shape)!=3 or any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in shape):raise ValueError('three probe dimensions >=2 required')
        n=int(np.prod(shape));estimate=8*n*(4*r.P.shape[1]+4*3+3)
        if estimate>max_bytes:raise MemoryError('probe-map cache budget exceeded')
        axes=[np.linspace(e[0],e[-1],k) for e,k in zip(s.edges,shape)]
        B=[sampling(e,s.p,x) for e,x in zip(s.edges,axes)]
        D=[sampling(e,s.p,x,True) for e,x in zip(s.edges,axes)]
        self.maps=[];self.offsets=[]
        for derivative in (-1,0,1,2):
            maps=[D[k] if k==derivative else B[k] for k in range(3)]
            carrier=kron3([a@b for a,b in zip(maps,s.prolong)])@s.oldA
            local=(kron3(maps)@s.raw)@s.transform
            full=np.column_stack((carrier,local))
            matrix=np.asarray(full@r.P);offset=np.asarray(full@r.offset)
            matrix.setflags(write=False);offset.setflags(write=False)
            self.maps.append(matrix);self.offsets.append(offset)
        self.X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1);self.X.setflags(write=False)
        self.identity=dict(reduction=r.signature,space=s.signature,axes=[array_digest(x) for x in axes],
            dtype='float64',geometry_only=True,gradient_convention='du_i/dX_j',stabilization_offset_included=True)
        self.signature=digest(self.identity);self.bytes=sum(a.nbytes+b.nbytes for a,b in zip(self.maps,self.offsets))+self.X.nbytes
        if self.bytes>max_bytes:raise MemoryError('actual probe-map bytes exceeded budget')
        self.build_seconds=time.perf_counter()-start

    def frame(self,state):
        m=self.model
        if state.child_states['identity'].get('model')!=m.reduction.signature:raise ValueError('foreign physical space for probe cache')
        m.boundary.validate(state)
        q,v=state.q,state.velocity
        u=(self.maps[0]@q+self.offsets[0]).reshape(*self.shape,3)
        velocity=(self.maps[0]@v).reshape(*self.shape,3)
        grad=np.stack([a@q+b for a,b in zip(self.maps[1:],self.offsets[1:])],axis=-1).reshape(*self.shape,3,3)
        gv=np.stack([a@v for a in self.maps[1:]],axis=-1).reshape(*self.shape,3,3)
        F=np.eye(3)+grad
        if not np.isfinite(F).all() or np.min(np.linalg.det(F))<=0:raise ValueError('invalid physical probe deformation')
        _,stress=material_response(F.reshape(-1,3,3),np.broadcast_to(m.parent.A,F.reshape(-1,3,3).shape),m.parent.params)
        return dict(time=np.array(state.time),step=np.array(state.step),X=self.X.copy(),x=self.X+u,F=F,
                    velocity=velocity,velocity_gradient=gv@np.linalg.inv(F),PK1=stress.reshape(F.shape))
