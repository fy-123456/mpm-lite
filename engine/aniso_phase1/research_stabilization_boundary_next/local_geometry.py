"""Exact cell-local tensor adjoints for the owned x-only RT0 fixture.

Only zero-weight quadrature points are removed from each cell contraction.
All nodal maps, reduced cross terms, current F, full mobility and H remain.
Static partition metadata is private; numerical cache entries are owned copies.
"""
import copy,time
from collections import OrderedDict
import numpy as np
import warp as wp
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_restoring_rt0_next.device_rt0 import DeviceGeometry
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel

def own_weights(weights):
    result=np.array(weights,dtype=np.float64,copy=True)
    if result.ndim!=1 or not np.isfinite(result).all() or np.any(result<0):raise ValueError('invalid owned integration weights')
    result.setflags(write=False);return result

def partition(geometry):
    g=geometry;t=g.topology
    if t.shape[1:]!=(1,1) or t.shape[0]>32:raise ValueError('cell-local route requires bounded x-only Cartesian grid')
    if g.count!=int(np.prod(g.shape)) or tuple(map(len,g.points))!=tuple(g.shape):raise ValueError('invalid tensor shape')
    nx,ny,nz=g.shape;ids=np.asarray(g.cell_ids).reshape(g.shape);x=np.asarray(g.points[0]);expected=np.searchsorted(t.cuts[0],x,side='right')-1
    if np.any(expected<0) or np.any(expected>=t.cells) or not np.array_equal(ids,np.broadcast_to(expected[:,None,None],g.shape)):raise ValueError('cell ownership differs from tensor slices')
    if not np.isfinite(g.total_weights).all() or np.any(g.total_weights<0):raise ValueError('invalid integration weights')
    result=[];cursor=0
    for c in range(t.cells):
        ix=np.flatnonzero(expected==c)
        if len(ix)==0 or ix[0]!=cursor or not np.array_equal(ix,np.arange(ix[0],ix[-1]+1)):raise ValueError('cell quadrature must form nonempty contiguous slices')
        start=int(ix[0]*ny*nz);stop=int((ix[-1]+1)*ny*nz);w=g.total_weights[start:stop]
        if not np.isclose(w.sum(),t.V0[c],atol=1e-13,rtol=1e-12):raise ValueError('cell volume not covered')
        result.append((int(ix[0]),int(ix[-1]+1),start,stop));cursor=int(ix[-1]+1)
    if cursor!=nx:raise ValueError('uncovered tensor points')
    return result

class LocalGeometry(DeviceGeometry):
    @classmethod
    def from_device(cls,original,*,cap_bytes):
        if not isinstance(original,DeviceGeometry):raise ValueError('authenticated device geometry required')
        tick=time.perf_counter();g=original;ranges=partition(g);maps=g.op.maps;p=g.model.parent.p
        # Both CSR directions for value and derivative: pessimistic nnz bound.
        estimated=g.count*16
        for lo,hi,start,stop in ranges:
            for n,nodes in zip((hi-lo,*g.shape[1:]),g.model.parent.shape):estimated+=2*(2*n*(p+1)*12+(n+nodes+2)*4)
        if estimated>cap_bytes:raise MemoryError('local tensor metadata exceeds frozen cap')
        guard=getattr(g.op,'memory_budget',None)
        if guard:guard.observe(estimated)
        obj=cls.__new__(cls);obj.__dict__=g.__dict__.copy();obj.cache=OrderedDict();obj.local=[];obj.total_weights=own_weights(g.total_weights);obj.local_bytes=obj.total_weights.nbytes
        for lo,hi,start,stop in ranges:
            layout=maps.layout((g.points[0][lo:hi],*g.points[1:]));weights=wp.array(np.array(obj.total_weights[start:stop],copy=True),dtype=wp.float64,device=g.op.device)
            obj.local_bytes+=weights.capacity+sum(a.bytes for pair in layout[0] for derivative in pair for a in derivative)
            obj.local.append((start,stop,layout,weights))
        if obj.local_bytes>cap_bytes:raise MemoryError('actual local metadata exceeds cap')
        obj.identity=dict(schema='bounded-local-cell-volume-gradient-v1',device_parent=copy.deepcopy(g.identity),ranges=[list(v) for v in ranges],metadata_bytes=int(obj.local_bytes),cap_bytes=int(cap_bytes),numerical_cache='original q key, copied results; maximum six entries; current F required')
        wp.synchronize_device(g.op.device);obj.local_setup_seconds=time.perf_counter()-tick;return obj

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:
            self.cache.move_to_end(key);return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.cache[key].items()}
        F=self.field(q);cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device)
        wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,unused,bad,.1],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('cell geometry leaves positive J range')
        jj=J.numpy();V=[];G=[]
        for start,stop,layout,weights in self.local:
            grad=self.op.maps.gradient_adjoint(cof[start:stop],layout,weights).numpy().reshape(self.model.parent.ndof,3)
            G.append(self.model.reduction.P.T@grad);V.append(float(self.total_weights[start:stop]@jj[start:stop]))
        H,minJ=self.assembler.assemble(F);value=dict(volume=np.array(V),gradient=np.array(G),H=H,min_detF=min(float(jj.min()),minJ));self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}

def install(coupling,previous_bridge=None):
    c=coupling;state=c.state;c.validate(state);before=physical_payload(state);parent=copy.deepcopy(c.identity);core=c.core;cap=min(256*2**20,int(.02*core.model.operator.memory_budget.initial_free));g=LocalGeometry.from_device(c.geometry,cap_bytes=cap)
    core.geometry=g;core.identity=dict(core.identity,geometry=g.identity,implementation='bounded-local-cell-volume-gradient-v1');core.signature=digest(core.identity)
    c.identity=dict(schema='stabilization-local-cell-fixture-v1',physical_parent=parent,core=core.identity);c.signature=digest(c.identity)
    state.child_states['fluid']['model']=core.signature;state.child_states['explicit_pressure_grid']=c.signature
    if physical_payload(state)!=before:raise ValueError('implementation bridge changed physical state')
    core._transaction=StateTransaction(state,validator=c.validate)
    return dict(status='passed_scoped',prior_bridge=previous_bridge,old_owner_validated=True,physical_state_exact=True,physical_payload_sha256=digest(before),new_digest=state.digest(),static_metadata_bytes=g.local_bytes)
