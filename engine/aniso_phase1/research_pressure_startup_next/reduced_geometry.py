"""Skip provably zero nodal columns in each cell's existing raw transpose.

No dense reduced nodal basis, value copies or discarded physical cross terms.
All CSR values/columns are shared immutable buffers; only segment metadata changes.
"""
import copy,time
from collections import OrderedDict
import numpy as np
import warp as wp
from engine.aniso_phase1.tensor_metrics import sampling
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedCSR
from engine.aniso_phase1.research_stabilization_boundary_next.local_geometry import LocalGeometry,partition
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction

def bounded_segments(ptr,cols,lo,hi,chunk=128):
    if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<1 or lo<0 or hi<=lo:raise ValueError('invalid column interval/chunk')
    starts=[];stops=[];rows=[0];used=0
    for a,b in zip(ptr[:-1],ptr[1:]):
        row=cols[a:b];left=int(a)+int(np.searchsorted(row,lo));right=int(a)+int(np.searchsorted(row,hi))
        used+=right-left
        for start in range(left,right,chunk):starts.append(start);stops.append(min(start+chunk,right))
        rows.append(len(starts))
    return tuple(np.asarray(x,dtype=np.int32) for x in (starts,stops,rows)),used

class BoundedCSR(SegmentedCSR):
    @classmethod
    def from_shared(cls,old,arrays):
        obj=cls.__new__(cls);obj.__dict__=old.__dict__.copy()
        obj.starts,obj.stops,obj.row_segments=[wp.array(a,dtype=wp.int32,device=old.device) for a in arrays]
        obj.segment_count=len(arrays[0]);obj.extra_bytes=sum(a.capacity for a in (obj.starts,obj.stops,obj.row_segments));obj.bytes=old.bytes+obj.extra_bytes
        return obj

class ReducedGeometry(LocalGeometry):
    @classmethod
    def from_device(cls,g,*,cap_bytes):
        if not isinstance(g,LocalGeometry):raise ValueError('authenticated BASE local geometry required')
        tick=time.perf_counter();ranges=partition(g);space=g.model.parent;maps=g.op.maps
        # This temporary transpose is CPU-only and released before any actual step.
        # The existing GPU raw transpose owns identical sorted columns and values.
        source=space.raw.tocsc();source.sort_indices();rowptr=maps.raw[1].ptr.numpy()
        if not np.array_equal(source.indptr,rowptr):raise ValueError('raw transpose row binding differs')
        plans=[];used=[];bounds=[]
        for lo,hi,start,stop in ranges:
            x=g.points[0][lo:hi];val=sampling(space.edges[0],space.p,x,False);der=sampling(space.edges[0],space.p,x,True)
            active=np.unique(np.r_[val.indices,der.indices]);nlo=int(active[0])*space.shape[1]*space.shape[2];nhi=(int(active[-1])+1)*space.shape[1]*space.shape[2]
            meta,n=bounded_segments(source.indptr,source.indices,nlo,nhi)
            plans.append(meta);used.append(n);bounds.append([nlo,nhi])
        extra=sum(a.nbytes for plan in plans for a in plan)
        if g.local_bytes+2*extra>cap_bytes:raise MemoryError('base plus device and staging metadata exceed cap')
        g.op.memory_budget.observe(extra)
        obj=cls.__new__(cls);obj.__dict__=g.__dict__.copy();obj.cache=OrderedDict();obj.local_maps=[];obj.local_bytes=g.local_bytes
        for plan in plans:
            mp=copy.copy(maps);mp.layouts=dict(maps.layouts);bounded=BoundedCSR.from_shared(maps.raw[1],plan)
            mp.raw=(maps.raw[0],bounded);obj.local_maps.append(mp);obj.local_bytes+=bounded.extra_bytes
        obj.identity=dict(schema='bounded-raw-column-cell-gradient-v1',parent=copy.deepcopy(g.identity),nodal_column_bounds=bounds,metadata_bytes=int(obj.local_bytes),construction_metadata_peak_bytes=int(g.local_bytes+2*extra),raw_total_nnz=int(source.nnz),used_nnz_per_cell=used,cap_bytes=int(cap_bytes),current_F=True,shared_values_immutable=True)
        del source,plans
        wp.synchronize_device(g.op.device);obj.local_setup_seconds=time.perf_counter()-tick
        return obj

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:
            self.cache.move_to_end(key);return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.cache[key].items()}
        F=self.field(q);cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device)
        wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,unused,bad,.1],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('cell geometry leaves positive J range')
        jj=J.numpy();V=[];G=[]
        for mp,(start,stop,layout,weights) in zip(self.local_maps,self.local):
            grad=mp.gradient_adjoint(cof[start:stop],layout,weights).numpy().reshape(self.model.parent.ndof,3)
            G.append(self.model.reduction.P.T@grad);V.append(float(self.total_weights[start:stop]@jj[start:stop]))
        H,minJ=self.assembler.assemble(F);value=dict(volume=np.array(V),gradient=np.array(G),H=H,min_detF=min(float(jj.min()),minJ));self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}

def install(coupling,previous_bridge=None):
    c=coupling;state=c.state;c.validate(state);before=physical_payload(state);parent=copy.deepcopy(c.identity);core=c.core
    cap=min(256*2**20,int(.02*core.model.operator.memory_budget.initial_free));g=ReducedGeometry.from_device(c.geometry,cap_bytes=cap)
    core.geometry=g;core.identity=dict(core.identity,geometry=g.identity,implementation='bounded-raw-column-cell-gradient-v1');core.signature=digest(core.identity)
    c.identity=dict(schema='pressure-startup-bounded-raw-fixture-v1',physical_parent=parent,core=core.identity);c.signature=digest(c.identity)
    state.child_states['fluid']['model']=core.signature;state.child_states['explicit_pressure_grid']=c.signature
    if physical_payload(state)!=before:raise ValueError('implementation bridge changed state')
    core._transaction=StateTransaction(state,validator=c.validate)
    return dict(status='passed_scoped',prior_bridge=previous_bridge,physical_state_exact=True,physical_payload_sha256=digest(before),new_digest=state.digest(),metadata_bytes=g.local_bytes)
