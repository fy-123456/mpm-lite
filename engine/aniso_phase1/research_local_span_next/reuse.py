"""Reuse already-built immutable CSR buffers when adding segmented row metadata.

No process-global cache, no skipped source verification, no changed summation.
Each model owns its layout dictionary and every action allocates its work arrays.
"""
import time
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedCSR,SegmentedSpace,SegmentedGPUOperator
from engine.aniso_phase1.research_sequential_next.profiling import ProfiledGPUOperator
from engine.aniso_phase1.research_sequential_next.model import PracticalModel
from engine.aniso_phase1.research_sequential.condensation import FullCoordinates
from engine.aniso_phase1.research_sequential_next.resources import GPUMemoryBudget
from engine.aniso_phase1.research_d.identity import digest

class ReusedSegmentedCSR(SegmentedCSR):
    @classmethod
    def from_csr(cls,original,source,chunk=128):
        if isinstance(chunk,bool) or not isinstance(chunk,int) or chunk<1:raise ValueError('positive chunk required')
        if original.shape!=source.shape or original.val.dtype!=wp.float64 or original.ptr.dtype!=wp.int32:raise ValueError('CSR layout/dtype mismatch')
        values=source.data if sp.issparse(source) else np.asarray(source)
        if not np.isfinite(values).all():raise ValueError('finite CSR required')
        ptr=original.ptr.numpy()
        if len(ptr)!=original.shape[0]+1 or ptr[0]!=0 or ptr[-1]!=original.val.size or np.any(np.diff(ptr)<0):raise ValueError('invalid existing CSR row pointers')
        obj=cls.__new__(cls);obj.shape=original.shape;obj.device=original.device
        obj.ptr=original.ptr;obj.col=original.col;obj.val=original.val
        starts=[];stops=[];rows=[0]
        for a,b in zip(ptr[:-1],ptr[1:]):
            for start in range(int(a),int(b),chunk):starts.append(start);stops.append(min(start+chunk,int(b)))
            rows.append(len(starts))
        obj.starts=wp.array(np.asarray(starts,dtype=np.int32),dtype=wp.int32,device=obj.device)
        obj.stops=wp.array(np.asarray(stops,dtype=np.int32),dtype=wp.int32,device=obj.device)
        obj.row_segments=wp.array(np.asarray(rows,dtype=np.int32),dtype=wp.int32,device=obj.device)
        obj.segment_count=len(starts);obj.chunk=chunk;obj.bytes=original.bytes+sum(x.capacity for x in (obj.starts,obj.stops,obj.row_segments));obj.reused_transpose_buffers=True
        return obj

class ReusedSegmentedSpace(SegmentedSpace):
    @classmethod
    def from_existing(cls,original):
        obj=cls.__new__(cls);obj.__dict__=original.__dict__.copy();obj.layouts=dict(original.layouts)
        obj.old=(original.old[0],ReusedSegmentedCSR.from_csr(original.old[1],np.asarray(original.space.oldA).T))
        obj.raw=(original.raw[0],ReusedSegmentedCSR.from_csr(original.raw[1],original.space.raw.T))
        replaced={id(original.old[1]),id(original.raw[1])};obj.pairs=[p for p in original.pairs if id(p) not in replaced]+[obj.old[1],obj.raw[1]]
        obj.static_bytes=sum(p.bytes for p in obj.pairs)+obj.ids.capacity+obj.lift.capacity
        # Identical chunking/order; this is an allocation change, not a new map.
        obj.cache_key=digest(dict(parent=original.cache_key,transpose_algorithm='segmented-FP64-128-v1'))
        return obj

class ReusedSegmentedGPUOperator(SegmentedGPUOperator):
    def __init__(self,*args,**kwargs):
        monitor=GPUMemoryBudget(kwargs.get('device','cuda:0'));ProfiledGPUOperator.__init__(self,*args,**kwargs);started=time.perf_counter();self.memory_budget=monitor
        self.maps=ReusedSegmentedSpace.from_existing(self.maps);self.layout=self.maps.layout(self.points);self.signature=digest(dict(original=self.signature,maps=self.maps.cache_key))
        wp.synchronize_device(self.device);self.build_seconds+=time.perf_counter()-started;self.memory_budget.observe()

class ReusedSegmentedModel(PracticalModel):
    def __init__(self,reduction,*,order=7,device='cuda:0',hold=None):
        if device=='cpu':raise ValueError('segmented GPU model needs CUDA')
        super().__init__(reduction,order=order,device='cpu',hold=hold)
        self.operator=ReusedSegmentedGPUOperator(FullCoordinates(self.parent),order=order,device=device);self.device=device
        self.identity=dict(self.identity,device=device);self.signature=digest(self.identity)
