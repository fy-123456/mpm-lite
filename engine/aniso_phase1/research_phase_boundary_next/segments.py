"""Vectorized construction of the unchanged ordered CSR segment metadata."""
import time
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.aniso_phase1.research_local_span_next.reuse import ReusedSegmentedCSR
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedSpace,SegmentedGPUOperator
from engine.aniso_phase1.research_sequential_next.profiling import ProfiledGPUOperator
from engine.aniso_phase1.research_sequential_next.model import PracticalModel
from engine.aniso_phase1.research_sequential_next.resources import GPUMemoryBudget
from engine.aniso_phase1.research_sequential.condensation import FullCoordinates
from engine.aniso_phase1.research_d.identity import digest

def segment_metadata(ptr,chunk=128):
    if isinstance(chunk,bool) or not isinstance(chunk,int) or chunk<1:raise ValueError('positive integer chunk required')
    p=np.asarray(ptr)
    limit=np.iinfo(np.int32).max
    if p.ndim!=1 or not len(p) or p.dtype.kind not in 'iu' or p[0]!=0 or np.any(p>limit):raise ValueError('int32 bounded row pointers required')
    p=p.astype(np.int64)
    lengths=np.diff(p)
    if np.any(lengths<0):raise ValueError('monotonic row pointers required')
    # Avoid overflow for unusually large Python integer chunks.
    counts=lengths//chunk+(lengths%chunk!=0) if chunk<=limit else (lengths>0).astype(np.int64)
    rows=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    n=int(rows[-1])
    if n>limit:raise ValueError('segment count exceeds int32 capacity')
    if chunk>limit:
        starts=p[:-1][lengths>0];stops=p[1:][lengths>0]
    else:
        starts=np.repeat(p[:-1],counts)+chunk*(np.arange(n,dtype=np.int64)-np.repeat(rows[:-1],counts))
        stops=np.minimum(starts+chunk,np.repeat(p[1:],counts))
    return tuple(np.asarray(x,dtype=np.int32) for x in (starts,stops,rows))

class VectorizedCSR(ReusedSegmentedCSR):
    @classmethod
    def from_csr(cls,original,source,chunk=128):
        if original.shape!=source.shape or original.val.dtype!=wp.float64 or original.ptr.dtype!=wp.int32:raise ValueError('CSR layout/dtype mismatch')
        values=source.data if sp.issparse(source) else np.asarray(source)
        if not np.isfinite(values).all():raise ValueError('finite CSR required')
        ptr=original.ptr.numpy()
        if len(ptr)!=original.shape[0]+1 or ptr[-1]!=original.val.size:raise ValueError('invalid existing CSR row pointers')
        starts,stops,rows=segment_metadata(ptr,chunk)
        obj=cls.__new__(cls);obj.shape=original.shape;obj.device=original.device
        obj.ptr=original.ptr;obj.col=original.col;obj.val=original.val
        obj.starts=wp.array(starts,dtype=wp.int32,device=obj.device)
        obj.stops=wp.array(stops,dtype=wp.int32,device=obj.device)
        obj.row_segments=wp.array(rows,dtype=wp.int32,device=obj.device)
        obj.segment_count=len(starts);obj.chunk=chunk;obj.bytes=original.bytes+sum(x.capacity for x in (obj.starts,obj.stops,obj.row_segments));obj.reused_transpose_buffers=True
        return obj

class VectorizedSpace(SegmentedSpace):
    @classmethod
    def from_existing(cls,original):
        obj=cls.__new__(cls);obj.__dict__=original.__dict__.copy();obj.layouts=dict(original.layouts)
        obj.old=(original.old[0],VectorizedCSR.from_csr(original.old[1],np.asarray(original.space.oldA).T))
        obj.raw=(original.raw[0],VectorizedCSR.from_csr(original.raw[1],original.space.raw.T))
        replaced={id(original.old[1]),id(original.raw[1])};obj.pairs=[p for p in original.pairs if id(p) not in replaced]+[obj.old[1],obj.raw[1]]
        obj.static_bytes=sum(p.bytes for p in obj.pairs)+obj.ids.capacity+obj.lift.capacity
        obj.cache_key=digest(dict(parent=original.cache_key,transpose_algorithm='segmented-FP64-128-v1'))
        return obj

class VectorizedGPUOperator(SegmentedGPUOperator):
    def __init__(self,*args,**kwargs):
        monitor=GPUMemoryBudget(kwargs.get('device','cuda:0'));ProfiledGPUOperator.__init__(self,*args,**kwargs);started=time.perf_counter();self.memory_budget=monitor
        self.maps=VectorizedSpace.from_existing(self.maps);self.layout=self.maps.layout(self.points);self.signature=digest(dict(original=self.signature,maps=self.maps.cache_key))
        wp.synchronize_device(self.device);self.build_seconds+=time.perf_counter()-started;self.memory_budget.observe()

class VectorizedModel(PracticalModel):
    def __init__(self,reduction,*,order=7,device='cuda:0',hold=None):
        if device=='cpu':raise ValueError('segmented GPU model needs CUDA')
        super().__init__(reduction,order=order,device='cpu',hold=hold)
        self.operator=VectorizedGPUOperator(FullCoordinates(self.parent),order=order,device=device);self.device=device
        self.identity=dict(self.identity,device=device);self.signature=digest(self.identity)
