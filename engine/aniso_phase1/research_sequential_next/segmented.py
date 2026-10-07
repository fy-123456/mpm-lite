"""Parallel partial sums for long transpose CSR rows; same FP64 linear map."""
from __future__ import annotations
import copy
import time
import numpy as np
import scipy.sparse as sp
import warp as wp
from ..research_d.stage2.gpu_space import CSR,GPUSpace
from ..research_d.identity import digest
from ..research_sequential.condensation import FullCoordinates
from .profiling import ProfiledGPUOperator
from .model import PracticalModel


@wp.kernel
def segment_kernel(starts:wp.array(dtype=wp.int32),stops:wp.array(dtype=wp.int32),
                   col:wp.array(dtype=wp.int32),val:wp.array(dtype=wp.float64),
                   x:wp.array(dtype=wp.float64),partial:wp.array(dtype=wp.float64),channels:int):
    i=wp.tid();segment=i//channels;c=i%channels;total=wp.float64(0)
    for k in range(starts[segment],stops[segment]):
        total+=val[k]*x[col[k]*channels+c]
    partial[i]=total


@wp.kernel
def finish_kernel(row_segments:wp.array(dtype=wp.int32),partial:wp.array(dtype=wp.float64),
                  out:wp.array(dtype=wp.float64),channels:int):
    i=wp.tid();row=i//channels;c=i%channels;total=wp.float64(0)
    for s in range(row_segments[row],row_segments[row+1]):total+=partial[s*channels+c]
    out[i]=total


class SegmentedCSR(CSR):
    def __init__(self,matrix,device,chunk=128):
        if isinstance(chunk,bool) or not isinstance(chunk,int) or chunk<1:raise ValueError('positive integer chunk required')
        matrix=sp.csr_matrix(matrix)
        if not np.isfinite(matrix.data).all():raise ValueError('finite CSR required')
        super().__init__(matrix,device)
        starts=[];stops=[];rows=[0]
        for a,b in zip(matrix.indptr[:-1],matrix.indptr[1:]):
            for start in range(int(a),int(b),chunk):starts.append(start);stops.append(min(start+chunk,int(b)))
            rows.append(len(starts))
        self.starts=wp.array(np.array(starts,dtype=np.int32),dtype=wp.int32,device=device)
        self.stops=wp.array(np.array(stops,dtype=np.int32),dtype=wp.int32,device=device)
        self.row_segments=wp.array(np.array(rows,dtype=np.int32),dtype=wp.int32,device=device)
        self.segment_count=len(starts);self.chunk=chunk
        self.bytes+=sum(v.capacity for v in (self.starts,self.stops,self.row_segments))

    def apply(self,x,shape,axis=0):
        if axis!=0:return super().apply(x,shape,axis)
        if shape[0]!=self.shape[1]:raise ValueError('segmented axis dimension mismatch')
        outshape=(self.shape[0],*shape[1:]);channels=int(np.prod(shape[1:]))
        output=wp.empty(int(np.prod(outshape)),dtype=wp.float64,device=self.device)
        partial=wp.empty(self.segment_count*channels,dtype=wp.float64,device=self.device)
        if partial.size:
            wp.launch(segment_kernel,dim=partial.size,inputs=[self.starts,self.stops,self.col,self.val,x,partial,channels],device=self.device)
        wp.launch(finish_kernel,dim=output.size,inputs=[self.row_segments,partial,output,channels],device=self.device)
        return output,outshape


class SegmentedSpace(GPUSpace):
    @classmethod
    def from_existing(cls,original):
        # Shared forward maps are immutable; transpose maps and layout-cache
        # dictionary are privately owned by this derived adapter.
        obj=cls.__new__(cls);obj.__dict__=original.__dict__.copy();obj.layouts=dict(original.layouts)
        obj.old=(original.old[0],SegmentedCSR(sp.csr_matrix(original.space.oldA).T,original.device))
        obj.raw=(original.raw[0],SegmentedCSR(original.space.raw.T,original.device))
        replaced={id(original.old[1]),id(original.raw[1])}
        obj.pairs=[p for p in original.pairs if id(p) not in replaced]+[obj.old[1],obj.raw[1]]
        obj.static_bytes=sum(p.bytes for p in obj.pairs)+obj.ids.capacity+obj.lift.capacity
        obj.cache_key=digest(dict(parent=original.cache_key,transpose_algorithm='segmented-FP64-128-v1'))
        return obj


class SegmentedGPUOperator(ProfiledGPUOperator):
    def __init__(self,*args,**kwargs):
        from .resources import GPUMemoryBudget
        monitor=GPUMemoryBudget(kwargs.get('device','cuda:0'))
        super().__init__(*args,**kwargs);started=time.perf_counter()
        self.memory_budget=monitor
        self.maps=SegmentedSpace.from_existing(self.maps)
        self.layout=self.maps.layout(self.points)
        self.signature=digest(dict(original=self.signature,maps=self.maps.cache_key))
        wp.synchronize_device(self.device);self.build_seconds+=time.perf_counter()-started
        self.memory_budget.observe()


class SegmentedModel(PracticalModel):
    def __init__(self,reduction,*,order=7,device='cuda:0',hold=None):
        if device=='cpu':raise ValueError('segmented GPU model requires a CUDA device')
        super().__init__(reduction,order=order,device='cpu',hold=hold)
        self.operator=SegmentedGPUOperator(FullCoordinates(self.parent),order=order,device=device)
        self.device=device;self.identity=dict(self.identity,device=device);self.signature=digest(self.identity)
