"""Fixed SPD overlapping Schwarz + optional separable and coarse corrections.

These corrections only act on residuals; they are never physical stiffness.
"""
from __future__ import annotations
import itertools
import time
import numpy as np
import scipy.linalg as la
import warp as wp


def tensor_blocks(shape,width=3,overlap=1):
    if width<1 or overlap<0 or overlap>=width: raise ValueError('0 <= overlap < width required')
    n=int(np.prod(shape)); blocks=[]
    starts=[range(0,k,max(1,width-overlap)) for k in shape]
    for start in itertools.product(*starts):
        ids=np.ravel_multi_index(np.array(np.meshgrid(*[np.arange(s,min(s+width,k))
            for s,k in zip(start,shape)],indexing='ij')).reshape(3,-1),shape)
        blocks.append(np.concatenate([ids+a*n for a in range(3)]))
    return blocks


class Schwarz:
    def __init__(self,A,blocks,*,base=None,coarse=None,max_block_dofs=768):
        start=time.perf_counter(); self.base=base; self.blocks=[]; self.size=A.shape[0]
        counts=np.zeros(self.size)
        for ids in blocks:
            ids=np.asarray(ids,dtype=np.int64)
            if len(ids)>max_block_dofs: raise MemoryError('local factor exceeds declared size budget')
            if len(ids)==0 or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=self.size):
                raise ValueError('invalid local restriction')
            counts[ids]+=1
        if np.any(counts==0): raise ValueError('Schwarz blocks must cover every free dof')
        for ids in blocks:
            ids=np.asarray(ids,dtype=np.int64); w=1/np.sqrt(counts[ids])
            block=A[ids][:,ids].toarray()
            if not np.allclose(block,block.T,rtol=1e-10,atol=1e-12): raise ValueError('nonsymmetric local block')
            L=la.cho_factor(block,lower=True)  # fails instead of adding a diagonal shift
            inverse=la.cho_solve(L,np.eye(len(ids)))
            self.blocks.append((ids,w,inverse))
        self.coarse=None
        if coarse is not None:
            Z=np.asarray(coarse,dtype=float)
            if Z.shape[0]!=self.size: raise ValueError('coarse-space size mismatch')
            gram=Z.T@(A@Z); factor=la.cho_factor(gram,lower=True)
            self.coarse=(Z,factor)
        self.build_seconds=time.perf_counter()-start
        self.memory_bytes=sum(ids.nbytes+w.nbytes+inv.nbytes for ids,w,inv in self.blocks)
        if self.coarse: self.memory_bytes+=sum(v.nbytes if isinstance(v,np.ndarray) else v[0].nbytes for v in self.coarse)

    def __call__(self,r):
        out=np.zeros_like(r) if self.base is None else self.base(r)
        for ids,w,inv in self.blocks: out[ids]+=w*(inv@(w*r[ids]))
        if self.coarse:
            Z,factor=self.coarse; out+=Z@la.cho_solve(factor,Z.T@r)
        return out


@wp.kernel
def batch_block(ids: wp.array2d(dtype=wp.int32), weights: wp.array2d(dtype=wp.float64),
                inverse: wp.array3d(dtype=wp.float64), r: wp.array(dtype=wp.float64),
                out: wp.array(dtype=wp.float64), width: int):
    block,i=wp.tid()
    if ids[block,i]>=0:
        value=wp.float64(0.)
        for j in range(width):
            if ids[block,j]>=0:
                value+=inverse[block,i,j]*weights[block,j]*r[ids[block,j]]
        wp.atomic_add(out,ids[block,i],weights[block,i]*value)


class BatchedSchwarz:
    """Local solves grouped by size, bounded host/device staging, float64.

    All factors remain resident after construction. batch_bytes bounds each
    padded batch; total_bytes independently bounds the whole factor cache.
    """
    def __init__(self,cpu,device='cuda:0',batch_bytes=32<<20,total_bytes=256<<20):
        if cpu.base is not None or cpu.coarse is not None:
            raise ValueError('GPU batch represents local blocks only; compose other terms explicitly')
        self.device=wp.get_device(device); self.size=cpu.size; self.batches=[]; self.memory_bytes=0
        groups={}
        for item in cpu.blocks: groups.setdefault(len(item[0]),[]).append(item)
        projected=sum(len(items)*(n*n*8+n*12) for n,items in groups.items())
        if projected>total_bytes: raise MemoryError('total local factors exceed declared device budget')
        for n,items in groups.items():
            one=n*n*8+n*12
            if one>batch_bytes: raise MemoryError('one local factor exceeds batch budget')
            limit=max(1,batch_bytes//one)
            for start in range(0,len(items),limit):
                chunk=items[start:start+limit]
                ids=wp.array(np.stack([q[0] for q in chunk]).astype(np.int32),dtype=wp.int32,device=self.device)
                w=wp.array(np.stack([q[1] for q in chunk]),dtype=wp.float64,device=self.device)
                inv=wp.array(np.stack([q[2] for q in chunk]),dtype=wp.float64,device=self.device)
                self.batches.append((ids,w,inv,n)); self.memory_bytes+=ids.capacity+w.capacity+inv.capacity

    def apply(self,r,out):
        out.zero_()
        for ids,w,inv,n in self.batches:
            wp.launch(batch_block,dim=ids.shape,inputs=[ids,w,inv,r,out,n],device=self.device)


@wp.kernel
def local_values(ids: wp.array2d(dtype=wp.int32), weights: wp.array2d(dtype=wp.float64),
                 inverse: wp.array3d(dtype=wp.float64), r: wp.array(dtype=wp.float64),
                 out: wp.array2d(dtype=wp.float64), width: int):
    block,i=wp.tid();value=wp.float64(0.)
    for j in range(width):
        value+=inverse[block,i,j]*weights[block,j]*r[ids[block,j]]
    out[block,i]=weights[block,i]*value


def local_corrections(batch,r):
    """Return individual weighted local corrections, including D2H explicitly.

    Order is grouped by local size, then original block order. Assembly of a
    space and its orthogonalization remain on CPU and must be timed by callers.
    """
    corrections=[]
    for ids,w,inv,n in batch.batches:
        values=wp.empty(ids.shape,dtype=wp.float64,device=batch.device)
        wp.launch(local_values,dim=ids.shape,inputs=[ids,w,inv,r,values,n],device=batch.device)
        corrections.extend(zip(ids.numpy(),values.numpy()))
    return corrections
