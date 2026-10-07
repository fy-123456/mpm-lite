"""Float64 tensor-axis backend and device-resident guarded PCG.

Tensor/BoxElastic defines the mathematics. This module only changes execution.
All vector storage is component first. No implicit CPU solve or precision change.
"""
from __future__ import annotations
import time
import numpy as np
import warp as wp
from .cpu import SolveInfo
from .identity import digest


@wp.kernel
def sparse_axis(row: wp.array(dtype=wp.int32), col: wp.array(dtype=wp.int32),
                val: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
                y: wp.array(dtype=wp.float64), width: int, stride: int):
    i = wp.tid(); r = (i//stride)%width; base = i-r*stride
    acc = wp.float64(0.)
    for j in range(row[r],row[r+1]): acc += val[j]*x[base+col[j]*stride]
    y[i] = acc


@wp.kernel
def dense_axis(A: wp.array2d(dtype=wp.float64), x: wp.array(dtype=wp.float64),
               y: wp.array(dtype=wp.float64), width: int, stride: int):
    i = wp.tid(); r = (i//stride)%width; base = i-r*stride
    acc = wp.float64(0.)
    for j in range(width): acc += A[r,j]*x[base+j*stride]
    y[i] = acc


@wp.kernel
def mix_components(H: wp.mat33d, x: wp.array(dtype=wp.float64),
                   out: wp.array(dtype=wp.float64), n: int):
    i = wp.tid(); a = i//n; node = i%n
    out[i] += H[a,0]*x[node]+H[a,1]*x[n+node]+H[a,2]*x[2*n+node]


@wp.kernel
def divide(x: wp.array(dtype=wp.float64), den: wp.array(dtype=wp.float64),
           y: wp.array(dtype=wp.float64)):
    i = wp.tid(); y[i] = x[i]/den[i]


@wp.kernel
def csr_apply(row: wp.array(dtype=wp.int32), col: wp.array(dtype=wp.int32),
              val: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
              y: wp.array(dtype=wp.float64)):
    i = wp.tid(); s = wp.float64(0.)
    for j in range(row[i],row[i+1]): s += val[j]*x[col[j]]
    y[i] = s


class TensorGPU:
    """Immutable numerical data, private scratch; never share between streams."""
    def __init__(self, problem, device='cuda:0'):
        self.device = wp.get_device(device)
        self.problem_key = problem.key
        self.key = digest([problem.key,str(self.device),str(self.device.context),str(wp.__version__)])
        self.size = problem.size; self.shape = problem.free_shape
        self.closed = False
        self.arrays = []; self.terms = []
        self.tmp1 = self._empty(self.size); self.tmp2 = self._empty(self.size)
        # Restrict every axis factor before transfer: homogeneous constraints
        # are exactly eliminated on input AND output, including artificial faces.
        grams = []
        for (_,gs), sl in zip(problem.op.axes,problem.op.slices):
            grams.append([self._csr(g[sl,sl].tocsr()) for g in (*gs,gs[2].T)])
        H = problem.H.reshape(3,3,3,3)
        for i in range(3):
            for j in range(3):
                h = H[:,i,:,j]
                if not np.any(h): continue
                factors = [grams[k][1 if i==j==k else 2 if k==i and i!=j
                                    else 3 if k==j and i!=j else 0] for k in range(3)]
                self.terms.append((wp.mat33d(h),factors))
        self.V = [self._array(v) for v in problem.op.vectors]
        self.VT = [self._array(v.T) for v in problem.op.vectors]
        self.den = self._array(problem.op.den.transpose(3,0,1,2).ravel())

    def _array(self,a,dtype=wp.float64):
        v = wp.array(np.ascontiguousarray(a),dtype=dtype,device=self.device)
        self.arrays.append(v); return v

    def _empty(self,n):
        v=wp.empty(n,dtype=wp.float64,device=self.device); self.arrays.append(v); return v

    def _csr(self,A):
        return (self._array(A.indptr.astype(np.int32),wp.int32),
                self._array(A.indices.astype(np.int32),wp.int32),self._array(A.data))

    @property
    def memory_bytes(self): return sum(a.capacity for a in self.arrays)

    def _check(self):
        if self.closed: raise RuntimeError('backend was closed; rebuild required')

    def apply(self,x,out):
        self._check(); out.zero_()
        for H, factors in self.terms:
            source=x
            for k,factor in enumerate(factors):
                dest=self.tmp1 if k%2==0 else self.tmp2
                wp.launch(sparse_axis,dim=self.size,inputs=[*factor,source,dest,self.shape[k],
                    int(np.prod(self.shape[k+1:]))],device=self.device)
                source=dest
            wp.launch(mix_components,dim=self.size,inputs=[H,source,out,self.size//3],device=self.device)

    def precondition(self,x,out):
        self._check(); source=x
        for k in range(3):
            dest=self.tmp1 if k%2==0 else self.tmp2
            wp.launch(dense_axis,dim=self.size,inputs=[self.VT[k],source,dest,self.shape[k],
                int(np.prod(self.shape[k+1:]))],device=self.device); source=dest
        wp.launch(divide,dim=self.size,inputs=[source,self.den,self.tmp2],device=self.device)
        source=self.tmp2
        for k in range(3):
            dest=out if k==2 else self.tmp1 if k==0 else self.tmp2
            wp.launch(dense_axis,dim=self.size,inputs=[self.V[k],source,dest,self.shape[k],
                int(np.prod(self.shape[k+1:]))],device=self.device); source=dest

    def host_apply(self,x,precondition=False):
        a=wp.array(np.asarray(x),dtype=wp.float64,device=self.device)
        b=wp.empty_like(a)
        (self.precondition if precondition else self.apply)(a,b)
        return b.numpy()

    def close(self):
        self.closed=True; self.terms=[]; self.V=[]; self.VT=[]; self.arrays=[]
        self.den=self.tmp1=self.tmp2=None


class CSRGPU:
    """Explicit assembled fallback for spatially varying tangents; cost is reported."""
    def __init__(self,A,device='cuda:0'):
        A=A.tocsr(); self.size=A.shape[0]; self.device=wp.get_device(device)
        if A.shape[1]!=self.size or not np.isfinite(A.data).all(): raise ValueError('finite square matrix required')
        if np.any(A.diagonal()<=0): raise ValueError('positive diagonal required for Jacobi')
        self.row=wp.array(A.indptr.astype(np.int32),dtype=wp.int32,device=self.device)
        self.col=wp.array(A.indices.astype(np.int32),dtype=wp.int32,device=self.device)
        self.val=wp.array(A.data,dtype=wp.float64,device=self.device)
        self.den=wp.array(A.diagonal(),dtype=wp.float64,device=self.device)
        self.key=digest([A.indptr,A.indices,A.data,str(self.device)]); self.closed=False

    @property
    def memory_bytes(self): return sum(a.capacity for a in (self.row,self.col,self.val,self.den))
    def apply(self,x,out):
        if self.closed: raise RuntimeError('closed backend')
        wp.launch(csr_apply,dim=self.size,inputs=[self.row,self.col,self.val,x,out],device=self.device)
    def precondition(self,x,out):
        if self.closed: raise RuntimeError('closed backend')
        wp.launch(divide,dim=self.size,inputs=[x,self.den,out],device=self.device)
    def host_apply(self,x):
        a=wp.array(x,dtype=wp.float64,device=self.device); b=wp.empty_like(a); self.apply(a,b); return b.numpy()
    def close(self):
        self.closed=True; self.row=self.col=self.val=self.den=None


@wp.kernel
def dot_chunks(a: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64),
               partial: wp.array(dtype=wp.float64), n: int):
    k=wp.tid(); s=wp.float64(0.)
    for i in range(k*128,wp.min((k+1)*128,n)): s+=a[i]*b[i]
    partial[k]=s


@wp.kernel
def sum_dot(partial: wp.array(dtype=wp.float64), state: wp.array(dtype=wp.float64), index: int):
    s=wp.float64(0.)
    for i in range(partial.shape[0]): s+=partial[i]
    state[index]=s


@wp.kernel
def initialize(state: wp.array(dtype=wp.float64), flags: wp.array(dtype=wp.int32),
               trace: wp.array(dtype=wp.float64), rtol: wp.float64, atol: wp.float64):
    flags[0]=0; flags[1]=0
    state[4]=wp.max(atol*atol,rtol*rtol*state[0]); state[3]=state[1]
    trace[0]=wp.sqrt(state[0])
    if not wp.isfinite(state[0]): flags[0]=5
    elif state[0]<=state[4]: flags[0]=1
    elif not wp.isfinite(state[1]) or state[1]<=wp.float64(0.): flags[0]=3


@wp.kernel
def curvature(state: wp.array(dtype=wp.float64), flags: wp.array(dtype=wp.int32)):
    if flags[0]==0:
        if not wp.isfinite(state[2]): flags[0]=6
        elif state[2]<=wp.float64(0.): flags[0]=2
        elif not wp.isfinite(state[3]) or state[3]<=wp.float64(0.): flags[0]=3
        else: state[5]=state[3]/state[2]


@wp.kernel
def update_xr(x: wp.array(dtype=wp.float64), r: wp.array(dtype=wp.float64),
              p: wp.array(dtype=wp.float64), Ap: wp.array(dtype=wp.float64),
              state: wp.array(dtype=wp.float64), flags: wp.array(dtype=wp.int32)):
    i=wp.tid()
    if flags[0]==0:
        x[i]+=state[5]*p[i]; r[i]-=state[5]*Ap[i]


@wp.kernel
def finish_iteration(state: wp.array(dtype=wp.float64), flags: wp.array(dtype=wp.int32),
                     trace: wp.array(dtype=wp.float64), maxiter: int):
    if flags[0]==0:
        flags[1]+=1; trace[flags[1]]=wp.sqrt(state[0])
        if not wp.isfinite(state[0]): flags[0]=5
        elif state[0]<=state[4]: flags[0]=1
        elif not wp.isfinite(state[1]) or state[1]<=wp.float64(0.): flags[0]=3
        elif flags[1]>=maxiter: flags[0]=4
        else:
            state[6]=state[1]/state[3]; state[3]=state[1]


@wp.kernel
def update_p(p: wp.array(dtype=wp.float64), z: wp.array(dtype=wp.float64),
             state: wp.array(dtype=wp.float64), flags: wp.array(dtype=wp.int32)):
    i=wp.tid()
    if flags[0]==0: p[i]=z[i]+state[6]*p[i]


@wp.kernel
def difference(b: wp.array(dtype=wp.float64), Ax: wp.array(dtype=wp.float64), r: wp.array(dtype=wp.float64)):
    i=wp.tid(); r[i]=b[i]-Ax[i]


class ResidentPCG:
    """Reusable workspace/graph owned by ONE immutable backend instance.

    Scalars stay on device; poll status once per chunk. Device guards freeze
    accepted/failed iterates within a chunk. True residual is always recomputed
    from the exact operator before returning success. A graph failure falls back
    to launches with the same guards and is included in diagnostics.
    """
    def __init__(self,backend,*,maxiter=2000,check_every=8,use_graph=True):
        if maxiter<1 or check_every<1: raise ValueError('positive iteration/check budgets required')
        self.backend=backend; self.device=backend.device; self.size=backend.size
        self.maxiter=maxiter; self.check_every=check_every; self.use_graph=use_graph
        self.backend_key=backend.key; self.closed=False; self.graph=None; self.graph_attempted=False
        self.graph_failure=None; self.capture_seconds=0.; self.warmed=False
        self.x,self.r,self.z,self.p,self.Ap,self.b=[wp.zeros(self.size,dtype=wp.float64,device=self.device) for _ in range(6)]
        self.state=wp.zeros(7,dtype=wp.float64,device=self.device)
        self.flags=wp.zeros(2,dtype=wp.int32,device=self.device)
        self.partial=wp.empty((self.size+127)//128,dtype=wp.float64,device=self.device)
        self.trace=wp.empty(maxiter+1,dtype=wp.float64,device=self.device)

    @property
    def memory_bytes(self):
        return sum(a.capacity for a in (self.x,self.r,self.z,self.p,self.Ap,self.b,self.state,self.flags,self.partial,self.trace))

    def dot(self,a,b,index):
        wp.launch(dot_chunks,dim=self.partial.shape[0],inputs=[a,b,self.partial,self.size],device=self.device)
        wp.launch(sum_dot,dim=1,inputs=[self.partial,self.state,index],device=self.device)

    def iteration(self):
        self.backend.apply(self.p,self.Ap); self.dot(self.p,self.Ap,2)
        wp.launch(curvature,dim=1,inputs=[self.state,self.flags],device=self.device)
        wp.launch(update_xr,dim=self.size,inputs=[self.x,self.r,self.p,self.Ap,self.state,self.flags],device=self.device)
        self.backend.precondition(self.r,self.z)
        self.dot(self.r,self.r,0); self.dot(self.r,self.z,1)
        wp.launch(finish_iteration,dim=1,inputs=[self.state,self.flags,self.trace,self.maxiter],device=self.device)
        wp.launch(update_p,dim=self.size,inputs=[self.p,self.z,self.state,self.flags],device=self.device)

    def solve(self,b,*,rtol=1e-7,atol=1e-12,diagnostic_true_residuals=False):
        if self.closed or self.backend.closed or self.backend.key!=self.backend_key:
            raise RuntimeError('stale or closed solver workspace')
        b=np.asarray(b,dtype=float)
        if b.shape!=(self.size,) or not np.isfinite(b).all(): raise ValueError('finite matching rhs required')
        if not np.isfinite([rtol,atol]).all() or min(rtol,atol)<0 or max(rtol,atol)==0:
            raise ValueError('positive tolerance required')
        start=time.perf_counter(); syncs=0; poll_bytes=0; launches=0; true_history=[]; exact_calls=0
        with wp.ScopedDevice(self.device):
            # Warm every kernel before graph capture; no trial state is committed.
            if not self.warmed:
                self.flags.assign(np.array([1,0],np.int32)); self.iteration()
                wp.launch(difference,dim=self.size,inputs=[self.b,self.Ap,self.r],device=self.device)
                self.warmed=True
            self.b.assign(b); self.x.zero_(); wp.copy(self.r,self.b)
            self.backend.precondition(self.r,self.z); wp.copy(self.p,self.z)
            self.dot(self.r,self.r,0); self.dot(self.r,self.z,1)
            wp.launch(initialize,dim=1,inputs=[self.state,self.flags,self.trace,rtol,atol],device=self.device)
            flags=self.flags.numpy(); syncs+=1; poll_bytes+=flags.nbytes
            if self.use_graph and self.device.is_cuda and not self.graph_attempted:
                capture_start=time.perf_counter(); self.graph_attempted=True
                try:
                    with wp.ScopedCapture(device=self.device) as capture:
                        for _ in range(self.check_every): self.iteration()
                    self.graph=capture.graph
                except Exception as error:
                    self.graph_failure=f'{type(error).__name__}: {error}'
                self.capture_seconds=time.perf_counter()-capture_start
            while flags[0]==0:
                if self.graph is not None: wp.capture_launch(self.graph)
                else:
                    for _ in range(self.check_every): self.iteration()
                launches+=self.check_every
                flags=self.flags.numpy(); syncs+=1; poll_bytes+=flags.nbytes
                if diagnostic_true_residuals:
                    # Profiling-only observation: preserve the recursive residual.
                    # z/Ap and the curvature slot are overwritten by the next iteration.
                    self.backend.apply(self.x,self.Ap); exact_calls+=1
                    wp.launch(difference,dim=self.size,inputs=[self.b,self.Ap,self.z],device=self.device)
                    self.dot(self.z,self.z,2)
                    observed=self.state.numpy(); syncs+=1; poll_bytes+=observed.nbytes
                    true_history.append(dict(iteration=int(flags[1]),norm=float(np.sqrt(observed[2]))))
            self.backend.apply(self.x,self.Ap)
            wp.launch(difference,dim=self.size,inputs=[self.b,self.Ap,self.r],device=self.device)
            self.dot(self.r,self.r,0)
            state=self.state.numpy(); syncs+=1; poll_bytes+=state.nbytes
            x=self.x.numpy(); syncs+=1
            trace=self.trace[:int(flags[1])+1].numpy().tolist(); syncs+=1
        normb=float(np.linalg.norm(b)); target=max(atol,rtol*normb)
        error=float(np.sqrt(max(0.,state[0]))) if np.isfinite(state[0]) else float('inf')
        status={1:'converged',2:'nonpositive_curvature',3:'invalid_preconditioner',
                4:'iteration_limit',5:'nonfinite_residual',6:'nonfinite_curvature'}[int(flags[0])]
        if not np.isfinite(error): status='nonfinite_residual'
        elif status=='converged' and error>target: status='true_residual_failure'
        info=SolveInfo(status,int(flags[1]),error,target,error/max(normb,atol,np.finfo(float).tiny),
            trace,true_history+[dict(iteration=int(flags[1]),norm=error)],launches+1+exact_calls,launches+1,
            int(flags[0])==2,time.perf_counter()-start,
            dict(device=str(self.device),dtype='float64',host_readbacks=syncs,
                 h2d_bytes=b.nbytes,d2h_bytes=poll_bytes+x.nbytes+8*len(trace),
                 graph_used=self.graph is not None,graph_failure=self.graph_failure,
                 capture_seconds=self.capture_seconds,workspace_bytes=self.memory_bytes,
                 backend_bytes=self.backend.memory_bytes,executed_iteration_slots=launches,
                 residual_history=('exact operator at chunk checkpoints and final' if diagnostic_true_residuals else 'recursive each iteration; exact operator at final acceptance')))
        return x,info

    def close(self):
        self.closed=True; self.graph=None
        for name in ('x','r','z','p','Ap','b','state','flags','partial','trace'): setattr(self,name,None)


class BackendCache:
    """Bounded cache scoped to a caller-owned instance token and device context.

    Dependency identity includes topology/coordinates, basis, BC/lifting,
    material tangent, precision, state, owner and device. Releasing destroys the
    workspace; no graph survives release/rebuild. Caller must serialize use.
    """
    def __init__(self,max_entries=2):
        if max_entries<1: raise ValueError('positive cache capacity required')
        self.max_entries=max_entries; self.entries={}; self.hits=0; self.misses=0
    def acquire(self,problem,owner,device='cuda:0'):
        dev=wp.get_device(device)
        key=(owner,problem.key,str(dev),str(getattr(dev,'uuid',dev)),str(dev.context),wp.__version__)
        if key in self.entries:
            backend=self.entries.pop(key)
            if not backend.closed:
                self.entries[key]=backend; self.hits+=1; return backend
        self.misses+=1; backend=TensorGPU(problem,dev)
        if len(self.entries)>=self.max_entries:
            old=next(iter(self.entries)); self.entries.pop(old).close()
        self.entries[key]=backend; return backend
    def release(self,owner=None):
        for key in list(self.entries):
            if owner is None or key[0]==owner: self.entries.pop(key).close()
