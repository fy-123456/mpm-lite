"""Frozen material-domain GPU probe using the production grouped kernels.

This is a total-reference material diagnostic, not a particle transfer solver.
Maps, volumes and direction moments stay fixed during all evaluations.
"""
import time
import numpy as np
import warp as wp
from scipy import sparse
from engine.types import real, vec3, mat33, mat99
from .aligned_quadrature import evaluate
from .grouped_quadrature import sample_residual,sample_tangent,sample_energy


def direction_data(points,field):
    n=len(points);A=np.zeros((n,3,3));M=np.zeros((n,9,9))
    if field=='crossed':
        A[:,0,0]=A[:,1,1]=.5;M[:,0,0]=M[:,4,4]=.5
    else:
        theta=np.zeros(n) if field=='uniform' else (points[:,1]-.4375)/.125*np.pi/2
        a=np.column_stack([np.cos(theta),np.sin(theta),np.zeros(n)])
        A=np.einsum('qi,qj->qij',a,a);flat=A.reshape(-1,9);M=np.einsum('qi,qj->qij',flat,flat)
    if field not in ('uniform','crossed','smooth'):raise ValueError(field)
    return A,M


class FrozenMaterial:
    def __init__(self,blend,rule,device,field='uniform',guard=lambda:None,permutation=None):
        start=time.perf_counter();self.blend=blend;self.rule=rule;self.device=device;self.n=len(blend.nodes)
        self.perm=np.arange(self.n) if permutation is None else np.asarray(permutation)
        inverse=np.argsort(self.perm);parts=[[],[],[]]
        for i in range(0,len(rule.weights),512):
            guard();_,G=evaluate(blend,rule.points[i:i+512])
            for d in range(3):parts[d].append(G[d])
        self.G=[sparse.vstack(p,format='csr') for p in parts]
        union=sum(abs(g) for g in self.G);width=np.diff(union.indptr).max()
        ids=np.full((len(rule.weights),width),-1,dtype=np.int32);B=np.zeros((len(rule.weights),width,3))
        for q in range(len(rule.weights)):
            cols=union.indices[union.indptr[q]:union.indptr[q+1]];ids[q,:len(cols)]=inverse[cols]
            for d in range(3):B[q,:len(cols),d]=self.G[d][q,cols].toarray().ravel()
        self.host_map_seconds=time.perf_counter()-start;A,M=direction_data(rule.points,field);self.host_A=A;self.host_M=M
        self.ids=wp.array(ids,dtype=int,device=device);self.B=wp.array(B,dtype=vec3,device=device)
        self.F0=wp.array(np.tile(np.eye(3),(len(rule.weights),1,1)),dtype=mat33,device=device)
        self.A=wp.array(A,dtype=mat33,device=device);self.M=wp.array(M,dtype=mat99,device=device)
        self.V=wp.array(rule.weights,dtype=real,device=device);self.F=wp.zeros_like(self.F0)
        self.v=wp.zeros(self.n,dtype=vec3,device=device);self.p=wp.zeros_like(self.v);self.out=wp.zeros_like(self.v)
        self.invalid=wp.zeros(1,dtype=int,device=device);self.energy_out=wp.zeros(2,dtype=real,device=device)
        self.host_v=wp.empty(self.n,dtype=vec3,device='cpu');self.host_p=wp.empty_like(self.host_v)
        self.map_bytes=int(ids.nbytes+B.nbytes)
        self.array_bytes=int(ids.nbytes+B.nbytes+len(rule.weights)*(9*8*3+81*8+8)+self.n*3*8*3+20)
        self.host_csr_bytes=sum(g.data.nbytes+g.indices.nbytes+g.indptr.nbytes for g in self.G)
        wp.synchronize_device(device);self.build_seconds=time.perf_counter()-start

    def residual_kernel(self):
        self.out.zero_();self.invalid.zero_()
        wp.launch(sample_residual,dim=len(self.rule.weights),inputs=[self.ids,self.B,self.v,self.F0,self.A,self.M,self.V,
            self.F,self.out,self.invalid,1.,10.,20.,200.],device=self.device)

    def energy_kernel(self):
        self.energy_out.zero_()
        wp.launch(sample_energy,dim=len(self.rule.weights),inputs=[self.F,self.A,self.M,self.V,self.energy_out,10.,20.,200.],device=self.device)

    def tangent_kernel(self,pd=False):
        self.out.zero_()
        wp.launch(sample_tangent,dim=len(self.rule.weights),inputs=[self.ids,self.B,self.p,self.F,self.A,self.M,self.V,self.out,
            1.,10.,20.,200.,int(pd)],device=self.device)

    def state(self,u):
        self.host_v.numpy()[:]=np.asarray(u).reshape(self.n,3)[self.perm];wp.copy(self.v,self.host_v)
        self.residual_kernel();r=self.out.numpy()[np.argsort(self.perm)].ravel()
        if self.invalid.numpy()[0]:return float('inf'),r
        self.energy_kernel();return float(self.energy_out.numpy()[0]),r

    def action(self,p,pd=False):
        self.host_p.numpy()[:]=np.asarray(p).reshape(self.n,3)[self.perm];wp.copy(self.p,self.host_p)
        self.tangent_kernel(pd);return self.out.numpy()[np.argsort(self.perm)].ravel()

    def host_state(self,u):
        from .material_snapshot import response
        from .constitutive import AnisotropicMaterialParams
        F=np.eye(3)+np.stack([g@np.asarray(u).reshape(self.n,3) for g in self.G],axis=2)
        E,P,_=response(F,self.host_A,self.host_M,AnisotropicMaterialParams(10,20,200))
        r=sum(g.T@(self.rule.weights[:,None]*P[:,:,d]) for d,g in enumerate(self.G))
        return float(self.rule.weights@E),r.ravel()

    def timings(self,repeats=3,batch=20):
        # State must be valid; warm every kernel, then measure synchronized batches.
        for fun in (self.residual_kernel,self.energy_kernel,self.tangent_kernel):fun()
        wp.synchronize_device(self.device);values={}
        for name,fun in [('residual',self.residual_kernel),('energy',self.energy_kernel),('matvec',self.tangent_kernel)]:
            times=[]
            for _ in range(repeats):
                wp.synchronize_device(self.device);start=time.perf_counter()
                for _ in range(batch):fun()
                wp.synchronize_device(self.device);times.append((time.perf_counter()-start)/batch)
            values[name]=dict(median_seconds=float(np.median(times)),min_seconds=min(times),max_seconds=max(times))
        return values


def checked_pcg(operator, rhs, precondition, rtol=1e-6, maxiter=200):
    """Small host diagnostic CG with per-iteration curvature checks."""
    x=np.zeros_like(rhs);r=rhs.copy();z=precondition@r;p=z.copy();rz=float(r@z)
    target=rtol*np.linalg.norm(rhs)
    if np.linalg.norm(r)<=target:return x,0,0
    for iteration in range(1,maxiter+1):
        Ap=operator@p;curvature=float(p@Ap)
        if not np.isfinite(curvature) or curvature<=0 or rz<=0:return x,-1,iteration
        alpha=rz/curvature;x+=alpha*p;r-=alpha*Ap
        if np.linalg.norm(r)<=target:return x,0,iteration
        z=precondition@r;new=float(r@z)
        if not np.isfinite(new) or new<=0:return x,-2,iteration
        p=z+(new/rz)*p;rz=new
    return x,1,maxiter


class MemoryMonitor:
    """Sample this process's GPU residency, separate from active array bytes."""
    def __enter__(self):
        import threading
        self.stop=threading.Event();self.observations=[];self.errors=[]
        self.thread=threading.Thread(target=self._sample,daemon=True);self.thread.start();return self

    def _sample(self):
        import os,subprocess
        while not self.stop.is_set():
            try:
                rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
                values=[float(line.split(',')[1]) for line in rows.splitlines() if line.split(',')[0].strip()==str(os.getpid())]
                if values:self.observations.append(sum(values))
            except Exception as error:self.errors.append(str(error))
            self.stop.wait(.25)

    def __exit__(self,*args):self.stop.set();self.thread.join(timeout=4)

    def result(self):
        import resource
        return dict(gpu_process_observed_peak_mib=max(self.observations) if self.observations else None,
            gpu_memory_samples=len(self.observations),cpu_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            errors=self.errors,scope='whole process including memory pools and sparse ordering setup; sampled, not exact peak')
