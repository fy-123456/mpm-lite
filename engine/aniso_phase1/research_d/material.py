"""Batched, frozen float64 material evaluation reusing production Warp laws."""
import copy
import numpy as np
import warp as wp
from engine.aniso_phase1.kernels import aniso_dpk1_wp
from engine.aniso_phase1.refinement_gpu import response_kernel


@wp.kernel
def tangent_kernel(F: wp.array(dtype=wp.mat33d), A: wp.array(dtype=wp.mat33d),
                   dF: wp.array(dtype=wp.mat33d), dP: wp.array(dtype=wp.mat33d),
                   mu: wp.float64, lam: wp.float64, kf: wp.float64):
    q=wp.tid()
    dP[q]=aniso_dpk1_wp(F[q],A[q],dF[q],mu,lam,kf,0)


class MaterialGPU:
    def __init__(self,A,params,device='cuda:0',max_batch=65536):
        A=np.asarray(A,dtype=float)
        if A.ndim!=3 or len(A)==0 or A.shape[1:]!=(3,3) or not np.isfinite(A).all(): raise ValueError('finite structure tensors required')
        if max_batch<1: raise ValueError('positive batch capacity required')
        self.A=A.copy(); self.params=copy.deepcopy(params); self.device=wp.get_device(device)
        self.max_batch=max_batch
        n=min(len(A),max_batch)
        self.a,self.f,self.df,self.p,self.dp=[wp.empty(n,dtype=wp.mat33d,device=self.device) for _ in range(5)]
        self.psi=wp.empty(n,dtype=wp.float64,device=self.device)
        self.workspace_bytes=sum(v.capacity for v in (self.a,self.f,self.df,self.p,self.dp,self.psi))

    def evaluate(self,F,dF=None):
        F=np.asarray(F,dtype=float)
        if F.shape!=self.A.shape or not np.isfinite(F).all() or np.any(np.linalg.det(F)<=0):
            raise ValueError('positive finite deformation required')
        # Same spectral valid domain as the production exact tangent.
        if np.linalg.svd(F,compute_uv=False).min()<1e-6: raise ValueError('deformation outside supported spectral domain')
        if dF is not None and (np.shape(dF)!=F.shape or not np.isfinite(dF).all()): raise ValueError('finite direction required')
        energies=[]; stresses=[]; tangents=[]; p=self.params
        for start in range(0,len(F),self.max_batch):
            stop=min(start+self.max_batch,len(F)); n=stop-start
            self.a[:n].assign(self.A[start:stop]); self.f[:n].assign(F[start:stop])
            wp.launch(response_kernel,dim=n,inputs=[self.f,self.a,p.mu,p.lam,p.k_f,self.psi,self.p],device=self.device)
            energies.append(self.psi[:n].numpy()); stresses.append(self.p[:n].numpy())
            if dF is not None:
                self.df[:n].assign(np.asarray(dF[start:stop]))
                wp.launch(tangent_kernel,dim=n,inputs=[self.f,self.a,self.df,self.dp,p.mu,p.lam,p.k_f],device=self.device)
                tangents.append(self.dp[:n].numpy())
        return dict(energy=np.concatenate(energies),stress=np.concatenate(stresses),
                    tangent=None if dF is None else np.concatenate(tangents),
                    batches=(len(F)+self.max_batch-1)//self.max_batch,workspace_bytes=self.workspace_bytes)
