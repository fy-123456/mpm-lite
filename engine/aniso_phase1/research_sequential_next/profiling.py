"""Synchronized coarse cost partitions of the unchanged GPU material operator."""
from __future__ import annotations
from collections import defaultdict
import time
import numpy as np
import warp as wp
from ..research_d.stage2.gpu_operator import GPUOperator,Linearization,response_kernel,reduce_kernel
from ..research_d.stage2.contracts import array_digest
from ..research_d.identity import digest
from ..research_sequential.condensation import FullCoordinates
from .model import PracticalModel


class ProfiledGPUOperator(GPUOperator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.timings=defaultdict(float);self.counts=defaultdict(int)

    def _stamp(self,name,started):
        wp.synchronize_device(self.device)
        self.timings[name]+=time.perf_counter()-started;self.counts[name]+=1
        if hasattr(self,'memory_budget'):self.memory_budget.observe()
        return time.perf_counter()

    def prepare(self,q):
        begun=time.perf_counter();s=self.space;m=self.maps;p=self.params
        if hasattr(self,'memory_budget'):
            self.memory_budget.observe(self.count*512+int(np.prod(s.shape))*256)
        full=s.expand(q);nodal=m.nodes(m.expand(q));F=m.gradient(nodal,self.layout,identity=True)
        t=self._stamp('mapping_and_upload',begun)
        Q=wp.empty(self.count,dtype=wp.mat33d,device=self.device);c=wp.empty(self.count,dtype=wp.vec3d,device=self.device)
        psi=wp.empty(self.count,dtype=wp.float64,device=self.device);P=wp.empty_like(F)
        det=wp.empty_like(psi);bad=wp.zeros(1,dtype=wp.int32,device=self.device)
        t=self._stamp('response_allocations',t)
        wp.launch(response_kernel,dim=self.count,inputs=[F,self.A,p.mu,p.lam,p.k_f,psi,P,Q,c,det,bad],device=self.device)
        t=self._stamp('constitutive_kernel',t)
        invalid=int(bad.numpy()[0]);t=self._stamp('validity_download',t)
        if invalid:raise ValueError('non-positive/non-finite detF or unsupported singular value')
        force_device=m.gradient_adjoint(P,self.layout,self.weights)
        t=self._stamp('gradient_adjoint',t)
        force=force_device.numpy().reshape(s.ndof,3)
        t=self._stamp('force_download',t)
        n=(self.count+255)//256;energies=wp.empty(n,dtype=wp.float64,device=self.device);mins=wp.empty_like(energies)
        wp.launch(reduce_kernel,dim=n,inputs=[psi,self.weights,det,energies,mins,self.count],device=self.device)
        t=self._stamp('energy_reduction',t)
        material_U=float(energies.numpy().sum());minimum=float(mins.numpy().min())
        t=self._stamp('scalars_download',t)
        material_force=force.copy();y=s.reference[:s.n]+full[:s.n];ks=s.Ks@y
        stabilization=.5*float(np.sum(y*ks));force[:s.n]+=ks
        r=dict(U=material_U+stabilization,energy_J=material_U+stabilization,material_U=material_U,
            stabilization_U=stabilization,full_force=force,material_force=material_force,force=s.restrict(force),
            min_detF=minimum,material_calls=self.count,rule_signature=self.rule.signature)
        if not np.isfinite(force).all() or not np.isfinite(r['U']):raise ValueError('non-finite material response')
        state_id=array_digest(q);self._stamp('host_stabilization_and_identity',t)
        r['response_seconds']=time.perf_counter()-begun
        return Linearization(self.signature,state_id,F,Q,c,r)

    def action(self,linearization,direction):
        begun=time.perf_counter();value=super().action(linearization,direction)
        self._stamp('exact_tangent_total',begun)
        return value

    def reset_profile(self):
        self.timings.clear();self.counts.clear()


class ProfiledModel(PracticalModel):
    def __init__(self,reduction,*,order=7,device='cuda:0',hold=None):
        if device=='cpu':raise ValueError('this profiling model is for the GPU operator')
        # Construct the shared host mass/boundary state once, then install the
        # explicitly selected derived operator; no old module is monkeypatched.
        super().__init__(reduction,order=order,device='cpu',hold=hold)
        self.operator=ProfiledGPUOperator(FullCoordinates(self.parent),order=order,device=device)
        self.device=device;self.identity=dict(self.identity,device=device);self.signature=digest(self.identity)
