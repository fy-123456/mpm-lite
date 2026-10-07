"""Avoid material eigenvector output storage when only energy/force is needed.

Exact tangent requests still use the original prepared spectral linearization.
"""
import time
import numpy as np
import warp as wp
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedGPUOperator
from engine.aniso_phase1.research_d.stage2.gpu_operator import reduce_kernel
from engine.aniso_phase1.research_d.stage2.contracts import array_digest
from engine.aniso_phase1.research_d.identity import digest

@wp.kernel
def force_response_kernel(F:wp.array(dtype=wp.mat33d),A:wp.mat33d,mu:wp.float64,lam:wp.float64,kf:wp.float64,
                    psi:wp.array(dtype=wp.float64),P:wp.array(dtype=wp.mat33d),
                    detout:wp.array(dtype=wp.float64),bad:wp.array(dtype=wp.int32)):
    k=wp.tid();f=F[k];det=wp.determinant(f);detout[k]=det
    if not wp.isfinite(det) or det<=wp.float64(0):
        wp.atomic_add(bad,0,1);psi[k]=wp.float64(0);P[k]=wp.mat33d();return
    U,sigma,V=wp.svd3(f)
    if wp.min(sigma)<wp.float64(1e-6):
        wp.atomic_add(bad,0,1);psi[k]=wp.float64(0);P[k]=wp.mat33d();return
    logs=wp.vec3d(wp.log(sigma[0]),wp.log(sigma[1]),wp.log(sigma[2]))
    tr=logs[0]+logs[1]+logs[2];principal=wp.vec3d();c=wp.vec3d()
    for i in range(3):
        principal[i]=(wp.float64(2)*mu*logs[i]+lam*tr)/sigma[i];c[i]=sigma[i]*sigma[i]
    fa=f@A;strain=wp.trace(wp.transpose(fa)@f)-wp.float64(1)
    psi[k]=mu*wp.dot(logs,logs)+wp.float64(.5)*lam*tr*tr+wp.float64(.5)*kf*strain*strain
    P[k]=U@wp.diag(principal)@wp.transpose(V)+wp.float64(2)*kf*strain*fa

class ForceOnlyOperator(SegmentedGPUOperator):
    @classmethod
    def from_existing(cls,original):
        obj=cls.__new__(cls);obj.__dict__=original.__dict__.copy()
        obj.signature=digest(dict(parent=original.signature,implementation='force-only-no-spectral-output-v1'))
        return obj

    def force_response(self,q):
        begun=time.perf_counter();s=self.space;m=self.maps;p=self.params
        if hasattr(self,'memory_budget'):
            self.memory_budget.observe(self.count*512+int(np.prod(s.shape))*256)
        full=s.expand(q);nodal=m.nodes(m.expand(q));F=m.gradient(nodal,self.layout,identity=True)
        t=self._stamp('mapping_and_upload',begun)
        psi=wp.empty(self.count,dtype=wp.float64,device=self.device);P=wp.empty_like(F)
        det=wp.empty_like(psi);bad=wp.zeros(1,dtype=wp.int32,device=self.device)
        t=self._stamp('response_allocations',t)
        wp.launch(force_response_kernel,dim=self.count,inputs=[F,self.A,p.mu,p.lam,p.k_f,psi,P,det,bad],device=self.device)
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
        return r

    def evaluate(self,q,direction=None):
        if direction is None:return self.force_response(q)
        return super().evaluate(q,direction)


def install_force_only(model):
    model.operator=ForceOnlyOperator.from_existing(model.operator)
    model._cached_q=None;model._cached_response=None
    return model
