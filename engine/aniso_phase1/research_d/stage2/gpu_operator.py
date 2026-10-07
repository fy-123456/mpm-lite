"""Original finite-strain potential on the actual common space.

Material and mappings run on the selected Warp device. The small original Ks
block and scalar diagnostics are finalized on the host, explicitly timed.
Exact spectral derivatives have continuous repeated-eigenvalue limits.
"""
from dataclasses import dataclass
import time
import numpy as np
import warp as wp
from ...tensor_metrics import quadrature_axis
from ...research_b.tensor import TensorRule
from .gpu_space import GPUSpace
from .contracts import array_digest,digest

@wp.func
def log_divided(ci:wp.float64,cj:wp.float64):
    r=(ci-cj)/cj
    out=wp.float64(0)
    if wp.abs(r)<wp.float64(1e-4):
        out=(wp.float64(1)-r/wp.float64(2)+r*r/wp.float64(3)-r*r*r/wp.float64(4)+r*r*r*r/wp.float64(5))/cj
    else:out=(wp.log(ci)-wp.log(cj))/(ci-cj)
    return out

@wp.kernel
def response_kernel(F:wp.array(dtype=wp.mat33d),A:wp.mat33d,mu:wp.float64,lam:wp.float64,kf:wp.float64,
                    psi:wp.array(dtype=wp.float64),P:wp.array(dtype=wp.mat33d),
                    Qout:wp.array(dtype=wp.mat33d),cout:wp.array(dtype=wp.vec3d),
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
    Qout[k]=V;cout[k]=c

@wp.kernel
def exact_tangent_kernel(F:wp.array(dtype=wp.mat33d),Q:wp.array(dtype=wp.mat33d),c:wp.array(dtype=wp.vec3d),
                         dF:wp.array(dtype=wp.mat33d),A:wp.mat33d,mu:wp.float64,lam:wp.float64,kf:wp.float64,
                         dP:wp.array(dtype=wp.mat33d)):
    k=wp.tid();f=F[k];q=Q[k];v=c[k];df=dF[k]
    logs=wp.vec3d(wp.log(v[0]),wp.log(v[1]),wp.log(v[2]));tr=logs[0]+logs[1]+logs[2]
    s=wp.vec3d()
    for i in range(3):s[i]=(mu*logs[i]+wp.float64(.5)*lam*tr)/v[i]
    local=wp.transpose(q)@(wp.transpose(f)@df+wp.transpose(df)@f)@q
    trace=wp.float64(0)
    for i in range(3):trace+=local[i,i]/v[i]
    ds=wp.mat33d()
    for i in range(3):
        for j in range(3):ds[i,j]=(mu*log_divided(v[i],v[j])-s[j])/v[i]*local[i,j]
        ds[i,i]+=wp.float64(.5)*lam*trace/v[i]
    S=q@wp.diag(s)@wp.transpose(q);fa=f@A
    strain=wp.trace(wp.transpose(fa)@f)-wp.float64(1)
    contraction=wp.trace(wp.transpose(fa)@df)
    dP[k]=df@S+f@q@ds@wp.transpose(q)+wp.float64(2)*kf*strain*(df@A)+wp.float64(4)*kf*contraction*fa

@wp.kernel
def reduce_kernel(psi:wp.array(dtype=wp.float64),weights:wp.array(dtype=wp.float64),det:wp.array(dtype=wp.float64),
                  energy:wp.array(dtype=wp.float64),minimum:wp.array(dtype=wp.float64),count:int):
    b=wp.tid();u=wp.float64(0);m=wp.float64(1e100)
    for i in range(b*256,wp.min((b+1)*256,count)):
        u+=weights[i]*psi[i];m=wp.min(m,det[i])
    energy[b]=u;minimum[b]=m

@dataclass
class Linearization:
    operator_sha256:str
    state_sha256:str
    F:object
    Q:object
    c:object
    response:dict

class GPUOperator:
    def __init__(self,space,*,order=6,device='cuda:0',max_workspace_bytes=16<<30):
        if isinstance(order,bool) or not isinstance(order,int) or order<1:
            raise ValueError('explicit positive integer material order required')
        count=int(np.prod([len(e)-1 for e in space.edges]))*order**3
        estimate=space.raw.nnz*24+space.oldA.nbytes*3+int(np.prod(space.shape))*3*8*12+count*600
        if estimate>max_workspace_bytes:raise MemoryError('declared GPU workspace budget exceeded')
        self.estimated_workspace_bytes=estimate
        t=time.perf_counter();self.space=space;self.maps=GPUSpace(space,device);self.device=self.maps.device
        self.rule=TensorRule.uniform(space.edges,order)
        qs=[quadrature_axis(e,order) for e in space.edges]
        self.points=[v[0] for v in qs];self.layout=self.maps.layout(self.points)
        weights=qs[0][1][:,None,None]*qs[1][1][None,:,None]*qs[2][1][None,None,:]
        self.weights=self.maps.upload(weights);self.count=weights.size
        self.A=wp.mat33d(*space.A.ravel());self.params=space.params;self.last=None
        self.signature=digest(dict(maps=self.maps.cache_key,rule=self.rule.signature,material=[space.params.mu,space.params.lam,space.params.k_f,array_digest(space.A)]))
        wp.synchronize_device(self.device);self.build_seconds=time.perf_counter()-t
    def prepare(self,q):
        t=time.perf_counter();s=self.space;m=self.maps;p=self.params
        full=s.expand(q);nodal=m.nodes(m.expand(q));F=m.gradient(nodal,self.layout,identity=True)
        Q=wp.empty(self.count,dtype=wp.mat33d,device=self.device);c=wp.empty(self.count,dtype=wp.vec3d,device=self.device)
        psi=wp.empty(self.count,dtype=wp.float64,device=self.device);P=wp.empty_like(F)
        det=wp.empty_like(psi);bad=wp.zeros(1,dtype=wp.int32,device=self.device)
        wp.launch(response_kernel,dim=self.count,inputs=[F,self.A,p.mu,p.lam,p.k_f,psi,P,Q,c,det,bad],device=self.device)
        if bad.numpy()[0]:raise ValueError('non-positive/non-finite detF or unsupported singular value')
        force=m.gradient_adjoint(P,self.layout,self.weights).numpy().reshape(s.ndof,3)
        n=(self.count+255)//256;energies=wp.empty(n,dtype=wp.float64,device=self.device);mins=wp.empty_like(energies)
        wp.launch(reduce_kernel,dim=n,inputs=[psi,self.weights,det,energies,mins,self.count],device=self.device)
        material_U=float(energies.numpy().sum());material_force=force.copy()
        y=s.reference[:s.n]+full[:s.n];ks=s.Ks@y;stab=.5*float(np.sum(y*ks));force[:s.n]+=ks
        r=dict(U=material_U+stab,energy_J=material_U+stab,material_U=material_U,stabilization_U=stab,
               full_force=force,material_force=material_force,force=s.restrict(force),
               min_detF=float(mins.numpy().min()),material_calls=self.count,rule_signature=self.rule.signature)
        if not np.isfinite(force).all() or not np.isfinite(r['U']):raise ValueError('non-finite material response')
        r['response_seconds']=time.perf_counter()-t
        return Linearization(self.signature,array_digest(q),F,Q,c,r)
    def action(self,linearization,direction):
        if not isinstance(linearization,Linearization) or linearization.operator_sha256!=self.signature:
            raise ValueError('foreign space/rule/material/device linearization')
        s=self.space;m=self.maps;p=self.params
        df=m.gradient(m.nodes(m.expand(direction,direction=True)),self.layout)
        dp=wp.empty_like(df)
        wp.launch(exact_tangent_kernel,dim=self.count,inputs=[linearization.F,linearization.Q,linearization.c,df,self.A,p.mu,p.lam,p.k_f,dp],device=self.device)
        out=m.gradient_adjoint(dp,self.layout,self.weights).numpy().reshape(s.ndof,3)
        out[:s.n]+=s.Ks@s.direction_coefficients(direction)[:s.n]
        if not np.isfinite(out).all():raise ValueError('non-finite tangent')
        return out
    def evaluate(self,q,direction=None):
        lin=self.prepare(q);r=lin.response.copy()
        if direction is not None:
            full=self.action(lin,direction);r.update(full_tangent_action=full,tangent_action=self.space.restrict(full))
        return r
