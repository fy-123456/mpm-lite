"""Optional center moments and displacement/corotated stabilization.

All particle visits are preparation between steps. Newton kernels only visit
centers/nodes. Stabilization uses transported reference positions and a frozen
small-strain tangent in legacy modes. The corotated mode freezes a material
reference map and includes rotation derivatives; see corotated.py.
"""
import itertools
import numpy as np
import warp as wp
from engine.types import real, vec3, mat33, mat99
from engine.sp_grid import B, unlin_IJK
from .kernels import mat_to_vec9_wp, vec9_to_mat_wp


@wp.func
def flat(b: int, i: int, j: int, k: int):
    return b*B*B*B+i*B*B+j*B+k


@wp.kernel
def scatter_moments(x: wp.array(dtype=vec3), A: wp.array(dtype=mat33), V: wp.array(dtype=real),
                    blocks: wp.array(dtype=int,ndim=3), M: wp.array(dtype=mat99), slots: wp.array(dtype=int), size: wp.vec3i, dx: real):
    p=wp.tid();pos=x[p]/dx-vec3(real(.5))
    base=wp.vec3i(int(wp.floor(pos[0])),int(wp.floor(pos[1])),int(wp.floor(pos[2])))
    f=pos-vec3(real(base[0]),real(base[1]),real(base[2]));av=mat_to_vec9_wp(A[p])
    outer=wp.outer(av,av)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c=base+wp.vec3i(i,j,k)
                if c[0]<0 or c[1]<0 or c[2]<0 or c[0]>=size[0] or c[1]>=size[1] or c[2]>=size[2]:continue
                b=blocks[c[0]//B,c[1]//B,c[2]//B]
                if b<0:continue
                w=(f[0] if i else real(1)-f[0])*(f[1] if j else real(1)-f[1])*(f[2] if k else real(1)-f[2])
                if w>real(0):wp.atomic_add(M,slots[flat(b,c[0]%B,c[1]%B,c[2]%B)]-1,w*V[p]*outer)


@wp.kernel
def normalize_moments(M: wp.array(dtype=mat99), slots: wp.array(dtype=int), volume: wp.array(dtype=real,ndim=4)):
    b,i,j,k=wp.tid();v=volume[0,b,i,j*B+k]
    if v>real(0):M[slots[flat(b,i,j,k)]-1]/=v


@wp.func
def moment_S(F: mat33,A: mat33,M: mat99):
    return vec9_to_mat_wp(M@mat_to_vec9_wp(wp.transpose(F)@F))-A


@wp.func
def moment_extra_P(F: mat33,A: mat33,M: mat99,kf: real):
    I4=wp.trace(A@wp.transpose(F)@F)
    return real(2)*kf*F@(moment_S(F,A,M)-(I4-real(1))*A)


@wp.func
def moment_extra_dP(F: mat33,A: mat33,M: mat99,dF: mat33,kf: real,pd: int):
    S=moment_S(F,A,M)
    I4=wp.trace(A@wp.transpose(F)@F)
    coefficient=I4-real(1)
    if pd!=0:
        Q,e=wp.eig3(S)
        e=vec3(wp.max(e[0],real(0)),wp.max(e[1],real(0)),wp.max(e[2],real(0)))
        S=Q@wp.diag(e)@wp.transpose(Q)
        coefficient=wp.max(coefficient,real(0))
    dC=wp.transpose(dF)@F+wp.transpose(F)@dF
    dS=vec9_to_mat_wp(M@mat_to_vec9_wp(dC))
    dI=wp.trace(A@dC)
    return real(2)*kf*(dF@(S-coefficient*A)+F@(dS-dI*A))


@wp.kernel
def moment_correction(cdof: wp.array(dtype=wp.vec2i), ids: wp.array(dtype=int,ndim=2),
                      F: wp.array(dtype=mat33,ndim=4), F0: wp.array(dtype=mat33,ndim=4),
                      A: wp.array(dtype=mat33,ndim=4), M: wp.array(dtype=mat99), slots: wp.array(dtype=int), V: wp.array(dtype=real,ndim=4),
                      out: wp.array(dtype=vec3), energy: wp.array(dtype=real), dx: real,dt: real,kf: real):
    c=wp.tid();b,l=cdof[c][0],cdof[c][1];i,j,k=unlin_IJK(l)
    f=F[0,b,i,j*B+k];a=A[0,b,i,j*B+k];m=M[slots[flat(b,i,j,k)]-1];vol=V[0,b,i,j*B+k]
    C=mat_to_vec9_wp(wp.transpose(f)@f);av=mat_to_vec9_wp(a)
    variance=wp.dot(C,m@C)-wp.dot(C,av)*wp.dot(C,av)
    wp.atomic_add(energy,0,real(.5)*kf*vol*variance)
    wp.atomic_add(energy,1,kf*vol*(wp.abs(wp.dot(C,m@C))+wp.dot(C,av)*wp.dot(C,av)))
    tau=moment_extra_P(f,a,m,kf)@wp.transpose(F0[0,b,i,j*B+k])
    for z in range(8):
        grad=vec3(real(2*(z//4)-1),real(2*((z//2)%2)-1),real(2*(z%2)-1))/(real(4)*dx)
        n=ids[c,z]
        if n>=0:wp.atomic_add(out,n,dt*vol*(tau@grad))


@wp.kernel
def moment_tangent(cdof: wp.array(dtype=wp.vec2i), ids: wp.array(dtype=int,ndim=2),
                   F: wp.array(dtype=mat33,ndim=4), F0: wp.array(dtype=mat33,ndim=4),
                   A: wp.array(dtype=mat33,ndim=4), M: wp.array(dtype=mat99), slots: wp.array(dtype=int), V: wp.array(dtype=real,ndim=4),
                   p: wp.array(dtype=vec3),out: wp.array(dtype=vec3),dx: real,dt: real,kf: real,pd: int):
    c=wp.tid();b,l=cdof[c][0],cdof[c][1];i,j,k=unlin_IJK(l);G=mat33(real(0))
    for z in range(8):
        grad=vec3(real(2*(z//4)-1),real(2*((z//2)%2)-1),real(2*(z%2)-1))/(real(4)*dx)
        n=ids[c,z]
        if n>=0:G+=wp.outer(p[n],grad)
    f0=F0[0,b,i,j*B+k];f=F[0,b,i,j*B+k];a=A[0,b,i,j*B+k]
    tau=moment_extra_dP(f,a,M[slots[flat(b,i,j,k)]-1],dt*G@f0,kf,pd)@wp.transpose(f0)
    for z in range(8):
        grad=vec3(real(2*(z//4)-1),real(2*((z//2)%2)-1),real(2*(z%2)-1))/(real(4)*dx)
        n=ids[c,z]
        if n>=0:wp.atomic_add(out,n,dt*V[0,b,i,j*B+k]*(tau@grad))


@wp.kernel
def center_nodes(cdof: wp.array(dtype=wp.vec2i), xyz: wp.array(dtype=wp.vec3i),blocks: wp.array(dtype=int,ndim=3),
                 dofs: wp.array(dtype=int,ndim=4),ids: wp.array(dtype=int,ndim=2)):
    c=wp.tid();b,l=cdof[c][0],cdof[c][1];i,j,k=unlin_IJK(l)
    base=xyz[b]*B+wp.vec3i(i,j,k)
    for z in range(8):
        n=base+wp.vec3i(z//4,(z//2)%2,z%2);nb=blocks[n[0]//B,n[1]//B,n[2]//B]
        ids[c,z]=dofs[nb,n[0]%B,n[1]%B,n[2]%B]


@wp.kernel
def scatter_reference(x: wp.array(dtype=vec3),X: wp.array(dtype=vec3),F: wp.array(dtype=mat33),mass: wp.array(dtype=real),
                      blocks: wp.array(dtype=int,ndim=3),dofs: wp.array(dtype=int,ndim=4),
                      sums: wp.array(dtype=vec3),weights: wp.array(dtype=real),dx: real):
    p=wp.tid();pos=x[p]/dx-vec3(real(.5))
    base=wp.vec3i(int(wp.floor(pos[0])),int(wp.floor(pos[1])),int(wp.floor(pos[2])))
    f=pos-vec3(real(base[0]),real(base[1]),real(base[2]));inverse=wp.inverse(F[p])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                w=(f[0] if i else real(1)-f[0])*(f[1] if j else real(1)-f[1])*(f[2] if k else real(1)-f[2])*mass[p]/real(8)
                for z in range(8):
                    n=base+wp.vec3i(i+z//4,j+(z//2)%2,k+z%2)
                    b=blocks[n[0]//B,n[1]//B,n[2]//B];d=dofs[b,n[0]%B,n[1]%B,n[2]%B]
                    if d>=0 and w>real(0):
                        xn=dx*vec3(real(n[0]),real(n[1]),real(n[2]))
                        u=xn-X[p]-inverse@(xn-x[p])
                        wp.atomic_add(sums,d,w*u);wp.atomic_add(weights,d,w)


@wp.kernel
def normalize_reference(u: wp.array(dtype=vec3),weights: wp.array(dtype=real)):
    i=wp.tid()
    if weights[i]>real(0):u[i]/=weights[i]


@wp.func
def reference_stress(D: mat33,A: mat33,M: mat99,mu: real,lam: real,kf: real,kind: int,fourth: int):
    if kind==2:return real(2)*mu*D
    fiber=wp.trace(wp.transpose(A)@D)*A
    if fourth!=0:fiber=vec9_to_mat_wp(M@mat_to_vec9_wp(D))
    return mu*(D+wp.transpose(D))+lam*wp.trace(D)*wp.identity(3,dtype=real)+real(4)*kf*fiber


@wp.kernel
def stabilization(cdof: wp.array(dtype=wp.vec2i),ids: wp.array(dtype=int,ndim=2),dg: wp.array(dtype=vec3,ndim=2),
                  u: wp.array(dtype=vec3),v: wp.array(dtype=vec3),A: wp.array(dtype=mat33,ndim=4),M: wp.array(dtype=mat99), slots: wp.array(dtype=int),
                  volume: wp.array(dtype=real,ndim=4),out: wp.array(dtype=vec3),energy: wp.array(dtype=real),
                  dt: real,eta: real,mu: real,lam: real,kf: real,kind: int,fourth: int,tangent: int):
    c,q=wp.tid();b,l=cdof[c][0],cdof[c][1];i,j,k=unlin_IJK(l);D=mat33(real(0))
    for z in range(8):
        n=ids[c,z]
        if n>=0:
            value=dt*v[n]
            if tangent==0:value+=u[n]
            D+=wp.outer(value,dg[q,z])
    m=mat99(real(0))
    if fourth!=0:m=M[slots[flat(b,i,j,k)]-1]
    stress=reference_stress(D,A[0,b,i,j*B+k],m,mu,lam,kf,kind,fourth)
    weight=eta*volume[0,b,i,j*B+k]/real(8)
    for z in range(8):
        n=ids[c,z]
        if n>=0:wp.atomic_add(out,n,dt*weight*(stress@dg[q,z]))
    if tangent==0:
        e=real(.5)*weight*wp.trace(wp.transpose(D)@stress)
        wp.atomic_add(energy,0,e);wp.atomic_add(energy,1,wp.abs(e))


class CenterEnhancements:
    def __init__(self,s):
        self.s=s;self.ready=False
        self.reference_valid=True
        # Particle quadrature keeps individual fibers for its material energy,
        # but its shared stabilizer needs the same averaged H0 as center M4.
        self.fourth_stabilization=(s.direction_model=='fourth_moment' or getattr(s,'quadrature_kind','center')=='particle')
        self.M=wp.zeros(1,dtype=mat99,device=s.device)
        self.slots=wp.zeros(1,dtype=int,device=s.device)
        self.energy=wp.zeros(2,dtype=real,device=s.device)
        self.hg_energy=wp.zeros(2,dtype=real,device=s.device)
        from .beam_reference import shape_gradients
        q=(.5-1/(2*np.sqrt(3)),.5+1/(2*np.sqrt(3)))
        dg=np.array([shape_gradients(p,s.dx)-shape_gradients((.5,)*3,s.dx) for p in itertools.product(q,repeat=3)])
        self.dg=wp.array(dg,dtype=vec3,device=s.device)
        if s.stabilization=='corotated':
            self.reference_g=wp.array(np.array([shape_gradients((.5,)*3,s.dx)]+[shape_gradients(p,s.dx) for p in itertools.product(q,repeat=3)]),dtype=vec3,device=s.device)

    def resample(self):
        s=self.s;self.ready=False
        self.reference_valid=True
        if s.direction_model=='fourth_moment' or (s.stabilization!='none' and self.fourth_stabilization):
            n=int(s.bcn)*B**3
            self.slots=wp.zeros(n,dtype=int,device=s.device)
            wp.launch(mark_occupied,dim=(int(s.bcn),B,B,B),inputs=[s.center_vol,self.slots],device=s.device)
            wp.utils.array_scan(self.slots,self.slots)
            count=int(self.slots[-1:].numpy()[0])
            if len(self.M)!=count:self.M=wp.zeros(count,dtype=mat99,device=s.device)
            else:self.M.zero_()
            wp.launch(scatter_moments,dim=s.n_ptc,inputs=[s.ptc_x,s.ptc_A0,s.ptc_vol0,s.block2bid,self.M,self.slots,s.center_size,s.dx],device=s.device)
            wp.launch(normalize_moments,dim=(int(s.bcn),B,B,B),inputs=[self.M,self.slots,s.center_vol],device=s.device)

    def prepare(self):
        if self.ready:return
        s=self.s;nc=int(s.n_active_centers.numpy()[0]);nn=int(s.n_active_nodes.numpy()[0])
        self.ids=wp.empty((nc,8),dtype=int,device=s.device)
        self.values=wp.zeros_like(s.node_residual);self.extra=wp.zeros_like(s.node_residual)
        self.u=wp.zeros_like(s.node_residual);weights=wp.zeros(len(self.u),dtype=real,device=s.device)
        wp.launch(center_nodes,dim=nc,inputs=[s.cdof2bijk,s.block_xyz_by_id,s.block2bid,s.node2dof,self.ids],device=s.device)
        if s.stabilization!='none':
            x=s.ptc_x.numpy()
            if np.any(x<.5*s.dx) or np.any(x>(np.array(tuple(s.center_size))-.5)*s.dx):
                raise ValueError('stabilization requires full interior particle support')
            if s.stabilization not in ('quadratic','material_quadratic'):
                wp.launch(scatter_reference,dim=s.n_ptc,inputs=[s.ptc_x,s.ptc_reference_x,s.ptc_F,s.ptc_m,s.block2bid,s.node2dof,self.u,weights,s.dx],device=s.device)
                wp.launch(normalize_reference,dim=nn,inputs=[self.u,weights],device=s.device)
        self.reference_ids=self.ids
        if s.stabilization=='corotated':
            from .corotated import prepare_reference
            self.reference_F0=wp.zeros((nc,9),dtype=mat33,device=s.device)
            self.reference_B=wp.zeros((nc,9,8),dtype=vec3,device=s.device)
            invalid=wp.zeros(1,dtype=int,device=s.device)
            wp.launch(prepare_reference,dim=(nc,9),inputs=[self.ids,self.u,self.reference_g,self.reference_F0,self.reference_B,invalid],device=s.device)
            self.reference_valid=int(invalid.numpy()[0])==0
        if s.stabilization in ('quadratic','material_quadratic'):
            from .quadratic import build_patch_maps
            try:
                ids,F0,gradient,self.reconstruction_stats=build_patch_maps(s)
                self.reference_ids=wp.array(ids,dtype=int,device=s.device)
                self.reference_F0=wp.array(F0,dtype=mat33,device=s.device)
                self.reference_B=wp.array(gradient,dtype=vec3,device=s.device)
            except (ValueError,np.linalg.LinAlgError) as error:
                self.reference_valid=False
                self.reconstruction_stats={'reconstruction_failure':str(error)}
        self.ready=True

    def correction(self,trial=True):
        s=self.s;self.prepare();self.extra.zero_();self.energy.zero_();self.hg_energy.zero_()
        wp.launch(gather_values,dim=int(s.n_active_nodes.numpy()[0]),inputs=[s.ndof2bijk,s.grid_v_it,self.values],device=s.device)
        if not trial:self.values.zero_()
        if s.direction_model=='fourth_moment' and getattr(s,'quadrature_kind','center')!='group4x8':
            wp.launch(moment_correction,dim=len(self.ids),inputs=[s.cdof2bijk,self.ids,s.aniso_trial_F if trial else s.aniso_committed_F,
                s.aniso_committed_F,s.aniso_A0,self.M,self.slots,s.center_vol,self.extra,self.energy,s.dx,s.dt,s.aniso_params.k_f],device=s.device)
        self._hg(self.values,self.extra,self.hg_energy,False)

    def _hg(self,v,out,energy,tangent,pd=False):
        s=self.s
        if s.stabilization=='none':return
        if s.stabilization in ('corotated','quadratic','material_quadratic'):
            from . import corotated
            if not self.reference_valid:
                raise ValueError('corotated reference mapping is inverted or singular')
            common=[s.cdof2bijk,self.reference_ids,self.reference_F0,self.reference_B]
            params=[s.dt,s.stabilization_strength,s.aniso_params.mu,s.aniso_params.lam,s.aniso_params.k_f,int(self.fourth_stabilization)]
            fields=[s.aniso_A0,self.M,self.slots,s.center_vol,out]
            if tangent:
                wp.launch(corotated.tangent,dim=(len(self.ids),8),inputs=common+[self.values,v]+fields+params+[int(pd)],device=s.device)
            else:
                wp.launch(corotated.residual,dim=(len(self.ids),8),inputs=common+[v]+fields+[energy,s.aniso_invalid_trial]+params,device=s.device)
            return
        wp.launch(stabilization,dim=(len(self.ids),8),inputs=[s.cdof2bijk,self.ids,self.dg,self.u,v,s.aniso_A0,self.M,self.slots,s.center_vol,out,energy,
            s.dt,s.stabilization_strength,s.aniso_params.mu,s.aniso_params.lam,s.aniso_params.k_f,
            1 if s.stabilization=='supplemental' else 2,int(self.fourth_stabilization),int(tangent)],device=s.device)

    def tangent(self,p,Ap,pd):
        s=self.s
        if s.direction_model=='fourth_moment' and getattr(s,'quadrature_kind','center')!='group4x8':
            wp.launch(moment_tangent,dim=len(self.ids),inputs=[s.cdof2bijk,self.ids,s.aniso_trial_F,s.aniso_committed_F,s.aniso_A0,
                self.M,self.slots,s.center_vol,p,Ap,s.dx,s.dt,s.aniso_params.k_f,int(pd)],device=s.device)
        self._hg(p,Ap,self.hg_energy,True,pd)

    def memory_bytes(self):
        from warp._src.types import type_size_in_bytes
        arrays={id(a):a for a in vars(self).values() if isinstance(a,wp.array)}
        return sum(a.size*type_size_in_bytes(a.dtype) for a in arrays.values())


@wp.kernel
def gather_values(ndof: wp.array(dtype=wp.vec2i),grid: wp.array(dtype=vec3,ndim=4),v: wp.array(dtype=vec3)):
    n=wp.tid();b,l=ndof[n][0],ndof[n][1];i,j,k=unlin_IJK(l);v[n]=grid[b,i,j,k]


@wp.kernel
def add_correction(extra: wp.array(dtype=vec3),out: wp.array(dtype=vec3)):
    i=wp.tid();out[i]+=extra[i]


@wp.kernel
def add_energy(a: wp.array(dtype=real),b: wp.array(dtype=real),out: wp.array(dtype=real)):
    i=wp.tid();out[i]+=a[i]+b[i]


@wp.kernel
def mark_occupied(volume: wp.array(dtype=real,ndim=4),slots: wp.array(dtype=int)):
    b,i,j,k=wp.tid()
    slots[flat(b,i,j,k)]=int(volume[0,b,i,j*B+k]>real(0))
