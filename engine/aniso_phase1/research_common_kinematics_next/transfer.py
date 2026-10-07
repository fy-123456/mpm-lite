"""Elastic-only candidate C2P: physical L and raw APIC use grad(S) H.

Local prescribed velocities have explicit reference gradients and are unchanged
between raw/new. This adapter is not an enriched implicit solve. Incomplete
support is rejected; no silent renormalization or deformation-history reset.
"""
import numpy as np
import warp as wp
from engine.types import vec3,mat33,real,Material
from engine.sp_grid import block_coords_from_node,local_coords_in_block
from engine.kernel.d3.kernel_lite import in_region

@wp.kernel
def common_c2p(
    block2bid:wp.array(dtype=int,ndim=3),x:wp.array(dtype=vec3),v:wp.array(dtype=vec3),
    F:wp.array(dtype=mat33),C:wp.array(dtype=mat33),Lout:wp.array(dtype=mat33),
    mass:wp.array(dtype=real,ndim=4),cv:wp.array(dtype=vec3,ndim=4),dv:wp.array(dtype=vec3,ndim=4),
    local_v:wp.array(dtype=vec3),local_DX:wp.array(dtype=mat33),
    size:wp.vec3i,dx:real,dt:real,beta:real,beta_C:real,status:wp.array(dtype=int)):
    p=wp.tid();q=x[p]/dx-vec3(real(.5))
    base=wp.vec3i(int(wp.floor(q[0])),int(wp.floor(q[1])),int(wp.floor(q[2])))
    f=q-vec3(real(base.x),real(base.y),real(base.z))
    vp=vec3(real(0.));delta=vec3(real(0.));L=mat33(real(0.));Lraw=mat33(real(0.));count=int(0)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c=base+wp.vec3i(i,j,k)
                if not in_region(c,size):continue
                bc=block_coords_from_node(c.x,c.y,c.z);bid=block2bid[bc.x,bc.y,bc.z]
                if bid<0:continue
                lc=local_coords_in_block(c.x,c.y,c.z)
                if mass[bid,lc.x,lc.y,lc.z]<=real(0.):continue
                count+=1
                a=f[0] if i==1 else real(1.)-f[0]
                b=f[1] if j==1 else real(1.)-f[1]
                cc=f[2] if k==1 else real(1.)-f[2]
                g=vec3(real(2*i-1)*b*cc,a*real(2*j-1)*cc,a*b*real(2*k-1))/dx
                vv=cv[bid,lc.x,lc.y,lc.z];dd=dv[bid,lc.x,lc.y,lc.z]
                vp+=a*b*cc*vv;delta+=a*b*cc*dd
                L+=wp.outer(vv,g);Lraw+=wp.outer(vv-dd,g)
    status[p]=count
    if count==8:
        Llocal=local_DX[p]@wp.inverse(F[p])
        L+=Llocal;Lraw+=Llocal;vp+=local_v[p]
        oldF=F[p]
        F[p]=(wp.identity(3,real)+dt*L)@oldF
        C[p]=L+beta_C*(C[p]-Lraw);Lout[p]=L
        v[p]=beta*(v[p]+delta)+(real(1.)-beta)*vp
        x[p]+=dt*vp


def transfer(s,dt,local_v=None,local_DX=None):
    """Trial arrays only; commit after support and admissibility checks."""
    if not np.isfinite(dt) or dt<=0:raise ValueError('positive dt required')
    params=s.psi_params.numpy();k=s.ptc_k.numpy()
    if np.any(params['mate_type'][k]!=Material.elastic):raise ValueError('elastic-only research adapter')
    n=s.n_ptc
    lv=np.zeros((n,3)) if local_v is None else np.asarray(local_v,float)
    ld=np.zeros((n,3,3)) if local_DX is None else np.asarray(local_DX,float)
    if lv.shape!=(n,3) or ld.shape!=(n,3,3) or not np.isfinite(lv).all() or not np.isfinite(ld).all():
        raise ValueError('finite matched material local field required')
    if np.any(np.linalg.det(s.ptc_F.numpy())<=.1):raise ValueError('invalid step-initial F')
    arrays=[wp.array(a.numpy(),dtype=a.dtype,device=s.device) for a in (s.ptc_x,s.ptc_v,s.ptc_F,s.ptc_C,s.ptc_L)]
    status=wp.zeros(n,dtype=int,device=s.device)
    wp.launch(common_c2p,dim=n,inputs=[s.block2bid,*arrays,s.center_m,s.center_v,s.center_dv,
        wp.array(lv,dtype=vec3,device=s.device),wp.array(ld,dtype=mat33,device=s.device),
        s.center_size,s.dx,dt,s.flip_ratio,s.flip_ratio if s.affine_flip_ratio is None else s.affine_flip_ratio,status],device=s.device)
    if np.any(status.numpy()!=8):raise ValueError('common map needs complete active center support')
    if any(not np.isfinite(a.numpy()).all() for a in arrays) or np.any(np.linalg.det(arrays[2].numpy())<=.1):
        raise ValueError('inadmissible common transfer trial')
    for target,trial in zip((s.ptc_x,s.ptc_v,s.ptc_F,s.ptc_C,s.ptc_L),arrays):wp.copy(target,trial)
