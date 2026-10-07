"""Separate APIC affine coefficients C from the deformation velocity gradient L.

Incremental mode: C_new = L_new + beta * (C_old - L_raw), where both L
samples use pre-advection particle positions and raw means before boundary
projection. The affine beta may be set independently from velocity PIC/FLIP;
None keeps the legacy shared beta. beta_C=1 preserves the full affine increment. The material F update continues to use L_new without blending.
"""
import warp as wp
from engine.types import mat33, real, vec3
from engine.sp_grid import B, block_coords_from_node, local_coords_in_block
from engine.kernel.d3.kernel_lite import lite_g2c_kernel, in_region


@wp.kernel
def affine_difference_kernel(
    block2bid: wp.array(dtype=int,ndim=3),
    x: wp.array(dtype=vec3), C: wp.array(dtype=mat33),
    mass: wp.array(dtype=real,ndim=4), raw_L: wp.array(dtype=mat33,ndim=4),
    difference: wp.array(dtype=mat33), size: wp.vec3i, dx: real,
):
    p=wp.tid();q=x[p]/dx-vec3(real(.5))
    base=wp.vec3i(int(wp.floor(q[0])),int(wp.floor(q[1])),int(wp.floor(q[2])))
    f=q-vec3(real(base.x),real(base.y),real(base.z));L=mat33(real(0.))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                c=base+wp.vec3i(i,j,k)
                if not in_region(c,size):continue
                bc=block_coords_from_node(c.x,c.y,c.z);bid=block2bid[bc.x,bc.y,bc.z]
                if bid<0:continue
                lc=local_coords_in_block(c.x,c.y,c.z)
                if mass[bid,lc.x,lc.y,lc.z]<=real(0.):continue
                w=(f[0] if i==1 else real(1.)-f[0])*(f[1] if j==1 else real(1.)-f[1])*(f[2] if k==1 else real(1.)-f[2])
                L+=w*raw_L[bid,lc.x,lc.y,lc.z]
    difference[p]=C[p]-L


@wp.kernel
def commit_affine_kernel(C: wp.array(dtype=mat33),L: wp.array(dtype=mat33),difference: wp.array(dtype=mat33),beta: real):
    p=wp.tid()
    # Legacy C2P has just stored L and updated F with it. Capture it before changing C.
    L[p]=C[p]
    C[p]=L[p]+beta*difference[p]


def prepare_incremental(s):
    """Call after solve success, before the regular G2C/C2P; scratch centers overwritten next."""
    if s._apic_difference is None or len(s._apic_difference)!=s.n_ptc:
        s._apic_difference=wp.zeros(s.n_ptc,dtype=mat33,device=s.device)
    wp.launch(lite_g2c_kernel,dim=(s.bcn,B,B,B),inputs=[s.block_count,s.block2bid,s.block_xyz_by_id,
        s.grid_v_raw,s.grid_v_raw,s.center_v,s.center_dv,s.center_G,s.center_size,s.dx],device=s.device)
    wp.launch(affine_difference_kernel,dim=s.n_ptc,inputs=[s.block2bid,s.ptc_x,s.ptc_C,s.center_m,
        s.center_G,s._apic_difference,s.center_size,s.dx],device=s.device)


def finish_transfer(s):
    if s.apic_transfer=='incremental':
        beta = s.flip_ratio if s.affine_flip_ratio is None else s.affine_flip_ratio
        wp.launch(commit_affine_kernel,dim=s.n_ptc,inputs=[s.ptc_C,s.ptc_L,s._apic_difference,beta],device=s.device)
    else:wp.copy(s.ptc_L,s.ptc_C)
