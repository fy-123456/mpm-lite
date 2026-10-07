"""Once-per-step particle history resampling; no particle visits inside Newton/PCG."""
import warp as wp
from engine.types import real, vec3, mat33
from engine.sp_grid import B


@wp.kernel
def scatter_particle_F(x: wp.array(dtype=vec3), F: wp.array(dtype=mat33),
                       volume: wp.array(dtype=real), blockmap: wp.array(dtype=int, ndim=3),
                       sums: wp.array(dtype=mat33, ndim=4), size: wp.vec3i, dx: real):
    p=wp.tid()
    pos=x[p]/dx-vec3(real(.5))
    base=wp.vec3i(int(wp.floor(pos[0])),int(wp.floor(pos[1])),int(wp.floor(pos[2])))
    f=pos-vec3(real(base[0]),real(base[1]),real(base[2]))
    for a in range(2):
        for b in range(2):
            for c in range(2):
                cell=base+wp.vec3i(a,b,c)
                if cell[0]<0 or cell[1]<0 or cell[2]<0 or cell[0]>=size[0] or cell[1]>=size[1] or cell[2]>=size[2]:
                    continue
                bid=blockmap[cell[0]//B,cell[1]//B,cell[2]//B]
                if bid<0:continue
                w=(f[0] if a else real(1)-f[0])*(f[1] if b else real(1)-f[1])*(f[2] if c else real(1)-f[2])
                wp.atomic_add(sums,0,bid,cell[0]%B,(cell[1]%B)*B+cell[2]%B,w*volume[p]*F[p])


@wp.kernel
def gather_history(volume: wp.array(dtype=real, ndim=4), sums: wp.array(dtype=mat33, ndim=4),
                   A_sum: wp.array(dtype=mat33, ndim=4), F: wp.array(dtype=mat33, ndim=4),
                   A: wp.array(dtype=mat33, ndim=4), invalid: wp.array(dtype=int)):
    b,i,j,k=wp.tid()
    jk=j*B+k
    vol=volume[0,b,i,jk]
    if vol>real(0):
        state=sums[0,b,i,jk]/vol
        determinant=wp.determinant(state)
        if not wp.isfinite(determinant) or determinant<=real(0):
            wp.atomic_max(invalid,0,1)
        F[0,b,i,jk]=state
        A[0,b,i,jk]=A_sum[0,b,i,jk]/vol


def resample_history(s):
    # The trial buffer is scratch before Newton, so no extra full sparse reserve.
    s.aniso_trial_F.zero_()
    s.aniso_invalid_trial.zero_()
    wp.launch(scatter_particle_F,dim=s.n_ptc,inputs=[s.ptc_x,s.ptc_F,s.ptc_vol0,
              s.block2bid,s.aniso_trial_F,s.center_size,s.dx],device=s.device)
    wp.launch(gather_history,dim=(int(s.bcn),B,B,B),inputs=[s.center_vol,s.aniso_trial_F,
              s.aniso_A0_sum,s.aniso_committed_F,s.aniso_A0,s.aniso_invalid_trial],device=s.device)
    return int(s.aniso_invalid_trial.numpy()[0])==0
