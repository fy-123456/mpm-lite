"""GPU reductions for the frozen-stencil incremental potential (not total energy)."""
import warp as wp
from engine.types import real, vec3, mat33
from engine.sp_grid import unlin_IJK, unlin_I_JK
from .kernels import aniso_energy_wp


@wp.func
def material_roundoff_scale(F: mat33, A0: mat33, mu: real, lam: real, kf: real) -> real:
    # Energy is quadratic in small log strains, while SVD roundoff perturbs
    # the singular values at an absolute scale set by F, not by that energy.
    U, sigma, V = wp.svd3(F)
    logs = vec3(wp.log(sigma[0]), wp.log(sigma[1]), wp.log(sigma[2]))
    condition = wp.max(sigma)/wp.min(sigma)
    iso_scale = condition*(real(2)*mu*(wp.abs(logs[0])+wp.abs(logs[1])+wp.abs(logs[2]))
                          + real(3)*wp.abs(lam)*wp.abs(logs[0]+logs[1]+logs[2]))
    I4 = wp.trace(A0 @ wp.transpose(F) @ F)
    return iso_scale + real(4)*kf*wp.abs(I4-real(1))*wp.max(real(1), I4)


@wp.kernel
def inertia_potential(ndof: wp.array(dtype=wp.vec2i), mass: wp.array(dtype=real, ndim=4),
                      old: wp.array(dtype=vec3, ndim=4), velocity: wp.array(dtype=vec3, ndim=4),
                      gravity: real, dt: real, out: wp.array(dtype=real)):
    i = wp.tid()
    b, l = ndof[i][0], ndof[i][1]
    a, c, d = unlin_IJK(l)
    dv = velocity[b,a,c,d] - old[b,a,c,d] - vec3(real(0), real(0), dt*gravity)
    wp.atomic_add(out, 0, real(.5)*mass[b,a,c,d]*wp.dot(dv,dv))
    wp.atomic_add(out, 1, mass[b,a,c,d]*wp.length(dv)*(wp.length(velocity[b,a,c,d])
                  + wp.length(old[b,a,c,d]) + wp.abs(dt*gravity)))


@wp.kernel
def center_potential(cdof: wp.array(dtype=wp.vec2i), volume: wp.array(dtype=real, ndim=4),
                     F: wp.array(dtype=mat33, ndim=4), A0: wp.array(dtype=mat33, ndim=4),
                     mu: real, lam: real, kf: real, out: wp.array(dtype=real)):
    i = wp.tid()
    b, l = cdof[i][0], cdof[i][1]
    a, c = unlin_I_JK(l)
    wp.atomic_add(out, 0, volume[0,b,a,c]*aniso_energy_wp(F[0,b,a,c], A0[0,b,a,c], mu, lam, kf))
    wp.atomic_add(out, 1, volume[0,b,a,c]*material_roundoff_scale(F[0,b,a,c], A0[0,b,a,c], mu, lam, kf))


@wp.kernel
def particle_potential(volume: wp.array(dtype=real), F: wp.array(dtype=mat33),
                       A0: wp.array(dtype=mat33), mu: real, lam: real, kf: real,
                       out: wp.array(dtype=real)):
    i = wp.tid()
    wp.atomic_add(out, 0, volume[i]*aniso_energy_wp(F[i], A0[i], mu, lam, kf))
    wp.atomic_add(out, 1, volume[i]*material_roundoff_scale(F[i], A0[i], mu, lam, kf))
