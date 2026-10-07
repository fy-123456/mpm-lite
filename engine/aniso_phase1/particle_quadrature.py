"""Particle quadrature ablation with identical constitutive law and Lite transfers.

The 27-node gradient is the composition of P2C trilinear weights and C2G
center gradients, so only the integration/state location changes. Every
residual/tangent evaluates the constitutive law at every particle.
"""
import warp as wp
from warp._src.types import type_size_in_bytes
from engine.types import real, vec3, mat33
from engine.sp_grid import B, local2global, unlin_IJK
from engine.boundary_utils import boundary_projection_kernel
from engine.kernel.constant import s_min, dPdF_eps_small, dPdF_eps_pd
from engine.kernel.d3.helper_constitutive_model import dPdF_StVK_Hencky_3D_analytic
from .kernels import aniso_dpk1_wp, aniso_pk1_wp, fiber_dpk1_wp, mat_to_vec9_wp, vec9_to_mat_wp
from .solver import AnisotropicLiteImplicitSolver


@wp.kernel
def particle_stencil(x: wp.array(dtype=vec3), blockmap: wp.array(dtype=int, ndim=3),
                     node2dof: wp.array(dtype=int, ndim=4), ids: wp.array(dtype=int, ndim=2),
                     grads: wp.array(dtype=vec3, ndim=2), dx: real, size: wp.vec3i):
    p = wp.tid()
    xc = x[p]/dx-vec3(real(.5))
    base = wp.vec3i(int(wp.floor(xc[0])), int(wp.floor(xc[1])), int(wp.floor(xc[2])))
    f = xc-vec3(real(base[0]), real(base[1]), real(base[2]))
    wx = vec3(real(1)-f[0], real(1), f[0])
    wy = vec3(real(1)-f[1], real(1), f[1])
    wz = vec3(real(1)-f[2], real(1), f[2])
    gx = vec3(f[0]-real(1), real(1)-real(2)*f[0], f[0])
    gy = vec3(f[1]-real(1), real(1)-real(2)*f[1], f[1])
    gz = vec3(f[2]-real(1), real(1)-real(2)*f[2], f[2])
    for a in range(3):
        for b in range(3):
            for c in range(3):
                slot = a*9+b*3+c
                node = base+wp.vec3i(a,b,c)
                ids[p,slot] = -1
                grads[p,slot] = vec3(real(0))
                if node[0]<0 or node[1]<0 or node[2]<0 or node[0]>=size[0] or node[1]>=size[1] or node[2]>=size[2]:
                    continue
                bid = blockmap[node[0]//B,node[1]//B,node[2]//B]
                if bid < 0:
                    continue
                ids[p,slot] = node2dof[bid,node[0]%B,node[1]%B,node[2]%B]
                grads[p,slot] = vec3(gx[a]*wy[b]*wz[c], wx[a]*gy[b]*wz[c], wx[a]*wy[b]*gz[c])/(real(4)*dx)


@wp.kernel
def node_inertia(ndof: wp.array(dtype=wp.vec2i), mass: wp.array(dtype=real, ndim=4),
                 old: wp.array(dtype=vec3, ndim=4), velocity: wp.array(dtype=vec3, ndim=4),
                 values: wp.array(dtype=vec3), residual: wp.array(dtype=vec3),
                 Hinv: wp.array(dtype=mat33), gravity: real, dt: real):
    i=wp.tid()
    bid,l=ndof[i][0],ndof[i][1]
    a,b,c=unlin_IJK(l)
    m=mass[bid,a,b,c]
    v=velocity[bid,a,b,c]
    values[i]=v
    residual[i]=m*(v-old[bid,a,b,c])-vec3(real(0),real(0),dt*m*gravity)
    Hinv[i]=wp.identity(3,dtype=real)/m


@wp.kernel
def node_tangent_mass(ndof: wp.array(dtype=wp.vec2i), mass: wp.array(dtype=real, ndim=4),
                      direction: wp.array(dtype=vec3), out: wp.array(dtype=vec3)):
    i=wp.tid()
    bid,l=ndof[i][0],ndof[i][1]
    a,b,c=unlin_IJK(l)
    out[i]=mass[bid,a,b,c]*direction[i]


@wp.kernel
def particle_residual(ids: wp.array(dtype=int, ndim=2), grads: wp.array(dtype=vec3, ndim=2),
                      values: wp.array(dtype=vec3), F0: wp.array(dtype=mat33), A0: wp.array(dtype=mat33),
                      volume: wp.array(dtype=real), trial: wp.array(dtype=mat33), residual: wp.array(dtype=vec3),
                      invalid: wp.array(dtype=int), dt: real, mu: real, lam: real, kf: real, variational: int):
    p=wp.tid()
    G=mat33(real(0))
    for j in range(27):
        i=ids[p,j]
        if i>=0:
            G+=wp.outer(values[i],grads[p,j])
    F=(wp.identity(3,dtype=real)+dt*G)@F0[p]
    trial[p]=F
    U, sigma, V = wp.svd3(F)
    if not wp.isfinite(wp.determinant(F)) or wp.determinant(F)<=real(0) or wp.min(sigma)<=s_min:
        wp.atomic_max(invalid,0,1)
        return
    pullback = F
    if variational != 0:
        pullback = F0[p]
    tau=aniso_pk1_wp(F,A0[p],mu,lam,kf)@wp.transpose(pullback)
    for j in range(27):
        i=ids[p,j]
        if i>=0:
            wp.atomic_add(residual,i,dt*volume[p]*(tau@grads[p,j]))


@wp.kernel
def particle_tangent(ids: wp.array(dtype=int, ndim=2), grads: wp.array(dtype=vec3, ndim=2),
                     direction: wp.array(dtype=vec3), F0: wp.array(dtype=mat33), A0: wp.array(dtype=mat33),
                     volume: wp.array(dtype=real), trial: wp.array(dtype=mat33), out: wp.array(dtype=vec3),
                     dt: real, mu: real, lam: real, kf: real, variational: int, project_pd: int):
    p=wp.tid()
    G=mat33(real(0))
    for j in range(27):
        i=ids[p,j]
        if i>=0:
            G+=wp.outer(direction[i],grads[p,j])
    F=trial[p]
    dF=dt*G@F0[p]
    dP=aniso_dpk1_wp(F,A0[p],dF,mu,lam,kf,project_pd)
    P=aniso_pk1_wp(F,A0[p],mu,lam,kf)
    dtau=dP@wp.transpose(F)+P@wp.transpose(dF)
    if variational != 0:
        dtau=dP@wp.transpose(F0[p])
    for j in range(27):
        i=ids[p,j]
        if i>=0:
            wp.atomic_add(out,i,dt*volume[p]*(dtau@grads[p,j]))


class ParticleQuadratureImplicitSolver(AnisotropicLiteImplicitSolver):
    """Controlled same-transfer baseline; shared center reserves remain allocated.

    Particle F is authoritative. Center reserves are common infrastructure,
    not claimed to be the optimized memory footprint of a conventional MPM.
    """
    quadrature_kind = "particle"

    def __init__(self, *args, **kwargs):
        if kwargs.get('direction_model','mean_tensor')!='mean_tensor':
            raise ValueError('particle quadrature uses individual fiber directions, not compressed moments')
        super().__init__(*args, **kwargs)
        if kwargs.get('energy_diagnostics', False):
            from .diagnostics import ParticleEnergyLedger
            self.energy_ledger=ParticleEnergyLedger()
        self._stencil_step = -1

    def _prepare_particle_arrays(self):
        if not hasattr(self, 'particle_trial_F') or len(self.particle_trial_F)!=self.n_ptc:
            self.particle_trial_F=wp.zeros(self.n_ptc,dtype=mat33,device=self.device)
            self.particle_node_ids=wp.empty((self.n_ptc,27),dtype=int,device=self.device)
            self.particle_grads=wp.empty((self.n_ptc,27),dtype=vec3,device=self.device)
        if not hasattr(self, 'particle_node_values') or len(self.particle_node_values)!=len(self.node_residual):
            self.particle_node_values=wp.zeros_like(self.node_residual)
        if self._stencil_step != self.sim_steps:
            wp.launch(particle_stencil,dim=self.n_ptc,inputs=[self.ptc_x,self.block2bid,self.node2dof,
                      self.particle_node_ids,self.particle_grads,self.dx,self.grid_size],device=self.device)
            self._stencil_step=self.sim_steps

    def step(self, **kwargs):
        self._stencil_step=-1
        return super().step(**kwargs)

    def _project(self, values):
        wp.launch(boundary_projection_kernel,dim=int(self.n_active_nodes.numpy()[0]),inputs=[
            self.n_active_nodes,self.ndof2bijk,self.block_xyz_by_id,values,
            self.bc_block2bid,self.bc_type,self.bc_norm,self.bc_velo,
            self.hf_bc_p,self.hf_bc_n,self.hf_bc_v,self.hf_bc_type,self.num_hf,
            self.grid_size,self.dx],device=self.device)

    def evaluate_residual(self):
        self._prepare_particle_arrays()
        self.node_residual.zero_()
        self.node_Hii_inv.zero_()
        self.aniso_invalid_trial.zero_()
        wp.launch(node_inertia,dim=int(self.n_active_nodes.numpy()[0]),inputs=[
            self.ndof2bijk,self.grid_m,self.grid_v,self.grid_v_it,self.particle_node_values,
            self.node_residual,self.node_Hii_inv,self.gravity,self.dt],device=self.device)
        wp.launch(particle_residual,dim=self.n_ptc,inputs=[self.particle_node_ids,self.particle_grads,
            self.particle_node_values,self.ptc_F,self.ptc_A0,self.ptc_vol0,self.particle_trial_F,
            self.node_residual,self.aniso_invalid_trial,self.dt,self.aniso_params.mu,self.aniso_params.lam,self.aniso_params.k_f,
            int(self.force_discretization == "variational")],device=self.device)
        self._project(self.node_residual)
        if self.enhancements is not None:
            from .enhancements import add_correction
            self.enhancements.correction()
            wp.launch(add_correction,dim=int(self.n_active_nodes.numpy()[0]),inputs=[self.enhancements.extra,self.node_residual],device=self.device)
            self._project(self.node_residual)
        return int(self.aniso_invalid_trial.numpy()[0])==0

    def _elastic_potential(self):
        from .potential import particle_potential
        wp.launch(particle_potential,dim=self.n_ptc,inputs=[self.ptc_vol0,self.particle_trial_F,
            self.ptc_A0,self.aniso_params.mu,self.aniso_params.lam,self.aniso_params.k_f,
            self._potential_sum],device=self.device)
        if self.enhancements is not None:
            from .enhancements import add_energy
            wp.launch(add_energy,dim=2,inputs=[self.enhancements.energy,self.enhancements.hg_energy,self._potential_sum],device=self.device)

    def apply_tangent(self,p,Ap,project_pd=False):
        if project_pd and self.force_discretization != "variational":
            raise ValueError("projected tangent requires variational forces")
        p = self.project_direction(p)
        Ap.zero_()
        wp.launch(node_tangent_mass,dim=int(self.n_active_nodes.numpy()[0]),inputs=[self.ndof2bijk,self.grid_m,p,Ap],device=self.device)
        wp.launch(particle_tangent,dim=self.n_ptc,inputs=[self.particle_node_ids,self.particle_grads,
            p,self.ptc_F,self.ptc_A0,self.ptc_vol0,self.particle_trial_F,Ap,self.dt,
            self.aniso_params.mu,self.aniso_params.lam,self.aniso_params.k_f,
            int(self.force_discretization == "variational"),int(project_pd)],device=self.device)
        if self.enhancements is not None:self.enhancements.tangent(p,Ap,project_pd)
        self._project(Ap)

    def _aniso_memory_bytes(self):
        common=super()._aniso_memory_bytes()
        if not hasattr(self,'particle_trial_F'):
            return common
        return common+sum(a.size*type_size_in_bytes(a.dtype) for a in
                          (self.particle_trial_F,self.particle_node_ids,self.particle_grads,self.particle_node_values))
