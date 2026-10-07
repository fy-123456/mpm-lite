"""Experimental group4x8 material integration with frozen P2+xyz MLS maps.

Grouping, Hermite history reconstruction and nodal maps run once between steps.
The solve visits only compact samples/nodes. Existing Lite particle transfers
are intentionally retained; their mismatch with this wider material map is
reported as a separate reconstruction jump, not hidden as integration loss.
"""
import time
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from warp._src.types import type_size_in_bytes
from engine.types import real, vec3, mat33, mat99
from engine.kernel.constant import s_min
from .solver import AnisotropicLiteImplicitSolver
from .particle_quadrature import node_inertia, node_tangent_mass, ParticleQuadratureImplicitSolver
from .material_snapshot import MaterialSnapshot, center_support, reconstructed_F, response
from .joint_sampling import build_rule
from .quadratic import history_polynomial, node_patch
from .kernels import aniso_energy_wp, aniso_pk1_wp, aniso_dpk1_wp, mat_to_vec9_wp, vec9_to_mat_wp
from .enhancements import add_correction, add_energy
from .potential import material_roundoff_scale
from .diagnostics import EnergyLedger, energy_density


def freeze_material(snapshot,h,params):
    """Reference weights, conditional moments and F0; no grid/solver mutation."""
    tree=cKDTree(snapshot.x);records=[]
    for key,ids,w in center_support(snapshot,h):
        center=(key+.5)*h
        coef,_=history_polynomial(snapshot.x,snapshot.X,snapshot.F,snapshot.volume,center,h,tree)
        x,V,A,M=build_rule(snapshot,ids,w,h,'group4x8')
        F=reconstructed_F(coef,x,center,h)
        records.append((center,x,V,A,M,F))
    start=0.
    for _,_,V,A,M,F in records:
        logs=np.log(np.linalg.svd(F,compute_uv=False));strain=(F.transpose(0,2,1)@F-np.eye(3)).reshape(-1,9)
        energy=params.mu*np.sum(logs*logs,axis=1)+.5*params.lam*logs.sum(axis=1)**2
        energy+=.5*params.k_f*np.einsum('qi,qij,qj->q',strain,M,strain)
        start+=float(V@energy)
    return records,start


def freeze_maps(records,nodes,h):
    tree=cKDTree(nodes);patches=[];gradients=[];affine=0.
    for center,x,V,A,M,F in records:
        ids,g,_,_=node_patch(nodes,center,h,tree,queries=x)
        affine=max(affine,float(np.max(abs(np.einsum('ni,qnj->qij',nodes[ids]-center,g)-np.eye(3)))))
        patches.extend([ids]*len(V));gradients.extend(np.einsum('qji,qnj->qni',F,g))
    if affine>1e-7:raise ValueError('grouped map lost affine reproduction')
    width=max(map(len,patches));ids=np.full((len(patches),width),-1,dtype=np.int32)
    B=np.zeros((len(patches),width,3))
    for q,(patch,g) in enumerate(zip(patches,gradients)):ids[q,:len(patch)]=patch;B[q,:len(patch)]=g
    return ids,B,affine


@wp.func
def grouped_energy(F: mat33,A: mat33,M: mat99,mu: real,lam: real,kf: real):
    # A4:I=A2 for unit fibers: the strain form avoids cancellation at rest.
    strain=mat_to_vec9_wp(wp.transpose(F)@F-wp.identity(3,dtype=real))
    return aniso_energy_wp(F,A,mu,lam,real(0))+real(.5)*kf*wp.dot(strain,M@strain)


@wp.func
def grouped_P(F: mat33,A: mat33,M: mat99,mu: real,lam: real,kf: real):
    S=vec9_to_mat_wp(M@mat_to_vec9_wp(wp.transpose(F)@F-wp.identity(3,dtype=real)))
    return aniso_pk1_wp(F,A,mu,lam,real(0))+real(2)*kf*F@S


@wp.func
def grouped_dP(F: mat33,A: mat33,M: mat99,dF: mat33,mu: real,lam: real,kf: real,pd: int):
    S=vec9_to_mat_wp(M@mat_to_vec9_wp(wp.transpose(F)@F-wp.identity(3,dtype=real)))
    if pd!=0:
        Q,e=wp.eig3(S)
        e=vec3(wp.max(e[0],real(0)),wp.max(e[1],real(0)),wp.max(e[2],real(0)))
        S=Q@wp.diag(e)@wp.transpose(Q)
    dS=vec9_to_mat_wp(M@mat_to_vec9_wp(wp.transpose(dF)@F+wp.transpose(F)@dF))
    return aniso_dpk1_wp(F,A,dF,mu,lam,real(0),pd)+real(2)*kf*(dF@S+F@dS)


@wp.kernel
def sample_residual(ids: wp.array(dtype=int,ndim=2), B: wp.array(dtype=vec3,ndim=2),
                    v: wp.array(dtype=vec3), F0: wp.array(dtype=mat33), A: wp.array(dtype=mat33),
                    M: wp.array(dtype=mat99), V: wp.array(dtype=real), Ftrial: wp.array(dtype=mat33),
                    out: wp.array(dtype=vec3), invalid: wp.array(dtype=int), dt: real,mu: real,lam: real,kf: real):
    q=wp.tid();F=F0[q]
    for j in range(ids.shape[1]):
        i=ids[q,j]
        if i>=0:F+=dt*wp.outer(v[i],B[q,j])
    Ftrial[q]=F
    U,sigma,W=wp.svd3(F)
    if not wp.isfinite(wp.determinant(F)) or wp.determinant(F)<=real(0) or wp.min(sigma)<=s_min:
        wp.atomic_max(invalid,0,1)
        return
    P=grouped_P(F,A[q],M[q],mu,lam,kf)
    for j in range(ids.shape[1]):
        i=ids[q,j]
        if i>=0:wp.atomic_add(out,i,dt*V[q]*(P@B[q,j]))


@wp.kernel
def sample_tangent(ids: wp.array(dtype=int,ndim=2), B: wp.array(dtype=vec3,ndim=2),
                   p: wp.array(dtype=vec3), Ftrial: wp.array(dtype=mat33), A: wp.array(dtype=mat33),
                   M: wp.array(dtype=mat99), V: wp.array(dtype=real), out: wp.array(dtype=vec3),
                   dt: real,mu: real,lam: real,kf: real,pd: int):
    q=wp.tid();dF=mat33(real(0))
    for j in range(ids.shape[1]):
        i=ids[q,j]
        if i>=0:dF+=dt*wp.outer(p[i],B[q,j])
    F=Ftrial[q]
    dP=grouped_dP(F,A[q],M[q],dF,mu,lam,kf,pd)
    for j in range(ids.shape[1]):
        i=ids[q,j]
        if i>=0:wp.atomic_add(out,i,dt*V[q]*(dP@B[q,j]))


@wp.kernel
def sample_energy(F: wp.array(dtype=mat33),A: wp.array(dtype=mat33),M: wp.array(dtype=mat99),
                  V: wp.array(dtype=real),out: wp.array(dtype=real),mu: real,lam: real,kf: real):
    q=wp.tid();C=mat_to_vec9_wp(wp.transpose(F[q])@F[q]);a=mat_to_vec9_wp(A[q])
    wp.atomic_add(out,0,V[q]*grouped_energy(F[q],A[q],M[q],mu,lam,kf))
    scale=material_roundoff_scale(F[q],A[q],mu,lam,kf)+kf*(wp.abs(wp.dot(C,M[q]@C))+wp.dot(C,a)*wp.dot(C,a))
    wp.atomic_add(out,1,V[q]*scale)


class GroupedEnergyLedger(EnergyLedger):
    """Budget uses solved frozen sample energy; rebuilding is a separate term."""
    reading_start=True

    def elastic_energy(self,s,snapshot=None):
        return s.group_start_energy if self.reading_start else self.solved_material

    def p2c(self,s):
        self.reading_start=True
        super().p2c(s)
        self.u_start=s.group_start_energy
        previous_hg=self.rows[-1].get('stabilization_energy',0.) if s.sim_steps else 0.
        previous_material=self.previous_elastic-previous_hg
        delta=self.u_start-previous_material
        self.current.update(state_reset_delta=0.,state_transport_delta=delta,
                            volume_remap_delta=-previous_hg,group_rebuild_delta=delta,
                            group_start_energy=self.u_start)

    def solved(self,s):
        super().solved(s)
        self.solved_material=s.group_material_energy()
        self.reading_start=False
        self.current['group_solved_energy']=self.solved_material
        # Preserve the unprojected *material* force for correct grip reactions.
        self.group_internal=s.group_internal_force()

    def finish(self,s):
        self.current['particle_reference_elastic']=float(s.ptc_vol0.numpy()@energy_density(s.ptc_F.numpy(),s.ptc_A0.numpy(),s.aniso_params))
        self.current['particle_minus_group_elastic']=self.current['particle_reference_elastic']-self.solved_material
        super().finish(s)


class GroupedQuadratureImplicitSolver(AnisotropicLiteImplicitSolver):
    """Opt-in research path; no assertion of converged beam accuracy or speedup."""
    quadrature_kind='group4x8'
    _project=ParticleQuadratureImplicitSolver._project

    def __init__(self,*args,**kwargs):
        if kwargs.get('force_discretization','variational')!='variational' or kwargs.get('history_mode','particle_resample')!='particle_resample':
            raise ValueError('group4x8 requires variational forces and particle_resample history')
        kwargs['direction_model']='fourth_moment'
        super().__init__(*args,**kwargs)
        if kwargs.get('energy_diagnostics',False):self.energy_ledger=GroupedEnergyLedger()
        self.group_ready=False;self.group_stats={}

    def _transfer_to_centers(self):
        self.group_ready=False
        if not super()._transfer_to_centers():return False
        start=time.perf_counter()
        snapshot=MaterialSnapshot(*(a.numpy() for a in (self.ptc_x,self.ptc_reference_x,self.ptc_F,self.ptc_A0,self.ptc_vol0)))
        if np.any(snapshot.x<.5*self.dx) or np.any(snapshot.x>(np.array(tuple(self.center_size))-.5)*self.dx):
            raise ValueError('group4x8 requires complete interior particle support')
        self.group_records,self.group_start_energy=freeze_material(snapshot,self.dx,self.aniso_params)
        self.group_stats=dict(group_history_seconds=time.perf_counter()-start,
            group_samples=sum(len(r[2]) for r in self.group_records),group_centers=len(self.group_records))
        return True

    def _prepare_groups(self):
        if self.group_ready:return
        from .stabilization_probe import node_coordinates
        start=time.perf_counter();ids,B,affine=freeze_maps(self.group_records,node_coordinates(self)*self.dx,self.dx)
        self.group_ids=wp.array(ids,dtype=int,device=self.device);self.group_B=wp.array(B,dtype=vec3,device=self.device)
        self.group_V,self.group_A,self.group_M,self.group_F0=[wp.array(np.concatenate([r[i] for r in self.group_records]),dtype=dtype,device=self.device) for i,dtype in ((2,real),(3,mat33),(4,mat99),(5,mat33))]
        self.group_trial=wp.clone(self.group_F0)
        self.group_values=wp.zeros_like(self.node_residual);self.group_force=wp.zeros_like(self.node_residual)
        self.group_energy=wp.zeros(2,dtype=real,device=self.device)
        self.group_stats.update(group_map_seconds=time.perf_counter()-start,group_map_width=ids.shape[1],group_affine_error=affine)
        self.group_ready=True

    def _material_residual(self,out):
        p=self.aniso_params
        wp.launch(sample_residual,dim=len(self.group_V),inputs=[self.group_ids,self.group_B,self.group_values,
            self.group_F0,self.group_A,self.group_M,self.group_V,self.group_trial,out,self.aniso_invalid_trial,
            self.dt,p.mu,p.lam,p.k_f],device=self.device)

    def evaluate_residual(self):
        self._prepare_groups();self.node_residual.zero_();self.node_Hii_inv.zero_();self.aniso_invalid_trial.zero_()
        wp.launch(node_inertia,dim=int(self.n_active_nodes.numpy()[0]),inputs=[self.ndof2bijk,self.grid_m,self.grid_v,
            self.grid_v_it,self.group_values,self.node_residual,self.node_Hii_inv,self.gravity,self.dt],device=self.device)
        self._material_residual(self.node_residual)
        self.enhancements.correction()
        wp.launch(add_correction,dim=int(self.n_active_nodes.numpy()[0]),inputs=[self.enhancements.extra,self.node_residual],device=self.device)
        self._project(self.node_residual)
        return int(self.aniso_invalid_trial.numpy()[0])==0

    def _material_energy(self,out):
        p=self.aniso_params
        wp.launch(sample_energy,dim=len(self.group_V),inputs=[self.group_trial,self.group_A,self.group_M,self.group_V,out,p.mu,p.lam,p.k_f],device=self.device)

    def _elastic_potential(self):
        self._material_energy(self._potential_sum)
        wp.launch(add_energy,dim=2,inputs=[self.enhancements.energy,self.enhancements.hg_energy,self._potential_sum],device=self.device)

    def group_material_energy(self):
        self.group_energy.zero_();self._material_energy(self.group_energy)
        return float(self.group_energy.numpy()[0])

    def group_internal_force(self):
        self.group_force.zero_();self._material_residual(self.group_force)
        return self.group_force[:int(self.n_active_nodes.numpy()[0])].numpy()/self.dt

    def apply_tangent(self,p,Ap,project_pd=False):
        p=self.project_direction(p);Ap.zero_();a=self.aniso_params
        wp.launch(node_tangent_mass,dim=int(self.n_active_nodes.numpy()[0]),inputs=[self.ndof2bijk,self.grid_m,p,Ap],device=self.device)
        wp.launch(sample_tangent,dim=len(self.group_V),inputs=[self.group_ids,self.group_B,p,self.group_trial,self.group_A,self.group_M,
            self.group_V,Ap,self.dt,a.mu,a.lam,a.k_f,int(project_pd)],device=self.device)
        self.enhancements.tangent(p,Ap,project_pd);self._project(Ap)

    def step(self,**kwargs):
        success=super().step(**kwargs)
        self.last_step_stats.update(self.group_stats)
        return success

    def _aniso_memory_bytes(self):
        return super()._aniso_memory_bytes()+sum(a.size*type_size_in_bytes(a.dtype) for k,a in vars(self).items() if k.startswith('group_') and isinstance(a,wp.array))
