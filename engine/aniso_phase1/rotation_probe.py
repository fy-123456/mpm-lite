"""Exact prescribed rotation of a nonuniform bent state, plus a host prototype.

The corotated Q1 potential is a reference-cell research prototype, NOT a new
production stabilizer. It differentiates the whole polar-dependent potential.
"""
import itertools
import numpy as np
import warp as wp
from engine.types import mat33, vec3
from .solver import AnisotropicLiteImplicitSolver
from .types import AnisotropicMaterialParams
from .controlled import affine_grips, rotation, weighted_particle_centers
from .stabilization_probe import bent_beam_state, node_coordinates
from .diagnostics import center_snapshot, energy_density
from .constitutive import pk1
from .beam_reference import shape_gradients, reference_hessian
from engine.sp_grid import B


def axis_rotation(angle,axis='z'):
    if axis=='z':return rotation(angle)
    if axis!='y':raise ValueError('supported rotation axes: y, z')
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def frozen_stencil_audit(s):
    """Rotate an existing spatial stencil and material state together.

    No rebinning, no time integration: Xhat stays in reference coordinates,
    x and gradients rotate. Since sum(x_i tensor dg_i)=0, D transforms as
    D Q^T for the production reconstructed u=x-Xhat representation.
    """
    e=s.enhancements;ids=e.ids.numpy();u=e.u.numpy()
    D=np.einsum('cni,qnj->cqij',u[ids],e.dg.numpy())
    cd=s.cdof2bijk[:len(ids)].numpy();volume=s.center_vol[:,:int(s.bcn)].numpy().reshape(-1)
    V=volume[cd[:,0]*B**3+cd[:,1]]
    H=20*np.eye(9) if s.stabilization=='hourglass' else reference_hessian()
    rows=[];initial=None
    for axis in ('z','y'):
        for degrees in (0,30,60,90):
            rotated=(D@axis_rotation(np.deg2rad(degrees),axis).T).reshape(len(ids),8,9)
            E=float(np.einsum('c,cqi,ij,cqj->',V,rotated,H,rotated)/16)
            if initial is None:initial=E
            rows.append(dict(axis=axis,angle=degrees,energy=E,relative_change=E/initial-1))
    return rows


def cell_potential(y, X, dx, kind='supplemental', corotated=False):
    """Reference Q1 cell; reference gradients/volume fixed throughout FD."""
    g0=shape_gradients((.5,)*3,dx)
    q=(.5-1/(2*np.sqrt(3)),.5+1/(2*np.sqrt(3)))
    dg=np.array([shape_gradients(p,dx)-g0 for p in itertools.product(q,repeat=3)])
    D=np.einsum('ni,qnj->qij',y-X,dg)
    if corotated:
        U,_,Vt=np.linalg.svd(y.T@g0)
        R=U@Vt
        if np.linalg.det(R)<=0:raise ValueError('polar prototype requires positive J')
        D=np.einsum('ji,qjk->qik',R,D)
    H=20*np.eye(9) if kind=='hourglass' else reference_hessian()
    return float(dx**3/16*np.einsum('qi,ij,qj->',D.reshape(8,9),H,D.reshape(8,9)))


def cell_force(y,X,dx,kind,corotated):
    # Diagnostic finite differences include dR, unlike a frozen-rotation force.
    eps=dx*1e-5;f=np.zeros_like(y)
    for i in range(y.size):
        d=np.zeros_like(y);d.flat[i]=eps
        f.flat[i]=-(cell_potential(y+d,X,dx,kind,corotated)-cell_potential(y-d,X,dx,kind,corotated))/(2*eps)
    return f


def reference_frame_audit():
    dx=1/16;X=np.array(list(itertools.product((0,1),repeat=3)))*dx+[.5,.4375,.4375]
    y,_=bent_beam_state(X,amplitude=.01);origin=X.mean(axis=0);rows=[]
    for kind in ('supplemental','hourglass'):
        for corotated in (False,True):
            E0=cell_potential(y,X,dx,kind,corotated);f0=cell_force(y,X,dx,kind,corotated)
            for degrees in (0,30,60,90):
                R=rotation(np.deg2rad(degrees));yr=(y-origin)@R.T+origin
                E=cell_potential(yr,X,dx,kind,corotated);f=cell_force(yr,X,dx,kind,corotated)
                rows.append(dict(stabilization=kind,corotated=corotated,angle=degrees,energy=E,
                    relative_energy_change=E/E0-1,force_covariance_error=float(np.linalg.norm(f-f0@R.T)/np.linalg.norm(f0)),
                    torque_z=float(np.cross(yr-origin,f).sum(axis=0)[2])))
    return rows


class BentRotation:
    """All velocities prescribed; finite strain transport, not free dynamics."""
    def __init__(self,grid=17,dt=.01,duration=.08,mode='none',device='cpu',angle=np.pi/2,axis='z'):
        self.dt,self.duration,self.angle=dt,duration,angle
        self.axis=axis
        self.origin=np.array([.5,.5,.5])
        lo=np.array([.25,.4375,.4375]);hi=np.array([.75,.5625,.5625])
        # Two samples per cell axis on both grids; total volume/mass fixed.
        counts=np.rint((hi-lo)*(grid-1)*2).astype(int)
        axes=[lo[d]+(np.arange(counts[d])+.5)*(hi[d]-lo[d])/counts[d] for d in range(3)]
        self.X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        self.x0,self.F0=bent_beam_state(self.X,amplitude=.01)
        self.solver=s=AnisotropicLiteImplicitSolver((grid,)*3,AnisotropicMaterialParams(10,20,200),dx=1/(grid-1),device=device,
            gravity=0,flip_ratio=0,energy_diagnostics=True,stabilization=mode,direction_model='fourth_moment')
        self.Q=axis_rotation(angle*dt/duration,axis);G=(self.Q-np.eye(3))/dt
        s.seed_particles(self.x0,density=1,vol0=.5*.125**2/len(self.X),deformation_gradient=self.F0,reference_positions=self.X,
            velocity=(self.x0-self.origin)@G.T,velocity_gradient=G)
        s.set_dt(dt)
        nodes=np.stack(np.meshgrid(*[np.arange(grid)]*3,indexing='ij'),axis=-1).reshape(-1,3).astype(np.int32)
        s.paint_boundary(nodes,np.ones(len(nodes),dtype=np.int32),boundary_v=(nodes*s.dx-self.origin)@G.T)
        self.nodes=wp.array(nodes,dtype=wp.vec3i,device=device)
        self.rows=[];self.frames=[self.x0.copy()];self.fibers=[self.F0[:,:,0].copy()]
        self.particle_energy0=float(np.dot(s.ptc_vol0.numpy(),energy_density(self.F0,s.ptc_A0.numpy(),s.aniso_params)))

    def step(self):
        s=self.solver;t=s.sim_time;R0=axis_rotation(self.angle*t/self.duration,self.axis);R=axis_rotation(self.angle*(t+self.dt)/self.duration,self.axis)
        oldx=(self.x0-self.origin)@R0.T+self.origin
        G=(self.Q-np.eye(3))/self.dt
        wp.launch(affine_grips,dim=len(self.nodes),inputs=[self.nodes,s.bc_block2bid,s.bc_velo,mat33(G),vec3(-G@self.origin),s.dx],device=s.device)
        if not s.step(max_iters=8,print_every=0,newton_atol=1e-8):raise RuntimeError(s.last_step_stats)
        coords,V,psi,active=center_snapshot(s)
        Fc=s.aniso_committed_F[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active]
        oracle=weighted_particle_centers(oldx,s.ptc_vol0.numpy(),self.F0,s.dx)
        base=np.array([oracle[tuple(c)] for c in coords]);target=R@base
        A=s.aniso_params.A0
        tau=np.array([pk1(f,A,s.aniso_params)@f.T for f in Fc])
        tau0=np.array([pk1(f,A,s.aniso_params)@f.T for f in base]);expected=R@tau0@R.T
        f=Fc[:,:,0].copy();ft=target[:,:,0].copy();f/=np.linalg.norm(f,axis=1)[:,None];ft/=np.linalg.norm(ft,axis=1)[:,None]
        Fp=s.ptc_F.numpy();x=s.ptc_x.numpy();Up=float(np.dot(s.ptc_vol0.numpy(),energy_density(Fp,s.ptc_A0.numpy(),s.aniso_params)))
        hg=float(s.enhancements.hg_energy.numpy()[0]);row=dict(time=s.sim_time,angle_degrees=self.angle*s.sim_time/self.duration*180/np.pi,
            stabilization_sampling_angle_degrees=self.angle*t/self.duration*180/np.pi,
            center_material_energy=float(np.dot(V,psi)),stabilization_energy=hg,particle_material_energy=Up,
            particle_energy_relative_change=Up/self.particle_energy0-1,
            particle_F_max_error=float(np.linalg.norm(Fp-R@self.F0,axis=(1,2)).max()),
            history_F_relative_error=float(np.linalg.norm(Fc-target)/np.linalg.norm(target)),
            stress_rotation_relative_error=float(np.linalg.norm(tau-expected)/np.linalg.norm(expected)),
            fiber_rotation_max_error=float(np.linalg.norm(f-ft,axis=1).max()),
            position_max_error=float(np.linalg.norm(x-((self.x0-self.origin)@R.T+self.origin),axis=1).max()),
            min_J=float(np.linalg.det(Fp).min()))
        out=wp.zeros_like(s.node_residual);e=wp.zeros_like(s.enhancements.hg_energy)
        s.enhancements._hg(s.enhancements.values,out,e,False)
        n=int(s.n_active_nodes.numpy()[0]);forces=-out[:n].numpy()/s.dt;xyz=node_coordinates(s)*s.dx
        row.update(stabilization_force_norm=float(np.linalg.norm(forces)),
            stabilization_net_force=float(np.linalg.norm(forces.sum(axis=0))),
            frozen_step_stabilization_torque=float(np.linalg.norm(np.cross(xyz-self.origin,forces).sum(axis=0))),
            trial_configuration_stabilization_torque=float(np.linalg.norm(np.cross(xyz+s.dt*s.enhancements.values[:n].numpy()-self.origin,forces).sum(axis=0))))
        self.rows.append(row);self.frames.append(x.copy());self.fibers.append(Fp[:,:,0].copy())
        return row
