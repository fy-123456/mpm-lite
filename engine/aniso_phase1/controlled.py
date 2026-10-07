"""Short prescribed-motion audits using production transfer and history kernels.

All grid velocities are prescribed. These are transport/patch tests, NOT free
vibration or unconstrained mechanical equilibria.
"""
import itertools
import numpy as np
import warp as wp
from engine.types import real, vec3, mat33
from engine.sp_grid import B
from .solver import AnisotropicLiteImplicitSolver
from .types import AnisotropicMaterialParams
from .diagnostics import center_snapshot, energy_density
from .constitutive import pk1
from .tensile import grid_values


@wp.kernel
def affine_grips(nodes: wp.array(dtype=wp.vec3i), blocks: wp.array(dtype=int,ndim=3),
                 velocities: wp.array(dtype=vec3,ndim=4), G: mat33, offset: vec3, dx: real):
    i=wp.tid();n=nodes[i]
    b=blocks[n[0]//B,n[1]//B,n[2]//B]
    velocities[b,n[0]%B,n[1]%B,n[2]%B]=G@vec3(real(n[0])*dx,real(n[1])*dx,real(n[2])*dx)+offset


def rotation(angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def weighted_particle_centers(x, volumes, values, dx):
    """Independent host resampling oracle, including normalized occupied support."""
    sums={};weights={}
    for xp,V,val in zip(x,volumes,values):
        q=xp/dx-.5;base=np.floor(q).astype(int);f=q-base
        for corner in itertools.product((0,1),repeat=3):
            w=float(np.prod(np.where(corner,f,1-f)))*V
            if w<=1e-20:continue
            key=tuple(base+corner)
            weights[key]=weights.get(key,0.)+w
            sums[key]=sums.get(key,np.zeros_like(val))+w*val
    return {key:sums[key]/weight for key,weight in weights.items()}


def transport_centers(old, new_coords, dx, increment, shift, origin):
    """Host semi-Lagrangian prototype with normalized support/nearest extension.

    F is left-multiplied by the prescribed increment; A0 is only transported.
    This compares center interpolation loss, not an implemented dynamics policy.
    """
    keys=np.array(list(old));inverse=np.linalg.inv(increment);out={};extended=0
    for key_array in new_coords:
        key=tuple(key_array);world=(key_array+.5)*dx
        depart=origin+inverse@(world-origin-shift)
        q=depart/dx-.5;base=np.floor(q).astype(int);f=q-base
        accum=np.zeros((2,3,3));weight=0.
        for corner in itertools.product((0,1),repeat=3):
            src=tuple(base+corner);w=float(np.prod(np.where(corner,f,1-f)))
            if src in old and w>0:accum+=w*old[src];weight+=w
        if weight<1e-12:
            src=tuple(keys[np.argmin(np.sum((keys-q)**2,axis=1))]);accum=old[src].copy();weight=1.;extended+=1
        accum/=weight;accum[0]=increment@accum[0];out[key]=accum
    return out,extended


class KinematicScene:
    def __init__(self, mode='translate', grid=65, dt=.025, duration=.4,
                 history_mode='particle_resample', device='cpu', rotation_rule='exact',solver_options=None):
        self.mode,self.grid,self.dt,self.duration=mode,grid,dt,duration
        self.rotation_rule=rotation_rule
        self.origin=np.array([.43,.5,.5]) if mode in ('translate','directions') else np.array([.5,.5,.5])
        a=np.linspace(-.035,.035,5)
        self.reference=np.stack(np.meshgrid(a,a,a,indexing='ij'),axis=-1).reshape(-1,3)+self.origin
        self.F0=np.diag([1.12,.96,1.03]) if mode in ('translate','rotate','directions') else np.eye(3)
        angles=(self.reference[:,0]-self.origin[0])/.07*np.pi/2+np.pi/4 if mode=='directions' else np.zeros(len(self.reference))
        self.directions=np.stack([np.cos(angles),np.sin(angles),np.zeros_like(angles)],axis=1)
        self.A=np.einsum('pi,pj->pij',self.directions,self.directions)
        self.solver=s=AnisotropicLiteImplicitSolver((grid,)*3,AnisotropicMaterialParams(10.,20.,200.),
                        dx=1/(grid-1),device=device,gravity=0.,flip_ratio=0.,energy_diagnostics=True,history_mode=history_mode,**(solver_options or {}))
        Q0=self.map_at(dt)
        if mode=='rotate' and rotation_rule=='euler':
            omega=np.pi/4/duration;Q0=np.eye(3)+dt*np.array([[0,-omega,0],[omega,0,0],[0,0,0]])
        G0=(Q0-np.eye(3))/dt
        v0=(self.reference-self.origin)@G0.T
        if mode in ('translate','directions'):v0[:,0]+=.14/duration
        s.seed_particles(self.reference,density=1000.,vol0=.07**3/len(self.reference),fiber_directions=self.directions,deformation_gradient=self.F0,
                         velocity=v0,velocity_gradient=G0,reference_positions=(self.reference-self.origin)@np.linalg.inv(self.F0).T+self.origin)
        s.set_dt(dt)
        nodes=np.stack(np.meshgrid(*[np.arange(grid)]*3,indexing='ij'),axis=-1).reshape(-1,3).astype(np.int32)
        s.paint_boundary(nodes,np.ones(len(nodes),dtype=np.int32))
        self.nodes=wp.array(nodes,dtype=wp.vec3i,device=device)
        s.prepare_centers()
        # Both policies start with the SAME nonzero center history.
        coords,vol,_,active=center_snapshot(s)
        Fs=s.aniso_committed_F[:,:int(s.bcn)].numpy().copy();Fs.reshape(-1,3,3)[active]=self.F0
        s.aniso_committed_F[:,:int(s.bcn)].assign(wp.array(Fs,dtype=mat33,device=device))
        As=s.aniso_A0[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active]
        self.transported={tuple(c):np.stack([self.F0,A]) for c,A in zip(coords,As)}
        self.reference_energy=float(np.dot(vol,energy_density(np.broadcast_to(self.F0,As.shape),As,s.aniso_params)))
        self.reference_F=self.F0.copy();self.reference_x=self.reference.copy();self.rows=[];self.frames=[]
        self.work=0.;self.pending_increment=np.eye(3);self.pending_shift=np.zeros(3)
        s.energy_ledger.begin(s);s.energy_ledger.initialize_centers(s)
        self.record(initial=True)

    def map_at(self,t):
        if self.mode=='rotate':return rotation(np.pi/4*t/self.duration)
        if self.mode=='tension':return np.diag([1+.06*t/self.duration,1.,1.])
        if self.mode=='shear':
            q=np.eye(3);q[0,1]=.10*t/self.duration;return q
        return np.eye(3)

    def step(self):
        s=self.solver;t=s.sim_time
        if t>=self.duration-1e-12:return False
        Q=self.map_at(t+self.dt)@np.linalg.inv(self.map_at(t))
        if self.mode=='rotate' and self.rotation_rule=='euler':
            omega=np.pi/4/self.duration;Q=np.eye(3)+self.dt*np.array([[0,-omega,0],[omega,0,0],[0,0,0]])
        shift=np.array([.14/self.duration*self.dt,0,0]) if self.mode in ('translate','directions') else np.zeros(3)
        G=(Q-np.eye(3))/self.dt;offset=shift/self.dt-G@self.origin
        wp.launch(affine_grips,dim=len(self.nodes),inputs=[self.nodes,s.bc_block2bid,s.bc_velo,mat33(G),vec3(offset),s.dx],device=s.device)
        if not s.step(max_iters=8,print_every=0,newton_atol=1e-8,cg_atol=1e-10):
            raise RuntimeError(str(s.last_step_stats))
        self.reference_x=(self.reference_x-self.origin)@Q.T+self.origin+shift
        self.reference_F=Q@self.reference_F
        self.increment,self.shift=Q,shift
        self.record()
        return True

    def record(self,initial=False):
        s=self.solver;coords,vol,psi,active=center_snapshot(s);n=int(s.bcn)
        Fs=s.aniso_committed_F[:,:n].numpy()[0].reshape(-1,3,3)[active]
        As=s.aniso_A0[:,:n].numpy()[0].reshape(-1,3,3)[active]
        # Centers describe the material sampled at BEGINNING of this time step.
        sampling_x=self.reference_x if initial else self.previous_x
        oracle=weighted_particle_centers(sampling_x,s.ptc_vol0.numpy(),self.A,s.dx)
        targetA=np.array([oracle[tuple(c)] for c in coords])
        targetF=np.broadcast_to(self.reference_F,Fs.shape)
        targetpsi=energy_density(targetF,targetA,s.aniso_params)
        elastic=float(np.dot(vol,psi));expected=float(np.dot(vol,targetpsi))
        tau=np.array([pk1(f,a,s.aniso_params)@f.T for f,a in zip(Fs,As)])
        targettau=np.array([pk1(f,a,s.aniso_params)@f.T for f,a in zip(targetF,targetA)])
        def rms(a):return float(np.sqrt(np.sum(vol[:,None,None]*a*a)/vol.sum()))
        row=dict(time=float(s.sim_time),F_relative_error=rms(Fs-targetF)/np.linalg.norm(self.reference_F),
                 A_rms_error=rms(As-targetA),stress_relative_error=rms(tau-targettau)/max(rms(targettau),1e-12),
                 elastic=elastic,expected_elastic=expected,elastic_relative_error=(elastic-expected)/max(abs(expected),1e-12),
                 particle_position_max_error=float(np.max(np.linalg.norm(s.ptc_x.numpy()-self.reference_x,axis=1))),
                 particle_F_max_error=float(np.max(np.linalg.norm(s.ptc_F.numpy()-targetF[0],axis=(1,2)))),
                 min_det_F=float(np.min(np.linalg.det(Fs))),active_blocks=int(s.bcn),
                 reactivation_events=int(s.aniso_reactivation_count.numpy()[0]))
        if not initial:
            # Transport stored center states to the quadrature positions at t_n,
            # then apply this step's deformation. No double advection.
            transported,extended=transport_centers(self.transported,coords,s.dx,self.pending_increment,self.pending_shift,self.origin)
            for item in transported.values():item[0]=self.increment@np.linalg.inv(self.pending_increment)@item[0]
            self.transported=transported
            self.pending_increment,self.pending_shift=self.increment,self.shift
            TF=np.array([transported[tuple(c)][0] for c in coords]);TA=np.array([transported[tuple(c)][1] for c in coords])
            row.update(center_transport_F_error=rms(TF-targetF)/np.linalg.norm(self.reference_F),
                       center_transport_A_error=rms(TA-targetA),center_transport_extension_count=extended)
            ledger=s.energy_ledger;grad=(2*ledger.corners-1)/(4*s.dx)
            F0=ledger.step_start_F[0].reshape(-1,3,3)[active]
            pull=np.array([pk1(f,a,s.aniso_params)@f0.T for f,a,f0 in zip(Fs,As,F0)])
            forces=np.einsum('p,pij,cj->pci',vol,pull,grad)
            internal=np.stack([np.bincount(ledger.inverse,weights=forces[:,:,d].reshape(-1)) for d in range(3)],axis=1)
            v=grid_values(s,s.grid_v_new,ledger.nodes)
            applied=internal+ledger.node_m[:,None]*(v-ledger.raw_v)/s.dt
            self.work+=float(s.dt*np.sum(applied*v))
            row.update(external_work=self.work,mechanical_minus_work=ledger.rows[-1]['mechanical']-ledger.rows[0]['mechanical']-self.work,
                       state_transport_delta=ledger.rows[-1]['state_transport_delta'],state_reset_delta=ledger.rows[-1]['state_reset_delta'])
        if self.mode=='rotate':
            exact=rotation(np.pi/4*s.sim_time/self.duration)@self.F0
            euler_energy=float(energy_density(np.array([self.reference_F]),np.array([s.aniso_params.A0]),s.aniso_params)[0])
            exact_energy=float(energy_density(np.array([exact]),np.array([s.aniso_params.A0]),s.aniso_params)[0])
            row['integration_energy_relative']=(euler_energy-exact_energy)/max(exact_energy,1e-12)
        self.previous_x=self.reference_x.copy()
        current=np.einsum('pij,pjk,plk->pil',Fs,As,Fs)
        values,vectors=np.linalg.eigh(current);directions=vectors[:,:,-1]
        self.frames.append(dict(points=s.ptc_x.numpy().copy(),centers=(coords+.5)*s.dx,directions=directions))
        self.rows.append(row)
