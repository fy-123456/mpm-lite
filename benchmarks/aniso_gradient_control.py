"""Experimental history-only gradient intervention; production solver is unchanged.

The optional intervention updates only particle F after a successful commit.
Position, velocity, APIC gradient, center solve, and force derivatives retain the
production formulas. This is a causal diagnostic, not a variational replacement.
"""
import itertools
import numpy as np
from engine.sp_grid import B
from engine.aniso_phase1.transfer_audit import TransferAuditLedger, center_weights, average, tensor_rms
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.constitutive import pk1


def sample_centers(x, coords, values, dx):
    """Independent complete-support trilinear interpolation and its derivative."""
    lookup={tuple(c):i for i,c in enumerate(coords)}
    q=x/dx-.5;base=np.floor(q).astype(int);fraction=q-base
    result=np.zeros((len(x),3));derivative=np.zeros((len(x),3,3))
    for corner in itertools.product((0,1),repeat=3):
        indices=np.array([lookup.get(tuple(c),-1) for c in base+corner])
        if np.any(indices<0):raise ValueError('gradient probe requires complete occupied center support')
        factor=np.where(corner,fraction,1-fraction)
        weights=np.prod(factor,axis=1)
        dw=np.stack([(2*corner[d]-1)/dx*np.prod(np.delete(factor,d,axis=1),axis=1) for d in range(3)],axis=1)
        v=values[indices]
        result+=weights[:,None]*v
        derivative+=v[:,:,None]*dw[:,None,:]
    return result,derivative


def sparse_center_velocities(s):
    n=int(s.bcn);mass=s.center_m[:n].numpy().reshape(-1);active=mass>0
    indices=np.flatnonzero(active)
    local=np.stack(np.unravel_index(indices%B**3,(B,B,B)),axis=1)
    coords=s.block_xyz_by_id[:n].numpy()[indices//B**3]*B+local
    return coords,s.center_v[:n].numpy().reshape(-1,3)[active].copy()


def stress(F,A,params):
    return np.array([pk1(f,a,params) for f,a in zip(F,A)])


class GradientControlLedger(TransferAuditLedger):
    def __init__(self,mode='baseline'):
        if mode not in ('baseline','pic_history'):raise ValueError('unknown gradient control mode')
        super().__init__();self.mode=mode

    def begin(self,s):
        super().begin(s)
        if self.mode!='baseline' or self.audit_next:
            self.history_x=s.ptc_x.numpy().copy()
            self.history_F=s.ptc_F.numpy().copy()

    def finish(self,s):
        # First record the untouched production update and its exact decomposition.
        super().finish(s)
        if self.mode=='baseline' and not self.audit_next:return
        coords,velocity=sparse_center_velocities(s)
        pic,Gpic=sample_centers(self.history_x,coords,velocity,s.dx)
        Fprod=s.ptc_F.numpy().copy()
        Fpic=(np.eye(3)+s.dt*Gpic)@self.history_F
        if not np.isfinite(Fpic).all() or np.any(np.linalg.det(Fpic)<=0):
            raise ValueError('invalid history-control F; abort diagnostic case')
        row=self.rows[-1]
        row['history_gradient_mode']=self.mode
        row['history_gradient_difference_rms']=float(np.sqrt(np.mean(np.sum((Gpic-s.ptc_L.numpy())**2,axis=(1,2)))))
        row['history_control_F_difference_rms']=float(np.sqrt(np.mean(np.sum((Fpic-Fprod)**2,axis=(1,2)))))
        row['pic_advection_oracle_max_error']=float(np.max(abs(s.ptc_x.numpy()-self.history_x-s.dt*pic)))
        if self.audit_next:
            # CPU Warp numpy arrays may alias live state. Freeze before intervention.
            self.snapshot={key:np.array(value,copy=True) for key,value in self.snapshot.items()}
            snap=self.snapshot;W=self.weights;vol=snap['volume'];A=snap['center_A0']
            Fc=snap['center_F_committed'];Ffrozen=average(W,Fprod);Falternative=average(W,Fpic)
            covariance=snap['covariance_increment'];gap=snap['gradient_gap_increment']
            variants={'committed':Fc,'frozen':Ffrozen,'without_gradient_gap':Fc+covariance,
                      'without_covariance':Fc+gap,'pic_history':Falternative}
            Ps={key:stress(F,A,s.aniso_params) for key,F in variants.items()}
            energies={key:float(vol@energy_density(F,A,s.aniso_params)) for key,F in variants.items()}
            self.audit.update(mode=self.mode,
                history_gradient_difference_rms=row['history_gradient_difference_rms'],
                pic_advection_oracle_max_error=row['pic_advection_oracle_max_error'],
                gradient_gap_rate_rms=tensor_rms(gap/s.dt,vol),
                covariance_rate_rms=tensor_rms(covariance/s.dt,vol),
                moving_weight_rate_rms=tensor_rms(snap['moving_weight_increment']/s.dt,vol),
                gradient_gap_stress_effect_rms_Pa=tensor_rms(Ps['frozen']-Ps['without_gradient_gap'],vol),
                covariance_stress_effect_rms_Pa=tensor_rms(Ps['frozen']-Ps['without_covariance'],vol),
                pic_history_stress_change_rms_Pa=tensor_rms(Ps['pic_history']-Ps['frozen'],vol),
                frozen_stress_gap_rms_Pa=tensor_rms(Ps['frozen']-Ps['committed'],vol),
                frozen_energy_gap_J=energies['frozen']-energies['committed'],
                gradient_gap_energy_effect_J=energies['frozen']-energies['without_gradient_gap'],
                covariance_energy_effect_J=energies['frozen']-energies['without_covariance'],
                pic_history_energy_change_J=energies['pic_history']-energies['frozen'],
                pic_history_frozen_F_gap_rms=tensor_rms(Falternative-Fc,vol))
            # Locate error in material currently inside grips versus free material.
            for name,mask in [('grip', (self.history_x[:,0]<=.25)|(self.history_x[:,0]>=.75)),
                              ('free',(self.history_x[:,0]>.25)&(self.history_x[:,0]<.75))]:
                self.audit[name+'_gradient_difference_rms']=float(np.sqrt(np.mean(np.sum((Gpic-s.ptc_L.numpy())[mask]**2,axis=(1,2)))))
            self.snapshot.update(center_velocity_coords=coords,center_velocity=velocity,
                particle_gradient_pic=Gpic,particle_F_pic=Fpic,
                particle_volume=s.ptc_vol0.numpy().copy(),
                frozen_F=Ffrozen,without_gradient_gap_F=variants['without_gradient_gap'],
                frozen_P=Ps['frozen'],without_gradient_gap_P=Ps['without_gradient_gap'],pic_history_P=Ps['pic_history'])
        if self.mode=='pic_history':
            # Keep the APIC gradient intact: isolate only the deformation history.
            s.ptc_F.assign(Fpic)
