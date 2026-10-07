"""Read-only transfer diagnostics for the center/particle_resample Lite path.

Host snapshots are opt-in; they are diagnostics, not a new integration rule.
"""
import itertools
import numpy as np
from .diagnostics import EnergyLedger, center_snapshot, energy_density
from .constitutive import pk1


def center_weights(x, particle_volume, coords, dx):
    """Independent volume-normalized trilinear P2C oracle on supplied centers."""
    q=x/dx-.5; base=np.floor(q).astype(int); f=q-base
    lookup={tuple(c):i for i,c in enumerate(coords)}
    weights=np.zeros((len(coords),len(x)))
    for corner in itertools.product((0,1),repeat=3):
        values=np.prod(np.where(corner,f,1-f),axis=1)*particle_volume
        for p,c in enumerate(base+corner):
            i=lookup.get(tuple(c))
            if i is not None: weights[i,p]+=values[p]
    volume=weights.sum(axis=1)
    return weights/np.maximum(volume[:,None],1e-300),volume


def average(W, tensor):
    return np.einsum('cp,pij->cij',W,tensor)


def tensor_rms(value, volume):
    return float(np.sqrt(np.sum(volume[:,None,None]*value**2)/volume.sum()))


class TransferAuditLedger(EnergyLedger):
    def __init__(self):
        super().__init__()
        self.audit_next=False
        self.snapshot=None
        self.previous_centers={}

    def begin(self,s):
        super().begin(s)
        self.momenta={'particle_start':np.sum(s.ptc_m.numpy()[:,None]*s.ptc_v.numpy(),axis=0)}
        self.snapshot=None
        if self.audit_next:
            self.before={k:getattr(s,'ptc_'+k).numpy().copy() for k in ('x','F','A0','vol0')}

    def p2c(self,s):
        super().p2c(s)
        coords,vol,_,active=center_snapshot(s);n=int(s.bcn)
        mass=s.center_m[:n].numpy().reshape(-1)[active]
        vel=s.center_v[:n].numpy().reshape(-1,3)[active]
        self.momenta['p2c']=np.sum(mass[:,None]*vel,axis=0)
        self.momenta['c2g_raw']=np.sum(self.node_m[:,None]*self.raw_v,axis=0)
        if not self.audit_next:return
        F=s.aniso_committed_F[:,:n].numpy()[0].reshape(-1,3,3)[active]
        A=s.aniso_A0[:,:n].numpy()[0].reshape(-1,3,3)[active]
        W,V=center_weights(self.before['x'],self.before['vol0'],coords,s.dx)
        Fbar=average(W,self.before['F'])
        P=np.array([pk1(f,a,s.aniso_params) for f,a in zip(F,A)])
        particle_P=np.array([pk1(f,a,s.aniso_params) for f,a in zip(self.before['F'],self.before['A0'])])
        psi=energy_density(F,A,s.aniso_params)
        particle_psi=energy_density(self.before['F'],self.before['A0'],s.aniso_params)
        common=np.array([tuple(c) in self.previous_centers for c in coords])
        old=np.array([self.previous_centers.get(tuple(c),f) for c,f in zip(coords,F)])
        oldP=np.array([pk1(f,a,s.aniso_params) for f,a in zip(old,A)])
        oldpsi=energy_density(old,A,s.aniso_params)
        self.audit=dict(time_start=float(s.sim_time),
            resample_oracle_F_max=float(np.max(abs(F-Fbar))),
            resample_oracle_volume_max=float(np.max(abs(V-vol))),
            stress_average_gap_rms_Pa=tensor_rms(P-average(W,particle_P),vol),
            compression_energy_gap_J=float(np.dot(vol,psi-W@particle_psi)),
            center_resample_F_rms=tensor_rms(F-old,vol),
            center_resample_P_rms_Pa=tensor_rms(P-oldP,vol),
            center_resample_energy_change_J=float(np.dot(vol,psi-oldpsi)),
            matched_previous_centers=int(common.sum()),centers=len(coords))
        self.snapshot=dict(coords=coords,volume=vol,center_F_before_resample=old,
            center_F_resampled=F,center_A0=A,center_P_before_resample=oldP,
            center_P_resampled=P,particle_mean_P=average(W,particle_P),
            center_psi_before_resample=oldpsi,center_psi_resampled=psi,
            particle_mean_psi=W@particle_psi,particle_x_before=self.before['x'],
            particle_F_before=self.before['F'],particle_A0=self.before['A0'])
        self.weights=W;self.Fbar=Fbar

    def _grid_momentum(self,s,field):
        n=int(s.bcn)
        return np.sum(s.grid_m[:n].numpy()[...,None]*field[:n].numpy(),axis=(0,1,2,3))

    def grid(self,s):
        super().grid(s);self.momenta['grid_projected']=self._grid_momentum(s,s.grid_v)

    def solved(self,s):
        super().solved(s);self.momenta['grid_solved']=self._grid_momentum(s,s.grid_v_it)

    def transferred_grid(self,s):
        super().transferred_grid(s);self.momenta['grid_final']=self._grid_momentum(s,s.grid_v_new)

    def finish(self,s):
        n=int(s.bcn)
        coords,vol,_,active=center_snapshot(s)
        center_F=s.aniso_committed_F[:,:n].numpy()[0].reshape(-1,3,3)[active]
        Gc=s.center_G[:n].numpy().reshape(-1,3,3)[active]
        mass=s.center_m[:n].numpy().reshape(-1)[active]
        vel=s.center_v[:n].numpy().reshape(-1,3)[active]
        self.momenta['g2c']=np.sum(mass[:,None]*vel,axis=0)
        self.momenta['particle_end']=np.sum(s.ptc_m.numpy()[:,None]*s.ptc_v.numpy(),axis=0)
        previous=self.momenta['particle_start']
        for stage in ('p2c','c2g_raw','grid_projected','grid_solved','grid_final','g2c','particle_end'):
            delta=self.momenta[stage]-previous
            for axis,value in zip('xyz',delta):self.current[f'momentum_{stage}_delta_{axis}']=float(value)
            previous=self.momenta[stage]
        if self.audit_next:
            W=self.weights;Gp=s.ptc_L.numpy();Fp=s.ptc_F.numpy();F0=self.before['F']
            Gbar=average(W,Gp)
            covariance=s.dt*(average(W,Gp@F0)-Gbar@self.Fbar)
            gradient_gap=s.dt*(Gbar-Gc)@self.Fbar
            discrepancy=average(W,Fp)-center_F
            Wnew,_=center_weights(s.ptc_x.numpy(),self.before['vol0'],coords,s.dx)
            moving=average(Wnew,Fp)-average(W,Fp)
            self.audit.update(time_end=float(s.sim_time),
                particle_update_max_error=float(np.max(abs(Fp-(np.eye(3)+s.dt*Gp)@F0))),
                frozen_update_covariance_F_rms=tensor_rms(covariance,vol),
                frozen_update_gradient_gap_F_rms=tensor_rms(gradient_gap,vol),
                frozen_center_particle_F_rms=tensor_rms(discrepancy,vol),
                frozen_decomposition_max_error=float(np.max(abs(discrepancy-covariance-gradient_gap))),
                moving_weights_F_rms=tensor_rms(moving,vol))
            self.snapshot.update(center_F_committed=center_F,particle_F_after=Fp,
                particle_x_after=s.ptc_x.numpy(),center_G=Gc,particle_G=Gp,particle_C=s.ptc_C.numpy(),
                covariance_increment=covariance,gradient_gap_increment=gradient_gap,
                moving_weight_increment=moving)
        self.previous_centers={tuple(c):f.copy() for c,f in zip(coords,center_F)}
        super().finish(s)
