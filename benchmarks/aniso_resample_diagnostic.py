"""Read-only same-time history-resampling decomposition and actual-kernel replay."""
import itertools
import numpy as np
from engine.aniso_phase1.transfer_audit import center_weights,average,tensor_rms
from engine.aniso_phase1.diagnostics import energy_density,center_snapshot
from engine.aniso_phase1.constitutive import pk1
from demos.aniso import Scene,Config


def occupied_coords(x,dx):
    base=np.floor(x/dx-.5).astype(int)
    return np.unique((base[:,None,:]+np.array(list(itertools.product((0,1),repeat=3)))).reshape(-1,3),axis=0)


def decompose(z,dt,params,dx=.125):
    """End-of-step reconstruction at fixed common physical time; no integration."""
    if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
    coords=z['coords'];x0=z['particle_x_before'];x1=z['particle_x_after'];vp=z['particle_volume']
    if set(map(tuple,occupied_coords(x1,dx)))!=set(map(tuple,coords)):
        raise ValueError('diagnostic currently requires unchanged occupied center support')
    A=z['center_A0'];Ap=z['particle_A0']
    if np.max(abs(A-Ap[0]))>1e-12 or np.max(abs(Ap-Ap[0]))>1e-12:
        raise ValueError('diagnostic requires uniform material direction')
    W,V=center_weights(x0,vp,coords,dx);W1,V1=center_weights(x1,vp,coords,dx)
    F0=z['particle_F_before'];F1=z['particle_F_after'];L=z['particle_L_after'];Lc=z['center_G']
    barF=average(W,F0);barL=average(W,L);Fc=z['center_F_committed']
    cov=dt*(average(W,L@F0)-barL@barF)
    grad=dt*(barL-Lc)@barF
    moving=average(W1,F1)-average(W,F1)
    rebuilt=average(W1,F1)
    Fs={'committed':Fc,'covariance':Fc+cov,'gradient':Fc+cov+grad,'moving':rebuilt}
    Ps={k:np.array([pk1(f,a,params) for f,a in zip(F,A)]) for k,F in Fs.items()}
    densities={k:energy_density(F,A,params) for k,F in Fs.items()}
    energies={k:float(V@v) for k,v in densities.items()};energies['new_volume']=float(V1@densities['moving'])
    Pp=np.array([pk1(f,a,params) for f,a in zip(F1,Ap)])
    psip=energy_density(F1,Ap,params)
    terms={};previous='committed'
    for name,delta in [('covariance',cov),('gradient',grad),('moving',moving)]:
        terms[name]=dict(F_rms=tensor_rms(delta,V),F_rate_rms=tensor_rms(delta/dt,V),
            P_sequential_change_rms_Pa=tensor_rms(Ps[name]-Ps[previous],V),
            energy_sequential_change_J=energies[name]-energies[previous],
            energy_sequential_rate_W=(energies[name]-energies[previous])/dt)
        previous=name
    full_energy=energies['new_volume']-energies['committed']
    energy_volume=energies['new_volume']-energies['moving']
    checks=dict(F_decomposition_max=float(np.max(abs(rebuilt-Fc-cov-grad-moving))),
        particle_update_max=float(np.max(abs(F1-(np.eye(3)+dt*L)@F0))),
        center_update_max=float(np.max(abs(Fc-(np.eye(3)+dt*Lc)@barF))),
        energy_decomposition_max=abs(full_energy-sum(r['energy_sequential_change_J'] for r in terms.values())-energy_volume),
        saved_resample_max=float(np.max(abs(barF-z['center_F_resampled']))),
        volume_mass_max=abs(V1.sum()-vp.sum()))
    record=dict(terms=terms,checks=checks,
        total_F_rms=tensor_rms(rebuilt-Fc,V),total_F_rate_rms=tensor_rms((rebuilt-Fc)/dt,V),
        total_P_change_rms_Pa=tensor_rms(Ps['moving']-Ps['committed'],V),
        total_energy_change_J=full_energy,total_energy_rate_W=full_energy/dt,
        volume_energy_change_J=energy_volume,
        nonlinear_stress_averaging_gap_rms_Pa=tensor_rms(Ps['moving']-average(W1,Pp),V1),
        nonlinear_energy_averaging_gap_J=float(V1@densities['moving']-vp@psip),
        before_resample_F_rms=tensor_rms(z['center_F_resampled']-z['center_F_before_resample'],V),
        before_resample_P_rms_Pa=tensor_rms(z['center_P_resampled']-z['center_P_before_resample'],V),
        before_resample_energy_change_J=float(V@(z['center_psi_resampled']-z['center_psi_before_resample'])),
        centers=len(coords),support_changes=0,
        nonlinear_attribution='ordered path: covariance, gradient, moving weights, volume; not unique causal shares')
    arrays=dict(coords=coords,old_volume=V,new_volume=V1,F_committed=Fc,F_rebuilt=rebuilt,
        covariance=cov,gradient=grad,moving=moving,P_committed=Ps['committed'],P_rebuilt=Ps['moving'],
        psi_committed=densities['committed'],psi_rebuilt=densities['moving'])
    return record,arrays


def replay(z,config,expected):
    """Two actual center rebuilds on identical particles, with no solve or time advance."""
    scene=Scene(Config(**config),'cpu');s=scene.solver
    s.energy_ledger=None
    for field,key in [('x','particle_x_after'),('F','particle_F_after'),('A0','particle_A0'),('C','particle_C_after'),('v','particle_velocity_after'),('L','particle_L_after')]:
        getattr(s,'ptc_'+field).assign(z[key])
    before={key:getattr(s,'ptc_'+key).numpy().copy() for key in ('x','F','v','C','L')}
    states=[]
    for _ in range(2):
        s.prepare_centers();coords,V,psi,active=center_snapshot(s)
        F=s.aniso_committed_F[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active].copy()
        lookup={tuple(c):i for i,c in enumerate(coords)};indices=[lookup[tuple(c)] for c in expected['coords']]
        states.append(dict(F=F[indices],V=V[indices].copy(),psi=psi[indices].copy()))
    r=dict(actual_resample_F_max=float(np.max(abs(states[0]['F']-expected['F_rebuilt']))),
        actual_resample_volume_max=float(np.max(abs(states[0]['V']-expected['new_volume']))),
        actual_resample_energy_density_max=float(np.max(abs(states[0]['psi']-expected['psi_rebuilt']))),
        repeated_F_max=float(np.max(abs(states[1]['F']-states[0]['F']))),
        repeated_energy_max=abs(float(states[1]['V']@states[1]['psi']-states[0]['V']@states[0]['psi'])),
        particle_mutation_max=max(float(np.max(abs(getattr(s,'ptc_'+key).numpy()-value))) for key,value in before.items()),
        sim_steps=s.sim_steps,sim_time=s.sim_time)
    del s,scene
    return r
