"""Bounded canonical-solver input sensitivity, distinct from time acceptance."""
import numpy as np
from benchmarks.aniso_v18_runs import ROOT,OUT,load,write,sha,initial
from engine.aniso_phase1.carrier_avf import CarrierAVFSolver
from engine.aniso_phase1.carrier_joint import Geometry
from benchmarks.aniso_v17_modes import weighted_rms

def main():
    assert not (OUT/'input-sensitivity.json').exists();spec=load(OUT/'protocol.json')['cases']['early-moving-L0'];s,e,m,h,_=initial(spec);g=Geometry(s,e,m,h)
    with np.load(OUT/'cases/early-moving-L0/stress.npz') as z:ref=z['P']
    direction=g.Q@np.random.default_rng(1801).normal(size=(g.Q.shape[1],3));direction/=np.max(abs(direction));records=[]
    write(OUT/'input-sensitivity-protocol.json',dict(source_sha256=sha(ROOT/'benchmarks/aniso_v18_sensitivity.py'),amplitudes=[1e-15,1e-13],seed=1801,steps=400,dt=spec['dt'],solver='unchanged original CarrierAVFSolver',perturbation='admissible carrier Y only; original v/C/x; physical initial perturbation measured explicitly'))
    for amp in (1e-15,1e-13):
        state=s.clone();state.Y+=amp*direction;so=CarrierAVFSolver(state,e,m,h,moving=True);P=[e.evaluate(state.Y)['P']];total=[]
        for k in range(400):
            row=so.step(spec['dt']);P.append(e.evaluate(so.state.Y)['P']);total.append(row['total_J'])
            assert row['history_commit_max']<1e-10 and abs(row['kinetic_force_work_defect_J'])<5e-13
            if (k+1)%100==0:print(amp,k+1,flush=True)
        P=np.array(P);norm=weighted_rms(ref,e.V);r=dict(amplitude_Y=amp,actual_initial_max_Y=float(np.max(abs(state.Y-s.Y))),initial_stress_relative=weighted_rms(P[0]-ref[0],e.V)/weighted_rms(ref[0],e.V),
            whole_stress_relative=weighted_rms(P-ref,e.V)/norm,terminal_stress_relative=weighted_rms(P[-1]-ref[-1],e.V)/weighted_rms(ref[-1],e.V))
        np.savez_compressed(OUT/f'sensitivity-{amp:g}.npz',P=P,total_J=total,Y_initial=state.Y);records.append(r);print(r,flush=True)
    write(OUT/'input-sensitivity.json',dict(completed=True,records=records,scope='Sensitivity to a measured tiny compatible initial-state change; does not identify the mechanism or replace formal dt convergence.'))
if __name__=='__main__':main()
