"""Same-input decomposition of velocity/affine PIC relaxation on v11 snapshots."""
import json
from pathlib import Path
import numpy as np
from benchmarks.aniso_mainline import write_json
from benchmarks.aniso_apic_frequency import Oracle
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/results/lite-aniso-mainline/v11'
OUT=ROOT/'docs/results/lite-aniso-mainline/v12-transfer-controls'


def rms(a,w):return float(np.sqrt(np.sum(w.reshape((-1,)+(1,)*(a.ndim-1))*a*a)/np.sum(w)))


def main():
    protocol=json.loads((SOURCE/'protocol.json').read_text());records=[]
    for name,cfg in protocol['configs'].items():
        if not name.startswith('F45-'):continue
        for file in sorted((SOURCE/'cases'/name).glob('audit-*.npz')):
            with np.load(file) as z:
                x=z['particle_x_before'];v=z['particle_velocity_before'];C=z['particle_C_before'];m=z['particle_mass'];dt=cfg['dt'];beta=cfg['flip_ratio'];dx=1/(cfg['grid']-1)
                o=Oracle(x,m,dx);raw=o.apply(v,C,0.)
                np.testing.assert_allclose(raw['grid_v'],z['grid_velocity_raw'],atol=1e-13,rtol=0)
                # After the SAME converged solve, removals differ only by these
                # discarded old-state components. x, F, L, current energy,
                # force and tangent stay unchanged at this step.
                dv=(1-beta)*(v-raw['pic']);dC=(1-beta)*(C-raw['G'])
                next_o=Oracle(z['particle_x_after'],m,dx)
                gv=next_o.apply(dv,np.zeros_like(dC),0.)['grid_v']
                gc=next_o.apply(np.zeros_like(dv),dC,0.)['grid_v']
                # Boundary projection for the following step is linear. The
                # prescribed part of a perturbation is exactly zero.
                fixed=(next_o.xn[:,0]<=.25)|(next_o.xn[:,0]>=.75)
                gv[fixed]=0.;gc[fixed]=0.
                G=next_o.S@next_o.D.reshape(len(next_o.c),-1)
                G=G.reshape(len(x),len(next_o.nodes),3)
                Lv=np.einsum('pni,nj->pji',G,gv);Lc=np.einsum('pni,nj->pji',G,gc)
                F=z['particle_F_after']
                dFv=dt*Lv@F;dFc=dt*Lc@F
                records.append(dict(case=name,snapshot=file.name,time=round(dt*int(file.stem.split('-')[-1]),12),dt=dt,
                    same_step_x_F_L_energy_force_tangent='identical: intervention happens after converged grid solve and material F commit',
                    velocity_removed_rms=rms(dv,m),affine_removed_rms=rms(dC,m),velocity_removed_rate=rms(dv,m)/dt,affine_removed_rate=rms(dC,m)/dt,
                    next_raw_grid_velocity_from_v=rms(gv,next_o.mn),next_raw_grid_velocity_from_C=rms(gc,next_o.mn),
                    next_material_gradient_from_v=rms(Lv,m),next_material_gradient_from_C=rms(Lc,m),
                    next_trial_F_from_v=rms(dFv,m),next_trial_F_from_C=rms(dFc,m),
                    next_gradient_sum=rms(Lv+Lc,m),next_gradient_inner_product=float(np.sum(m[:,None,None]*Lv*Lc)/np.sum(m)),
                    velocity_momentum_change=float(np.linalg.norm(m@dv))))
    result=dict(completed=True,snapshots=len(records),records=records,
        interpretation='same-state causal path probe, not a solved next-step trajectory: perturb the transfer return only, advance P2G at the unchanged next positions, project prescribed velocities, then sample material gradient; full nonlinear attribution requires factorial loads',
        kappa_per_second=-np.log(.9)/.001,max_momentum_change=max(r['velocity_momentum_change'] for r in records))
    write_json(OUT/'same-input-attribution.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    for r in records:
        if r['time']==.5:print(r['case'],'next L v/C',r['next_material_gradient_from_v'],r['next_material_gradient_from_C'])

if __name__=='__main__':main()
