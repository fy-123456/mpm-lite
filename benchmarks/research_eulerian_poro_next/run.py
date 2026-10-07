"""Small descendant cycles; this command does not advance original Lite."""
import argparse,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from engine.aniso_phase1.research_eulerian_poro_next.bridge import ImprovedBridge
from engine.aniso_phase1.research_eulerian_poro_next import checkpoint
from .diagnose import write


def run_case(output,case,ppc=3,steps=12,dt=.005,rule='director-log'):
    folder=Path(output)/case
    if folder.exists():raise ValueError('new case directory required; never overwrite evidence')
    if not 1<=steps<=24 or not 0<dt<=.005:raise ValueError('bounded short-window run required')
    folder.mkdir(parents=True)
    b=ImprovedBridge(Space(),ppc,rule);rows=[];frames=[];start=time.perf_counter()
    for i in range(steps):
        attempt={'index':i,'accepted':False};write(folder/f'attempt-{i:03d}.json',attempt)
        try:
            m=b.step(dt);m.update(time=b.state.time,step=b.state.step)
            energy_scale=max(abs(m['energy_initial']),abs(m['energy_final']),abs(m['external_work']),1e-9)
            if abs(m['energy_defect'])>.01*energy_scale+1e-10:raise RuntimeError('scene energy drift')
            if m['min_detF']<=.1:raise RuntimeError('scene geometry invalid')
            attempt.update(accepted=True,metrics=m);rows.append(m)
            if i in {0,steps//3,2*steps//3,steps-1}:
                frames.append(dict(time=b.state.time,x=b.state.particles.x.tolist(),pressure=b.state.p.tolist()))
        except Exception as error:
            attempt['error']=repr(error);write(folder/f'attempt-{i:03d}.json',attempt);raise
        write(folder/f'attempt-{i:03d}.json',attempt)
    op,_,_,_=b.prepare();op.save(folder/'frozen.npz');checkpoint.save(b,folder/'checkpoint.npz')
    np.savez_compressed(folder/'terminal.npz',q=b.state.q,v=b.state.v,p=b.state.p,X=b.state.particles.X,x=b.state.particles.x)
    summary=dict(case=case,ppc=ppc,rule=rule,steps=steps,dt=dt,seconds=time.perf_counter()-start,
        max_displacement_m=max(m['max_particle_displacement'] for m in rows),min_detF=min(m['min_detF'] for m in rows),
        max_energy_defect_J=max(abs(m['energy_defect']) for m in rows),
        max_content_defect_m3=max(abs(m['actual_content_defect']) for m in rows),
        Nq=len(op.points),state_digest=b.state.digest(),eulerian_integration=False)
    write(folder/'rows.json',rows);write(folder/'frames.json',frames);write(folder/'summary.json',summary)
    print(summary,flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--case',required=True)
    p.add_argument('--ppc',type=int,default=3);p.add_argument('--steps',type=int,default=12);p.add_argument('--dt',type=float,default=.005)
    p.add_argument('--rule',choices=['director-log','director-or-positive','fixed-positive'],default='director-log')
    a=p.parse_args();run_case(a.output,a.case,a.ppc,a.steps,a.dt,a.rule)
