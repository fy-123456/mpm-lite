"""Bounded sequential scenario runner; raw data and failed attempts retained."""
import argparse,json,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from engine.aniso_phase1.research_unified_lite_poro.model import Bridge
from engine.aniso_phase1.research_unified_lite_poro.checkpoint import save


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False))


def run_case(root,name,ppc,steps,rule="moments"):
    folder=root/name
    if folder.exists():raise ValueError('case output already exists; never overwrite evidence')
    folder.mkdir(parents=True);b=Bridge(Space(),ppc,rule=rule);rows=[];frames=[]
    start=time.perf_counter()
    for step in range(steps):
        attempt=dict(case=name,index=step,accepted=False)
        write(folder/f'attempt-{step:03d}.json',attempt)
        try:
            m=b.step();m.update(time=b.state.time,step=b.state.step)
            if abs(m['energy_defect'])>1e-7 or m['min_detF']<.1:raise RuntimeError('scene energy/geometry budget failed')
            attempt.update(accepted=True,metrics=m);rows.append(m)
            if step in {0,steps//3,2*steps//3,steps-1}:
                frames.append(dict(time=b.state.time,x=b.state.particles.x.tolist(),pressure=b.state.p.tolist()))
        except Exception as e:
            attempt['error']=repr(e);write(folder/f'attempt-{step:03d}.json',attempt);raise
        write(folder/f'attempt-{step:03d}.json',attempt)
    save(b,folder/'checkpoint.npz');op,_,_,_=b.prepare();op.save(folder/'frozen.npz')
    np.savez_compressed(folder/'terminal.npz',q=b.state.q,v=b.state.v,p=b.state.p,X=b.state.particles.X,x=b.state.particles.x)
    summary=dict(case=name,rule=rule,ppc_axis=ppc,particles=len(b.state.particles.X),Nq=len(op.points),
       accepted=len(rows),seconds=time.perf_counter()-start,max_displacement_m=max(x['max_particle_displacement'] for x in rows),
       min_detF=min(x['min_detF'] for x in rows),max_energy_defect_J=max(abs(x['energy_defect']) for x in rows),
       max_mass_defect_m3=max(x['actual_content_defect'] for x in rows),state_digest=b.state.digest(),
       reference_coordinate_scope=True,eulerian_grid_transfer=False)
    write(folder/'summary.json',summary);write(folder/'rows.json',rows);write(folder/'frames.json',frames)
    print(json.dumps(summary),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--ppc',type=int,default=3);p.add_argument('--steps',type=int,default=12);p.add_argument('--case',default='cycle');p.add_argument('--rule',choices=['moments','fixed-positive'],default='fixed-positive');a=p.parse_args()
    if not 1<=a.steps<=16:raise ValueError('bounded run supports 1..16 steps')
    run_case(a.output,a.case,a.ppc,a.steps,a.rule)

if __name__=='__main__':main()
