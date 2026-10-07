"""Bounded research run; actual frames and full state, never overwrite a case."""
import argparse,json,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_common_kinematics_next.model import CommonBridge
from engine.aniso_phase1.research_common_kinematics_next.checkpoint import save

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def run(output,steps=4,dt=.00125,ppc=4):
    if output.exists():raise ValueError('case output must not already exist')
    if not 1<=steps<=48:raise ValueError('bounded experiment: 1..48 steps')
    output.mkdir(parents=True);b=CommonBridge(ppc=ppc);op,v,fit=b.prepare()
    op.save(output/'frozen-initial.npz');np.savez_compressed(output/'solve-input.npz',v=v,p=b.state.p,dt=dt,time=b.state.time)
    frames=[b.state.particles.x.copy()];F=[b.state.particles.F.copy()];vel=[b.state.particles.v.copy()]
    increments=[];pressure=[b.state.p.copy()];rows=[];ticks=[0.];qdisp=[np.zeros_like(b.state.qx)];qvel=[b.state.qv.copy()]
    t=time.perf_counter()
    for i in range(steps):
        m=b.step(dt);rows.append(m);increments.append(b.state.last_increment.copy());frames.append(b.state.particles.x.copy());F.append(b.state.particles.F.copy())
        vel.append(b.state.particles.v.copy());pressure.append(b.state.p.copy());ticks.append(b.state.time)
        qdisp.append(b.state.qx-op.gX);qvel.append(b.state.qv.copy())
        if i==min(3,steps-1):save(b,output/'checkpoint-short.npz')
        if abs(b.state.time-.045)<1e-10:save(b,output/'checkpoint-unload.npz')
    save(b,output/'checkpoint.npz')
    np.savez_compressed(output/'frames.npz',X=b.state.particles.X,x=np.array(frames),F=np.array(F),
        velocity=np.array(vel),pressure=np.array(pressure),time=np.array(ticks),qdisp=np.array(qdisp),qvel=np.array(qvel),increments=np.array(increments))
    write(output/'metrics.json',rows)
    summary=dict(steps=steps,dt=dt,seconds=time.perf_counter()-t,particles=len(b.state.particles.X),Nq=128,
        max_displacement=max(r['max_particle_displacement'] for r in rows),
        min_detF=min(min(r['min_detF'],r['min_particle_detF']) for r in rows),
        max_mass_defect=max(r['actual_content_defect'] for r in rows),
        max_exchange_work_defect=max(abs(r['exchange_work_defect']) for r in rows),
        max_energy_defect=max(abs(r['energy_defect']) for r in rows),
        max_velocity_projection=max(r['prepare']['relative_velocity_projection'] for r in rows),
        cumulative_projection_work=b.state.projection_work,darcy_dissipation=b.state.dissipation,
        final_digest=b.state.digest(),scope='updated modal SH + material bubbles; material pressure cells; not formal144 or production implicit Lite')
    write(output/'summary.json',summary);print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=4);p.add_argument('--dt',type=float,default=.00125);p.add_argument('--ppc',type=int,default=4)
    a=p.parse_args();run(a.output,a.steps,a.dt,a.ppc)
