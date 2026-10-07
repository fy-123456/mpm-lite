"""Controlled relocation audit of real sparse-center reactivation (not a physical trajectory)."""
import json
from pathlib import Path
import numpy as np
import warp as wp
from engine.types import vec3
from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams, select_lowest_memory_device
from demos.aniso import DATA_ROOT
from utils.resource_guard import prepare_warp_cache


def main():
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    wp.init()
    device=select_lowest_memory_device('auto')
    s=AnisotropicLiteImplicitSolver((65,)*3,AnisotropicMaterialParams(10,20,200,[1,0,0]),
                                   dx=1/64,gravity=0,device=device,energy_diagnostics=True)
    axis=np.array([.48,.49])
    reference=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    G=np.diag([.08,0,0])
    s.seed_particles(reference,density=1000,vol0=.001,velocity=reference@G.T,velocity_gradient=G)
    s.set_dt(.001)
    interventions=[]
    for shift in (None,None,.12,None,-.12):
        wp.config.kernel_cache_dir=prepare_warp_cache(wp.config.kernel_cache_dir,DATA_ROOT)
        if shift is not None:
            before=sum(s.energy_ledger.kinetic(s))
            x=s.ptc_x.numpy().copy();x[:,0]+=shift
            s.ptc_x.assign(wp.array(x,dtype=vec3,device=device))
            interventions.append(dict(time=s.sim_time,shift=shift,kinetic_jump=sum(s.energy_ledger.kinetic(s))-before))
        if not s.step(max_iters=16,print_every=0,cg_tol=1e-5,cg_atol=1e-12):
            raise RuntimeError(s.last_step_stats)
    out=Path('output/energy-reactivation');out.mkdir(parents=True,exist_ok=True)
    (out/'history.csv').write_text(s.energy_ledger.csv())
    data=dict(device=device,interventions=interventions,final=s.energy_ledger.rows[-1],
              note='Artificial position relocation isolates history reset; relocation kinetic jumps are external interventions.')
    (out/'summary.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data))


if __name__=='__main__':main()
