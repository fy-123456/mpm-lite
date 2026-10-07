"""Isolated-process, matched-physics center/particle quadrature comparison.

The loading is the same prescribed initial affine velocity in a free block.
A finer particle simulation is a numerical reference, not an analytic truth.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from demos.aniso import DATA_ROOT
from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams, select_lowest_memory_device, query_gpu_memory
from engine.aniso_phase1.particle_quadrature import ParticleQuadratureImplicitSolver
from engine.aniso_phase1.diagnostics import energy_density
from utils.resource_guard import prepare_warp_cache


def sample_lattice(grid, samples):
    if (grid-1)%4 or samples<1:
        raise ValueError('grid-1 must be divisible by four and samples >=1')
    n=(grid-1)//2*samples
    axis=.25+(np.arange(n)+.5)*.5/n
    return np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)


def worker(args):
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache', DATA_ROOT)
    wp.init()
    device=select_lowest_memory_device(args.device)
    cls=AnisotropicLiteImplicitSolver if args.method=='center' else ParticleQuadratureImplicitSolver
    params=AnisotropicMaterialParams(10.,20.,200.,[1.,1.,0.])
    x=sample_lattice(args.grid,args.samples)
    G=np.array([[.08,.02,.01],[0.,-.03,.015],[0.,0.,.04]])
    s=cls((args.grid,)*3,params,dx=1/(args.grid-1),device=device,gravity=0.,flip_ratio=.9)
    s.seed_particles(x,density=1.,vol0=.125/len(x),velocity=(x-.5)@G.T,velocity_gradient=G)
    s.set_dt(args.dt)
    steps=round(args.time/args.dt)
    if steps<2 or not np.isclose(steps*args.dt,args.time,rtol=0,atol=1e-12):
        raise ValueError('time must be a multiple of dt and at least 2 steps')
    times=[]; iterations=[]; newton=[]; matvec=[]; residual=[]
    gpu_before=query_gpu_memory().get(device,{})
    for _ in range(steps):
        wp.config.kernel_cache_dir=prepare_warp_cache(wp.config.kernel_cache_dir,DATA_ROOT)
        wp.synchronize_device(device)
        start=time.perf_counter()
        success=s.step(max_iters=20,print_every=0,cg_tol=1e-5,cg_atol=1e-12,newton_atol=1e-10)
        wp.synchronize_device(device)
        times.append(time.perf_counter()-start)
        if not success: raise RuntimeError(f'{args.method} failed at {s.sim_time}: {s.last_step_stats}')
        iterations.append(s.last_step_stats['linear_iterations'])
        newton.append(s.last_step_stats['newton_iterations'])
        matvec.append(s.last_step_stats['matvec_calls'])
        residual.append(float(s.last_step_stats['last_residual_norm']))
    # Same fixed reference-coordinate observation locations for every resolution/PPC.
    axis=np.linspace(.3,.7,5)
    probes=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    distance,indices=cKDTree(x).query(probes,k=min(8,len(x)))
    weights=1/np.maximum(distance,1e-14)**2
    weights/=weights.sum(axis=1,keepdims=True)
    def observe(values): return np.einsum('pi,pi...->p...',weights,values[indices])
    disp=observe(s.ptc_x.numpy()-x)
    velocity=observe(s.ptc_v.numpy())
    F=observe(s.ptc_F.numpy())
    arrays=[a for a in vars(s).values() if isinstance(a,wp.array)]
    owned={a.ptr: a.capacity for a in arrays if a.ptr}
    pool_peak=wp.get_mempool_used_mem_high(device) if wp.get_device(device).is_cuda else None
    data=dict(method=args.method,grid=args.grid,ppc=args.samples**3,particles=len(x),
              device=device,dt=args.dt,physical_time=float(s.sim_time),volume=float(s.ptc_vol0.numpy().sum()),mass=float(s.ptc_m.numpy().sum()),
              first_step_seconds=times[0],warm_step_mean_seconds=float(np.mean(times[1:])),
              warm_step_std_seconds=float(np.std(times[1:])),total_seconds=sum(times),
              linear_iterations_total=sum(iterations),newton_iterations_total=sum(newton),matvec_total=sum(matvec),
              max_final_residual=max(residual),active_nodes=s.last_step_stats['active_nodes'],
              active_centers=s.last_step_stats['active_centers'],aniso_state_bytes=s._aniso_memory_bytes(),
              owned_array_bytes=sum(owned.values()),cuda_pool_peak_bytes=pool_peak,
              gpu_before=gpu_before,gpu_after=query_gpu_memory().get(device,{}),
              linear_solver=s.last_step_stats['linear_solver'],diagnostics_enabled=False,
              particle_elastic_diagnostic=float(np.sum(s.ptc_vol0.numpy()*energy_density(s.ptc_F.numpy(),s.ptc_A0.numpy(),params))))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(data,indent=2))
    np.savez(args.out.with_suffix('.npz'),displacement=disp,velocity=velocity,F=F,probe_reference=probes)
    print(json.dumps(data),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto')
    p.add_argument('--method',choices=('center','particle'),default='center')
    p.add_argument('--grid',type=int,default=17)
    p.add_argument('--samples',type=int,default=2)
    p.add_argument('--dt',type=float,default=.001)
    p.add_argument('--time',type=float,default=.01)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,default=Path('output/fair-comparison'))
    args=p.parse_args()
    if args.worker:
        worker(args);return
    prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    args.out.mkdir(parents=True,exist_ok=True)
    # Full grid and PPC comparisons plus a finer time-step reference.
    configs=[(g,2,args.dt) for g in (9,17,33)]
    configs += [(17,s,args.dt) for s in (1,3,4)]
    if args.repeats < 1: raise ValueError('repeats must be positive')
    device=select_lowest_memory_device(args.device)
    cases=[('particle',33,2,args.dt/2,0)]
    cases += [(m,g,s,dt,repeat) for repeat in range(args.repeats) for g,s,dt in configs for m in ('center','particle')]
    paths=[]
    for method,grid,samples,dt,repeat in cases:
        prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
        target=args.out/f'{method}-grid{grid}-ppc{samples**3}-dt{dt}-r{repeat}.json'
        command=[sys.executable,'-m','benchmarks.aniso_fair_comparison','--worker','--device',device,
                 '--method',method,'--grid',str(grid),'--samples',str(samples),'--dt',str(dt),
                 '--time',str(args.time),'--out',str(target)]
        with target.with_suffix('.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        paths.append(target)
        print(f'completed {target.name}',flush=True)
    reference=np.load(paths[0].with_suffix('.npz'))
    rows=[]
    for path in paths:
        row=json.loads(path.read_text())
        observed=np.load(path.with_suffix('.npz'))
        for key in ('displacement','velocity','F'):
            row[key+'_rmse_vs_reference']=float(np.sqrt(np.mean((observed[key]-reference[key])**2)))
        row['reference']=paths[0].name
        rows.append(row)
    (args.out/'summary.json').write_text(json.dumps(rows,indent=2))
    aggregates=[]
    for method in ('center','particle'):
        for grid,samples,dt in configs:
            group=[r for r in rows if r['method']==method and r['grid']==grid and r['ppc']==samples**3 and r['dt']==dt]
            item=dict(method=method,grid=grid,ppc=samples**3,dt=dt,device=device,repeats=len(group))
            for key in ('warm_step_mean_seconds','linear_iterations_total','newton_iterations_total','matvec_total','cuda_pool_peak_bytes','displacement_rmse_vs_reference','velocity_rmse_vs_reference','F_rmse_vs_reference'):
                values=[r[key] for r in group if r[key] is not None]
                item[key]=float(np.median(values)) if values else None
            item['run_time_min_seconds']=min(r['warm_step_mean_seconds'] for r in group)
            item['run_time_max_seconds']=max(r['warm_step_mean_seconds'] for r in group)
            aggregates.append(item)
    (args.out/'aggregate.json').write_text(json.dumps(aggregates,indent=2))


if __name__=='__main__':main()
