"""Reproducible variational derivative/spectrum and matched trajectory comparisons."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import warp as wp
from demos.aniso import Config, Scene, DATA_ROOT
from engine.aniso_phase1 import select_lowest_memory_device, query_gpu_memory
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.particle_quadrature import ParticleQuadratureImplicitSolver
from engine.aniso_phase1.linear import guarded_pcg, nonsymmetric_solve
from engine.kernel.d3.kernel_lite_implicit import lite_implicit_precond_kernel_Hii
from engine.types import vec3
from utils.resource_guard import prepare_warp_cache, inspect_storage


def preconditioner(s):
    def apply(a, out):
        wp.launch(lite_implicit_precond_kernel_Hii, dim=s.MAX_DOF, inputs=[
            s.n_active_nodes,s.ndof2bijk,s.block_xyz_by_id,s.node_Hii_inv,a,out,
            s.bc_block2bid,s.bc_type,s.bc_norm,s.bc_velo,s.hf_bc_p,s.hf_bc_n,
            s.hf_bc_v,s.hf_bc_type,s.num_hf,s.dx],device=s.device)
    return apply


def linear_compare(probe):
    s = probe.s
    Z, matrices, _ = probe.last_dense
    result = {}
    # Fixed normalized manufactured solution. Include negative-eigenvector RHS
    # separately to guarantee that the curvature guard sees the known instability.
    p = probe.project(np.random.default_rng(12).normal(size=(probe.n,3)))
    p /= np.linalg.norm(p)
    rhs, x = wp.zeros_like(s.node_residual), wp.zeros_like(s.node_residual)
    for method in ('pcg','pcg_projected','bicgstab','gmres'):
        modified = method == 'pcg_projected'
        apply = lambda a, b: s.apply_tangent(a,b,project_pd=modified)
        rhs.zero_()
        wp.copy(rhs,wp.array(probe.tangent(p,modified),dtype=vec3,device=s.device),count=probe.n)
        runs = []
        for repeat in range(4):
            x.zero_(); wp.synchronize_device(s.device); start=time.perf_counter()
            if method.startswith('pcg'):
                iterations,error,tol,reason=guarded_pcg(apply,rhs,x,preconditioner(s),1e-8,1e-14,500)
            else:
                iterations,error,tol=nonsymmetric_solve(apply,rhs,x,preconditioner(s),1e-8,1e-14,500,method)
                reason='converged' if error<=tol else 'linear_failure'
            wp.synchronize_device(s.device)
            runs.append(dict(milliseconds=1e3*(time.perf_counter()-start),iterations=int(iterations),
                             residual=float(error),tolerance=float(tol),status=reason,
                             solution_error=float(np.linalg.norm(x[:probe.n].numpy()-p))))
        result[method]=dict(runs=runs[1:],median_ms=float(np.median([r['milliseconds'] for r in runs[1:]])))
    if np.linalg.eigvalsh(matrices['exact'])[0]<0:
        eig, vec=np.linalg.eigh(matrices['exact'])
        rhs.zero_();wp.copy(rhs,wp.array((Z@vec[:,0]).reshape(probe.n,3),dtype=vec3,device=s.device),count=probe.n)
        iterations,error,tol,reason=guarded_pcg(s.apply_tangent,rhs,x,preconditioner(s),1e-8,1e-14,500)
        result['negative_mode_guard']=dict(status=reason,iterations=iterations,residual=error)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--out',type=Path,default=Path('docs/results/variational'))
    parser.add_argument('--part',choices=('operator','trajectories','nonlinear','all'),default='all')
    args=parser.parse_args()
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    wp.init(); before=query_gpu_memory();device=select_lowest_memory_device(args.device)
    args.out.mkdir(parents=True,exist_ok=True)
    def guard():
        wp.config.kernel_cache_dir=prepare_warp_cache(wp.config.kernel_cache_dir,DATA_ROOT)
    def save(name,records):
        (args.out/(name+'.json')).write_text(json.dumps(records,indent=2))
    save('environment',dict(device=device,gpu_before=before,storage=vars(inspect_storage(DATA_ROOT))))
    if args.part in ('operator','all'):
        records=[]
        for state,F in [('strained',None),('compressed',np.diag([.65,.85,1.]))]:
            for kf in (0.,200.,20000.):
                for dt in (.0005,.005,.05):
                    guard();p=SparseProbe(device,kf,dt)
                    r=p.variational_check(F)
                    r.update(state=state,kf=kf,dt=dt,quadrature='center')
                    if (kf,dt) in ((200.,.005),(20000.,.05)):
                        r['linear_comparison']=linear_compare(p)
                    records.append(r);save('operator',records)
                    print(json.dumps(r),flush=True);del p;gc.collect()
        guard();p=SparseProbe(device,200.,.005,solver_cls=ParticleQuadratureImplicitSolver)
        r=p.variational_check();r.update(state='strained',kf=200.,dt=.005,quadrature='particle')
        records.append(r);save('operator',records);del p;gc.collect()
    if args.part in ('nonlinear','all'):
        records=[]
        for kf,dt in ((200.,.005),(20000.,.005),(20000.,.05)):
            guard();p=SparseProbe(device,kf,dt);p.set_deformation(np.diag([.65,.85,1.]))
            success=p.s.step(max_iters=80,max_cg_iters=500,cg_tol=1e-5,cg_atol=1e-12,newton_atol=1e-10,print_every=0)
            r=dict(kf=kf,dt=dt,success=bool(success),stats=p.s.last_step_stats,trace=p.s.last_newton_trace)
            records.append(r);save('nonlinear',records);print(json.dumps(r),flush=True);del p;gc.collect()
    if args.part in ('trajectories','all'):
        records=[]
        for formulation in ('legacy_kirchhoff','variational'):
            for kind in ('fixed','affine','tensile'):
                configs=[dict(dt=dt,flip_ratio=.9) for dt in (.0005,.001,.002)] if kind!='tensile' else [dict(dt=.002,fiber_angle=a) for a in (0.,45.,90.)]
                if kind!='tensile':
                    configs += [dict(dt=.001,flip_ratio=f) for f in (0.,1.)]
                    configs += [dict(dt=.001,cg_tol=.5,residual_atol=1e-5),dict(dt=.001,cg_tol=1e-6,residual_atol=1e-12)]
                for cfg in configs:
                    guard();config=Config(scene=kind,grid=9 if kind=='tensile' else 8,loading_time=.2,force_discretization=formulation,**cfg)
                    scene=Scene(config,device)
                    total=.4 if kind=='tensile' else .02
                    iterations=0;modified=0;backtracks=0;trace_error=0.;max_residual=0.
                    for _ in range(round(total/config.dt)):
                        guard()
                        if not scene.step():raise RuntimeError((formulation,kind,cfg,scene.solver.last_step_stats))
                        stats=scene.solver.last_step_stats;iterations+=stats['linear_iterations'];modified+=stats['projected_tangent_solves']
                        backtracks+=stats['line_search_backtracks'];max_residual=max(max_residual,stats['last_residual_norm'])
                        for trace in scene.solver.last_newton_trace:
                            trace_error=max(trace_error,trace['potential_after']-trace['potential_before']-1e-4*trace['alpha']*trace['slope']-trace['armijo_slack'])
                    ledger=scene.solver.energy_ledger;rows=ledger.rows;last=rows[-1];E0=rows[0]['mechanical']
                    name=f'{formulation}-{kind}-dt{config.dt}-flip{config.flip_ratio}-cg{config.cg_tol}-atol{config.residual_atol}-a{config.fiber_angle:g}'
                    (args.out/(name+'.csv')).write_text(ledger.csv())
                    r=dict(name=name,formulation=formulation,scene=kind,dt=config.dt,flip=config.flip_ratio,cg_tol=config.cg_tol,newton_atol=config.residual_atol,angle=config.fiber_angle,
                           final_time=last['time'],initial_energy=E0,final_energy=last['mechanical'],
                           relative_change=(last['mechanical']-E0)/E0 if E0 else None,
                           max_abs_relative_change=max(abs(row['mechanical']-E0) for row in rows)/E0 if E0 else None,
                           linear_iterations=iterations,projected_solves=modified,line_search_backtracks=backtracks,max_armijo_violation=trace_error,max_nonlinear_residual=max_residual,
                           max_budget_closure=max(abs(row.get('budget_closure',0.)) for row in rows),
                           budget={key:sum(row.get(key,0.) for row in rows[1:]) for key in last if key.endswith('_delta') and key!='cumulative_delta'})
                    if kind=='tensile':
                        peak=max(scene.loading_rows,key=lambda a:a['displacement'])
                        loading=[row for row in scene.loading_rows if row['loading_velocity']>0 and row['displacement']>=.2*peak['displacement']]
                        r.update(peak_force=peak['right_force'],secant_stiffness=peak['effective_stiffness'],loading_work=peak['loading_work'],cycle_work=last['loading_work'],
                                 fitted_loading_stiffness=float(np.polyfit([row['displacement'] for row in loading],[row['right_force'] for row in loading],1)[0]),
                                 max_free_residual=max(row['free_force_residual_norm'] for row in scene.loading_rows),
                                 max_momentum_balance_error=max(abs(row['momentum_balance_error']) for row in scene.loading_rows))
                    records.append(r);save('trajectories',records);print(json.dumps(r),flush=True)
                    del scene,ledger;gc.collect()


if __name__=='__main__':main()
