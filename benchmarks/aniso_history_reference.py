"""Short transport, mechanics-reference and direction-compression experiments."""
import argparse,csv,gc,json
from pathlib import Path
import numpy as np
import warp as wp
from demos.aniso import Config,Scene,DATA_ROOT
from engine.aniso_phase1 import (AnisotropicLiteImplicitSolver, AnisotropicMaterialParams,
                               select_lowest_memory_device,query_gpu_memory)
from engine.aniso_phase1.controlled import KinematicScene
from engine.aniso_phase1.diagnostics import center_snapshot
from engine.aniso_phase1.direction_moments import audit_direction_mixture
from engine.aniso_phase1.beam_reference import beam_audit
from utils.resource_guard import prepare_warp_cache,inspect_storage


def analytic_PK1(F,params):
    # Independent spectral expression in C, rather than the production F-SVD.
    values,Q=np.linalg.eigh(F.T@F)
    logU=Q@np.diag(.5*np.log(values))@Q.T
    iso=np.linalg.inv(F).T@(2*params.mu*logU+params.lam*np.trace(logU)*np.eye(3))
    A=params.A0;I4=np.sum(A*(F.T@F))
    return iso+2*params.k_f*(I4-1)*F@A


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='auto');p.add_argument('--out',type=Path,default=Path('docs/results/history-reference'))
    p.add_argument('--part',choices=('migration','reference','directions','all'),default='all')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    def guard():wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    guard();wp.init();device=select_lowest_memory_device(args.device)
    def save(name,data):(args.out/(name+'.json')).write_text(json.dumps(data,indent=2))
    def csvsave(name,rows):
        with (args.out/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),restval=0.)
            writer.writeheader();writer.writerows(rows)
    save('environment-'+args.part,dict(device=device,gpus=query_gpu_memory(),storage=vars(inspect_storage(DATA_ROOT))))
    if args.part in ('migration','all'):
        records=[]
        cases=[(mode,65,.05,policy,'exact') for mode in ('translate','rotate','directions') for policy in ('grid_locked','particle_resample')]
        cases += [(mode,grid,dt,'particle_resample','exact') for mode in ('translate','rotate','directions') for grid,dt in ((33,.05),(65,.025))]
        cases += [('rotate',65,dt,'particle_resample','euler') for dt in (.05,.025)]
        for mode,grid,dt,policy,rule in cases:
            guard();scene=KinematicScene(mode,grid,dt,history_mode=policy,device=device,rotation_rule=rule)
            for _ in range(round(scene.duration/dt)):guard();scene.step()
            name=f'{mode}-g{grid}-dt{dt}-{policy}-{rule}'
            csvsave(name,scene.rows)
            arrays={f'{key}_{i}':value for i,frame in enumerate(scene.frames) for key,value in frame.items()}
            np.savez_compressed(args.out/(name+'.npz'),**arrays)
            r=dict(name=name,mode=mode,grid=grid,dt=dt,history_mode=policy,rotation_rule=rule,steps=len(scene.rows)-1,final=scene.rows[-1],
                   max_F_error=max(row['F_relative_error'] for row in scene.rows),max_A_error=max(row['A_rms_error'] for row in scene.rows),
                   max_energy_jump=max(abs(b['elastic']-a['elastic']) for a,b in zip(scene.rows,scene.rows[1:])),
                   max_abs_energy_error=max(abs(row['elastic_relative_error']) for row in scene.rows))
            records.append(r);save('migration',records);print(json.dumps(r),flush=True);del scene;gc.collect()
    if args.part in ('reference','all'):
        patches=[]
        for mode in ('tension','shear'):
            for grid,dt in ((17,.05),(33,.05),(33,.025)):
                guard();scene=KinematicScene(mode,grid,dt,device=device)
                for _ in range(round(scene.duration/dt)):guard();scene.step()
                s=scene.solver
                from engine.aniso_phase1.sparse_kernels import aniso_center_tau_kernel
                from engine.sp_grid import B
                wp.launch(aniso_center_tau_kernel,dim=(int(s.bcn),B,B,B),inputs=[s.block_count,s.block_xyz_by_id,s.center_vol,s.aniso_committed_F,s.aniso_A0,s.center_tau,s.center_size,s.aniso_params.mu,s.aniso_params.lam,s.aniso_params.k_f],device=device)
                coords,vol,_,active=center_snapshot(s)
                F=s.aniso_committed_F[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active]
                tau=s.center_tau[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active]
                actual=np.einsum('pij,pjk->pik',tau,np.linalg.inv(F).transpose(0,2,1));mean=np.einsum('p,pij->ij',vol,actual)/vol.sum()
                expected=analytic_PK1(scene.map_at(scene.duration),s.aniso_params)
                r=dict(mode=mode,grid=grid,dt=dt,PK1_mean=mean.tolist(),PK1_reference=expected.tolist(),stress_relative_error=float(np.linalg.norm(mean-expected)/np.linalg.norm(expected)),final=scene.rows[-1])
                patches.append(r);save('patches',patches);print(json.dumps(r),flush=True);del scene;gc.collect()
        beams=[]
        for grid in (17,33):
            guard();r=beam_audit(grid,device=device);beams.append(r);save('beam',beams);print(json.dumps(r),flush=True);gc.collect()
        cycles=[]
        # Same maximum displacement, two complete cycles, modest frame counts.
        for grid,T,per_leg in ((9,.04,8),(9,.08,16),(9,.16,16),(9,.32,16),(9,.16,32),(17,.16,16)):
            guard();dt=T/per_leg
            scene=Scene(Config('tensile',grid,dt,loading_time=T,loading_speed=.002/T,loading_cycles=2,smooth_loading=True,residual_atol=1e-8,cg_tol=1e-3),device)
            for _ in range(4*per_leg):
                guard()
                if not scene.step():raise RuntimeError(str(scene.solver.last_step_stats))
            rows=scene.loading_rows
            first=min(rows,key=lambda r:abs(r['time']-T));reload=min(rows,key=lambda r:abs(r['time']-3*T))
            name=f'cycles-g{grid}-T{T}-dt{dt}';csvsave(name,scene.solver.energy_ledger.rows)
            r=dict(name=name,grid=grid,loading_time=T,dt=dt,steps=len(rows),first_peak_force=first['right_force'],reload_peak_force=reload['right_force'],
                   first_peak_elastic_force=first['right_elastic_force'],reload_peak_elastic_force=reload['right_elastic_force'],
                   loading_work=first['loading_work'],cycle_work=rows[-1]['loading_work'],final_mechanical=scene.solver.energy_ledger.rows[-1]['mechanical'],
                   final_work_balance=rows[-1]['mechanical_minus_loading_work'],max_free_force_residual=max(r['free_force_residual_norm'] for r in rows),
                   min_det_F=scene.metrics()['min_det_F'])
            cycles.append(r);save('cycles',cycles);print(json.dumps(r),flush=True);del scene;gc.collect()
        guard();scene=Scene(Config('fixed',8,.002,residual_atol=1e-8,cg_tol=1e-3),device)
        for _ in range(80):
            guard()
            if not scene.step():raise RuntimeError(str(scene.solver.last_step_stats))
        csvsave('fixed-extended',scene.solver.energy_ledger.rows)
        rows=scene.solver.energy_ledger.rows;E0=rows[0]['mechanical']
        save('extended',dict(steps=80,final_time=rows[-1]['time'],relative_energy_change=rows[-1]['mechanical']/E0-1,
                            max_abs_relative_energy_change=max(abs(r['mechanical']/E0-1) for r in rows),min_det_F=scene.metrics()['min_det_F']))
        del scene;gc.collect();guard()
        # Unconstrained motion complements prescribed-velocity history tests.
        s=AnisotropicLiteImplicitSolver((65,)*3,AnisotropicMaterialParams(10,20,200),dx=1/64,
                                       device=device,gravity=0,energy_diagnostics=True)
        a=np.linspace(.4,.46,4)
        x=np.stack(np.meshgrid(a,a+.05,a+.05,indexing='ij'),axis=-1).reshape(-1,3)
        s.seed_particles(x,density=1000.,vol0=.06**3/len(x),velocity=[1.,0.,0.],
                         deformation_gradient=np.diag([1.08,.98,1.02]))
        s.set_dt(.01);iterations=0
        for _ in range(8):
            guard()
            if not s.step(max_iters=16,cg_tol=1e-3,cg_atol=1e-10,newton_atol=1e-8,print_every=0):
                raise RuntimeError(str(s.last_step_stats))
            iterations+=s.last_step_stats['linear_iterations']
        csvsave('free-advection',s.energy_ledger.rows)
        save('free-advection',dict(device=device,steps=8,time=float(s.sim_time),linear_iterations=iterations,
             min_particle_det=float(np.linalg.det(s.ptc_F.numpy()).min()),
             max_particle_displacement=float(np.linalg.norm(s.ptc_x.numpy()-x,axis=1).max()),
             energy_relative_change=s.energy_ledger.rows[-1]['mechanical']/s.energy_ledger.rows[0]['mechanical']-1,
             max_budget_closure=max(abs(r.get('budget_closure',0)) for r in s.energy_ledger.rows)))
    if args.part in ('directions','all'):
        records=[]
        for name,angles in [('uniform',np.zeros(16)),('crossed',np.array([0.,90.]*8)),('smooth',np.linspace(0,90,16))]:
            a=np.stack([np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles)),np.zeros(len(angles))],axis=1)
            for label,F in [('stretch',np.diag([1.1,1.,1.])),('shear',np.array([[1.,.1,0],[0,1.,0],[0,0,1.]])),('balanced',np.diag(np.sqrt([1.1,.9,1.])) )]:
                r=audit_direction_mixture(F,a);r.update(direction_field=name,deformation=label)
                records.append(r);print(json.dumps(r),flush=True)
        save('direction-mixtures',records)


if __name__=='__main__':main()
