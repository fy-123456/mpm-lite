"""Short exact-rotation, release-budget and mixed-fiber cycle comparisons."""
import argparse,csv,gc,json
from pathlib import Path
import numpy as np
import warp as wp
from demos.aniso import Config,Scene,DATA_ROOT
from engine.aniso_phase1 import select_lowest_memory_device,query_gpu_memory
from engine.aniso_phase1.rotation_probe import BentRotation,reference_frame_audit,frozen_stencil_audit
from engine.aniso_phase1.diagnostics import energy_density
from utils.resource_guard import prepare_warp_cache,inspect_storage


def guard():
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)


def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),restval=0.)
        writer.writeheader();writer.writerows(rows)


def run(args):
    guard();wp.init();device=select_lowest_memory_device(args.device);args.out.mkdir(parents=True,exist_ok=True)
    def save(name,data):
        (args.out/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
    save('environment-'+args.part,dict(device=device,gpus=query_gpu_memory(),storage=vars(inspect_storage(DATA_ROOT))))
    if args.part in ('rotation','all'):
        save('reference-cell-rotation',reference_frame_audit());records=[]
        for grid,dt,axis in ((17,.01,'z'),(33,.01,'z'),(17,.005,'z'),(17,.01,'y')):
            for mode in ('none','supplemental','hourglass'):
                guard();scene=BentRotation(grid,dt,mode=mode,device=device,axis=axis)
                for step in range(round(scene.duration/dt)):
                    guard();scene.step()
                    if step==0 and grid==17 and dt==.01 and axis=='z' and mode!='none':
                        save('frozen-spatial-stencil-'+mode,frozen_stencil_audit(scene.solver))
                s=scene.solver;name=f'rotation-g{grid}-dt{dt}-{mode}'+('' if axis=='z' else '-y')
                table(args.out/(name+'.csv'),scene.rows);table(args.out/(name+'-budget.csv'),s.energy_ledger.rows)
                np.savez_compressed(args.out/(name+'.npz'),positions=scene.frames,fibers=scene.fibers,reference=scene.X,
                    times=np.arange(len(scene.frames))*dt)
                E0=s.energy_ledger.rows[0]['elastic'];energies=np.array([r['center_material_energy']+r['stabilization_energy'] for r in scene.rows])
                rr=dict(name=name,grid=grid,dt=dt,axis=axis,particles=s.n_ptc,stabilization=mode,initial_elastic=E0,
                    relative_energy_min=float(energies.min()/E0-1),relative_energy_max=float(energies.max()/E0-1),
                    max_relative_step_jump=float(np.abs(np.diff(np.r_[E0,energies])).max()/E0),
                    stabilization_energy_range=[min(r['stabilization_energy'] for r in scene.rows),max(r['stabilization_energy'] for r in scene.rows)],
                    max_particle_energy_error=max(abs(r['particle_energy_relative_change']) for r in scene.rows),
                    max_stress_rotation_error=max(r['stress_rotation_relative_error'] for r in scene.rows),
                    max_fiber_rotation_error=max(r['fiber_rotation_max_error'] for r in scene.rows),
                    max_history_error=max(r['history_F_relative_error'] for r in scene.rows),
                    max_position_error=max(r['position_max_error'] for r in scene.rows),
                    max_frozen_stabilization_torque=max(r['frozen_step_stabilization_torque'] for r in scene.rows))
                records.append(rr);save('rotation',records);print('ROTATION',json.dumps(rr),flush=True)
                del scene,s;gc.collect()
    if args.part in ('release','all'):
        records=[]
        for mode in ('none','supplemental'):
            for dt,flip in ((.001,.9),(.0005,.9),(.001,0.),(.001,1.)):
                guard();scene=Scene(Config('beam',17,dt,flip_ratio=flip,stabilization=mode,direction_model='fourth_moment',residual_atol=1e-8,cg_tol=1e-3),device)
                s=scene.solver;Up0=float(np.dot(s.ptc_vol0.numpy(),energy_density(s.ptc_F.numpy(),s.ptc_A0.numpy(),s.aniso_params)))
                frames=[s.ptc_x.numpy()]
                for _ in range(round(.008/dt)):
                    guard()
                    if not scene.step():raise RuntimeError(s.last_step_stats)
                    s.energy_ledger.rows[-1]['particle_material_energy']=float(np.dot(s.ptc_vol0.numpy(),energy_density(s.ptc_F.numpy(),s.ptc_A0.numpy(),s.aniso_params)))
                    frames.append(s.ptc_x.numpy())
                rows=s.energy_ledger.rows;E0=rows[0]['mechanical'];name=f'release-{mode}-dt{dt}-flip{flip}'
                table(args.out/(name+'.csv'),rows)
                np.savez_compressed(args.out/(name+'.npz'),positions=frames,reference=scene.reference)
                groups=dict(solve=['solve_delta'],velocity_transfer=['p2c_delta','c2g_delta','g2p_delta'],
                    history_reconstruction=['state_reset_delta','state_transport_delta','volume_remap_delta'],
                    boundary=['boundary_projection_delta','final_projection_damping_delta'])
                rr=dict(name=name,stabilization=mode,dt=dt,flip=flip,physical_time=s.sim_time,initial_energy=E0,
                    relative_energy_change=rows[-1]['mechanical']/E0-1,
                    budget_relative={k:sum(r.get(term,0.) for r in rows[1:] for term in terms)/E0 for k,terms in groups.items()},
                    max_budget_closure=max(abs(r.get('budget_closure',0.)) for r in rows),
                    particle_material_energy_initial=Up0,particle_material_energy_final=rows[-1]['particle_material_energy'],
                    min_J=float(np.linalg.det(s.ptc_F.numpy()).min()),max_motion=float(np.linalg.norm(frames[-1]-frames[0],axis=1).max()))
                records.append(rr);save('release',records);print('RELEASE',json.dumps(rr),flush=True);del scene,s;gc.collect()
    if args.part in ('tensile','all'):
        records=[]
        for mode,T in (('none',.08),('none',.16),('supplemental',.16)):
            for kind in ('center','particle'):
                guard();dt=.005
                scene=Scene(Config('tensile',9,dt,quadrature=kind,direction_model='fourth_moment' if kind=='center' else 'mean_tensor',
                    stabilization=mode,fiber_field='crossed',loading_cycles=2,smooth_loading=True,loading_time=T,loading_speed=.002/T),device)
                s=scene.solver;frames=[s.ptc_x.numpy()]
                for _ in range(round(4*T/dt)):
                    guard()
                    if not scene.step():raise RuntimeError(s.last_step_stats)
                    frames.append(s.ptc_x.numpy())
                name=f'tensile-{kind}-{mode}-T{T}';rows=scene.loading_rows;energies=s.energy_ledger.rows
                table(args.out/(name+'.csv'),energies);np.savez_compressed(args.out/(name+'.npz'),positions=frames,reference=scene.reference)
                peak=[min(rows,key=lambda r:abs(r['time']-t)) for t in (T,3*T)]
                n=round(2*T/dt);force=np.array([r['right_force'] for r in rows]);fel=np.array([r['right_elastic_force'] for r in rows])
                displacements=np.r_[0.,[r['displacement'] for r in rows]];dwork=force*np.diff(displacements)
                denom=max(np.linalg.norm(force[:n]),1e-15)
                rr=dict(name=name,quadrature=kind,stabilization=mode,loading_time=T,speed=.002/T,dt=dt,steps=len(rows),maximum_displacement=.002,
                    peak_reactions=[r['right_force'] for r in peak],peak_elastic_reactions=[r['right_elastic_force'] for r in peak],
                    loading_work=[float(dwork[i*n:i*n+n//2].sum()) for i in range(2)],
                    cycle_net_work=[float(dwork[i*n:(i+1)*n].sum()) for i in range(2)],
                    repeat_force_relative=float(np.linalg.norm(force[n:]-force[:n])/denom),
                    inertia_force_relative=float(np.linalg.norm(force-fel)/max(np.linalg.norm(force),1e-15)),
                    final_mechanical_minus_work=rows[-1]['mechanical_minus_loading_work'],
                    max_free_force_residual=max(r['free_force_residual_norm'] for r in rows),
                    max_budget_closure=max(abs(r.get('budget_closure',0.)) for r in energies),
                    min_J=float(np.linalg.det(s.ptc_F.numpy()).min()))
                records.append(rr);save('tensile',records);print('TENSILE',json.dumps(rr),flush=True);del scene,s;gc.collect()
        comparisons=[]
        for a,b in zip(records[::2],records[1::2]):
            ca=np.genfromtxt(args.out/(a['name']+'.csv'),delimiter=',',names=True)[1:]
            cb=np.genfromtxt(args.out/(b['name']+'.csv'),delimiter=',',names=True)[1:]
            comparisons.append(dict(loading_time=a['loading_time'],stabilization=a['stabilization'],
                force_curve_relative=float(np.linalg.norm(ca['right_force']-cb['right_force'])/np.linalg.norm(cb['right_force'])),
                elastic_force_curve_relative=float(np.linalg.norm(ca['right_elastic_force']-cb['right_elastic_force'])/np.linalg.norm(cb['right_elastic_force'])),
                loading_work_relative=[x/y-1 for x,y in zip(a['loading_work'],b['loading_work'])]))
        save('tensile-comparison',comparisons)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='auto')
    p.add_argument('--part',choices=('rotation','release','tensile','all'),default='all')
    p.add_argument('--out',type=Path,default=Path('docs/results/rotation-dissipation'))
    run(p.parse_args())
