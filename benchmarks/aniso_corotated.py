"""Short production corotated stabilization validation and regression report."""
import argparse,gc,json,time
from pathlib import Path
from functools import partial
import numpy as np
import warp as wp
from engine.types import vec3
from demos.aniso import Config,Scene,DATA_ROOT
from engine.aniso_phase1 import AnisotropicLiteImplicitSolver,select_lowest_memory_device,query_gpu_memory
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.rotation_probe import BentRotation,axis_rotation
from engine.aniso_phase1.stabilization_probe import static_audit,node_coordinates
from engine.aniso_phase1.controlled import KinematicScene
from benchmarks.aniso_rotation_dissipation import guard,table
from utils.resource_guard import inspect_storage


def operator_audit(device,stabilization='corotated'):
    p=SparseProbe(device=device,dt=.005,boundary=True,
        solver_cls=partial(AnisotropicLiteImplicitSolver,stabilization=stabilization,direction_model='fourth_moment'))
    rng=np.random.default_rng(92);v=p.project(rng.normal(size=(p.n,3))*.15)
    d=p.project(rng.normal(size=v.shape));d/=np.linalg.norm(d)
    r=p.residual(v);H=p.tangent(d);eps=1e-5
    rp=p.residual(v+eps*d);ep=p.s.incremental_potential()
    rm=p.residual(v-eps*d);em=p.s.incremental_potential()
    gradient_error=abs((ep-em)/(2*eps)-np.sum(r*d))/max(abs(np.sum(r*d)),1e-15)
    tangent_error=np.linalg.norm((rp-rm)/(2*eps)-H)/np.linalg.norm(H)
    p.residual(v)
    basis=np.eye(3*p.n)
    Q=np.column_stack([p.project(x.reshape(p.n,3)).ravel() for x in basis])
    eig,Z=np.linalg.eigh((Q+Q.T)/2);Z=Z[:,eig>.5];spectra={}
    for pd,label in ((False,'exact'),(True,'gauss_newton')):
        J=Z.T@np.column_stack([p.tangent(d.reshape(p.n,3),pd).ravel() for d in Z.T])
        spectra[label]=dict(symmetry_error=float(np.linalg.norm(J-J.T)/np.linalg.norm(J)),min_eigenvalue=float(np.linalg.eigvalsh((J+J.T)/2)[0]))
    # Isolate the stabilizer; rotate the trial configuration on the SAME map.
    s=p.s;e=s.enhancements;x=node_coordinates(s)*s.dx
    def evaluate(values):
        e.values.zero_();wp.copy(e.values,wp.array(values,dtype=vec3,device=device),count=p.n)
        out=wp.zeros_like(e.extra);energy=wp.zeros_like(e.hg_energy);s.aniso_invalid_trial.zero_()
        e._hg(e.values,out,energy,False)
        if int(s.aniso_invalid_trial.numpy()[0]):raise ValueError('invalid objective probe')
        return float(energy.numpy()[0]),out[:p.n].numpy()
    E,g=evaluate(v);objective=[]
    for axis in ('z','y'):
        R=axis_rotation(.9,axis);vr=((x+s.dt*v-.5)@R.T+.5-x)/s.dt
        Er,gr=evaluate(vr)
        objective.append(dict(axis=axis,energy_relative_error=abs(Er/E-1),force_covariance_error=float(np.linalg.norm(gr-g@R.T)/np.linalg.norm(g))))
    return dict(active_blocks=int(s.bcn),free_dofs=Z.shape[1],potential_gradient_error=float(gradient_error),exact_tangent_fd_error=float(tangent_error),spectra=spectra,objective=objective)


def reference_rebuild_audit(device):
    """Replace nodal history interpolation by the analytic inverse bend.

    Still uses the same fixed Cartesian Q1 cells/particle volumes. This isolates
    history reconstruction from spatial interpolation/quadrature orientation.
    The prescribed particle motion and material state are never modified.
    """
    from engine.aniso_phase1.corotated import prepare_reference
    rows=[]
    for grid in (17,33):
        scene=BentRotation(grid,.01,mode='corotated',device=device)
        for step in range(5):
            guard();scene.step()
            if step not in (0,4):continue
            s=scene.solver;e=s.enhancements;xyz=node_coordinates(s)*s.dx
            angle=step*np.pi/16;Q=axis_rotation(angle)
            bent=(xyz-.5)@Q+.5
            xb=bent[:,0]-.25;yb=bent[:,1]-.5;a=.01
            local=xb.copy()
            for _ in range(8):local-=(local*(1+8*a*yb+32*a*a*local**2)-xb)/(1+8*a*yb+96*a*a*local**2)
            X=bent.copy();X[:,0]=.25+local;X[:,1]=.5+yb+4*a*local**2
            # Independent inverse check, including the extended support nodes.
            from engine.aniso_phase1.stabilization_probe import bent_beam_state
            recovered,_=bent_beam_state(X,amplitude=a)
            inverse_error=float(np.linalg.norm(recovered-bent,axis=1).max())
            old_energy=float(e.hg_energy.numpy()[0])
            wp.copy(e.u,wp.array(xyz-X,dtype=vec3,device=device),count=len(xyz))
            invalid=wp.zeros(1,dtype=int,device=device)
            wp.launch(prepare_reference,dim=(len(e.ids),9),inputs=[e.ids,e.u,e.reference_g,e.reference_F0,e.reference_B,invalid],device=device)
            if int(invalid.numpy()[0]):raise ValueError('analytic reference map invalid')
            zero=wp.zeros_like(e.values);out=wp.zeros_like(e.extra);energy=wp.zeros_like(e.hg_energy)
            e._hg(zero,out,energy,False)
            rows.append(dict(grid=grid,sampling_angle=angle*180/np.pi,particle_reconstructed_stabilization_energy=old_energy,
                analytic_reference_stabilization_energy=float(energy.numpy()[0]),analytic_inverse_error=inverse_error))
        del scene;gc.collect()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='auto')
    p.add_argument('--part',choices=('operators','rotation','scenes','reconstruction','all'),default='all')
    p.add_argument('--out',type=Path,default=Path('docs/results/corotated'))
    args=p.parse_args();guard();wp.init();device=select_lowest_memory_device(args.device);args.out.mkdir(parents=True,exist_ok=True)
    def save(name,data):(args.out/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
    save('environment-'+args.part,dict(device=device,gpus=query_gpu_memory(),storage=vars(inspect_storage(DATA_ROOT))))
    if args.part in ('reconstruction','all'):
        save('reference-reconstruction',reference_rebuild_audit(device));gc.collect()
    if args.part in ('operators','all'):
        save('operator',operator_audit(device));gc.collect()
        records=[]
        for grid in (17,33):
            guard();records+=static_audit(grid,device,include_corotated=True);save('static',records);gc.collect()
        patches=[]
        for mode in ('tension','shear'):
            guard();scene=KinematicScene(mode,17,.025,.1,device=device,solver_options=dict(stabilization='corotated',direction_model='fourth_moment'))
            for _ in range(4):scene.step()
            patches.append(dict(mode=mode,final=scene.rows[-1],stabilization_energy=scene.solver.energy_ledger.rows[-1]['stabilization_energy']))
            del scene;gc.collect()
        save('affine-patches',patches)
    if args.part in ('rotation','all'):
        records=[]
        for grid,dt,axis in ((17,.01,'z'),(33,.01,'z'),(17,.005,'z'),(17,.01,'y'),(33,.01,'y')):
            guard();scene=BentRotation(grid,dt,mode='corotated',device=device,axis=axis)
            for _ in range(round(.08/dt)):guard();scene.step()
            s=scene.solver;name=f'rotation-g{grid}-dt{dt}-corotated'+('' if axis=='z' else '-y')
            table(args.out/(name+'.csv'),scene.rows);table(args.out/(name+'-budget.csv'),s.energy_ledger.rows)
            np.savez_compressed(args.out/(name+'.npz'),positions=scene.frames,fibers=scene.fibers,reference=scene.X,times=np.arange(len(scene.frames))*dt)
            rows=s.energy_ledger.rows;E0=rows[0]['elastic'];energy=np.array([r['elastic'] for r in rows]);initial_hg=rows[0]['stabilization_energy']
            rr=dict(name=name,grid=grid,dt=dt,axis=axis,steps=len(rows)-1,initial_energy=E0,
                relative_energy_min=float(energy.min()/E0-1),relative_energy_max=float(energy.max()/E0-1),
                max_frozen_step_stabilization_energy_error=max(abs(r['stabilization_solve_delta'])/max(r['stabilization_start_energy'],1e-15) for r in rows[1:]),
                stabilization_rebuild_sum_relative=sum(r.get('stabilization_rebuild_delta',0) for r in rows)/E0,
                max_history_error=max(r['history_F_relative_error'] for r in scene.rows),max_particle_energy_error=max(abs(r['particle_energy_relative_change']) for r in scene.rows),
                max_trial_torque=max(r['trial_configuration_stabilization_torque'] for r in scene.rows))
            records.append(rr);save('rotation',records);print('ROTATION',json.dumps(rr),flush=True);del scene,s;gc.collect()
    if args.part in ('scenes','all'):
        records=[]
        for dt in (.001,.0005):
            guard();scene=Scene(Config('beam',17,dt,stabilization='corotated',direction_model='fourth_moment',residual_atol=1e-8,cg_tol=1e-3),device)
            s=scene.solver;frames=[s.ptc_x.numpy()];iterations=0;times=[]
            for _ in range(round(.008/dt)):
                guard();wp.synchronize_device(device);start=time.perf_counter()
                if not scene.step():raise RuntimeError(s.last_step_stats)
                wp.synchronize_device(device);times.append(time.perf_counter()-start);iterations+=s.last_step_stats['linear_iterations'];frames.append(s.ptc_x.numpy())
            name=f'beam-corotated-dt{dt}';rows=s.energy_ledger.rows
            table(args.out/(name+'.csv'),rows);np.savez_compressed(args.out/(name+'.npz'),positions=frames,reference=scene.reference)
            rr=dict(name=name,dt=dt,steps=len(rows)-1,relative_energy_change=rows[-1]['mechanical']/rows[0]['mechanical']-1,
                min_J=float(np.linalg.det(s.ptc_F.numpy()).min()),max_motion=float(np.linalg.norm(frames[-1]-frames[0],axis=1).max()),linear_iterations=iterations,
                median_diagnostic_step_ms=1000*float(np.median(times[2:])),enhancement_bytes=s.enhancements.memory_bytes(),max_budget_closure=max(abs(r.get('budget_closure',0)) for r in rows))
            records.append(rr);save('beam',records);print('BEAM',json.dumps(rr),flush=True);del scene,s;gc.collect()
        records=[]
        for kind in ('center','particle'):
            guard();scene=Scene(Config('tensile',9,.005,quadrature=kind,stabilization='corotated',direction_model='fourth_moment' if kind=='center' else 'mean_tensor',
                fiber_field='crossed',loading_cycles=2,smooth_loading=True,loading_time=.16,loading_speed=.0125),device)
            s=scene.solver;frames=[s.ptc_x.numpy()]
            for _ in range(128):
                guard()
                if not scene.step():raise RuntimeError(s.last_step_stats)
                frames.append(s.ptc_x.numpy())
            name=f'tensile-{kind}-corotated';rows=s.energy_ledger.rows;table(args.out/(name+'.csv'),rows)
            np.savez_compressed(args.out/(name+'.npz'),positions=frames,reference=scene.reference)
            peak=[min(scene.loading_rows,key=lambda r:abs(r['time']-t)) for t in (.16,.48)]
            rr=dict(name=name,quadrature=kind,steps=s.sim_steps,peak_reactions=[r['right_force'] for r in peak],loading_work=scene.loading_work,
                min_J=float(np.linalg.det(s.ptc_F.numpy()).min()),max_budget_closure=max(abs(r.get('budget_closure',0)) for r in rows))
            records.append(rr);save('tensile',records);print('TENSILE',json.dumps(rr),flush=True);del scene,s;gc.collect()


if __name__=='__main__':main()
