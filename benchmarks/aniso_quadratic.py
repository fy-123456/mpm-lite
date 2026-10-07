"""Short quadratic reconstruction comparisons, with static checks without mass."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from engine.types import vec3
from engine.sp_grid import B
from engine.aniso_phase1.quadratic import MODES,node_patch
from engine.aniso_phase1.beam_reference import beam_matrices,reference_hessian
from engine.aniso_phase1.stabilization_probe import node_coordinates
from engine.aniso_phase1.rotation_probe import BentRotation
from engine.aniso_phase1 import AnisotropicLiteImplicitSolver,AnisotropicMaterialParams,select_lowest_memory_device,query_gpu_memory
from engine.aniso_phase1.controlled import KinematicScene
from demos.aniso import Scene,Config,DATA_ROOT
from benchmarks.aniso_rotation_dissipation import guard,table
from benchmarks.aniso_corotated import operator_audit
from utils.resource_guard import inspect_storage


def static_audit(grid,device):
    """Assemble the linearized potential independently of GPU polar kernels.

    The MLS coefficients are shared; assembly, solve and eigenanalysis are host
    reference operations. Remove mass before comparing the production matvec.
    """
    nodes,centers,free,tip,force,matrices=beam_matrices(grid)
    h=1/(grid-1);x=nodes*h;tree=cKDTree(x);H=reference_hessian()
    rows=[];cols=[];vals=[]
    for center in centers:
        ids,g,_,_=node_patch(x,center,h,tree);K=np.zeros((3*len(ids),)*2)
        for dg in g[1:]-g[:1]:
            mapping=np.zeros((9,3*len(ids)))
            for a in range(3):mapping[a*3:a*3+3,a::3]=dg.T
            K+=h**3/8*mapping.T@H@mapping
        dofs=(ids[:,None]*3+np.arange(3)).ravel()
        rows.extend(np.repeat(dofs,len(dofs)));cols.extend(np.tile(dofs,len(dofs)));vals.extend(K.ravel())
    K=matrices[0]+coo_matrix((vals,(rows,cols)),shape=matrices[0].shape).tocsr()
    eigen=np.linalg.eigvalsh(K[free][:,free].toarray())
    u=np.zeros(len(force));u[free]=spsolve(K[free][:,free],force[free]);tip_u=float(np.mean(u[3*tip+1]))
    full=np.zeros(len(force));full[free]=spsolve(matrices[1][free][:,free],force[free]);full_tip=float(np.mean(full[3*tip+1]))
    result=dict(grid=grid,soft_modes=int(np.sum(eigen<1e-9*max(eigen[-1],1))),min_eigenvalue=float(eigen[0]),
        tip_displacement=tip_u,full_reference_tip=full_tip,relative_to_full=abs(tip_u/full_tip-1),production=[])
    for mode in MODES:
        s=AnisotropicLiteImplicitSolver((grid,)*3,AnisotropicMaterialParams(10,20,200),dx=h,device=device,gravity=0,stabilization=mode)
        s.seed_particles(centers,density=1,vol0=h**3)
        fixed=nodes[nodes[:,0]==nodes[:,0].min()];s.paint_boundary(fixed,np.ones(len(fixed),dtype=np.int32))
        s.set_dt(.01);s.step(max_iters=0,print_every=0);s.evaluate_residual()
        actual=node_coordinates(s);lookup={tuple(c):i for i,c in enumerate(nodes)};perm=np.array([lookup[tuple(c)] for c in actual]);n=len(actual)
        p=np.random.default_rng(4).normal(size=(len(nodes),3));p[nodes[:,0]==nodes[:,0].min()]=0
        d=wp.zeros_like(s.node_residual);out=wp.zeros_like(d);wp.copy(d,wp.array(p[perm],dtype=vec3,device=device),count=n)
        s.apply_tangent(d,out);nd=s.ndof2bijk[:n].numpy();m=s.grid_m[:int(s.bcn)].numpy()
        mass=np.array([m[b,l//(B*B),(l//B)%B,l%B] for b,l in nd])
        measured=(out[:n].numpy()-mass[:,None]*p[perm])/s.dt**2
        target=(K@p.ravel()).reshape(-1,3)[perm];target[actual[:,0]==nodes[:,0].min()]=0
        result['production'].append(dict(mode=mode,matvec_relative_error=float(np.linalg.norm(measured-target)/np.linalg.norm(target))))
        del s;gc.collect()
    return result


def rotation_run(grid,axis,mode,device,out):
    scene=BentRotation(grid,.01,mode=mode,device=device,axis=axis);preparation=[]
    for _ in range(8):guard();scene.step();preparation.append(scene.solver.last_step_stats['reconstruction_seconds'])
    s=scene.solver;rows=s.energy_ledger.rows;E0=rows[0]['elastic'];name=f'rotation-g{grid}-dt0.01-{mode}'+('' if axis=='z' else '-y')
    table(out/(name+'.csv'),scene.rows);table(out/(name+'-budget.csv'),rows)
    np.savez_compressed(out/(name+'.npz'),positions=scene.frames,fibers=scene.fibers,reference=scene.X,times=np.arange(9)*.01)
    energy=np.array([r['elastic'] for r in rows]);hg=np.array([r['stabilization_energy'] for r in rows])
    material=energy-hg;material0=material[0]
    return dict(name=name,grid=grid,axis=axis,mode=mode,steps=8,initial_energy=E0,initial_stabilization=hg[0],
        relative_energy_min=float(energy.min()/E0-1),relative_energy_max=float(energy.max()/E0-1),
        max_stabilization_relative_change=float(np.max(np.abs(hg/hg[0]-1))),
        max_material_relative_change=float(np.max(np.abs(material/material0-1))),
        max_frozen_step_energy_error=max(abs(r['stabilization_solve_delta'])/max(r['stabilization_start_energy'],1e-15) for r in rows[1:]),
        max_particle_F_error=max(r['particle_F_max_error'] for r in scene.rows),
        max_particle_energy_error=max(abs(r['particle_energy_relative_change']) for r in scene.rows),
        max_stress_rotation_error=max(r['stress_rotation_relative_error'] for r in scene.rows),
        max_fiber_rotation_error=max(r['fiber_rotation_max_error'] for r in scene.rows),
        max_trial_torque=max(r['trial_configuration_stabilization_torque'] for r in scene.rows),
        median_reconstruction_ms=1000*float(np.median(preparation)),enhancement_bytes=s.enhancements.memory_bytes())


def translation_run(grid,mode,device,out):
    """Prescribed bent beam translation by two coarse cells; F is unchanged."""
    from engine.aniso_phase1.diagnostics import energy_density
    scene=BentRotation(grid,.01,mode=mode,device=device);s=scene.solver
    velocity=np.array([1.5625,0.,0.]);nodes=scene.nodes.numpy()
    s.ptc_v.assign(wp.array(np.tile(velocity,(s.n_ptc,1)),dtype=vec3,device=device));s.ptc_G.zero_()
    s.paint_boundary(nodes,np.ones(len(nodes),dtype=np.int32),boundary_v=np.tile(velocity,(len(nodes),1)))
    rows=[];frames=[s.ptc_x.numpy()]
    for _ in range(8):
        guard()
        if not s.step(max_iters=8,print_every=0):raise RuntimeError(s.last_step_stats)
        x=s.ptc_x.numpy();F=s.ptc_F.numpy();e=s.energy_ledger.rows[-1]
        rows.append(dict(time=s.sim_time,elastic=e['elastic'],stabilization_energy=e['stabilization_energy'],
            particle_energy=float(np.dot(s.ptc_vol0.numpy(),energy_density(F,s.ptc_A0.numpy(),s.aniso_params))),
            shape_error=float(np.linalg.norm(x-scene.x0-s.sim_time*velocity,axis=1).max()),F_error=float(np.linalg.norm(F-scene.F0,axis=(1,2)).max())))
        frames.append(x)
    name=f'translation-g{grid}-{mode}';table(out/(name+'.csv'),rows);table(out/(name+'-budget.csv'),s.energy_ledger.rows)
    np.savez_compressed(out/(name+'.npz'),positions=frames,reference=scene.X)
    energies=np.array([r['elastic'] for r in s.energy_ledger.rows]);hg=np.array([r['stabilization_energy'] for r in s.energy_ledger.rows])
    return dict(name=name,grid=grid,mode=mode,translation=.125,relative_energy_min=float(energies.min()/energies[0]-1),
        relative_energy_max=float(energies.max()/energies[0]-1),max_stabilization_relative_change=float(np.max(np.abs(hg/hg[0]-1))),
        max_shape_error=max(r['shape_error'] for r in rows),max_F_error=max(r['F_error'] for r in rows),
        max_relative_step_jump=float(np.max(np.abs(np.diff(energies)))/energies[0]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='auto')
    p.add_argument('--part',choices=('operators','rotation','translation','scenes','all'),default='all')
    p.add_argument('--out',type=Path,default=Path('docs/results/quadratic'));args=p.parse_args()
    guard();wp.init();device=select_lowest_memory_device(args.device);args.out.mkdir(parents=True,exist_ok=True)
    def save(name,data):(args.out/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
    save('environment-'+args.part,dict(device=device,gpus=query_gpu_memory(),storage=vars(inspect_storage(DATA_ROOT))))
    if args.part in ('operators','all'):
        records=[]
        for mode in MODES:
            guard();r=operator_audit(device,mode);r['mode']=mode;records.append(r);save('operator',records);gc.collect()
        records=[]
        for grid in (17,33):guard();records.append(static_audit(grid,device));save('static',records);gc.collect()
        records=[]
        for mode in MODES:
            for patch in ('tension','shear'):
                guard();s=KinematicScene(patch,17,.025,.1,device=device,solver_options=dict(stabilization=mode,direction_model='fourth_moment'))
                for _ in range(4):s.step()
                records.append(dict(mode=mode,patch=patch,final=s.rows[-1],stabilization_energy=s.solver.energy_ledger.rows[-1]['stabilization_energy']))
                save('patches',records);del s;gc.collect()
    if args.part in ('rotation','all'):
        records=[]
        for grid in (17,33):
            for axis in ('z','y'):
                for mode in MODES:
                    guard();r=rotation_run(grid,axis,mode,device,args.out);records.append(r);save('rotation',records);print('ROTATION',json.dumps(r),flush=True);gc.collect()
    if args.part in ('translation','all'):
        records=[]
        for grid in (17,33):
            for mode in ('corotated',)+MODES:
                guard();r=translation_run(grid,mode,device,args.out);records.append(r);save('translation',records);print('TRANSLATION',json.dumps(r),flush=True);gc.collect()
    if args.part in ('scenes','all'):
        records=[]
        for mode in MODES:
            for kind in ('beam','tensile'):
                guard();s=Scene(Config(kind,17 if kind=='beam' else 9,.0005 if kind=='beam' else .005,
                    stabilization=mode,direction_model='fourth_moment',fiber_field='uniform' if kind=='beam' else 'crossed',
                    loading_cycles=1,smooth_loading=True,loading_time=.08,loading_speed=.025),device)
                count=16 if kind=='beam' else 32;cost=[];prep=[];iters=0
                for _ in range(count):
                    guard();start=time.perf_counter()
                    if not s.step():raise RuntimeError(s.solver.last_step_stats)
                    wp.synchronize_device(device);cost.append(time.perf_counter()-start);stats=s.solver.last_step_stats
                    prep.append(stats['reconstruction_seconds']);iters+=stats['linear_iterations']
                rows=s.solver.energy_ledger.rows;name=kind+'-'+mode;table(args.out/(name+'.csv'),rows)
                r=dict(name=name,steps=count,physical_time=s.solver.sim_time,min_J=float(np.linalg.det(s.solver.ptc_F.numpy()).min()),
                    relative_mechanical_change=rows[-1]['mechanical']/rows[0]['mechanical']-1 if rows[0]['mechanical'] else None,
                    max_budget_closure=max(abs(r.get('budget_closure',0)) for r in rows),linear_iterations=iters,
                    median_step_ms=1000*float(np.median(cost[2:])),median_reconstruction_ms=1000*float(np.median(prep[2:])),
                    enhancement_bytes=s.solver.enhancements.memory_bytes())
                records.append(r);save('scenes',records);print('SCENE',json.dumps(r),flush=True);del s;gc.collect()


if __name__=='__main__':main()
