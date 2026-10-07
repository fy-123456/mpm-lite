"""Short optional stabilization/moment validation and matched performance audit."""
import argparse,csv,gc,json,time
from pathlib import Path
from functools import partial
import numpy as np
import warp as wp
from engine.types import mat33,vec3
from engine.sp_grid import B
from demos.aniso import Config,Scene,DATA_ROOT
from engine.aniso_phase1 import AnisotropicLiteImplicitSolver,AnisotropicMaterialParams,select_lowest_memory_device,query_gpu_memory
from engine.aniso_phase1.stabilization_probe import static_audit,bent_beam_state,node_coordinates
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.controlled import weighted_particle_centers
from engine.aniso_phase1.diagnostics import center_snapshot,energy_density
from engine.aniso_phase1.constitutive import pk1
from engine.aniso_phase1.particle_quadrature import ParticleQuadratureImplicitSolver
from utils.resource_guard import prepare_warp_cache,inspect_storage


def moments_at_centers(s,F,active):
    A=s.aniso_A0[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active]
    P=np.array([pk1(f,a,s.aniso_params) for f,a in zip(F,A)])
    if s.direction_model=='fourth_moment':
        M=s.enhancements.M.numpy();C=np.einsum('pji,pjk->pik',F,F).reshape(-1,9)
        covariance=np.einsum('pij,pj->pi',M,C)-A.reshape(-1,9)*np.einsum('pi,pi->p',A.reshape(-1,9),C)[:,None]
        P+=2*s.aniso_params.k_f*np.einsum('pij,pjk->pik',F,covariance.reshape(-1,3,3))
    return P


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='auto')
    p.add_argument('--out',type=Path,default=Path('docs/results/stabilization-moments'))
    p.add_argument('--part',choices=('beam','moments','migration','performance','all'),default='all');args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    def guard():wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    guard();wp.init();device=select_lowest_memory_device(args.device)
    def save(name,data):(args.out/(name+'.json')).write_text(json.dumps(data,indent=2))
    def table(name,rows):
        with (args.out/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,list(dict.fromkeys(k for r in rows for k in r)),restval=0.);writer.writeheader();writer.writerows(rows)
    save('environment-'+args.part,dict(device=device,gpus=query_gpu_memory(),storage=vars(inspect_storage(DATA_ROOT))))
    if args.part in ('beam','all'):
        records=[]
        for grid in (17,33):
            guard();records+=static_audit(grid,device);save('static-beam',records);print('STATIC',json.dumps(records[-3:]),flush=True);gc.collect()
        records=[]
        for grid,dt in ((17,.001),(33,.001),(17,.0005)):
            for mode in ('none','supplemental','hourglass'):
                guard();scene=Scene(Config('beam',grid,dt,stabilization=mode,direction_model='fourth_moment',residual_atol=1e-8,cg_tol=1e-3),device)
                frames=[scene.solver.ptc_x.numpy()];iters=0
                for _ in range(round(.008/dt)):
                    guard()
                    if not scene.step():raise RuntimeError(scene.solver.last_step_stats)
                    frames.append(scene.solver.ptc_x.numpy());iters+=scene.solver.last_step_stats['linear_iterations']
                rows=scene.solver.energy_ledger.rows;E0=rows[0]['mechanical'];name=f'beam-g{grid}-dt{dt}-{mode}'
                table(name,rows);np.savez_compressed(args.out/(name+'.npz'),positions=np.array(frames),reference=scene.reference)
                r=dict(name=name,grid=grid,dt=dt,stabilization=mode,steps=len(rows)-1,relative_energy_change=rows[-1]['mechanical']/E0-1,
                    max_abs_energy_change=max(abs(x['mechanical']/E0-1) for x in rows),min_J=float(np.linalg.det(scene.solver.ptc_F.numpy()).min()),
                    max_motion=float(np.linalg.norm(frames[-1]-frames[0],axis=1).max()),linear_iterations=iters,max_budget_closure=max(abs(x.get('budget_closure',0)) for x in rows),
                    final_stabilization_energy=rows[-1].get('stabilization_energy',0.))
                records.append(r);save('dynamic-beam',records);print('DYNAMIC',json.dumps(r),flush=True);del scene;gc.collect()
    if args.part in ('moments','all'):
        records=[]
        for field in ('uniform','crossed','smooth'):
            for model in ('mean_tensor','fourth_moment'):
                guard();probe=SparseProbe(device=device,boundary=False,solver_cls=partial(AnisotropicLiteImplicitSolver,direction_model=model,stabilization='supplemental'))
                s=probe.s;n=s.n_ptc
                angles=np.zeros(n) if field=='uniform' else (np.where(np.arange(n)%2,90.,0.) if field=='crossed' else np.linspace(0,90,n))
                a=np.stack([np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles)),np.zeros(n)],axis=1);A=np.einsum('pi,pj->pij',a,a)
                s.ptc_A0.assign(wp.array(A,dtype=mat33,device=device));s.step(max_iters=0,print_every=0)
                F=np.diag([1.1,1.,1.]);probe.set_deformation(F);v=np.zeros((probe.n,3));r0=probe.residual(v);phi=s.incremental_potential()
                expected=float(np.dot(s.ptc_vol0.numpy(),energy_density(np.broadcast_to(F,(n,3,3)),A,s.aniso_params)))
                # Aggregate stress from the actual nodal residual's affine virtual work.
                xyz=node_coordinates(s)*s.dx
                P=np.array([[np.sum(r0[:,i]*xyz[:,j])/(s.dt*s.ptc_vol0.numpy().sum()) for j in range(3)] for i in range(3)])
                Pref=sum(pk1(F,aa,s.aniso_params) for aa in A)/n
                # J response to affine directions extracts material H, with mass removed.
                H=np.empty((9,9));F0inv=np.linalg.inv(F);ndof=s.ndof2bijk[:probe.n].numpy();mass=s.grid_m[:int(s.bcn)].numpy()
                m=np.array([mass[b,l//(B*B),(l//B)%B,l%B] for b,l in ndof])
                for j,dF in enumerate(np.eye(9).reshape(9,3,3)):
                    dG=dF@F0inv;d=xyz@dG.T;Ap=probe.tangent(d)-m[:,None]*d
                    dPF=np.array([[np.sum(Ap[:,i]*xyz[:,k])/(s.dt**2*s.ptc_vol0.numpy().sum()) for k in range(3)] for i in range(3)])
                    H[:,j]=(dPF@F0inv.T).ravel()
                from engine.aniso_phase1.constitutive import dP_apply_finite_difference
                Href=np.column_stack([sum(dP_apply_finite_difference(F,aa,dF,s.aniso_params) for aa in A).ravel()/n for dF in np.eye(9).reshape(9,3,3)])
                # r virtual work yields P F0^T, so undo the frozen pullback.
                P=P@F0inv.T
                rr=dict(field=field,direction_model=model,energy_relative_error=abs(phi/expected-1),stress_relative_error=float(np.linalg.norm(P-Pref)/np.linalg.norm(Pref)),
                        tangent_relative_error=float(np.linalg.norm(H-Href)/np.linalg.norm(Href)),enhancement_bytes=s.enhancements.memory_bytes())
                records.append(rr);save('moments',records);print('MOMENTS',json.dumps(rr),flush=True);del probe,s;gc.collect()
        # Same affine motion with and without stabilization: gradients fluctuate by zero.
        from engine.aniso_phase1.controlled import KinematicScene
        patches=[]
        for mode in ('tension','shear'):
            for stabilizer in ('none','supplemental','hourglass'):
                guard();scene=KinematicScene(mode,17,.025,.1,device=device,solver_options=dict(stabilization=stabilizer,direction_model='fourth_moment'))
                for _ in range(4):scene.step()
                rr=dict(mode=mode,stabilization=stabilizer,final=scene.rows[-1],stabilization_energy=scene.solver.energy_ledger.rows[-1].get('stabilization_energy',0.))
                patches.append(rr);save('patches',patches);del scene;gc.collect()
    if args.part in ('migration','all'):
        records=[]
        lo=np.array([.25,.4375,.4375]);hi=np.array([.75,.5625,.5625]);counts=(32,8,8)
        axes=[lo[d]+(np.arange(counts[d])+.5)*(hi[d]-lo[d])/counts[d] for d in range(3)]
        X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3);x0,F=bent_beam_state(X)
        for grid,dt in ((33,.01),(65,.01),(33,.005)):
            for mode in ('none','supplemental'):
                guard();s=AnisotropicLiteImplicitSolver((grid,)*3,AnisotropicMaterialParams(10,20,200),dx=1/(grid-1),device=device,gravity=0,energy_diagnostics=True,stabilization=mode,direction_model='fourth_moment')
                s.seed_particles(x0,density=1,vol0=.5*.125**2/len(X),deformation_gradient=F,reference_positions=X,velocity=[1.625,0,0])
                s.set_dt(dt);nodes=np.stack(np.meshgrid(*[np.arange(grid)]*3,indexing='ij'),axis=-1).reshape(-1,3).astype(np.int32)
                s.paint_boundary(nodes,np.ones(len(nodes),dtype=np.int32),boundary_v=np.tile([1.625,0,0],(len(nodes),1)))
                s.step(max_iters=0,print_every=0);rows=[];frames=[x0.copy()];particleP=np.array([pk1(f,s.aniso_params.A0,s.aniso_params) for f in F])
                for _ in range(round(.08/dt)):
                    guard();sample=x0+np.array([s.sim_time*1.625,0,0])
                    if not s.step(max_iters=8,print_every=0,newton_atol=1e-8):raise RuntimeError(s.last_step_stats)
                    coords,vol,psi,active=center_snapshot(s);Fc=s.aniso_committed_F[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active]
                    oracle=weighted_particle_centers(sample,s.ptc_vol0.numpy(),F,s.dx);target=np.array([oracle[tuple(c)] for c in coords])
                    stressoracle=weighted_particle_centers(sample,s.ptc_vol0.numpy(),particleP,s.dx);targetP=np.array([stressoracle[tuple(c)] for c in coords]);P=moments_at_centers(s,Fc,active)
                    row=dict(time=s.sim_time,history_F_error=float(np.linalg.norm(Fc-target)/np.linalg.norm(target)),stress_averaging_error=float(np.linalg.norm(P-targetP)/np.linalg.norm(targetP)),
                             shape_error=float(np.linalg.norm(s.ptc_x.numpy()-x0-np.array([s.sim_time*1.625,0,0]),axis=1).max()),elastic=float(np.dot(vol,psi)),
                             stabilization_energy=s.energy_ledger.rows[-1].get('stabilization_energy',0.),min_J=float(np.linalg.det(Fc).min()))
                    rows.append(row);frames.append(s.ptc_x.numpy())
                name=f'bent-translation-g{grid}-dt{dt}-{mode}';table(name,rows);table(name+'-budget',s.energy_ledger.rows)
                np.savez_compressed(args.out/(name+'.npz'),positions=np.array(frames),reference=X)
                initial=s.energy_ledger.rows[0]['elastic'];r=dict(name=name,grid=grid,dt=dt,stabilization=mode,steps=len(rows),final=rows[-1],
                     elastic_relative_change=(rows[-1]['elastic']+rows[-1]['stabilization_energy'])/initial-1,
                     max_elastic_step_jump=max(abs(b['elastic']+b['stabilization_energy']-a['elastic']-a['stabilization_energy']) for a,b in zip(rows,rows[1:]))/initial)
                records.append(r);save('migration',records);print('MIGRATION',json.dumps(r),flush=True);del s;gc.collect()
    if args.part in ('performance','all'):
        records=[]
        for samples in (2,4):
            axis=.375+(np.arange(4*samples)+.5)*.25/(4*samples)
            points=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
            for kind,model in (('center','mean_tensor'),('center','fourth_moment'),('particle','mean_tensor')):
                guard();cls=AnisotropicLiteImplicitSolver if kind=='center' else ParticleQuadratureImplicitSolver
                s=cls((17,)*3,AnisotropicMaterialParams(10,20,200),dx=1/16,device=device,gravity=0,stabilization='supplemental',direction_model=model)
                G=np.diag([.08,-.02,-.01]);s.seed_particles(points,density=1,vol0=.25**3/len(points),velocity=(points-.5)@G.T,velocity_gradient=G);s.set_dt(.001)
                times=[];iterations=[]
                for step in range(7):
                    guard();wp.synchronize_device(device);start=time.perf_counter()
                    if not s.step(max_iters=16,print_every=0,newton_atol=1e-8,cg_tol=1e-3,cg_atol=1e-10):raise RuntimeError(s.last_step_stats)
                    wp.synchronize_device(device);elapsed=time.perf_counter()-start
                    if step>=2:times.append(elapsed*1000);iterations.append(s.last_step_stats['linear_iterations'])
                Fp=s.ptc_F.numpy();Xp=s.ptc_x.numpy()
                endpoint=args.out/f'performance-s{samples}-{kind}-{model}.npz';np.savez_compressed(endpoint,positions=Xp,F=Fp)
                from warp._src.types import type_size_in_bytes
                arrays={id(a):a for a in vars(s).values() if isinstance(a,wp.array)}
                if s.enhancements is not None:arrays.update({id(a):a for a in vars(s.enhancements).values() if isinstance(a,wp.array)})
                owned=sum(a.size*type_size_in_bytes(a.dtype) for a in arrays.values())
                r=dict(samples_per_axis=samples,particles=s.n_ptc,quadrature=kind,direction_model=model,stabilization='supplemental',duration=s.sim_time,
                    physical_volume=.25**3,mass=float(s.ptc_m.numpy().sum()),median_step_ms=float(np.median(times)),step_ms=times,linear_iterations=iterations,
                    owned_array_bytes=owned,enhancement_bytes=s.enhancements.memory_bytes(),min_J=float(np.linalg.det(Fp).min()))
                records.append(r);save('performance',records);print('PERFORMANCE',json.dumps(r),flush=True);del s,arrays;gc.collect()
        for r in records:
            ref=np.load(args.out/f"performance-s{r['samples_per_axis']}-particle-mean_tensor.npz")
            cur=np.load(args.out/f"performance-s{r['samples_per_axis']}-{r['quadrature']}-{r['direction_model']}.npz")
            r['position_rms_vs_particle']=float(np.sqrt(np.mean((cur['positions']-ref['positions'])**2)))
            r['F_relative_vs_particle']=float(np.linalg.norm(cur['F']-ref['F'])/np.linalg.norm(ref['F']))
            r['strain_relative_vs_particle']=float(np.linalg.norm(cur['F']-ref['F'])/max(np.linalg.norm(ref['F']-np.eye(3)),1e-15))
            samples=r['samples_per_axis'];axis=.375+(np.arange(4*samples)+.5)*.25/(4*samples)
            X=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
            r['displacement_relative_vs_particle']=float(np.linalg.norm(cur['positions']-ref['positions'])/max(np.linalg.norm(ref['positions']-X),1e-15))
            ref.close();cur.close()
        save('performance',records)


if __name__=='__main__':main()
