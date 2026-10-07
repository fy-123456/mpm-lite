"""Fixed group4x8: factorial clamp/load tests, then a separate field change.

All static operators exclude mass and stabilization. Constraints use nullspaces,
not tuned penalty terms. The nonlinear production solver is not reconfigured.
"""
import argparse,gc,json,itertools,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from engine.aniso_phase1.trace_probe import (BlendedMLS,ConstrainedStatic,grouped_maps,stiffness,
    lite_values,face_points,CORNERS)
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,center_support
from engine.aniso_phase1.beam_reference import beam_matrices,reference_hessian
from benchmarks.aniso_material_snapshot import make_snapshot
from benchmarks.aniso_rotation_dissipation import guard,table
from demos.aniso import DATA_ROOT
from utils.resource_guard import inspect_storage


def q1_values(nodes,points,h):
    integer=np.rint(nodes/h).astype(int);lookup={tuple(p):i for i,p in enumerate(integer)}
    low,high=integer.min(0),integer.max(0)-1;rows=[];cols=[];vals=[]
    for q,p in enumerate(points):
        base=np.clip(np.floor(p/h).astype(int),low,high);f=p/h-base
        for c in CORNERS:
            rows.append(q);cols.append(lookup[tuple(base+c)]);vals.append(np.prod(np.where(c,f,1-f)))
    return sparse.coo_matrix((vals,(rows,cols)),shape=(len(points),len(nodes))).tocsr()


def load_vector(N,total=-1e-4):
    f=np.zeros((N.shape[1],3));f[:,1]=total*np.asarray(N.mean(axis=0)).ravel()
    return f.ravel()


def reference(grid,particles):
    nodes,_,free,_,_,K=beam_matrices(grid);h=1/(grid-1);x=nodes*h
    root=face_points(grid,.25,7,True);tip=face_points(grid,.75)
    N=q1_values(x,tip,h);f=load_vector(N);u=np.zeros(len(f));u[free]=spsolve(K[1][free][:,free],f[free])
    R=K[1]@u-f
    tip_u=float(u@f/-1e-4)
    return dict(tip_displacement=tip_u,reaction=R.reshape(-1,3).sum(0).tolist(),
        mean_displacement_target=-5e-4,force_for_mean_displacement=-1e-4*(-5e-4/tip_u),
        effective_stiffness=-1e-4/tip_u,
        strain_energy=float(.5*u@K[1]@u),load='uniform face traction; area-average conjugate displacement'),q1_values(x,particles,h)@u.reshape(-1,3)


def run_grid(grid,out):
    guard();h=1/(grid-1);snap=make_snapshot(grid=grid)
    snap=MaterialSnapshot(snap.X,snap.X,np.tile(np.eye(3),(len(snap.X),1,1)),snap.A,snap.volume)
    nodes=np.unique(np.concatenate([key+CORNERS for key,_,_ in center_support(snap,h)]),axis=0)*h
    x,V,M,G=grouped_maps(snap,nodes,h);blend=BlendedMLS(nodes,h)
    bg=blend.evaluate(x)[1];matrices={'local':stiffness(G,V,M),'global_blend':stiffness(bg,V,M)}
    tip=face_points(grid,.75);root_check=face_points(grid,.25,7,True)
    check=blend.evaluate(root_check)[0];local_check=blend.local_traces(root_check)
    root=face_points(grid,.25,6,True);root_sparse=face_points(grid,.25)
    Cdict=dict(nodes=np.eye(len(nodes))[nodes[:,0]<=.25],
        blended_face=blend.evaluate(root)[0],patch_faces=blend.local_traces(root),sparse_face=blend.evaluate(root_sparse)[0])
    loads=dict(lite=lite_values(nodes,tip,h),mls=blend.evaluate(tip)[0])
    particle_readers=dict(mls=blend.evaluate(snap.x)[0],lite=lite_values(nodes,snap.x,h))
    ref,reference_u=reference(grid,snap.x);records=[];states={}
    # K is identical across every 'local' run. Global-blend changes the field,
    # and hence K, explicitly; sample positions/volumes/directions stay fixed.
    cases=[('local',c) for c in Cdict]+[('global_blend',c) for c in ('blended_face','sparse_face')]
    for field,clamp in cases:
        guard();K=matrices[field];solver=ConstrainedStatic(K,Cdict[clamp])
        for load,L in loads.items():
            if clamp=='sparse_face' and load=='lite':continue
            f=load_vector(L);u,R=solver.solve(f);uv=u.reshape(-1,3)
            name=f'g{grid}-{field}-{clamp}-{load}';states[name]=u
            clamp_error=float(np.max(np.linalg.norm(check@uv,axis=1)));patch_error=float(np.max(np.linalg.norm(local_check@uv,axis=1)))
            reaction=R.reshape(-1,3).sum(0);tip_u=float(u@f/-1e-4);energy=float(.5*u@K@u)
            constraint=float(np.max(np.abs(solver.C@uv)));free_res=float(np.linalg.norm(solver.Z.T@R)/np.linalg.norm(f))
            record=dict(name=name,grid=grid,field=field,clamp=clamp,load=load,tip_displacement=tip_u,
                mean_displacement_target=-5e-4,force_for_mean_displacement=-1e-4*(-5e-4/tip_u),
                reaction_y_for_mean_displacement=float(reaction[1]*(-5e-4/tip_u)),effective_stiffness=-1e-4/tip_u,
                relative_to_reference=float(tip_u/ref['tip_displacement']-1),
                tip_mls=float(np.mean((loads['mls']@uv)[:,1])),tip_lite=float(np.mean((loads['lite']@uv)[:,1])),
                root_mls_max=clamp_error,root_local_patch_max=patch_error,constrained_values_max=constraint,
                reaction_x=float(reaction[0]),reaction_y=float(reaction[1]),reaction_z=float(reaction[2]),
                force_balance_relative=float(np.linalg.norm(reaction+f.reshape(-1,3).sum(0))/1e-4),
                torque_balance_relative=float(np.linalg.norm(np.cross(nodes,(R+f).reshape(-1,3)).sum(0))/5e-5),
                free_residual_relative=free_res,work_identity_relative=abs(2*energy-f@u)/max(abs(f@u),1e-30),
                energy=energy,constraint_rank=solver.rank,soft_modes=solver.soft,min_eigenvalue=float(solver.eigen[0]),
                symmetry_error=solver.symmetry,nodes=len(nodes),samples=len(V))
            records.append(record)
            np.savez_compressed(out/(name+'.npz'),reference=snap.X,nodes=nodes,node_displacement=uv,
                displacement_mls=particle_readers['mls']@uv,displacement_lite=particle_readers['lite']@uv,
                reference_displacement=reference_u,clamp_points=root_check,clamp_displacement=check@uv,
                tip_points=tip,tip_displacement=loads['mls']@uv)
            print('CASE',json.dumps(record),flush=True)
    # Boundary-density control, not another parameter fit to a reference tip.
    C8=blend.evaluate(face_points(grid,.25,8,True))[0];checksolver=ConstrainedStatic(matrices['local'],C8)
    u8,_=checksolver.solve(load_vector(loads['mls']));u6=states[f'g{grid}-local-blended_face-mls']
    density=dict(rank6=next(r['constraint_rank'] for r in records if r['clamp']=='blended_face'),rank8=checksolver.rank,
        displacement_relative=float(np.linalg.norm(u8-u6)/np.linalg.norm(u6)))
    result=dict(grid=grid,reference=ref,boundary_density=density,records=records,
        volume=float(V.sum()),physical_size=[.5,.125,.125],material=[10,20,200],total_tip_force=-1e-4)
    (out/f'grid{grid}.json').write_text(json.dumps(result,indent=2,allow_nan=False));table(out/f'grid{grid}.csv',records)
    return result


def global_energy_control(out):
    """Read a fixed anomalous global-blend solution with denser template rules.

    This is an integration diagnostic, not a new reference continuum or solve.
    Covariance cubes retain the same means/second moments and positive volumes.
    """
    grid=17;h=1/16;snap=make_snapshot(grid=grid)
    snap=MaterialSnapshot(snap.X,snap.X,np.tile(np.eye(3),(len(snap.X),1,1)),snap.A,snap.volume)
    with np.load(out/'g17-global_blend-blended_face-mls.npz') as a:nodes=a['nodes'];u=a['node_displacement']
    blend=BlendedMLS(nodes,h);H=reference_hessian().reshape(3,3,3,3)
    def energy(points,weights):
        total=0.
        for start in range(0,len(points),256):
            guard();G=blend.evaluate(points[start:start+256])[1];D=np.stack([g@u for g in G],axis=2)
            total+=.5*np.einsum('q,qai,aibj,qbj->',weights[start:start+256],D,H,D)
        return total
    x,V,_,_=grouped_maps(snap,nodes,h);result=dict(group8_energy=energy(x,V),particle_reading=energy(snap.x,snap.volume),refined_templates=[])
    for order in (3,4):
        positions=[];weights=[];z,wg=np.polynomial.legendre.leggauss(order)
        coords=np.array(list(itertools.product(z*np.sqrt(3),repeat=3)));qw=np.prod(np.array(list(itertools.product(wg/2,repeat=3))),axis=1)
        for _,ids,w in center_support(snap,h):
            wn=w/w.sum();mean=wn@snap.x[ids];dx=snap.x[ids]-mean;cov=np.einsum('p,pi,pj->ij',wn,dx,dx)
            e,U=np.linalg.eigh(cov);L=U*np.sqrt(np.maximum(e,0));positions.extend(mean+coords@L.T);weights.extend(w.sum()*qw)
        E=energy(np.array(positions),np.array(weights));result['refined_templates'].append(dict(order=order,samples=len(weights),energy=E,ratio_to_group8=E/result['group8_energy']))
    (out/'global-energy-control.json').write_text(json.dumps(result,indent=2));return result


def gpu_audit(device):
    """Actual frozen sample kernels with nonlocal homogeneous face constraints.

    Inject the candidate frozen B only into this disposable diagnostic solver.
    No production boundary hooks, transfer or default material maps are changed.
    """
    import warp as wp
    from engine.types import vec3,real
    from engine.sp_grid import B
    from engine.aniso_phase1.grouped_quadrature import GroupedQuadratureImplicitSolver
    from engine.aniso_phase1 import AnisotropicMaterialParams
    from engine.aniso_phase1.stabilization_probe import node_coordinates
    from engine.aniso_phase1.operator_probe import assign_velocity
    guard();grid=17;h=1/16;snap=make_snapshot(grid=grid)
    snap=MaterialSnapshot(snap.X,snap.X,np.tile(np.eye(3),(len(snap.X),1,1)),snap.A,snap.volume)
    nodes=np.unique(np.concatenate([k+CORNERS for k,_,_ in center_support(snap,h)]),axis=0)*h
    x,V,M,G=grouped_maps(snap,nodes,h);blend=BlendedMLS(nodes,h);results=[]
    C=blend.evaluate(face_points(grid,.25,6,True))[0].toarray();_,sing,Vh=np.linalg.svd(C,full_matrices=True)
    Z=Vh[int(np.sum(sing>1e-10*sing[0])):].T
    def project(v):return Z@(Z.T@v)
    for mode,g in [('local',G),('global_blend',blend.evaluate(x)[1])]:
        guard();s=GroupedQuadratureImplicitSolver((grid,)*3,AnisotropicMaterialParams(10,20,200),dx=h,gravity=0,device=device)
        s.seed_particles(snap.x,density=1,vol0=float(snap.volume[0]));s.set_dt(.0005);s.step(max_iters=0,print_every=0);s.evaluate_residual()
        actual=node_coordinates(s);lookup={tuple(np.rint(p/h).astype(int)):i for i,p in enumerate(nodes)}
        perm=np.array([lookup[tuple(p)] for p in actual]);inverse=np.argsort(perm);n=len(nodes)
        np.testing.assert_allclose(s.group_V.numpy(),V,atol=1e-16)
        if mode=='global_blend':
            dense=np.stack([gd.toarray() for gd in g],axis=2);patches=[np.flatnonzero(np.any(dense[q]!=0,axis=1)) for q in range(len(V))]
            width=max(map(len,patches));ids=np.full((len(V),width),-1,dtype=np.int32);grad=np.zeros((len(V),width,3))
            for q,p in enumerate(patches):ids[q,:len(p)]=inverse[p];grad[q,:len(p)]=dense[q,p]
            s.group_ids=wp.array(ids,dtype=int,device=device);s.group_B=wp.array(grad,dtype=vec3,device=device)
        nd=s.ndof2bijk[:n].numpy();mass=s.grid_m[:int(s.bcn)].numpy();m=np.array([mass[b,l//B**2,(l//B)%B,l%B] for b,l in nd])[inverse]
        def residual(v):
            wp.launch(assign_velocity,dim=n,inputs=[s.ndof2bijk,wp.array(v[perm],dtype=vec3,device=device),s.grid_v_it],device=device)
            if not s.evaluate_residual():raise ValueError('invalid probe')
            return project(s.node_residual[:n].numpy()[inverse])
        rng=np.random.default_rng(109);v=project(rng.normal(size=(n,3))*.02);d=project(rng.normal(size=(n,3)));d/=np.linalg.norm(d)
        r=residual(v);direction=wp.zeros_like(s.node_residual);out=wp.zeros_like(direction)
        wp.copy(direction,wp.array(d[perm],dtype=vec3,device=device),count=n);s.apply_tangent(direction,out);Jd=project(out[:n].numpy()[inverse])
        eps=1e-5;rp=residual(v+eps*d);ep=s.incremental_potential();rm=residual(v-eps*d);em=s.incremental_potential()
        residual(np.zeros_like(v));s.apply_tangent(direction,out)
        measured=project((out[:n].numpy()[inverse]-m[:,None]*d)/s.dt**2);target=project((stiffness(g,V,M)@d.ravel()).reshape(n,3))
        results.append(dict(mode=mode,device=device,constraint_direction_error=float(np.max(abs(C@d))),
            potential_fd=abs((ep-em)/(2*eps)-np.sum(r*d))/max(abs(np.sum(r*d)),1e-20),
            tangent_fd=float(np.linalg.norm((rp-rm)/(2*eps)-Jd)/np.linalg.norm(Jd)),
            static_matvec_relative=float(np.linalg.norm(measured-target)/np.linalg.norm(target))))
        del s;gc.collect()
    return results


def plot(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=['local-nodes-lite','local-nodes-mls','local-blended_face-mls','local-patch_faces-mls','global_blend-blended_face-mls']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for result in results:
        lookup={r['name'].split('-',1)[1]:r for r in result['records']}
        axes[0].plot(range(len(cases)),[100*lookup[c]['relative_to_reference'] for c in cases],'o-',label=f"grid {result['grid']}")
        axes[1].semilogy(range(len(cases)),[max(lookup[c]['root_mls_max'],1e-20) for c in cases],'o-',label=f"grid {result['grid']}")
    labels=['node / Lite','node / MLS','face / MLS','patch faces / MLS','global field / MLS']
    for ax in axes:ax.set_xticks(range(len(cases)),labels,rotation=20,ha='right');ax.grid(alpha=.3);ax.legend()
    axes[0].axhline(0,color='gray',linewidth=.8)
    axes[0].set_ylabel('Tip displacement error vs matched Q1 (%)');axes[1].set_ylabel('Independent clamp-face displacement (m)')
    fig.tight_layout();fig.savefig(out/'summary.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grids',type=int,nargs='+',default=[17,33]);p.add_argument('--device',default='auto');p.add_argument('--gpu-check',action='store_true');p.add_argument('--energy-control',action='store_true');p.add_argument('--out',type=Path,default=Path('docs/results/boundary-consistency'));args=p.parse_args()
    if any(g not in (17,33) for g in args.grids):p.error('supported controlled beam grids: 17, 33')
    guard();args.out.mkdir(parents=True,exist_ok=True);results=[]
    for g in args.grids:
        results.append(run_grid(g,args.out));(args.out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False));gc.collect()
    plot(args.out,results)
    if args.energy_control:
        if 17 not in args.grids:p.error('energy control requires grid 17')
        print('ENERGY_CONTROL',global_energy_control(args.out),flush=True)
    environment=dict(storage=vars(inspect_storage(DATA_ROOT)),static_device='cpu',production_defaults_changed=False)
    if args.gpu_check:
        import warp as wp
        from engine.aniso_phase1 import select_lowest_memory_device,query_gpu_memory
        wp.init();device=select_lowest_memory_device(args.device);environment.update(gpu_device=device,gpus=query_gpu_memory())
        (args.out/'gpu.json').write_text(json.dumps(gpu_audit(device),indent=2,allow_nan=False))
    (args.out/'environment.json').write_text(json.dumps(environment,indent=2))


if __name__=='__main__':main()
