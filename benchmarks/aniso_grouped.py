"""Mass-free grouped stiffness, frozen sparse derivatives and short regressions."""
import argparse,gc,itertools,json,time
from pathlib import Path
from functools import partial
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import spsolve
from engine.types import vec3
from engine.sp_grid import B
from engine.aniso_phase1.grouped_quadrature import GroupedQuadratureImplicitSolver,freeze_material,freeze_maps
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,center_support,moments
from engine.aniso_phase1.joint_sampling import build_rule
from engine.aniso_phase1.quadratic import node_patch
from engine.aniso_phase1.beam_reference import reference_hessian,element_stiffness,beam_matrices
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.stabilization_probe import node_coordinates
from engine.aniso_phase1 import AnisotropicMaterialParams,select_lowest_memory_device,query_gpu_memory
from benchmarks.aniso_material_snapshot import make_snapshot
from benchmarks.aniso_rotation_dissipation import guard,table
from demos.aniso import Scene,Config,DATA_ROOT
from utils.resource_guard import inspect_storage


def static_system(grid=17,field='uniform'):
    """Actual P2C volumes/support and MLS maps; every matrix excludes mass.

    Dense reference samples the same material particles with the same MLS map.
    Weighted Q1 uses identical grid, center volumes, moments and physical load.
    Independent sharp-domain full Q1 is also reported, with a different support.
    """
    h=1/(grid-1);s=make_snapshot(grid=grid,field=field)
    s=MaterialSnapshot(s.X,s.X,np.tile(np.eye(3),(len(s.X),1,1)),s.A,s.volume)
    groups=center_support(s,h);corners=np.array(list(itertools.product((0,1),repeat=3)))
    nodes=np.unique(np.concatenate([key+corners for key,_,_ in groups]),axis=0)
    lookup={tuple(n):i for i,n in enumerate(nodes)};x=nodes*h;tree=cKDTree(x);Hiso=reference_hessian(kf=0)
    rows=[];cols=[];values=[[],[],[]];qr=[];qc=[];qv=[];hv=[];hs=[]
    for key,ids,w in groups:
        center=(key+.5)*h;points,W,A,M=build_rule(s,ids,w,h,'group4x8')
        patch,g,_,_=node_patch(x,center,h,tree,queries=np.r_[points,s.x[ids]])
        _,gh,_,_=node_patch(x,center,h,tree)
        _,Mcenter=moments(s.A[ids],w);a=s.A[ids].reshape(-1,9)
        def stiffness(grad,weight,H):
            # Contract samples first; avoid forming sample x dof x dof tensors.
            return np.einsum('q,qni,qmj,qaibj->namb',weight,grad,grad,np.broadcast_to(H,(len(weight),9,9)).reshape(-1,3,3,3,3),optimize=True).reshape(3*len(patch),-1)
        Ks=[stiffness(g[:len(W)],W,Hiso+800*M),
            stiffness(g[len(W):],w,Hiso+800*np.einsum('pi,pj->pij',a,a)),
            stiffness(gh[1:]-gh[:1],np.full(8,w.sum()/8),Hiso+800*Mcenter)]
        dof=(3*patch[:,None]+np.arange(3)).ravel();rows.extend(np.repeat(dof,len(dof)));cols.extend(np.tile(dof,len(dof)))
        for v,K in zip(values,Ks):v.extend(K.ravel())
        qids=np.array([lookup[tuple(n)] for n in key+corners]);dof=(3*qids[:,None]+np.arange(3)).ravel()
        qr.extend(np.repeat(dof,24));qc.extend(np.tile(dof,24))
        H=Hiso+800*Mcenter;full=element_stiffness(h,H,True)*w.sum()/h**3
        qv.extend(full.ravel());hs.extend((full-element_stiffness(h,H,False)*w.sum()/h**3).ravel())
        # Old hourglass mode uses 2 mu ||D||^2 rather than elastic sym(D).
        HG=element_stiffness(h,20*np.eye(9),True)-element_stiffness(h,20*np.eye(9),False)
        hv.extend((HG*w.sum()/h**3).ravel())
    size=(len(x)*3,)*2
    Ks=[coo_matrix((v,(rows,cols)),shape=size).tocsr() for v in values]
    for v in (qv,hs,hv):Ks.append(coo_matrix((v,(qr,qc)),shape=size).tocsr())
    free=np.flatnonzero(np.repeat(x[:,0]>.25+1e-9,3));force=np.zeros(len(x)*3)
    # Work-conjugate force/displacement at material tip, using actual Lite
    # P2C+C2G interpolation. Loading low-volume halo nodes directly is unfair.
    tippts=s.X[s.X[:,0]==s.X[:,0].max()].copy();tippts[:,0]=.75
    for point in tippts:
        z=point/h-.5;base=np.floor(z).astype(int);frac=z-base
        for c in corners:
            wt=np.prod(np.where(c,frac,1-frac))/8/len(tippts)
            for nc in corners:force[3*lookup[tuple(base+c+nc)]+1]-=1e-4*wt
    return s,nodes,free,force,Ks


def static_audit(grid=17,field='uniform',device=None):
    guard();s,nodes,free,force,K=static_system(grid,field);records=[]
    variants=[('group_none',K[0]),('dense_particles_same_MLS',K[1]),('weighted_full_Q1',K[3])]
    for name,stabilizer in zip(('quadratic','supplemental','hourglass'),(K[2],K[4],K[5])):
        for eta in (.1,1.):variants.append((f'group_{name}_{eta}',K[0]+eta*stabilizer))
    for name,A in variants:
        reduced=A[free][:,free];e=eigvalsh(reduced.toarray());threshold=1e-9*max(1,e[-1]);soft=int(sum(e<threshold))
        u=np.zeros(len(force));u[free]=spsolve(reduced,force[free])
        records.append(dict(name=name,soft_modes=soft,min_eigenvalue=float(e[0]),max_eigenvalue=float(e[-1]),
            tip_displacement=float(u@force/-1e-4),strain_energy=float(.5*u@A@u)))
    result=dict(grid=grid,field=field,particles=len(s.x),nodes=len(nodes),free_dofs=len(free),results=records,
                reference_scope='P2C cloud, same nodes and Lite work-conjugate tip load; no mass')
    if field=='uniform':
        _,_,ff,tip,f,mat=beam_matrices(grid);u=np.zeros(len(f));u[ff]=spsolve(mat[1][ff][:,ff],f[ff])
        result['independent_sharp_Q1_tip']=float(u[3*tip+1].mean())
    if device is not None:
        result['production']=[]
        for mode in ('none','quadratic'):
            sol=GroupedQuadratureImplicitSolver((grid,)*3,AnisotropicMaterialParams(10,20,200),dx=1/(grid-1),gravity=0,device=device,stabilization=mode)
            sol.seed_particles(s.x,density=1,vol0=float(s.volume[0]));from engine.types import mat33
            sol.ptc_A0.assign(wp.array(s.A.copy(),dtype=mat33,device=device))
            fixed=nodes[nodes[:,0]/(grid-1)<=.25+1e-9];sol.paint_boundary(fixed,np.ones(len(fixed),dtype=np.int32))
            sol.set_dt(.005);sol.step(max_iters=0,print_every=0);sol.evaluate_residual()
            actual=node_coordinates(sol);lookup={tuple(n):i for i,n in enumerate(nodes)};perm=np.array([lookup[tuple(n)] for n in actual])
            p=np.random.default_rng(63).normal(size=(len(nodes),3));p[nodes[:,0]/(grid-1)<=.25+1e-9]=0
            d=wp.zeros_like(sol.node_residual);out=wp.zeros_like(d);wp.copy(d,wp.array(p[perm],dtype=vec3,device=device),count=len(actual));sol.apply_tangent(d,out)
            nd=sol.ndof2bijk[:len(actual)].numpy();mass=sol.grid_m[:int(sol.bcn)].numpy()
            m=np.array([mass[b,l//B**2,(l//B)%B,l%B] for b,l in nd]);measured=(out[:len(actual)].numpy()-m[:,None]*p[perm])/sol.dt**2
            target=((K[0]+(K[2] if mode=='quadratic' else 0))@p.ravel()).reshape(-1,3)[perm];target[actual[:,0]/(grid-1)<=.25+1e-9]=0
            result['production'].append(dict(mode=mode,matvec_relative_error=float(np.linalg.norm(measured-target)/np.linalg.norm(target))))
            del sol;gc.collect()
    return result


def operator_audit(device,mode='none'):
    guard();p=SparseProbe(device=device,dt=.005,boundary=True,solver_cls=partial(GroupedQuadratureImplicitSolver,stabilization=mode))
    # Nonuniform directions and prestrain exercise the full conditional M4,
    # including the terms that vanish for a pure single fiber at F=I.
    from engine.types import mat33
    F=np.array([[1.08,.06,0],[0,.96,.02],[.01,0,1.01]])
    x=p.s.ptc_x.numpy();a=np.linspace(0,1.4,len(x));directions=np.stack((np.cos(a),np.sin(a),np.zeros_like(a)),axis=1)
    p.s.ptc_A0.assign(wp.array(np.einsum('pi,pj->pij',directions,directions),dtype=mat33,device=device))
    p.s.ptc_F.assign(wp.array(np.tile(F,(len(x),1,1)),dtype=mat33,device=device))
    p.s.ptc_reference_x.assign(wp.array((x-.5)@np.linalg.inv(F).T+.5,dtype=vec3,device=device))
    p.s.step(max_iters=0,print_every=0)
    rng=np.random.default_rng(29);coordinates=node_coordinates(p.s)
    order=np.lexsort((coordinates[:,2],coordinates[:,1],coordinates[:,0]))
    v=np.empty((p.n,3));v[order]=rng.normal(size=v.shape)*.15;v=p.project(v)
    d=np.empty_like(v);d[order]=rng.normal(size=d.shape);d=p.project(d);d/=np.linalg.norm(d)
    r=p.residual(v);Jd=p.tangent(d);eps=1e-5
    rp=p.residual(v+eps*d);ep=p.s.incremental_potential();rm=p.residual(v-eps*d);em=p.s.incremental_potential()
    gradient=abs((ep-em)/(2*eps)-np.sum(r*d))/max(abs(np.sum(r*d)),1e-20)
    tangent=float(np.linalg.norm((rp-rm)/(2*eps)-Jd)/np.linalg.norm(Jd));p.residual(v)
    Q=np.column_stack([p.project(e.reshape(p.n,3)).ravel() for e in np.eye(3*p.n)])
    e,Z=np.linalg.eigh((Q+Q.T)/2);Z=Z[:,e>.5];spectra={}
    for pd,name in ((False,'exact'),(True,'projected')):
        J=Z.T@np.column_stack([p.tangent(d.reshape(p.n,3),pd).ravel() for d in Z.T])
        spectra[name]=dict(symmetry=float(np.linalg.norm(J-J.T)/np.linalg.norm(J)),min_eigenvalue=float(np.linalg.eigvalsh((J+J.T)/2)[0]))
    from engine.aniso_phase1.rotation_probe import axis_rotation
    s=p.s;E=s.group_material_energy();force=s.group_internal_force();R=axis_rotation(.7);nodes=node_coordinates(s)*s.dx
    rotated=((nodes+s.dt*v-.5)@R.T+.5-nodes)/s.dt
    p.residual(rotated);Er=s.group_material_energy();fr=s.group_internal_force()
    objective=dict(energy_error=abs(Er/E-1),force_error=float(np.linalg.norm(fr-force@R.T)/np.linalg.norm(force)))
    # A deliberately compressed snapshot is a solver-selection control, not a
    # long physical experiment. The exact *dynamic* tangent may be indefinite.
    F=np.diag([.8,1.,1.]);s.ptc_F.assign(wp.array(np.tile(F,(len(x),1,1)),dtype=mat33,device=device))
    s.ptc_reference_x.assign(wp.array((x-.5)@np.linalg.inv(F).T+.5,dtype=vec3,device=device))
    s.step(max_iters=0,print_every=0);p.residual(np.zeros_like(v));compressed={};exact_vector=None
    # Sparse compaction can reorder dofs even when positions are unchanged.
    # Rebuild the free basis after rebuilding the grid, not just the material.
    Q=np.column_stack([p.project(e.reshape(p.n,3)).ravel() for e in np.eye(3*p.n)])
    e,Z=np.linalg.eigh((Q+Q.T)/2);Z=Z[:,e>.5]
    for pd,name in ((False,'exact'),(True,'projected')):
        J=Z.T@np.column_stack([p.tangent(d.reshape(p.n,3),pd).ravel() for d in Z.T])
        eig,vectors=np.linalg.eigh((J+J.T)/2);compressed[name+'_min_eigenvalue']=float(eig[0])
        if not pd:exact_vector=Z@vectors[:,0]
    from engine.aniso_phase1.linear import guarded_pcg
    b=wp.zeros_like(s.node_residual);solution=wp.zeros_like(b)
    wp.copy(b,wp.array(exact_vector.reshape(p.n,3),dtype=vec3,device=device),count=p.n)
    def identity(a,b):wp.copy(b,a)
    for pd,name in ((False,'exact'),(True,'projected')):
        result=guarded_pcg(lambda a,b:s.apply_tangent(a,b,pd),b,solution,identity,1e-7,1e-12,200)
        compressed[name+'_pcg_status']=result[3]
    return dict(mode=mode,blocks=int(s.bcn),samples=len(s.group_V),free_dofs=Z.shape[1],potential_fd=float(gradient),tangent_fd=tangent,spectra=spectra,material_objectivity=objective,compressed=compressed)


def scene_run(device,kind,mode,out,steps=None):
    config=Config(kind,17 if kind=='beam' else 9,.0005 if kind=='beam' else .005,quadrature='group4x8',stabilization=mode,
        direction_model='fourth_moment',fiber_field='uniform' if kind=='beam' else 'smooth',loading_time=.04,loading_speed=.025,smooth_loading=True)
    scene=Scene(config,device);times=[];frames=[scene.solver.ptc_x.numpy()];iterations=0
    for _ in range(steps or (8 if kind=='beam' else 16)):
        guard();start=time.perf_counter()
        if not scene.step():raise RuntimeError(scene.solver.last_step_stats)
        times.append(time.perf_counter()-start);iterations+=scene.solver.last_step_stats['linear_iterations'];frames.append(scene.solver.ptc_x.numpy())
    s=scene.solver;rows=s.energy_ledger.rows;name=f'{kind}-{mode}'
    table(out/(name+'.csv'),rows);np.savez_compressed(out/(name+'.npz'),positions=frames,reference=scene.reference)
    scale=max(rows[0]['mechanical'],max(r['mechanical'] for r in rows),1e-20)
    return dict(name=name,steps=len(times),min_J=float(np.linalg.det(s.ptc_F.numpy()).min()),physical_time=s.sim_time,
        relative_mechanical_change=rows[-1]['mechanical']/rows[0]['mechanical']-1 if rows[0]['mechanical']>1e-20 else None,
        max_rebuild_jump_fraction=max(abs(r.get('group_rebuild_delta',0)) for r in rows)/scale,
        sum_rebuild_fraction=sum(r.get('group_rebuild_delta',0) for r in rows)/scale,
        max_budget_closure=max(abs(r.get('budget_closure',0)) for r in rows),linear_iterations=iterations,
        median_step_ms=1000*float(np.median(times[2:])),memory_bytes=s._aniso_memory_bytes(),
        peak_reaction=max(abs(r.get('right_force',0)) for r in rows),samples=len(s.group_V),
        initial_stabilization=rows[0].get('stabilization_energy',0),initial_material=rows[0]['elastic']-rows[0].get('stabilization_energy',0))


def rebuild_audit():
    """Exact prescribed translation/rotation; no velocity or F integration."""
    from engine.aniso_phase1.rotation_probe import axis_rotation
    params=AnisotropicMaterialParams(10,20,200);results=[]
    for grid in (17,33):
        for motion in ('translate','rotate'):
            energies=[];counts=[]
            original=make_snapshot(grid=grid,field='smooth')
            for t in np.linspace(0,1,5):
                guard();Q=axis_rotation(t*np.pi/4) if motion=='rotate' else np.eye(3)
                shift=np.array([t*2/(grid-1),0,0]) if motion=='translate' else np.zeros(3)
                snap=MaterialSnapshot((original.x-.5)@Q.T+.5+shift,original.X,Q@original.F,original.A,original.volume)
                records,E=freeze_material(snap,1/(grid-1),params);energies.append(E);counts.append(sum(len(r[2]) for r in records))
            results.append(dict(grid=grid,motion=motion,energies=energies,samples=counts,
                max_relative_change=float(np.max(np.abs(np.array(energies)/energies[0]-1))),
                max_relative_step_jump=float(np.max(abs(np.diff(energies)))/energies[0])))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--part',choices=('static','operators','scenes','rebuild','all'),default='all');p.add_argument('--device',default='auto');p.add_argument('--out',type=Path,default=Path('docs/results/grouped-quadrature'));args=p.parse_args()
    guard();args.out.mkdir(parents=True,exist_ok=True)
    def save(name,data):(args.out/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False))
    device=None
    if args.part!='rebuild':wp.init();device=select_lowest_memory_device(args.device)
    save('environment-'+args.part,dict(device=device,storage=vars(inspect_storage(DATA_ROOT)),gpus=query_gpu_memory() if device else []))
    if args.part in ('static','all'):
        rows=[]
        for g,field in ((17,'uniform'),(33,'uniform'),(17,'smooth')):
            rows.append(static_audit(g,field,device));save('static',rows);print('STATIC',json.dumps(rows[-1]),flush=True);gc.collect()
    if args.part in ('operators','all'):
        rows=[]
        for mode in ('none','quadratic'):
            rows.append(operator_audit(device,mode));save('operators',rows);print('OPERATOR',rows[-1],flush=True);gc.collect()
    if args.part in ('scenes','all'):
        rows=[]
        for kind in ('beam','tensile'):
            for mode in ('none','quadratic'):
                rows.append(scene_run(device,kind,mode,args.out));save('scenes',rows);print('SCENE',rows[-1],flush=True);gc.collect()
    if args.part in ('rebuild','all'):save('rebuild',rebuild_audit())


if __name__=='__main__':main()
