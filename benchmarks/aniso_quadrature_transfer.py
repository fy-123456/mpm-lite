"""Production-transfer gate before promoting aligned MLS material quadrature."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import warp as wp
from engine.aniso_phase1 import AnisotropicLiteImplicitSolver,AnisotropicMaterialParams,select_lowest_memory_device,query_gpu_memory
from engine.aniso_phase1.aligned_quadrature import box_rule,evaluate
from engine.aniso_phase1.transfer_compatibility import transfer,history_read,mass_audit
from engine.aniso_phase1.stabilization_probe import node_coordinates
from engine.aniso_phase1.trace_probe import lite_values
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.particle_quadrature import particle_stencil
from engine.types import real,vec3
from benchmarks.aniso_quadrature_validation import setup,save
from benchmarks.aniso_material_snapshot import make_snapshot
from benchmarks.aniso_rotation_dissipation import guard,table


def run(grid,device,out,ppc=2):
    guard();_,nodes,blend=setup(grid);s0=make_snapshot(grid=grid,ppc_axis=ppc);x=s0.X
    identity=np.tile(np.eye(3),(len(x),1,1));h=1/(grid-1);params=AnisotropicMaterialParams(10,20,200)
    mass=mass_audit(blend,x,s0.volume,grid)
    s=AnisotropicLiteImplicitSolver((grid,)*3,params,dx=h,device=device,gravity=0)
    s.seed_particles(x,density=1.,vol0=float(s0.volume[0]));s.set_dt(.001);s.step(max_iters=0,print_every=0)
    actual=node_coordinates(s)*h;lookup={tuple(np.rint(p/h).astype(int)):i for i,p in enumerate(nodes)}
    perm=np.array([lookup[tuple(np.rint(p/h).astype(int))] for p in actual])
    if len(perm)!=len(nodes):raise ValueError('sparse node set differs')
    N,G=evaluate(blend,x);Lite=lite_values(nodes,x,h)
    ids=wp.empty((len(x),27),dtype=int,device=device);grad=wp.empty((len(x),27),dtype=vec3,device=device)
    wp.launch(particle_stencil,dim=len(x),inputs=[s.ptc_x,s.block2bid,s.node2dof,ids,grad,h,s.grid_size],device=device)
    ids_host=ids.numpy();grad_host=grad.numpy()
    rule=box_rule(h,3);Nq,Gq=evaluate(blend,rule.points)
    path=Path('docs/results/quadrature-validation')/('aligned-v1' if grid==17 else 'aligned-v1-fine')
    x0,y0,z0=(nodes-np.array([.25,.5,.5])).T
    fields=dict(affine=np.column_stack([x0,0*x0,0*x0]),quadratic_bend=np.column_stack([-2*x0*y0,x0*x0,0*x0]))
    with np.load(path/f'g{grid}-gauss3.npz') as a:fields['solved_mls']=a['node_displacement']
    with np.load(path/f'g{grid}-gauss3-mode.npz') as a:fields['weakest_mls']=a['node_displacement']
    records=[]
    for name,v in fields.items():
        v=.05*v/max(np.linalg.norm(v,axis=1).max(),1e-20)
        Gmls=np.stack([g@v for g in G],axis=2);Gsample=np.stack([g@v for g in Gq],axis=2)
        safe=np.maximum(ids_host,0);Gexpected=np.einsum('pna,pnb->pab',v[perm][safe],grad_host*(ids_host>=0)[:,:,None])
        for dt in (.001,.0005):
            guard();after,Fa,Ga,seconds=transfer(s,x,identity,v[perm],dt)
            F_error=float(np.max(np.abs(Fa-(np.eye(3)+dt*Gexpected))))
            if F_error>1e-12:raise AssertionError('actual particle F update is not the audited elastic stencil')
            Fm=np.eye(3)+dt*Gmls;xm=x+dt*(N@v)
            solved=np.eye(3)+dt*Gsample;sample_after=rule.points+dt*(Nq@v)
            U=float(rule.weights@energy_density(solved,np.tile(np.diag([1.,0.,0.]),(len(solved),1,1)),params))
            actual_rebuild,st=history_read(after,x,Fa,s0.volume,sample_after,h)
            matched_rebuild,st_match=history_read(xm,x,Fm,s0.volume,sample_after,h)
            Aq=np.tile(np.diag([1.,0.,0.]),(len(solved),1,1))
            Ua=float(rule.weights@energy_density(actual_rebuild,Aq,params));Um=float(rule.weights@energy_density(matched_rebuild,Aq,params))
            row=dict(grid=grid,ppc_axis=ppc,field=name,dt=dt,particles=len(x),samples=len(rule.weights),
                actual_F_vs_expected=F_error,actual_vs_particle_stencil=float(np.linalg.norm(Ga-Gexpected)/max(np.linalg.norm(Gexpected),1e-20)),
                actual_position_vs_lite=float(np.linalg.norm(after-(x+dt*(Lite@v)))),
                gradient_relative_mismatch=float(np.linalg.norm(Ga-Gmls)/max(np.linalg.norm(Gmls),1e-20)),
                pic_velocity_relative_mismatch=float(np.linalg.norm(Lite@v-N@v)/max(np.linalg.norm(N@v),1e-20)),
                particle_energy_lite=float(s0.volume@energy_density(Fa,s0.A,params)),
                particle_energy_mls=float(s0.volume@energy_density(Fm,s0.A,params)),
                particle_energy_relative=float((s0.volume@energy_density(Fa,s0.A,params))/(s0.volume@energy_density(Fm,s0.A,params))-1),
                sample_solved_energy=U,actual_history_energy=Ua,matched_history_energy=Um,
                actual_history_delta=Ua-U,matched_history_delta=Um-U,
                actual_history_relative=Ua/U-1,matched_history_relative=Um/U-1,
                transfer_seconds=seconds,**st,matched_history_seconds=st_match['history_seconds'],
                min_det=float(np.linalg.det(Fa).min()))
            records.append(row);print('TRANSFER',json.dumps(row),flush=True)
    controls=[r for r in records if r['field'] in ('affine','quadratic_bend')]
    if max(r['actual_vs_particle_stencil'] for r in records)>1e-9:raise AssertionError('actual Lite stencil audit failed')
    if max(r['gradient_relative_mismatch'] for r in controls)>1e-9:raise AssertionError('affine/quadratic positive control failed')
    if max(r['actual_position_vs_lite'] for r in records)>1e-12:raise AssertionError('actual advection audit failed')
    result=dict(grid=grid,ppc_axis=ppc,mass=mass,records=records,decision='direct production promotion rejected: material/transfer/history maps differ; naive MLS mass replacement is unsafe')
    save(out/f'g{grid}-ppc{ppc}.json',result);table(out/f'g{grid}-ppc{ppc}.csv',records)
    del s;gc.collect();return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grids',nargs='+',type=int,default=[17,33]);p.add_argument('--ppc',nargs='+',type=int,default=[2,4]);p.add_argument('--device',default='auto')
    p.add_argument('--out',type=Path,default=Path('docs/results/quadrature-validation/transfer-v1'));args=p.parse_args()
    guard();wp.init();device=select_lowest_memory_device(args.device);args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'environment.json',dict(device=device,gpus=query_gpu_memory(),production_defaults_changed=False))
    results=[]
    for grid in args.grids:
        for ppc in args.ppc:
            results.append(run(grid,device,args.out,ppc));save(args.out/'results.json',results)


if __name__=='__main__':main()
