"""Finite-strain and cost audit of frozen physical-domain sampling (no transfer)."""
import argparse,gc,json,os,subprocess,time
from pathlib import Path
import numpy as np
import warp as wp
from scipy.linalg import null_space
from engine.aniso_phase1 import select_lowest_memory_device,query_gpu_memory,AnisotropicMaterialParams
from engine.aniso_phase1.aligned_quadrature import box_rule,Rule,evaluate
from engine.aniso_phase1.quadrature_probe import FrozenMaterial
from engine.aniso_phase1.trace_probe import face_points
from benchmarks.aniso_quadrature_validation import setup,save,fingerprint
from benchmarks.aniso_material_snapshot import make_snapshot
from benchmarks.aniso_rotation_dissipation import guard,table


def actual_order(grid,nodes,device):
    from engine.aniso_phase1.grouped_quadrature import GroupedQuadratureImplicitSolver
    from engine.aniso_phase1.stabilization_probe import node_coordinates
    snap,_,_=setup(grid)
    s=GroupedQuadratureImplicitSolver((grid,)*3,AnisotropicMaterialParams(10,20,200),dx=1/(grid-1),gravity=0,device=device)
    s.seed_particles(snap.X,density=1,vol0=float(snap.volume[0]));s.set_dt(.0005);s.step(max_iters=0,print_every=0)
    coords=node_coordinates(s);lookup={tuple(np.rint(p*(grid-1)).astype(int)):i for i,p in enumerate(nodes)}
    perm=np.array([lookup[tuple(p)] for p in coords])
    if len(perm)!=len(nodes) or len(np.unique(perm))!=len(nodes):raise ValueError('actual sparse node set changed')
    del s;gc.collect();return perm


def run(grid,device,out):
    guard();snap,nodes,blend=setup(grid);h=blend.h
    perm=actual_order(grid,nodes,device)
    C=evaluate(blend,face_points(grid,.25,6,True))[0].toarray();Z=np.kron(null_space(C,rcond=1e-10),np.eye(3))
    x,y,z=(nodes-np.array([.25,.5,.5])).T
    fields_u=dict(bend=np.column_stack([-.16*x*y,.08*x*x,0*x]).ravel(),
        stretch_shear=np.column_stack([.04*x,.025*x,0*x]).ravel())
    rng=np.random.default_rng(196);p=Z@rng.normal(size=Z.shape[1]);p/=np.linalg.norm(p)
    q=Z@rng.normal(size=Z.shape[1]);q/=np.linalg.norm(q)
    records=[];costs=[]
    for field in ('uniform','crossed','smooth'):
        models={}
        names=['gauss5','gauss4','gauss3']+(['particle2','particle4','particle8'] if field=='uniform' else [])
        for name in names:
            guard()
            if name.startswith('gauss'):rule=box_rule(h,int(name[5:]))
            else:
                s=make_snapshot(grid=grid,ppc_axis=int(name[8:]));rule=Rule(s.X,s.volume,np.full(len(s.X),-1))
            models[name]=FrozenMaterial(blend,rule,device,field,guard=guard,permutation=perm)
            print('FROZEN',grid,field,name,len(rule.weights),flush=True)
        for state,u in fields_u.items():
            Uref,rref=models['gauss5'].state(u);Jref=models['gauss5'].action(p)
            for name,m in models.items():
                U,r=m.state(u);Jp=m.action(p);Jq=m.action(q)
                hostU,hostr=m.host_state(u)
                fd=[]
                for eps in (1e-4,1e-5):
                    ep,rp=m.state(u+eps*p);em,rm=m.state(u-eps*p)
                    fd.append(dict(eps=eps,energy_gradient_relative=abs((ep-em)/(2*eps)-r@p)/max(abs(r@p),1e-12),
                        tangent_relative=float(np.linalg.norm((rp-rm)/(2*eps)-Jp)/np.linalg.norm(Jp))))
                m.state(u)
                angle=np.pi/4;Q=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
                rotated=(nodes+u.reshape(-1,3)-.5)@Q.T+.5-nodes
                rotated_U,rotated_r=m.state(rotated.ravel())
                rotation_force=float(np.linalg.norm(rotated_r.reshape(-1,3)-r.reshape(-1,3)@Q.T)/np.linalg.norm(r))
                m.state(u)
                record=dict(grid=grid,field=field,state=state,rule=name,samples=len(m.rule.weights),energy=U,
                    energy_relative=U/Uref-1,rotation_energy_relative=abs(rotated_U/U-1),rotation_force_relative=rotation_force,free_force_relative=float(np.linalg.norm(Z.T@(r-rref))/np.linalg.norm(Z.T@rref)),
                    action_relative=float(np.linalg.norm(Jp-Jref)/np.linalg.norm(Jref)),
                    symmetry_relative=abs(p@Jq-q@Jp)/max(np.linalg.norm(Jp)*np.linalg.norm(q),1e-20),
                    host_energy_relative=abs(U-hostU)/max(abs(U),1e-20),host_force_relative=float(np.linalg.norm(r-hostr)/np.linalg.norm(r)),
                    min_det=float(np.linalg.det(m.F.numpy()).min()),finite_differences=fd)
                records.append(record);print('FINITE',json.dumps(record),flush=True)
        if field=='uniform':
            # Interleave warm synchronized measurements on the same shared device.
            timings={name:[] for name in models}
            for repeat in range(3):
                for name in (list(models) if repeat%2==0 else list(models)[::-1]):
                    models[name].state(fields_u['bend']);models[name].action(p)
                    timings[name].append(models[name].timings(repeats=1,batch=20))
            for name,m in models.items():
                cost=dict(rule=name,samples=len(m.rule.weights),build_seconds=m.build_seconds,
                    host_map_seconds=m.host_map_seconds,map_bytes=m.map_bytes,device_array_bytes=m.array_bytes,
                    host_gradient_csr_bytes=m.host_csr_bytes,point_rule_hash=fingerprint(m.rule.points,m.rule.weights),timings=timings[name])
                costs.append(cost)
        save(out/'nonlinear.json',records);save(out/'kernel-cost.json',costs)
        del models;gc.collect()
    return records,costs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grid',type=int,default=17);p.add_argument('--device',default='auto')
    p.add_argument('--out',type=Path,default=Path('docs/results/quadrature-validation/nonlinear-v1'));args=p.parse_args()
    guard();wp.init();device=select_lowest_memory_device(args.device);args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'environment.json',dict(device=device,gpus=query_gpu_memory(),mode='frozen total-reference material; no particle transfer',pid=os.getpid()))
    from engine.aniso_phase1.quadrature_probe import MemoryMonitor
    with MemoryMonitor() as memory:
        records,costs=run(args.grid,device,args.out)
    save(args.out/'memory-observed.json',memory.result())
    assert max(r['host_force_relative'] for r in records)<1e-7
    assert max(min(f['tangent_relative'] for f in r['finite_differences']) for r in records)<1e-4
    print('NONLINEAR_AUDIT_OK',flush=True)


if __name__=='__main__':main()
