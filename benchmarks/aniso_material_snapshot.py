"""Frozen material-quadrature comparisons; never advances a physical time step."""
import argparse,csv,gc,json,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,compare_snapshot
from engine.aniso_phase1.stabilization_probe import bent_beam_state
from engine.aniso_phase1.rotation_probe import axis_rotation
from engine.aniso_phase1 import AnisotropicMaterialParams,select_lowest_memory_device,query_gpu_memory
from benchmarks.aniso_rotation_dissipation import guard,table
from demos.aniso import DATA_ROOT
from utils.resource_guard import inspect_storage


def make_snapshot(grid=17,ppc_axis=2,kind='bend',field='uniform',angle=0.,shift=(0.,0.,0.)):
    lo=np.array([.25,.4375,.4375]);hi=np.array([.75,.5625,.5625]);counts=np.rint((hi-lo)*(grid-1)*ppc_axis).astype(int)
    axes=[lo[d]+(np.arange(counts[d])+.5)*(hi[d]-lo[d])/counts[d] for d in range(3)]
    X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    if kind=='bend':x,F=bent_beam_state(X,amplitude=.01)
    else:
        F0=np.diag([1.04,.99,1.]) if kind=='tension' else np.array([[1,.04,0],[0,1,0],[0,0,1.]])
        F=np.broadcast_to(F0,(len(X),3,3)).copy();x=(X-.5)@F0.T+.5
    angles=np.zeros(len(X))
    if field=='crossed':angles=np.where(np.arange(len(X))%2,np.pi/2,0.)
    if field=='smooth':angles=(X[:,1]-lo[1])/(hi[1]-lo[1])*np.pi/2
    a=np.stack((np.cos(angles),np.sin(angles),np.zeros_like(angles)),axis=1)
    Q=axis_rotation(np.deg2rad(angle));x=(x-.5)@Q.T+.5+shift;F=Q@F
    return MaterialSnapshot(x,X,F,np.einsum('pi,pj->pij',a,a),np.full(len(X),np.prod(hi-lo)/len(X)))


def analytic_bend_F(points,angle=0.,shift=(0.,0.,0.)):
    """Exact inverse of the manufactured bend, used only to isolate fit error."""
    Q=axis_rotation(np.deg2rad(angle));bent=(points-.5-shift)@Q+.5
    a=.01;xb=bent[:,0]-.25;yb=bent[:,1]-.5;s=xb.copy()
    for _ in range(10):s-=(s*(1+8*a*yb+32*a*a*s*s)-xb)/(1+8*a*yb+96*a*a*s*s)
    X=bent.copy();X[:,0]=.25+s;X[:,1]=.5+yb+4*a*s*s
    restored,F=bent_beam_state(X,amplitude=a)
    if np.max(np.abs(restored-bent))>1e-10:raise ValueError('analytic bend inverse did not converge')
    return Q@F


def production_check(snapshot,grid,params,device,reference):
    """Compare the host center reading with actual production GPU P2C/M4.

    max_iters=0 prepares history but does not advance/commit the particles.
    No diagnostic-only reconstructed quadrature is installed in the solver.
    """
    import warp as wp
    from engine.types import mat33
    from engine.aniso_phase1 import AnisotropicLiteImplicitSolver
    from engine.aniso_phase1.diagnostics import center_snapshot
    from engine.sp_grid import B
    s=AnisotropicLiteImplicitSolver((grid,)*3,params,dx=1/(grid-1),device=device,gravity=0,stabilization='none',direction_model='fourth_moment')
    s.seed_particles(snapshot.x,density=1,vol0=float(snapshot.volume[0]),deformation_gradient=snapshot.F,reference_positions=snapshot.X)
    s.ptc_A0.assign(wp.array(snapshot.A.copy(),dtype=mat33,device=device));s.set_dt(.001)
    s.step(max_iters=0,print_every=0)
    keys,V,psi,active=center_snapshot(s);E=float(V@psi)
    from engine.aniso_phase1.material_snapshot import response
    F=s.aniso_committed_F[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active]
    A=s.aniso_A0[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active]
    M=s.enhancements.M.numpy();_,P,tau=response(F,A,M,params)
    target=reference['methods']['center'];Ptotal=np.einsum('p,pij->ij',V,P)
    result=dict(device=device,active_centers=len(V),energy_relative_error=abs(E/target['energy']-1),
        integrated_P_relative_error=float(np.linalg.norm(Ptotal-np.array(target['integrated_P']))/max(np.linalg.norm(target['integrated_P']),1e-20)),
        particle_x_unchanged=bool(np.array_equal(s.ptc_x.numpy(),snapshot.x)),particle_F_unchanged=bool(np.array_equal(s.ptc_F.numpy(),snapshot.F)),physical_time=float(s.sim_time))
    del s;gc.collect();return result


def plot(root,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=('center','grid8','moment8','reconstructed_particles');colors=('#9855aa','#d67d25','#287bbc','#16936b')
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for grid,ax in zip((17,33),axes[0]):
        subset=[r for r in records if r['kind']=='bend' and r['field']=='uniform' and r['grid']==grid and r['ppc_axis']==2 and r['shift']==[0.,0.,0.]]
        subset.sort(key=lambda r:r['angle'])
        for method,color in zip(names,colors):ax.plot([r['angle'] for r in subset],[100*r['methods'][method]['energy_relative_error'] for r in subset],'-o',label=method,color=color)
        ax.set(title=f'Fixed bent snapshots, grid {grid}',xlabel='Rigid rotation (degrees)',ylabel='Material energy error vs particles (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    subset=[r for r in records if r['kind']=='bend' and r['grid']==17 and r['angle']==0 and r['shift']==[0.,0.,0.] and r['ppc_axis']==2]
    for ax,key,title in ((axes[1,0],'energy_relative_error','Material energy error (%)'),(axes[1,1],'center_P_rms_relative_error','Center stress RMS error (%)')):
        for i,(method,color) in enumerate(zip(names,colors)):
            ax.bar(np.arange(len(subset))+(i-1.5)*.18,[100*r['methods'][method][key] for r in subset],width=.18,label=method,color=color)
        ax.set_xticks(np.arange(len(subset)),[r['field'] for r in subset]);ax.set(title=title,xlabel='Reference fiber field');ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'summary.png',dpi=160);fig.savefig(root/'summary.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('docs/results/material-snapshot'))
    p.add_argument('--device',default='auto');p.add_argument('--production-check',action='store_true')
    args=p.parse_args();guard();args.out.mkdir(parents=True,exist_ok=True);params=AnisotropicMaterialParams(10,20,200)
    env=dict(storage=vars(inspect_storage(DATA_ROOT)),diagnostics_device='cpu',production_solver_changed=False)
    if args.production_check:env.update(production_device=select_lowest_memory_device(args.device),gpus=query_gpu_memory())
    (args.out/('environment-production.json' if args.production_check else 'environment.json')).write_text(json.dumps(env,indent=2));cases=[]
    for kind,field in (('tension','uniform'),('tension','crossed'),('shear','uniform')):cases.append((17,2,kind,field,0.,(0.,0.,0.)))
    for grid in (17,33):
        for angle,shift in ((0.,(0.,0.,0.)),(45.,(0.,0.,0.)),(90.,(0.,0.,0.)),(0.,(.037,.013,0.))):cases.append((grid,2,'bend','uniform',angle,shift))
    for field in ('crossed','smooth'):
        for angle in (0.,45.):cases.append((17,2,'bend',field,angle,(0.,0.,0.)))
    cases.append((17,4,'bend','uniform',0.,(0.,0.,0.)))
    records=[];checks=[];flat=[]
    for grid,ppc,kind,field,angle,shift in cases:
        guard();snapshot=make_snapshot(grid,ppc,kind,field,angle,shift)
        name=f'{kind}-{field}-g{grid}-ppc{ppc}-r{angle:g}'+('-shift' if any(shift) else '')
        oracle=(lambda points:analytic_bend_F(points,angle,shift)) if kind=='bend' and field=='uniform' else None
        snapshot.save(args.out/(name+'.npz'));start=time.perf_counter();result,details=compare_snapshot(snapshot,1/(grid-1),params,oracle)
        result.update(name=name,grid=grid,ppc_axis=ppc,kind=kind,field=field,angle=angle,shift=list(shift),diagnostic_seconds=time.perf_counter()-start)
        table(args.out/(name+'-centers.csv'),details);records.append(result)
        for method,r in result['methods'].items():flat.append(dict(case=name,method=method,energy=r['energy'],energy_relative_error=r['energy_relative_error'],center_P_rms_relative_error=r['center_P_rms_relative_error'],center_tau_rms_relative_error=r['center_tau_rms_relative_error']))
        (args.out/'snapshots.json').write_text(json.dumps(records,indent=2,allow_nan=False));table(args.out/'summary.csv',flat)
        print(name,{k:round(100*v['energy_relative_error'],5) for k,v in result['methods'].items()},flush=True)
        if args.production_check and grid==17 and ppc==2 and ((kind=='tension' and field=='crossed') or (kind=='bend' and field=='uniform' and angle==45)):
            guard();check=production_check(snapshot,grid,params,env['production_device'],result);check['name']=name;checks.append(check)
            (args.out/'production-check.json').write_text(json.dumps(checks,indent=2,allow_nan=False))
    plot(args.out,records)


if __name__=='__main__':main()
