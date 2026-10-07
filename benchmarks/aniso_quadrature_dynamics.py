"""Short total-reference implicit dynamics of the validated frozen MLS field.

No particle transfer or regrouping is performed: this isolates sampling in a
nonlinear solve. It is explicitly not a replacement for production MPM tests.
"""
import argparse,time
from pathlib import Path
import numpy as np
import warp as wp
from scipy import sparse
from scipy.linalg import null_space,cho_factor,cho_solve
from scipy.sparse.linalg import LinearOperator,cg
from engine.aniso_phase1 import select_lowest_memory_device,query_gpu_memory
from engine.aniso_phase1.aligned_quadrature import box_rule,face_rule,evaluate,assemble,Rule
from engine.aniso_phase1.quadrature_probe import FrozenMaterial,checked_pcg
from engine.aniso_phase1.trace_probe import face_points
from benchmarks.aniso_quadrature_validation import setup,save
from benchmarks.aniso_rotation_dissipation import guard,table


def run(device,out,steps=12,dt=.005,scenes=('release','load_unload'),rules=('gauss4','gauss3'),time_pair=True):
    setup_start=time.perf_counter()
    snap,nodes,blend=setup(17);n=len(nodes);C=evaluate(blend,face_points(17,.25,6,True))[0].toarray()
    Z=np.kron(null_space(C,rcond=1e-10),np.eye(3));nr=Z.shape[1]
    rule4=box_rule(blend.h,4);mass=sparse.csr_matrix((n,n))
    for i in range(0,len(rule4.weights),512):
        N,_=evaluate(blend,rule4.points[i:i+512]);mass+=N.T@N.multiply(rule4.weights[i:i+512,None])
    # rho=1, same consistent mass integrated independently for every candidate.
    M=Z.T@(sparse.kron(mass,sparse.eye(3))@Z);mass_factor=cho_factor(M)
    K,_=assemble(blend,rule4,guard=guard);A=Z.T@(K@Z);A=(A+A.T)/2
    tip,w=face_rule(blend.h);L=np.asarray(w@evaluate(blend,tip)[0]).ravel()/w.sum()
    load=np.zeros((n,3));load[:,1]=-1e-4*L;load=Z.T@load.ravel()
    initial=cho_solve(cho_factor(A),load)*10
    check=evaluate(blend,face_points(17,.25,7,True))[0];reader=evaluate(blend,snap.X)[0]
    shared_setup_seconds=time.perf_counter()-setup_start;results=[]
    # dt pair has equal physical time. Loading is a short dynamic ramp, not a
    # claim of quasi-static convergence. The explicit status is saved below.
    for scene in scenes:
        for rule_name in rules:
            order=int(rule_name[5:]) if rule_name.startswith('gauss') else None
            for factor in ((1,2) if scene=='release' and time_pair else (1,)):
                guard();case_start=time.perf_counter();step_dt=(4/steps if scene=='slow_cycle' else 8/steps if scene=='slower_cycle' else dt)/factor;count=steps*factor
                if order is not None:rule=box_rule(blend.h,order)
                else:
                    from benchmarks.aniso_material_snapshot import make_snapshot
                    s=make_snapshot(grid=17,ppc_axis=int(rule_name[8:]));rule=Rule(s.X,s.volume,np.full(len(s.X),-1))
                model=FrozenMaterial(blend,rule,device,'uniform' if scene=='release' else 'smooth',guard=guard)
                pre=cho_factor(M/step_dt**2+A);precondition=LinearOperator((nr,nr),matvec=lambda x:cho_solve(pre,x))
                u=initial.copy() if scene=='release' else np.zeros(nr);v=np.zeros(nr);E,g=model.state(Z@u)
                previous=.5*v@M@v+E;initial_energy=previous;rows=[];frames=[reader@(Z@u).reshape(n,3)]
                start_run=time.perf_counter();old_force=np.zeros(nr)
                for step in range(count):
                    guard();t=(step+1)*step_dt
                    external=np.zeros(nr) if scene=='release' else 10*np.sin(np.pi*t/(count*step_dt))**2*load
                    predictor=u+step_dt*v;iterate=u.copy();total_cg=0;line_backtracks=0;fallbacks=0;converged=False
                    def objective(x):
                        energy,force=model.state(Z@x);d=x-predictor
                        return .5*d@M@d/step_dt**2+energy-external@x, M@d/step_dt**2+Z.T@force-external,energy
                    for newton in range(12):
                        phi,r,elastic=objective(iterate);rnorm=np.linalg.norm(r)
                        if rnorm<max(1e-9,1e-6*max(np.linalg.norm(external),np.linalg.norm(load))):converged=True;break
                        direction=None
                        for pd in (False,True):
                            counter=[0]
                            operator=LinearOperator((nr,nr),matvec=lambda p:M@p/step_dt**2+Z.T@model.action(Z@p,pd=pd))
                            d,info,iterations=checked_pcg(operator,-r,precondition,rtol=1e-6,maxiter=200)
                            total_cg+=iterations
                            true=np.linalg.norm(operator@d+r)/rnorm
                            if info==0 and r@d<0 and true<1e-4 and d@(operator@d)>0:direction=d;fallbacks+=int(pd);break
                        if direction is None:raise RuntimeError(f'linear direction failed: {scene}, order {order}, step {step}')
                        alpha=1.;accepted=False
                        for ls in range(16):
                            trial=iterate+alpha*direction;trial_phi,_,_=objective(trial)
                            if np.isfinite(trial_phi) and trial_phi<=phi+1e-4*alpha*(r@direction):accepted=True;break
                            alpha*=.5;line_backtracks+=1
                        if not accepted:
                            detail=dict(scene=scene,order=order,dt=step_dt,step=step,newton=newton,residual=rnorm,phi=phi,trial_phi=trial_phi,slope=float(r@direction),alpha=alpha)
                            save(out/'line-search-failure.json',detail)
                            raise RuntimeError(f'original-potential line search failed: {detail}')
                        iterate=trial
                    if not converged:
                        _,r,elastic=objective(iterate)
                        if np.linalg.norm(r)>max(1e-9,1e-6*max(np.linalg.norm(external),np.linalg.norm(load))):raise RuntimeError(f'Newton failed, residual {np.linalg.norm(r)}')
                    E,_=model.state(Z@iterate);vnew=(iterate-u)/step_dt;kinetic=.5*vnew@M@vnew
                    work=float(external@(iterate-u));trapwork=float(.5*(old_force+external)@(iterate-u))
                    total=E+kinetic;change=total-previous
                    rows.append(dict(step=step+1,time=t,elastic=E,kinetic=float(kinetic),mechanical=float(total),
                        backward_euler_external_work=work,trapezoid_external_work=trapwork,balance_defect=float(change-work),
                        rule_rebuild_delta=0.,history_rebuild_delta=0.,newton_iterations=newton,pcg_iterations=total_cg,
                        line_backtracks=line_backtracks,pd_fallbacks=fallbacks,residual=float(np.linalg.norm(r)),
                        min_det=float(np.linalg.det(model.F.numpy()).min()),
                        clamp_max=float(np.max(np.linalg.norm(check@(Z@iterate).reshape(n,3),axis=1))),
                        tip_displacement=float(load@iterate/-1e-4)))
                    u,v,previous,old_force=iterate,vnew,total,external
                    frames.append(reader@(Z@u).reshape(n,3))
                    if (step+1)%4==0:print("STEP",scene,order,step_dt,step+1,"residual",rows[-1]["residual"],flush=True)
                name=f'{scene}-{rule_name}-dt{step_dt:g}'
                result=dict(name=name,scene=scene,order=order,rule=rule_name,samples=len(rule.weights),dt=step_dt,steps=count,physical_time=count*step_dt,shared_setup_seconds=shared_setup_seconds,
                    model='frozen total-reference MLS; no MPM transfers or history reconstruction',quasistatic_validated=False,
                    initial_energy=float(initial_energy),final_energy=float(previous),nonlinear_absolute_tolerance=1e-9,nonlinear_relative_tolerance=1e-6,
                    whole_case_seconds=time.perf_counter()-case_start,whole_run_seconds=time.perf_counter()-start_run,build_seconds=model.build_seconds,
                    device_array_bytes=model.array_bytes,rows=rows)
                results.append(result);table(out/(name+'.csv'),rows)
                np.savez_compressed(out/(name+'.npz'),reference=snap.X,displacements=np.asarray(frames),times=np.arange(count+1)*step_dt)
                save(out/'dynamics.json',results);print('DYNAMIC',name,'min_det',min(r['min_det'] for r in rows),'energy',previous,flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='auto');p.add_argument('--steps',type=int,default=12);p.add_argument('--dt',type=float,default=.005)
    p.add_argument('--rules',nargs='+',choices=['gauss4','gauss3','particle4','particle8'],default=['gauss4','gauss3'])
    p.add_argument('--no-time-pair',action='store_true')
    p.add_argument('--scenes',nargs='+',choices=['release','load_unload','slow_cycle','slower_cycle'],default=['release','load_unload'])
    p.add_argument('--out',type=Path,default=Path('docs/results/quadrature-validation/dynamics-v1'));args=p.parse_args()
    if args.steps<1 or not np.isfinite(args.dt) or args.dt<=0:p.error('positive steps and finite positive dt required')
    guard();wp.init();device=select_lowest_memory_device(args.device);args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'environment.json',dict(device=device,gpus=query_gpu_memory(),production_defaults_changed=False))
    run(device,args.out,args.steps,args.dt,args.scenes,args.rules,not args.no_time_pair)


if __name__=='__main__':main()
