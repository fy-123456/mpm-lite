"""Independent CPU reference check and explicit separation of projection error."""
import argparse,time
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from .provenance import *
from .runtime import update
from engine.aniso_phase1.research_coupled_reference_next.reference import affine_operator,exact,interval_average

def main(run):
    run=Path(run);mutable(run);tick=time.perf_counter();z=dict(np.load(run/'S1/linear-blocks.npz'));probes=dict(np.load(run/'S1/probes.npz'))
    if read(run/'S1/derivative-check.json')['status']!='passed_scoped':raise ValueError('unverified derivatives')
    times=z['times'];mf=la.cho_factor(z['M']);hf=la.cho_factor(z['H']);Lmass=la.cholesky(z['M'],lower=True)
    results={};checks=[];validity=[]
    for r in (8,12):
        U=z['U'][:,:r];Kfull=(z['Ksolid']+z['Kpressure'])[:,:r];G=z['G']@U;D=z['D'][:,:r]
        A,drift=affine_operator(U.T@Kfull,G,z['C'],z['L'],D,U.T@z['M']@z['acc'],z['pdot'])
        projected=z['actual_q']@z['M']@U;projected_v=z['actual_v']@z['M']@U
        scale=np.r_[np.maximum(np.max(abs(projected),axis=0),1e-10),np.maximum(np.max(abs(projected_v),axis=0),1e-5),np.full(len(z['C']),.2)]
        n=len(A);aug=np.zeros((2*n,2*n));aug[:n,:n]=A;aug[n:,:n]=np.eye(n)
        yy=exact(aug,np.r_[drift,np.zeros(n)],times,scale=np.r_[scale,scale*75e-6]);y=yy[:,:n];integ=yy[:,n:]
        pressure=z['p0']+y[:,2*r:];u=np.einsum('tr,r...->t...',y[:,:r],probes['basis'][:r]);v=np.einsum('tr,r...->t...',y[:,r:2*r],probes['basis'][:r])
        Zp=la.cho_solve(hf,z['B'].T);Zq=-la.cho_solve(hf,np.column_stack([dh@z['z0'] for dh in z['dH'][:r]]))
        cumulative=times[:,None]*z['z0']+integ[:,:r]@Zq.T+integ[:,2*r:]@Zp.T
        flux=interval_average(cumulative,times)
        mode_weights=z['top_volume'][:,None]*z['eta']/np.sum(z['top_volume'][:,None]*z['eta']**2,axis=0)
        modes=pressure@mode_weights;actual_modes=z['actual_p']@mode_weights
        mass=(pressure-z['p0'])*z['C']+.8*y[:,:r]@G.T+cumulative@z['B'].T
        # The DH correction contributes to linearized flux and thus to content.
        rec=dict(rank=r,initial_error=float(np.max(abs(y[0]))),max_cell_mass_defect_m3=float(np.max(abs(mass))),min_pressure_Pa=float(pressure.min()),nonzero_affine_drift=bool(np.linalg.norm(drift)>0),new_GPU_steps=0)
        if r==12:
            scaled=A*scale[None,:]/scale[:,None];ds=drift/scale
            sol=solve_ivp(lambda t,w:scaled@w+ds,(0,float(times[2])),np.zeros(n),method='BDF',jac=scaled,rtol=2e-7,atol=1e-10,t_eval=times[1:3])
            if not sol.success:raise ValueError('independent BDF failed '+sol.message)
            independent=sol.y.T*scale;error=float(np.max(abs(independent[:,2*r:]-y[1:3,2*r:])))
            rec.update(independent_method='scaled BDF, independent of augmented expm',independent_pressure_error_Pa=error,independent_rhs_calls=sol.nfev,independent_times_s=times[1:3].tolist())
            # Fixed solid degeneration uses identical H,C,B,gb and initial p.
            fixed=exact(-z['L']/z['C'][:,None],z['pdot'],times)+z['p0']
            old=dict(np.load(APP/'S1/reference-YZ128.npz'));ids=[int(np.argmin(abs(old['times']-t))) for t in times]
            rec['fixed_skeleton_degenerate_error_Pa']=float(np.max(abs(fixed-old['pressure'][ids])))
        rec['passed']=rec['initial_error']==0 and rec['max_cell_mass_defect_m3']<1e-10 and rec['min_pressure_Pa']>=0 and rec.get('independent_pressure_error_Pa',0)<1e-6 and rec.get('fixed_skeleton_degenerate_error_Pa',0)<1e-8;checks.append(rec)
        residual=[]
        for k,t in enumerate(times[1:],1):
            ar=y[k,:r];dp=y[k,2*r:];acc_full=z['acc']+la.cho_solve(mf,-Kfull@ar+.8*z['G'].T@dp)
            acc_reduced=U@(A@y[k]+drift)[r:2*r]
            residual.append(dict(time_s=float(t),acceleration_projection_relative=float(la.norm(Lmass.T@(acc_full-acc_reduced))/max(la.norm(Lmass.T@acc_full),1e-30))))
        err_p=np.max(abs(pressure-z['actual_p']),axis=1);err_u=np.max(abs(u-probes['actual_displacement']),axis=tuple(range(1,u.ndim)));err_v=np.max(abs(v-probes['actual_velocity']),axis=tuple(range(1,v.ndim)))
        validity.append(dict(rank=r,pressure_error_Pa=err_p.tolist(),displacement_error_m=err_u.tolist(),velocity_error_m_s=err_v.tolist(),maximum_actual_displacement_m=float(np.max(abs(probes['actual_displacement']))),projection_residual=residual,scope='differences include old trajectory time error; no continuum or nonlinear truth claim'))
        results[r]=dict(y=y,pressure=pressure,displacement=u,velocity=v,Ay=modes[:,0],Az=modes[:,1],flux=flux,cumulative=cumulative)
        np.savez_compressed(run/'S1'/f'reference-r{r}.npz',times=times,actual_Ay=actual_modes[:,0],actual_Az=actual_modes[:,1],actual_pressure=z['actual_p'],actual_displacement=probes['actual_displacement'],**results[r])
    rankdiff=float(np.max(abs(results[8]['pressure']-results[12]['pressure'])));maxres=max(v['acceleration_projection_relative'] for v in validity[-1]['projection_residual'])
    write(run/'S1/reference-self-check.json',dict(status='passed_scoped' if all(x['passed'] for x in checks) else 'limited',scope='projected affine model only',records=checks,rank8_to12_pressure_difference_Pa=rankdiff,seconds=time.perf_counter()-tick))
    write(run/'S1/local-validity.json',dict(status='diagnostic_only',records=validity,rank_pressure_difference_Pa=rankdiff,maximum_acceleration_projection_relative=maxres,full_space_reference_certified=False,mechanical_relative_accuracy=False,linearization_remainders=read(run/'S1/local-remainders.json'),reason='rank agreement alone does not control full-space dynamic projection error; reference is a mechanism diagnostic'))
    decision=dict(status='limited',decision='retain_short_window',half_trigger=False,extension_trigger=False,reason='projected reference has no certified full-space error bound; no new isolated time-error evidence; preserve current nonlinear scene and study independent BD cost',existing_end_s=75e-6,reference_scope='diagnostic_only',pressure_rank_difference_Pa=rankdiff,projection_relative=maxres)
    register(run,'S1/window-decision.json',decision)
    write(run/'S2/time-review.json',dict(status='not_triggered',new_steps=0,reason=decision['reason']))
    write(run/'S2/window-review.json',dict(status='not_triggered',new_steps=0,new_frames=0,reason=decision['reason']))
    update(f'S1参考自检：{checks[-1]}。r8→r12压力差{rankdiff:.3g}Pa，最大全空间加速度投影残差比{maxres:.3g}；仍标记diagnostic_only。S2扩窗/h2不触发，不把投影模型当真实非线性真值。')
    print('REFERENCE_REVIEW',checks,rankdiff,maxres,flush=True)
    if not all(x['passed'] for x in checks):raise ValueError('reference self-check requires inspection')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
