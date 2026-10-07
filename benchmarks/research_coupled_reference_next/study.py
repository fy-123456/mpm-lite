"""Budgeted coupled-reference assembly: true geometric derivatives, scoped rank."""
import argparse,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .fixture import setup,inputs
from .runtime import update
from engine.aniso_phase1.research_coupled_reference_next.reference import mass_basis

def main(run):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,cfg,ident=setup(run);built=time.perf_counter()-tick;h=inputs();s=h[0]['state'];q=s.q.copy();g=c.geometry
    if s.digest()!=c.initial_digest or np.any(s.velocity):raise ValueError('reference requires authenticated stationary initial anchor')
    a=g.evaluate(q);p=np.array(s.child_states['fluid']['pressure_Pa']);C=c.core.capacity;B=c.core.B;H=a['H'];hf=la.cho_factor(H)
    G=a['gradient'][:,m.free].reshape(g.cells,-1);M=m.M3ff;mf=la.cho_factor(M);K=m.rest_K[np.ix_(m.ids,m.ids)]
    z=la.cho_solve(hf,B.T@p-c.core.gb);L=B@la.cho_solve(hf,B.T)
    raw=m.evaluate(q);f=raw['force'][m.free].ravel();acc=la.cho_solve(mf,-f+.8*G.T@p);pdot=-(B@z)/C
    centres=np.mean(np.asarray(g.topology.cell_bounds),axis=2);eta=2*(centres[:,1:]-.5)/.25
    columns=[acc,la.cho_solve(mf,G.T@np.ones(g.cells)),la.cho_solve(mf,G.T@eta[:,0]),la.cho_solve(mf,G.T@eta[:,1])]
    for step in (8,12,18):columns.extend([h[step]['state'].q[m.free].ravel(),h[step]['state'].velocity[m.free].ravel()])
    w=acc.copy()
    for _ in range(10):
        w=la.cho_solve(mf,K@w);w/=max(la.norm(w),1e-100);columns.append(w.copy())
    U=mass_basis(M,np.array(columns).T,12);r=U.shape[1]
    register(run,'S1/reference-cost-decision.json',dict(status='diagnostic_only',route='mass-orthonormal r8/r12 solid, all128 pressure',free_solid_dofs=len(M),full_state_dofs=2*len(M)+len(C),projected_state_dofs=2*r+len(C),reason='full DH/dq and pressure geometric Hessian not available as cheap complete matrix; bounded directional assembly avoids all-DOF differences',assembly_directions=r,validation_directions=4,build_s=built,full_space_temporal_accuracy=False))
    register(run,'S1/reference-protocol.json',dict(anchor_digest=s.digest(),anchor_step=0,non_equilibrium_affine_drift=True,zero_velocity_anchor=True,pressure_geometric_derivative=True,Darcy_geometry_derivative=True,source_zero=True,rank=[8,12],finite_difference_gradient_amplitude=1e-5,validation_half_amplitude=5e-6,full_matrix_symmetrization=False,independent_times_s=[25e-6,37.5e-6]))
    (run/'S1/equation-map.md').write_text('M vdot = -fint(q)+alpha G(q)^T p; C pdot=-alpha G(q)v-Bz; H(q)z=B^Tp-gb.\nAnchor is authenticated rest q,v with nonuniform p. Drift retained.\nK=Ksolid-alpha Dq(G^Tp); Dq(Bz)=-B H^-1 DH z.\nThe original theta_matrix is an iteration matrix, not the continuous Jacobian.\nDiagnostic projection only; no formal-space modification.\n')
    checks=[];dpress=[];D=[];dG=[];dH=[];epsilons=[];Ks=[];probeU=[]
    axes=tuple(np.linspace(e[0],e[-1],9) for e in m.parent.edges)
    for j in range(r):
        direction=np.zeros_like(q);direction[m.free]=U[:,j].reshape(-1,3)
        disp,grad=m.parent._sample(m.parent.nodes(m.reduction.velocity(direction)),axes);probeU.append(disp)
        eps=1e-5/max(float(np.max(abs(grad))),1e-12);epsilons.append(eps)
        plus=g.evaluate(q+eps*direction);minus=g.evaluate(q-eps*direction)
        dg=(plus['gradient'][:,m.free].reshape(g.cells,-1)-minus['gradient'][:,m.free].reshape(g.cells,-1))/(2*eps)
        dh=(plus['H']-minus['H'])/(2*eps)
        kg=-.8*(dg.T@p);dd=-B@la.cho_solve(hf,dh@z)
        dpress.append(kg);D.append(dd);dG.append(dg);dH.append(dh)
        # Actual material tangent includes the original stabilization. Existing
        # rest_K is checked, not assumed equal to the force derivative.
        kv=m.evaluate(q,direction)['tangent_action'][m.free].ravel();Ks.append(kv)
        if j<4:
            pp=g.evaluate(q+.5*eps*direction);mm=g.evaluate(q-.5*eps*direction)
            dg2=(pp['gradient'][:,m.free].reshape(g.cells,-1)-mm['gradient'][:,m.free].reshape(g.cells,-1))/eps
            dh2=(pp['H']-mm['H'])/eps
            zplus=la.cho_solve(la.cho_factor(pp['H']),B.T@p-c.core.gb);zminus=la.cho_solve(la.cho_factor(mm['H']),B.T@p-c.core.gb)
            direct=B@(zplus-zminus)/eps
            def err(x,y):return float(la.norm(x-y)/max(la.norm(y),1e-30))
            row=dict(direction=j,eps=eps,solid_rest_tangent_relative=err(K@U[:,j],kv),G_half_relative=err(dg,dg2),DH_half_relative=err(dh,dh2),Darcy_action_relative=err(dd,direct),volume_chain_relative=err((pp['volume']-mm['volume'])/eps,G@U[:,j]))
            row['passed']=all(row[k]<1e-3 for k in ('solid_rest_tangent_relative','G_half_relative','DH_half_relative','Darcy_action_relative','volume_chain_relative'));checks.append(row)
        print('REFERENCE_DIRECTION',j+1,flush=True)
    Ksolid=np.array(Ks).T;Kpressure=np.array(dpress).T;D=np.array(D).T
    passed=all(x['passed'] for x in checks)
    write(run/'S1/derivative-check.json',dict(status='passed_scoped' if passed else 'limited',records=checks,geometry_evaluations=2*r+4*2,material_tangent_calls=r,assembly_basis_directions=r,independent_direction_checks=4,seconds=time.perf_counter()-tick))
    np.savez_compressed(run/'S1/probes.npz',basis=np.array(probeU),actual_displacement=np.array([m.parent._sample(m.parent.nodes(m.reduction.velocity(h[i]['state'].q)),axes)[0] for i in (0,8,12,18)]),actual_velocity=np.array([m.parent._sample(m.parent.nodes(m.reduction.velocity(h[i]['state'].velocity)),axes)[0] for i in (0,8,12,18)]))
    # Full-space block actions retained for a posteriori projection residuals.
    np.savez_compressed(run/'S1/linear-blocks.npz',U=U,M=M,Ksolid=Ksolid,Kpressure=Kpressure,G=G,C=C,H=H,B=B,L=L,D=D,acc=acc,pdot=pdot,z0=z,p0=p,f0=f,q0=q,free=m.free,ids=m.ids,eps=np.array(epsilons),dG=np.array(dG),dH=np.array(dH),times=np.array([h[i]['state'].time for i in (0,8,12,18)]),actual_q=np.array([h[i]['state'].q[m.free].ravel() for i in (0,8,12,18)]),actual_v=np.array([h[i]['state'].velocity[m.free].ravel() for i in (0,8,12,18)]),actual_p=np.array([h[i]['state'].child_states['fluid']['pressure_Pa'] for i in (0,8,12,18)]),V0=a['volume'],top_volume=g.V0,eta=eta,gb=c.core.gb)
    # Quantify Taylor remainder on inherited actual states, not on the fit alone.
    local=[]
    for step in (8,12,18):
        state=h[step]['state'];actual=g.evaluate(state.q);pp=np.array(state.child_states['fluid']['pressure_Pa']);vf=state.velocity[m.free].ravel();x=U.T@M@state.q[m.free].ravel();vp=U.T@M@vf
        GG=actual['gradient'][:,m.free].reshape(g.cells,-1);zz=la.cho_solve(la.cho_factor(actual['H']),B.T@pp-c.core.gb)
        fp=m.evaluate(state.q)['force'][m.free].ravel();true_acc=la.cho_solve(mf,-fp+.8*GG.T@pp)
        lin_acc=acc+la.cho_solve(mf,-(Ksolid+Kpressure)@x+.8*G.T@(pp-p))
        true_pd=(-.8*GG@vf-B@zz)/C;lin_pd=pdot+(-.8*G@U@vp-D@x-L@(pp-p))/C
        local.append(dict(step=step,time_s=state.time,projected_q_mass_error=float(la.norm(la.cholesky(M,lower=True).T@(state.q[m.free].ravel()-U@x))),pressure_derivative_remainder_Pa_s=float(np.max(abs(true_pd-lin_pd))),pressure_remainder_over_12_5us_Pa=float(np.max(abs(true_pd-lin_pd))*12.5e-6),acceleration_remainder_mass_norm=float(np.sqrt(max(0,(true_acc-lin_acc)@M@(true_acc-lin_acc)))),min_detF=actual['min_detF']))
    write(run/'S1/local-remainders.json',dict(status='diagnostic_only',records=local,scope='includes projection and nonlinear remainder, independent inherited states; not time-error attribution'))
    write(run/'S1/assembly.json',dict(status='passed_scoped' if passed else 'limited',seconds=time.perf_counter()-tick,build_s=built,rank=r,free_solid_dofs=len(M),initial_acceleration_mass_norm=float(np.sqrt(acc@M@acc)),initial_pressure_derivative_max_Pa_s=float(np.max(abs(pdot))),mass_orthogonality=float(la.norm(U.T@M@U-np.eye(r))),geometry_profile=g.profile,execution_identity=ident))
    update(f'S1：已构造含完整质量投影、压力几何项、Darcy几何项和非零初始驱动的r8/r12诊断参考，压力保持128格。4方向导数检查 {"通过" if passed else "受限"}；正式空间未变，完整空间参考不作认证。')
    if not passed:raise ValueError('derivative checks require inspection before reference adoption')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
