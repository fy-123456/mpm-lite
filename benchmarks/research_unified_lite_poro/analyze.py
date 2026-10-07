"""Shared-site diagnostics and bounded fixed-skeleton reference, no new frames."""
from pathlib import Path
import json,time
import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from engine.aniso_phase1.research_unified_lite_poro.model import Bridge,FrozenOperator
from .run import write


def main(root):
    root=Path(root);s=Space();X,w,_=s.rule(4);N,D=s.basis(X)
    comparisons=[]
    for prefix in ['', 'fixed-']:
        a=np.load(root/'S3'/f'{prefix}short-ppc2'/'terminal.npz')
        b=np.load(root/'S3'/f'{prefix}short-ppc4'/'terminal.npz')
        ua,ub=N@a['q'],N@b['q']
        comparisons.append(dict(rule='moments' if not prefix else 'fixed-positive',
          physical_time_s=.02,same_observation_points=True,
          displacement_relative=float(la.norm(ua-ub)/la.norm(ub)),displacement_max_m=float(abs(ua-ub).max()),
          pressure_max_Pa=float(abs(a['p']-b['p']).max()),comparison='PPC2 vs PPC4, not exact solution'))
    write(root/'S6/common-site-ppc.json',comparisons)
    # Compare per-call material cost and count at identical generalized state;
    # preparation/linear solve are excluded and never called end-to-end speedups.
    q=np.load(root/'S3/fixed-cycle-ppc3/terminal.npz')['q'];rows=[]
    for ppc in (2,3,4,8):
        t=time.perf_counter();b=Bridge(s,ppc,rule='fixed-positive');op,_,_,_=b.prepare();prep=time.perf_counter()-t
        op.material(q);times=[];count0=op.material_evaluations
        for k in range(3):
            t=time.perf_counter();op.material(q);times.append(time.perf_counter()-t)
        rows.append(dict(ppc=ppc,Np=len(b.state.particles.X),Nq=len(op.points),prepare_and_construct_s=prep,
          per_call_material_evaluations=(op.material_evaluations-count0)//3,median_material_seconds=float(np.median(times)),
          direct_array_bytes_excluding_space_topology=int(sum(v.nbytes for v in vars(op).values() if isinstance(v,np.ndarray)))))
    write(root/'S6/operator-scaling.json',dict(scope='CPU per-call only; fixed reference-coordinate template',rows=rows))
    # Same-space material integration reference: analytic continuous direction
    # is used only here, never in the particle reconstruction implementation.
    refrows=[];refs=[]
    for order in (5,7):
        xr,wr,_=s.rule(order);angle=np.deg2rad(20+50*xr[:,1]/.25+15*xr[:,0]);a=np.c_[np.cos(angle),np.sin(angle),np.zeros(len(angle))]
        A=a[:,:,None]*a[:,None,:];v=A.reshape(-1,9)
        op=FrozenOperator(s,xr,wr,A,v[:,:,None]*v[:,None,:]);E,f,P=op.material(q);refs.append((E,f))
        refrows.append(dict(order=order,Nq=len(xr),energy_J=E,force_norm_N=float(la.norm(f))))
    relative=float(la.norm(refs[0][1]-refs[1][1])/la.norm(refs[1][1]))
    write(root/'S4/material-reference.json',dict(scope='same-space integration convergence only',rows=refrows,force_order5_order7_relative=relative,full_space_reference=False))
    # Fixed skeleton is a degenerate subproblem, not a substitute for moving
    # skeleton accuracy. Independent BDF checks exponential time evolution.
    b=Bridge(s,rule='fixed-positive');op,_,_,_=b.prepare();q0=b.state.q
    V,G,H,_=op.geometry(q0);D=op.top.B;L=D@la.solve(H,D.T,assume_a='pos')
    A=-L/op.capacity[:,None];p0=b.state.p;ts=np.array([0.,.01,.02,.06])
    exact=np.array([la.expm(t*A)@p0 for t in ts])
    sol=solve_ivp(lambda t,p:A@p,(0.,ts[-1]),p0,method='BDF',t_eval=ts,rtol=1e-7,atol=1e-10)
    if not sol.success:raise RuntimeError(sol.message)
    err=float(np.max(abs(exact-sol.y.T)))
    write(root/'S4/fixed-skeleton-reference.json',dict(scope='fixed skeleton only; two pressure cells, no spatial accuracy',
         pressure_difference_Pa=err,passed=err<1e-6,times_s=ts.tolist(),pressure=exact.tolist()))
    write(root/'S3/adoption-decision.json',dict(selected_rule='fixed-positive',Nq=128,scope='structured reference-particle CPU bridge',
         reason='same-site PPC displacement difference falls below preregistered 5% engineering scale; positive moments; stable short cycle',
         original_rule_retained_as='moments diagnostic baseline',spatial_accuracy=False,
         remaining_material_force_error='13.4% at PPC4 in registered diagnostic state; do not certify full stress accuracy',
         production_default_changed=False,eulerian_lite_integration=False))
    print(json.dumps(dict(common_sites=comparisons,material_reference_relative=relative,fixed_skeleton_error=err),indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
