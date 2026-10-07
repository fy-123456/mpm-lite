"""Measure the inherited quasi-Newton matrix against the actual mixed residual."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .coupling import setup
from benchmarks.research_phase_stress_next.time_study import history


def probe(run):
    run=Path(run);h=history(run/'cases/observable-coarse-h');s0,s1=h[0]['state'],h[1]['state'];c,m,cfg=setup(run);dt=s1.time-s0.time;N=len(m.ids);nc=c.cells;p0=np.array(s0.child_states['fluid']['pressure_Pa']);q0=s0.q;v0=s0.velocity;geo0=c.geometry.evaluate(q0)
    x=np.r_[((s1.q-q0)/dt)[m.free].ravel(),s1.child_states['fluid']['pressure_Pa'],s1.child_states['fluid']['flux_interval_m3_s']]
    def residual(x):
        W=np.zeros_like(q0);W[m.free]=x[:N].reshape(-1,3);p1=x[N:N+nc];z=x[N+nc:];pbar=.5*(p0+p1);q1=q0+dt*W;g1=c.geometry.evaluate(q1);G=c.geometry.discrete(q0,q1);H=c.geometry.evaluate((q0+q1)/2)['H'];path=c.avf.path(q0,W,dt);fp=-c.alpha*np.einsum('k,kij->ij',pbar,G);solid=2*m.M@(W-v0)+dt*(path['force']+fp);mass=c.alpha*(g1['volume']-geo0['volume'])+c.capacity*(p1-p0)+dt*c.B@z-dt*c.source;darcy=H@z-c.B.T@pbar+c.gb
        return np.r_[solid[m.free].ravel(),mass,darcy],c.matrix(dt,G,g1['gradient'],H)
    base,A=residual(x);rng=np.random.default_rng(14);d=np.r_[rng.normal(size=N)*1e-5,rng.normal(size=nc)*.01,rng.normal(size=c.nflux)*1e-9];records=[]
    for eps in (1e-3,5e-4):
        plus,_=residual(x+eps*d);minus,_=residual(x-eps*d);fd=(plus-minus)/(2*eps);approx=A@d;parts={}
        for label,sl in [('solid',slice(0,N)),('mass',slice(N,N+nc)),('darcy',slice(N+nc,None))]:
            parts[label]=dict(fd_norm=float(np.linalg.norm(fd[sl])),approximation_error_norm=float(np.linalg.norm(fd[sl]-approx[sl])),relative_difference=float(np.linalg.norm(fd[sl]-approx[sl])/max(np.linalg.norm(fd[sl]),1e-30)))
        records.append(dict(epsilon=eps,blocks=parts))
    old=read(run/'S3/quasi-newton-scope.json');old.update(direction_probes=records,actual_equations_evaluated=True,dynamic_steps=0,exact_nonlinear_jacobian=False);write(run/'S3/quasi-newton-scope.json',old)
    print('MIXED_QUASI_NEWTON_PROBE',records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):probe(a.run)
