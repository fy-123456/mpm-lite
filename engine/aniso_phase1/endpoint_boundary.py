"""v19 research constraint impulse, preserving free and full-null history.

The midpoint material solve is unchanged. A boundary velocity constraint at
actual end geometry supplies a second, explicitly recorded impulse. This is
not an output reaction filter: its momentum is included in the next state and
in the interval reaction. Its constraint work and kinetic loss are separate.
"""
import copy
import numpy as np
import scipy.linalg as la
from .carrier_driven import DrivenAVF,FastGeometry,projection,kinetic_metric
from .unresolved_velocity import pack,unpack


def prescribed_speed(t):
    if 0<t<.5:return .005*np.pi*np.sin(2*np.pi*t)
    if .6<t<1.1:return -.005*np.pi*np.sin(2*np.pi*(t-.6))
    return 0.


def right_lift(g,h):
    grid=np.zeros((len(g.nodes),3));right=g.nodes[:,0]*h>=.75;fixed=(g.nodes[:,0]*h<=.25)|right;grid[right,0]=1.
    lift=g.N@grid if g.fast_square else la.lstsq(g.E[fixed],grid[fixed],cond=1e-11)[0]
    if np.max(abs(g.E[fixed]@lift-grid[fixed]))>1e-9:raise ValueError('inaccurate boundary lift')
    return lift


def endpoint_impulse(g,z,speed,h):
    """Orthogonal velocity constraint; no change to admissible or null motion."""
    lift=right_lift(g,h);target=g.J@lift*speed;sqrt=np.sqrt(g.metric)[:,None]
    def basis(A):
        Q,R=la.qr(sqrt*A,mode='economic');U,sv,_=la.svd(R,full_matrices=False)
        return Q@U[:,sv>1e-12*sv[0]]
    full=basis(g.J);free=basis(g.J@g.Q)
    Z=sqrt*z;T=sqrt*target;delta_w=T-full@(full.T@Z)+free@(free.T@(Z-T))
    delta=delta_w/sqrt;updated=z+delta;impulse=g.J.T@(g.metric[:,None]*delta)
    external=float(np.sum(impulse*lift)*speed);loss=.5*float(np.sum(delta_w*delta_w))
    energy_change=.5*float(np.sum(g.metric[:,None]*(updated*updated-z*z)))
    free_error=float(la.norm(g.Q.T@impulse));null_delta=full@(full.T@delta_w)/sqrt
    end=sqrt*(updated-target);constraint=float(la.norm(full@(full.T@end)-free@(free.T@end)));rank=full.shape[1]
    if max(free_error,la.norm(delta-null_delta),constraint)>1e-9:raise ValueError('inexact endpoint constraint impulse')
    if abs(energy_change-external+loss)>1e-14:raise ValueError('endpoint impulse work identity')
    return updated,dict(impulse=impulse,delta=delta,lift=lift,rank=rank,external_work_J=external,kinetic_loss_J=loss,energy_change_J=energy_change,
        free_impulse_error=free_error,null_change_error=float(la.norm(delta-null_delta)),constraint_norm=constraint)


class EndpointAVF(DrivenAVF):
    def step(self,dt,max_iters=20):
        # Work copy keeps failure transactional, including Newton predictors.
        work=copy.copy(self);row=DrivenAVF.step(work,dt,max_iters);s=work.state
        end=FastGeometry(s,work.energy,work.m,work.h,work.clamped) if work.moving else work.geometry
        trial_z=pack(s.v,s.C);speed=prescribed_speed(s.time) if work.mode=='driven' else 0.
        if work.clamped:
            updated,diag=endpoint_impulse(end,trial_z,speed,work.h)
        else:
            updated=trial_z;diag=dict(impulse=np.zeros_like(s.Y),delta=np.zeros_like(trial_z),lift=np.zeros_like(s.Y),rank=end.Q.shape[1],external_work_J=0.,kinetic_loss_J=0.,energy_change_J=0.,free_impulse_error=0.,null_change_error=0.,constraint_norm=0.)
        s.v,s.C=unpack(updated);correction=float(np.sum(diag['impulse']*diag['lift'])/dt)
        row.update(midpoint_reaction_N=row['reaction_N'],endpoint_reaction_N=correction,midpoint_boundary_work_J=row['boundary_work_J'],
            endpoint_boundary_work_J=diag['external_work_J'],constraint_kinetic_loss_J=diag['kinetic_loss_J'],
            endpoint_velocity_constraint=diag['constraint_norm'],endpoint_free_impulse_error=diag['free_impulse_error'],endpoint_null_change_error=diag['null_change_error'],
            endpoint_speed=speed,endpoint_quadrature_work_J=diag['external_work_J']-dt*correction*row['loading_speed'],
            endpoint_momentum_change=np.sum(work.m[:,None]*diag['delta'][:len(work.m)],axis=0).tolist())
        row['reaction_N']+=correction;row['boundary_work_J']+=diag['external_work_J'];row['kinetic_J']+=diag['energy_change_J'];row['total_J']+=diag['energy_change_J'];row['delta_total_J']+=diag['energy_change_J']
        row['budget_defect_J']=row['delta_total_J']-row['boundary_work_J']-row['metric_change_J']-row['kinetic_force_work_defect_J']-row['potential_quadrature_error_J']+row['constraint_kinetic_loss_J']
        if abs(row['budget_defect_J'])>1e-13:raise ValueError('endpoint step energy budget')
        work.endpoint_geometry=end;work.endpoint_data=diag;work.midpoint_trial_z=trial_z
        self.__dict__.update(work.__dict__)
        return row
