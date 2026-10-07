"""v19 bounded compatible-carrier AVF. Material grip coordinates are fixed.

Moving physical quadrature points use x=N_comp Y and L=grad(N_comp) F^-1.
This deliberately tests a conforming material representation, not production
Eulerian particle-grid remapping. Kinetic metric remains the APIC functional.
"""
import copy
import numpy as np
import scipy.linalg as la
from .carrier_driven import DrivenAVF,EquivalentEnergy,projection,kinetic_metric,displacement
from .carrier_joint import State,gradient
from .carrier_avf import SITES
from .unresolved_velocity import pack,unpack
from .endpoint_boundary import endpoint_impulse,prescribed_speed

class CompatibleGeometry:
    def __init__(self,s,e,m,h,clamped=True):
        if len(m)>50000 or e.n>512:raise ValueError('bounded compatible quadrature size exceeded')
        self.nodes=np.rint(e.reconstruction.nodes/h).astype(int);self.N=np.eye(e.n);self.E=self.N;self.fast_square=True
        fixed=((self.nodes[:,0]*h<=.25)|(self.nodes[:,0]*h>=.75)) if clamped else np.zeros(e.n,bool)
        self.Q=self.N[:,~fixed];self.metric=kinetic_metric(s.x,m,h);self.T=e.position_basis
        F=gradient(e.B,s.Y);inv=np.linalg.inv(F);L=[sum(b*inv[:,k,j,None] for k,b in enumerate(e.B)) for j in range(3)];self.J=np.vstack([self.T]+L)
        error=max(float(np.max(abs(self.T@s.Y-s.x))),float(np.max(abs(gradient(L,s.Y)-np.eye(3)))))
        if error>1e-9:raise ValueError('incompatible position/gradient history')
        self.info=dict(carriers=e.n,grid_nodes=e.n,free_scalar_dofs=self.Q.shape[1],right_inverse_error=0.,constraint_error=0.,affine_error=error)

class CompatibleAVF(DrivenAVF):
    def __init__(self,s,e,m,h,moving=True,mode='hold',clamped=True,chord=True):
        if not moving:raise ValueError('compatible x=N Y requires moving quadrature points')
        if mode not in ('hold','driven'):raise ValueError(mode)
        self.state=s.clone();self.energy=EquivalentEnergy(e);self.m=m;self.h=h;self.moving=True;self.mode=mode;self.clamped=clamped;self.chord=chord
        self.geometry=CompatibleGeometry(s,e,m,h,clamped);self.steps=0;self.previous=None;self.Kcache=None;self.cache_time=None

    # AVF residual/energy implementation copied unchanged from sealed v18.
    def midpoint_step(self,dt,max_iters=20):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        s=self.state;e=self.energy;g=CompatibleGeometry(s,e,self.m,self.h,self.clamped) if self.moving else self.geometry
        Q=g.Q;J=g.J;Jr=J@Q;q=g.metric;z=pack(s.v,s.C);Wb,speed=self.boundary(g,s.time,dt);initial=e.evaluate(s.Y);n=Q.shape[1]
        projected,rank=projection(Jr if self.mode=='hold' else J,q,z)
        # Previous physical carrier velocity is merely a Newton predictor.
        u=la.lstsq(Q,self.previous-Wb,cond=1e-12)[0] if self.previous is not None else np.zeros((n,3))
        Mr=Jr.T@(q[:,None]*Jr);mass=2*la.block_diag(Mr,Mr,Mr)
        def trial(u):
            W=Wb+Q@u;d=J@W-z;val=float(np.sum(q[:,None]*d*d));force=np.zeros_like(s.Y)
            for a in SITES:
                out=e.evaluate(s.Y+a*dt*W);val+=.5/a*out['U'];force+=.5*out['force']
            residual=2*Jr.T@(q[:,None]*d)+dt*Q.T@force
            return val,residual,force,W
        if self.chord:
            if self.Kcache is None or s.time-self.cache_time>.01:
                self.Kcache=e.tangent(s.Y);self.cache_time=s.time
            blocks=[[Q.T@self.Kcache[a*e.n:(a+1)*e.n,b*e.n:(b+1)*e.n]@Q for b in range(3)] for a in range(3)]
            H=mass+.5*dt*dt*np.block(blocks);factor=la.cho_factor(H)
        accepted=False;backs=0
        for it in range(max_iters+1):
            val,r,force,W=trial(u);rn=float(la.norm(r))
            if rn<=1e-17:accepted=True;break
            if it==max_iters:break
            if not self.chord or it>=4:
                H=mass+sum(.5*a*dt*dt*e.tangent(s.Y+a*dt*W,Q) for a in SITES)
                du=la.solve(H,-r.T.ravel(),assume_a='sym').reshape(3,n).T
            else:du=la.cho_solve(factor,-r.T.ravel()).reshape(3,n).T
            descent=float(np.sum(r*du))
            if descent>=0:raise ValueError('non-descent unmodified AVF residual')
            for ls in range(25):
                alpha=2.**(-ls)
                try:nval=trial(u+alpha*du)[0]
                except ValueError:continue
                if nval<=val+1e-4*alpha*descent+1e-18:u+=alpha*du;backs+=ls;break
            else:raise RuntimeError('AVF line search failed')
        if not accepted:raise RuntimeError(f'AVF residual failure: {rn}')
        Y=s.Y+dt*W;out=e.evaluate(Y);zp=z+2*(J@W-projected);v,C=unpack(zp)
        x=s.x+dt*g.T@W if self.moving else s.x.copy();q1=kinetic_metric(x,self.m,self.h) if self.moving else q
        K0=.5*float(np.sum(q[:,None]*z*z));Kf=.5*float(np.sum(q[:,None]*zp*zp));K1=.5*float(np.sum(q1[:,None]*zp*zp))
        work=float(np.sum(force*(Y-s.Y)));impulse=2*J.T@(q[:,None]*(J@W-z))+dt*force
        boundary_work=float(np.sum(impulse*Wb)) if self.mode=='driven' else 0.
        defect=Kf-K0+work-boundary_work;quad=out['U']-initial['U']-work
        # Generalized force dual to the unit right-grip velocity, independent
        # of current loading speed, so force is measured during holds too.
        grid=np.zeros((len(g.nodes),3));right=g.nodes[:,0]*self.h>=.75;grid[right,0]=1.
        fixed=(g.nodes[:,0]*self.h<=.25)|right
        lift=g.N@grid if g.fast_square else la.lstsq(g.E[fixed],grid[fixed],cond=1e-11)[0]
        reaction=float(np.sum(impulse*lift)/dt)
        bc=g.E@W;expected=np.zeros_like(bc);expected[right,0]=speed
        bcerr=float(np.max(abs(bc[fixed]-expected[fixed]))) if self.clamped else 0.
        row=dict(step=self.steps+1,time=s.time+dt,mode=self.mode,moving=self.moving,dt=dt,
            material_J=out['Um'],stabilization_J=out['Us'],kinetic_J=K1,total_J=K1+out['U'],delta_total_J=K1-K0+out['U']-initial['U'],
            metric_change_J=K1-Kf,boundary_work_J=boundary_work,reaction_N=reaction,loading_speed=speed,command_displacement=displacement(s.time+dt),
            kinetic_force_work_defect_J=defect,potential_quadrature_error_J=quad,newton_residual=rn,newton_iterations=it,line_search_backtracks=backs,
            history_commit_max=float(np.max(abs(out['F']-initial['F']-gradient(e.B,Y-s.Y)))),grip_velocity_error=bcerr,
            stress_rms_Pa=float(np.sqrt(np.mean(np.sum(out['P']**2,axis=(1,2))))),min_det_F=float(np.linalg.det(out['F']).min()),kinetic_projection_rank=rank,**g.info)
        if bcerr>1e-8:raise ValueError('grip velocity constraint failure')
        self.state=State(x,Y,v,C,s.time+dt);self.geometry=g;self.previous=W;self.steps+=1;self.last=dict(old=s,force=force,W=W,impulse=impulse,projected=projected)
        return row

    def step(self,dt,max_iters=20):
        # Work copy keeps failure transactional, including Newton predictors.
        work=copy.copy(self);row=work.midpoint_step(dt,max_iters);s=work.state
        end=CompatibleGeometry(s,work.energy,work.m,work.h,work.clamped) if work.moving else work.geometry
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
