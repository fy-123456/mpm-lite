"""v20 residual-merit fallback near energy-rounding level; unchanged final gates, with optional variational material-null condensation.

The null directions Z have B_material Z=0 and zero prescribed carrier values.
Eliminating Z by Z^T Ks Y=0 minimizes the ORIGINAL patch energy, without any
coefficient change. Arbitrary old states off this manifold are rejected, not
silently projected. Full cycles start consistently from rest. No mass shift.
"""
import copy
import numpy as np
import scipy.linalg as la
from .carrier_driven import DrivenAVF,EquivalentEnergy,projection,kinetic_metric,displacement
from .carrier_joint import State,gradient
from .carrier_avf import SITES
from .unresolved_velocity import pack,unpack
from .endpoint_boundary import right_lift,prescribed_speed
from .separate_kinetic import KineticGeometry
from .thin_projection import ThinProjector,projection


def material_null_condensation(e,reference,h):
    fixed=(reference[:,0]<=.25)|(reference[:,0]>=.75);Q=np.eye(e.n)[:,~fixed];A=np.vstack([b@Q for b in e.B]);_,sv,Vh=la.svd(A,full_matrices=False);Z=Q@Vh[sv<1e-12*sv[0]].T
    if not Z.shape[1]:return np.eye(e.n),np.eye(e.n),Z,np.zeros((0,e.n))
    C=Z.T@e.Ks;Kzz=C@Z;la.cholesky(Kzz);R=np.eye(e.n)-Z@la.solve(Kzz,C,assume_a='pos');H=la.null_space(C,rcond=1e-12)
    if max(la.norm(np.vstack(e.B)@Z),la.norm(C@R))>1e-9:raise ValueError('inexact material-null condensation')
    return H,R,Z,C


def boundary_impulse(g,z,speed,h):
    lift=g.R@right_lift(g,h);target=g.J@lift*speed;root=np.sqrt(g.metric)[:,None]
    full=getattr(g,'full_projector',None)
    if full is None:full=ThinProjector(root*(g.J@g.H));g.full_projector=full
    free=getattr(g,'free_projector',None)
    if free is None:free=ThinProjector(root*(g.J@g.Q));g.free_projector=free
    Z=root*z;T=root*target;delta_w=T-full.project(Z)+free.project(Z-T);delta=delta_w/root;updated=z+delta
    impulse=g.J.T@(g.metric[:,None]*delta);work=float(np.sum(impulse*lift)*speed);loss=.5*float(np.sum(delta_w**2));change=.5*float(np.sum(g.metric[:,None]*(updated**2-z**2)))
    end=root*(updated-target);error=float(la.norm(full.project(end)-free.project(end)));free_error=float(la.norm(g.Q.T@impulse));nullerr=float(la.norm(delta_w-full.project(delta_w)))
    if max(error,free_error,nullerr)>1e-9 or abs(change-work+loss)>1e-13:raise ValueError('boundary work/constraint failure')
    return updated,dict(impulse=impulse,delta=delta,lift=lift,external_work_J=work,kinetic_loss_J=loss,energy_change_J=change,constraint_norm=error,free_impulse_error=free_error,null_change_error=nullerr)

class RobustIntegratedAVF(DrivenAVF):
    def __init__(self,s,e,m,h,moving=True,mode='driven',condense=False,clamped=True,chord=True):
        if mode not in ('hold','driven'):raise ValueError(mode)
        self.state=s.clone();self.energy=EquivalentEnergy(e);self.m=m;self.h=h;self.moving=moving;self.mode=mode;self.clamped=clamped;self.chord=chord;self.condense=condense
        if condense:
            if not clamped:raise ValueError('this condensation is defined for the prescribed hard grips')
            ref=getattr(e,'carrier_reference',None)
            if ref is None:raise ValueError('reference carriers required for condensation')
            self.H,self.R,self.Z,self.constraint=material_null_condensation(e,ref,h)
            if la.norm(self.constraint@s.Y)>1e-10:raise ValueError('old stabilization history not on condensation manifold; no silent reset')
        else:self.H=self.R=np.eye(e.n);self.Z=np.empty((e.n,0));self.constraint=np.empty((0,e.n))
        self.geometry=self.make_geometry(s);self.steps=0;self.previous=None;self.Kcache=None;self.cache_time=None
    def make_geometry(self,s):
        g=KineticGeometry(s,self.energy,self.m,self.h,self.clamped);g.H=self.H;g.R=self.R
        if self.condense:
            g.Q=g.Q@la.null_space(self.constraint@g.Q,rcond=1e-12);g.info['free_scalar_dofs']=g.Q.shape[1]
            fixed=(g.nodes[:,0]*self.h<=.25)|(g.nodes[:,0]*self.h>=.75)
            if la.norm(g.E[fixed]@self.Z)>1e-9:raise ValueError('condensation no longer preserves current grips')
        return g
    def boundary(self,g,t,dt):
        W,speed=DrivenAVF.boundary(self,g,t,dt);return self.R@W,speed

    def midpoint_step(self,dt,max_iters=20):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        s=self.state;e=self.energy;g=getattr(self,'endpoint_geometry',self.geometry) if self.moving else self.geometry
        Q=g.Q;J=g.J;Jr=J@Q;q=g.metric;z=pack(s.v,s.C);Wb,speed=self.boundary(g,s.time,dt);initial=e.evaluate(s.Y);n=Q.shape[1]
        basis=getattr(g,'free_projector' if self.mode=='hold' else 'full_projector',None)
        if basis is None:basis=ThinProjector(np.sqrt(q)[:,None]*(Jr if self.mode=='hold' else J@self.H))
        projected=basis.project(np.sqrt(q)[:,None]*z)/np.sqrt(q)[:,None];rank=basis.rank
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
        accepted=False;backs=0;roundoff_merit=0
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
                try:nval,nres,_,_=trial(u+alpha*du)
                except ValueError:continue
                if nval<=val+1e-4*alpha*descent+1e-18:u+=alpha*du;backs+=ls;break
                # Energy differences can fall below material-energy rounding.
                # Near the root only, accept a decreasing true residual; the
                # final 1e-17 residual and 1e-13 energy ledger are unchanged.
                if rn<1e-12 and abs(nval-val)<1e-14 and la.norm(nres)<rn:
                    u+=alpha*du;backs+=ls;roundoff_merit+=1;break
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
        lift=self.R@lift
        reaction=float(np.sum(impulse*lift)/dt)
        bc=g.E@W;expected=np.zeros_like(bc);expected[right,0]=speed
        bcerr=float(np.max(abs(bc[fixed]-expected[fixed]))) if self.clamped else 0.
        row=dict(step=self.steps+1,time=s.time+dt,mode=self.mode,moving=self.moving,dt=dt,
            material_J=out['Um'],stabilization_J=out['Us'],kinetic_J=K1,total_J=K1+out['U'],delta_total_J=K1-K0+out['U']-initial['U'],
            metric_change_J=K1-Kf,boundary_work_J=boundary_work,reaction_N=reaction,loading_speed=speed,command_displacement=displacement(s.time+dt),
            kinetic_force_work_defect_J=defect,potential_quadrature_error_J=quad,newton_residual=rn,newton_iterations=it,line_search_backtracks=backs,roundoff_merit_accepts=roundoff_merit,
            history_commit_max=float(np.max(abs(out['F']-initial['F']-gradient(e.B,Y-s.Y)))),grip_velocity_error=bcerr,
            stress_rms_Pa=float(np.sqrt(np.mean(np.sum(out['P']**2,axis=(1,2))))),min_det_F=float(np.linalg.det(out['F']).min()),kinetic_projection_rank=rank,**g.info)
        if bcerr>1e-8:raise ValueError('grip velocity constraint failure')
        self.state=State(x,Y,v,C,s.time+dt);self.geometry=g;self.previous=W;self.steps+=1;self.last=dict(old=s,force=force,W=W,impulse=impulse,projected=projected)
        return row

    def step(self,dt,max_iters=20):
        # Work copy keeps failure transactional, including Newton predictors.
        work=copy.copy(self);row=work.midpoint_step(dt,max_iters);s=work.state
        end=work.make_geometry(s) if work.moving else work.geometry
        trial_z=pack(s.v,s.C);speed=prescribed_speed(s.time) if work.mode=='driven' else 0.
        if work.clamped:
            updated,diag=boundary_impulse(end,trial_z,speed,work.h)
        else:
            updated=trial_z;diag=dict(impulse=np.zeros_like(s.Y),delta=np.zeros_like(trial_z),lift=np.zeros_like(s.Y),rank=end.Q.shape[1],external_work_J=0.,kinetic_loss_J=0.,energy_change_J=0.,free_impulse_error=0.,null_change_error=0.,constraint_norm=0.)
        s.v,s.C=unpack(updated);correction=float(np.sum(diag['impulse']*diag['lift'])/dt)
        row.update(midpoint_reaction_N=row['reaction_N'],endpoint_reaction_N=correction,midpoint_boundary_work_J=row['boundary_work_J'],
            endpoint_boundary_work_J=diag['external_work_J'],constraint_kinetic_loss_J=diag['kinetic_loss_J'],
            endpoint_velocity_constraint=diag['constraint_norm'],endpoint_free_impulse_error=diag['free_impulse_error'],endpoint_null_change_error=diag['null_change_error'],
            endpoint_speed=speed,endpoint_quadrature_work_J=diag['external_work_J']-dt*correction*row['loading_speed'],
            endpoint_momentum_change=np.sum(work.m[:,None]*diag['delta'][:len(work.m)],axis=0).tolist())
        row['reaction_N']+=correction;row['boundary_work_J']+=diag['external_work_J'];row['kinetic_J']+=diag['energy_change_J'];row['total_J']+=diag['energy_change_J'];row['delta_total_J']+=diag['energy_change_J']
        row['material_null_constraint']=float(la.norm(self.constraint@s.Y))
        if row['material_null_constraint']>1e-9:raise ValueError('condensed history drift')
        row['budget_defect_J']=row['delta_total_J']-row['boundary_work_J']-row['metric_change_J']-row['kinetic_force_work_defect_J']-row['potential_quadrature_error_J']+row['constraint_kinetic_loss_J']
        if abs(row['budget_defect_J'])>1e-13:raise ValueError('endpoint step energy budget')
        work.endpoint_geometry=end;work.endpoint_data=diag;work.midpoint_trial_z=trial_z
        self.__dict__.update(work.__dict__)
        return row
