"""Two-point average-vector-field time update of the SAME carrier potential.

Midpoint velocity advances Y/x; particle v/C receive twice the midpoint impulse.
The force is Gauss-integrated along Y0 -> Y1, including the same patch energy.
Forces and tangents derive from one quadrature incremental potential. Quadratic
patch and quartic fiber energy increments integrate exactly; Hencky quadrature
error is measured, not removed by an offset. No added damping or mass shift.
"""
import numpy as np
import scipy.linalg as la
from .carrier_joint import CarrierJointSolver,Geometry,State,gradient
from .unresolved_velocity import maps as apic_maps,pack,unpack
SITES=(.5-np.sqrt(3)/6,.5+np.sqrt(3)/6)


def trial_terms(energy,Y,z,g,u,dt,tangent=False):
    Q=g.Q;W=Q@u;Jr=g.J@Q;d=Jr@u-z;q=g.metric
    value=float(np.sum(q[:,None]*d*d));r=2*Jr.T@(q[:,None]*d)
    force=np.zeros_like(Y);H=2*la.block_diag(*([Jr.T@(q[:,None]*Jr)]*3)) if tangent else None
    for a in SITES:
        at=Y+a*dt*W;v=energy.evaluate(at)
        value+=.5/a*v['U'];force+=.5*v['force']
        if tangent:H+=.5*a*dt*dt*energy.tangent(at,Q)
    r+=dt*Q.T@force
    return value,r,H,force,W


class CarrierAVFSolver(CarrierJointSolver):
    def __init__(self,state,energy,m,h,moving=False,clamped=True):
        super().__init__(state,energy,m,h,mode='joint',moving=moving,clamped=clamped)

    def step(self,dt,max_iters=15):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        s=self.state;e=self.energy;g=Geometry(s,e,self.m,self.h,self.clamped) if self.moving else self.geometry
        z=pack(s.v,s.C);q=g.metric;Q=g.Q;Jr=g.J@Q;n=Q.shape[1];initial=e.evaluate(s.Y)
        Uj,sj,_=la.svd(np.sqrt(q)[:,None]*Jr,full_matrices=False);Uj=Uj[:,sj>1e-12*sj[0]]
        projected=(Uj@(Uj.T@(np.sqrt(q)[:,None]*z)))/np.sqrt(q)[:,None]
        Paff=np.column_stack((np.ones(e.n),(s.Y-s.Y.mean(0))/self.h))
        coeff=la.lstsq(np.sqrt(q)[:,None]*(g.J@Paff),np.sqrt(q)[:,None]*z,cond=1e-12)[0]
        u=Q.T@(Paff@coeff);backtracks=0;accepted=False
        for iteration in range(max_iters+1):
            val,r,_,force,W=trial_terms(e,s.Y,z,g,u,dt);rn=float(la.norm(r))
            if rn<=max(1e-14,dt*1e-9):accepted=True;break
            if iteration==max_iters:break
            H=trial_terms(e,s.Y,z,g,u,dt,True)[2]
            du=la.solve(H,-r.T.ravel(),assume_a='sym').reshape(3,n).T;descent=float(np.sum(r*du))
            if not descent<0:raise ValueError('non-descent unmodified AVF tangent')
            for ls in range(25):
                alpha=2.**(-ls)
                try:nval=trial_terms(e,s.Y,z,g,u+alpha*du,dt)[0]
                except ValueError:continue
                if nval<=val+1e-4*alpha*descent+1e-18:
                    u+=alpha*du;backtracks+=ls;break
            else:raise RuntimeError('AVF line search failed')
        if not accepted:raise RuntimeError(f'AVF Newton did not converge: {rn}')
        Y=s.Y+dt*W;new=e.evaluate(Y);zp=z+2*(Jr@u-projected);vp,Cp=unpack(zp)
        x=s.x+dt*g.T@W if self.moving else s.x.copy()
        qnext=apic_maps(x,self.m,self.h)['metric'] if self.moving else q
        K0=.5*float(np.sum(q[:,None]*z*z));Kf=.5*float(np.sum(q[:,None]*zp*zp));K1=.5*float(np.sum(qnext[:,None]*zp*zp))
        dy=Y-s.Y;work=float(np.sum(force*dy));defect=Kf-K0+work;quadrature_error=new['U']-initial['U']-work
        row=dict(step=self.steps+1,time=s.time+dt,mode='avf_gauss2',moving=self.moving,
            material_J=new['Um'],stabilization_J=new['Us'],kinetic_J=K1,total_J=K1+new['U'],delta_total_J=K1-K0+new['U']-initial['U'],
            metric_change_J=K1-Kf,potential_quadrature_error_J=quadrature_error,kinetic_force_work_defect_J=defect,
            newton_residual=rn,newton_iterations=iteration,line_search_backtracks=backtracks,
            momentum_change_norm=float(la.norm(self.m@(vp-s.v))),reference_rebuild_delta_J=0.,
            history_commit_max=float(np.max(abs(new['F']-(initial['F']+gradient(e.B,dy))))),
            stress_rms_Pa=float(np.sqrt(np.mean(np.sum(new['P']**2,axis=(1,2))))),min_det_F=float(np.linalg.det(new['F']).min()),
            max_carrier_speed=float(np.max(la.norm(W,axis=1))),**g.info)
        self.state=State(x,Y,vp,Cp,s.time+dt);self.geometry=g;self.steps+=1
        return row
