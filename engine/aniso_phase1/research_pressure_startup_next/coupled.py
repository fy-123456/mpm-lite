"""Solid AVF with explicitly registered pressure theta and separate numerical dissipation.

Retains full mass/material, current tensor H, pressure-work cancellation and owned transactions.
"""
import copy
import numpy as np
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianGeometry as ParentGeometry
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import ScaledTensorCellAVF,balanced_solve
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from engine.aniso_phase1.research_phase_stress_next.coupled import ExplicitGridCoupling
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_d.identity import digest


def time_tolerance(*values):
    scale=max(abs(float(x)) for x in values)
    return 8*abs(float(np.spacing(scale)))


def residual_budgets(model,h,capacity,V0,p0,work,external,window):
    eps=np.finfo(float).eps
    scale=2*abs(model.M)@(abs(work['W'])+abs(work['v0']))+h*(abs(work['path']['force'])+abs(work['pressure_force'])+abs(external))
    solid=h*1e-7+64*eps*scale
    mass=1e-10*(V0/V0.sum())*(h/window)+64*eps*(.8*V0+capacity*(abs(p0)+abs(work['p1'])))
    darcy=np.full(len(work['z']),1e-10)
    return np.r_[solid[model.free].ravel(),mass,darcy]


class OwnedGeometry(ParentGeometry):
    def evaluate(self,q):
        # A caller owns its returned arrays; neither dict assignment nor an
        # in-place write can corrupt a subsequent cache hit or another model.
        value=super().evaluate(q)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}


class StartupAVF(ScaledTensorCellAVF):
    def __init__(self,model,config,*,cuts,window,alpha=.8,storage=.0002,mobility=MOBILITY,fixed_solid=False,pressure0=.01,reservoir=.002,state=None,theta_schedule=()):
        if model.boundary.hold!=0:raise ValueError('zero fixed grips required')
        if not all(np.isfinite(x) for x in (alpha,storage,reservoir,window)) or alpha!=.8 or storage<=0 or window<=0:raise ValueError('invalid registered fluid parameters')
        if fixed_solid:raise ValueError('this research adapter validates actual two-way equations')
        self.thetas=np.array(theta_schedule,dtype=float,copy=True)
        if self.thetas.ndim!=1 or not len(self.thetas) or not np.isin(self.thetas,[.5,1.]).all():raise ValueError('registered pressure theta schedule required')
        self.thetas.setflags(write=False)
        self.model=model;self.config=copy.deepcopy(config);self.alpha=float(alpha);self.storage=float(storage);self.drained=True;self.fixed_solid=False;self.reservoir=float(reservoir);self.window=float(window)
        self.geometry=OwnedGeometry(model,cuts,mobility);self.cells=self.geometry.cells;self.capacity=storage*self.geometry.V0;self.active=np.arange(self.geometry.topology.nflux);self.B=self.geometry.B;self.nflux=len(self.active);self.gb=self.geometry.topology.boundary_term(reservoir)
        self.identity=dict(schema='pressure-startup-cartesian-AVF-v1',theta_schedule=self.thetas.tolist(),solid=model.identity,mass=digest(model.M.tolist()),geometry=self.geometry.identity,alpha=alpha,storage=storage,boundary_pressure_Pa=reservoir,window_s=window,linear_solver='row-column general LU, original system residual',residual_policy='h*1e-7N + roundoff; time/volume allocated mass; 1e-10Pa Darcy',time='unchanged AVF solid, registered pressure theta, current full tensor H, discrete volume gradient')
        self.signature=digest(self.identity)
        if state is None:
            state=model.rest();p=np.broadcast_to(pressure0,(self.cells,)).astype(float).copy();content=alpha*(self.geometry.evaluate(state.q)['volume']-self.geometry.V0)+self.capacity*p
            state.child_states['fluid']=dict(model=self.signature,pressure_Pa=p.tolist(),flux_interval_m3_s=np.zeros(self.nflux).tolist(),content_m3=content.tolist(),cumulative_boundary_m3=0.,cumulative_source_m3=np.zeros(self.cells).tolist(),cumulative_numerical_dissipation_J=0.)
        self.validate(state);self._transaction=StateTransaction(state,validator=self.validate);self.avf=ValidatedAVF(model,config,state)

    def validate(self,state):
        super().validate(state)
        value=state.child_states['fluid'].get('cumulative_numerical_dissipation_J',float('nan'))
        if not np.isfinite(value) or value<0 or not 0<=state.step<=len(self.thetas):
            raise ValueError('invalid pressure numerical history')

    def theta_matrix(self,h,G0,G1,H,theta):
        if theta not in (.5,1.):raise ValueError('unregistered theta')
        A=self.matrix(h,G0,G1,H);n=len(self.model.ids);c=self.cells
        A[:n,n:n+c]*=2*theta;A[n+c:,n:n+c]*=2*theta
        return A

    def _compute(self,s,h,source,external):
        theta=float(self.thetas[s.step])
        linear=[];m=self.model;n=len(m.ids);c=self.cells;p0=np.array(s.child_states['fluid']['pressure_Pa']);q0=s.q;v0=s.velocity
        geo0=self.geometry.evaluate(q0);initial=m.evaluate(q0);ext=np.zeros_like(q0) if external is None else np.array(external,copy=True)
        if ext.shape!=q0.shape or not np.isfinite(ext).all():raise ValueError('finite generalized external force required')
        x=np.zeros(n+c+self.nflux);x[:n]=v0[m.free].ravel();x[n:n+c]=p0+h*source/self.capacity
        def residual(x):
            W=np.zeros_like(q0);W[m.free]=x[:n].reshape(-1,3);p1=x[n:n+c];z=x[n+c:];pbar=(1-theta)*p0+theta*p1;q1=q0+h*W
            g1=self.geometry.evaluate(q1);gb=self.geometry.discrete(q0,q1);H=self.geometry.evaluate((q0+q1)/2)['H']
            path=self.avf.path(q0,W,h);pressure_force=-self.alpha*np.einsum('k,kij->ij',pbar,gb)
            solid=2*m.M@(W-v0)+h*(path['force']+pressure_force-ext)
            mass=self.alpha*(g1['volume']-geo0['volume'])+self.capacity*(p1-p0)+h*self.B@z-h*source
            darcy=H@z-self.B.T@pbar+self.gb
            work=dict(W=W,v0=v0,q1=q1,p1=p1,pbar=pbar,z=z,g1=g1,gb=gb,H=H,path=path,solid=solid,pressure_force=pressure_force)
            return np.r_[solid[m.free].ravel(),mass,darcy],work
        for it in range(12):
            res,w=residual(x);tolerances=residual_budgets(m,h,self.capacity,self.geometry.V0,p0,w,ext,self.window);norm=float(np.max(abs(res)/tolerances))
            if norm<=1:break
            if it==11:raise ValueError('dimensioned mixed residual did not converge')
            A=self.theta_matrix(h,w['gb'],w['g1']['gradient'],w['H'],theta);dx,diagnostic=balanced_solve(A,-res,atol=.01*tolerances,diagnose=not linear);linear.append(diagnostic)
            for ls in range(10):
                trial=x+2.**(-ls)*dx
                try:new,ww=residual(trial)
                except ValueError:continue
                if np.max(abs(new)/residual_budgets(m,h,self.capacity,self.geometry.V0,p0,ww,ext,self.window))<norm:x=trial;break
            else:raise ValueError('dimensioned mixed line search failed')
        q1=w['q1'];W=w['W'];v1=2*W-v0;p1=w['p1'];pbar=w['pbar'];z=w['z'];final=m.evaluate(q1);dv=w['g1']['volume']-geo0['volume'];dq=q1-q0
        pressure_solid=-self.alpha*pbar*np.einsum('kij,ij->k',w['gb'],dq);pressure_fluid=self.alpha*pbar*dv
        E0=m.kinetic(v0)+initial['U']+.5*np.sum(self.capacity*p0*p0);Ef=.5*np.sum(self.capacity*p1*p1);E1=m.kinetic(v1)+final['U']+Ef
        D=h*float(z@w['H']@z);external_work=h*float(np.sum(ext*W));source_work=h*float(pbar@source);boundary_work=-h*float(self.gb@z)
        path_error=final['U']-initial['U']-h*float(np.sum(w['path']['force']*W));solve_work=float(np.sum(w['solid']*W)+pbar@res[n:n+c]+h*z@res[n+c:])
        Dnum=float((theta-.5)*np.sum(self.capacity*(p1-p0)**2))
        balance=E1-E0+D+Dnum-external_work-source_work-boundary_work;closure=balance-path_error-solve_work;scale=max(abs(E0),abs(E1),abs(source_work),abs(boundary_work),1e-8)
        if abs(closure)>1e-10+.01*scale or abs(balance)>1e-9+.01*scale:raise ValueError('mixed energy budget failed')
        if D<0 or np.max(abs(pressure_solid+pressure_fluid))>1e-10:raise ValueError('pressure work or dissipation failed')
        if not np.isfinite(p1).all() or p1.min()<0:raise ValueError('positive-pressure fixture undershot zero')
        candidate=s.clone();candidate.q=q1;candidate.velocity=v1;candidate.predictor=W.copy();candidate.time+=h;candidate.step+=1
        content=self.alpha*(w['g1']['volume']-self.geometry.V0)+self.capacity*p1
        pressure_reaction=float(np.sum(w['pressure_force']*m.boundary.unit));inertial_reaction=float(np.sum((2*m.M@(W-v0))*m.boundary.unit)/h);skeleton_reaction=float(np.sum(w['path']['force']*m.boundary.unit))
        row=dict(step=candidate.step,time=candidate.time,dt=h,iterations=it,true_scaled_residual=norm,linear_scaling=linear,
            solid_force_residual_N=float(np.max(abs(res[:n]))/h),solid_impulse_residual_N_s=float(np.max(abs(res[:n]))),residual_budgets=tolerances.tolist(),
            pressure_force_max_N=float(np.max(abs(w['pressure_force'][m.free]))),pressure_impulse_over_old_atol=float(np.max(abs(h*w['pressure_force'][m.free]))/1e-10),
            pressure_Pa=p1.tolist(),flux_interval_m3_s=z.tolist(),content_m3=content.tolist(),cell_volume_m3=w['g1']['volume'].tolist(),delta_volume_m3=dv.tolist(),mass_defect_m3=res[n:n+c].tolist(),darcy_residual_Pa=res[n+c:].tolist(),
            pressure_solid_work_J=pressure_solid.tolist(),pressure_fluid_work_J=pressure_fluid.tolist(),pressure_work_defect_J=(pressure_solid+pressure_fluid).tolist(),
            solid_material_J=final['material_U'],stabilization_J=final['stabilization_U'],kinetic_J=m.kinetic(v1),fluid_storage_J=float(Ef),total_energy_J=float(E1),darcy_dissipation_J=D,numerical_dissipation_J=Dnum,theta=theta,physical_energy_balance_J=float(balance-Dnum),external_work_J=external_work,source_work_J=source_work,reservoir_work_J=boundary_work,energy_balance_J=float(balance),path_error_J=float(path_error),solve_work_J=solve_work,ledger_closure_J=float(closure),min_detF=min(final['min_detF'],w['path']['min_detF'],w['g1']['min_detF']),reaction_N=pressure_reaction+inertial_reaction+skeleton_reaction,pressure_reaction_N=pressure_reaction,inertial_reaction_N=inertial_reaction,skeleton_reaction_N=skeleton_reaction,endpoint_impulse_N_s=0.,fixed_zero_grips=True)
        f=s.child_states['fluid'];candidate.child_states['fluid']=dict(model=self.signature,pressure_Pa=p1.tolist(),flux_interval_m3_s=z.tolist(),content_m3=content.tolist(),cumulative_boundary_m3=f['cumulative_boundary_m3']+h*float(np.sum(self.B@z)),cumulative_source_m3=(np.array(f['cumulative_source_m3'])+h*source).tolist(),cumulative_numerical_dissipation_J=float(f['cumulative_numerical_dissipation_J']+Dnum),last_ledger=copy.deepcopy(row))
        return candidate,row


class StartupCoupling(ExplicitGridCoupling):
    def __init__(self,model,config,times,*,source_m3_s=None,state=None,method="backward-euler",**kwargs):
        ts=np.array(times,dtype=float,copy=True)
        if ts.ndim!=1 or len(ts)<2 or ts[0]!=0 or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):raise ValueError('increasing finite zero-origin times required')
        from .theta import schedule
        thetas=schedule(ts,method)
        self.times=ts;self.times.setflags(write=False);self.core=StartupAVF(model,config,window=float(ts[-1]),state=state,theta_schedule=thetas,**kwargs)
        self.source=self.core.geometry.topology.source() if source_m3_s is None else np.broadcast_to(source_m3_s,(self.core.cells,)).astype(float).copy()
        if not np.isfinite(self.source).all():raise ValueError('finite source required')
        self.source.setflags(write=False);self.identity=dict(schema='pressure-startup-exact-grid-v1',method=method,core=self.core.identity,times_s=ts.tolist(),source_m3_s=self.source.tolist(),time_validation='8 ULP and exact frozen index');self.signature=digest(self.identity)
        if state is None:
            owned=self.core.state;owned.child_states['explicit_pressure_grid']=self.signature
            self.core._transaction=StateTransaction(owned,validator=self.validate)
        self.validate(self.core.state)

    def validate(self,state):
        self.core.validate(state)
        numerical=state.child_states['fluid'].get('cumulative_numerical_dissipation_J',float('nan'))
        if not np.isfinite(numerical) or numerical<0:raise ValueError('invalid cumulative numerical dissipation')
        if state.child_states.get('explicit_pressure_grid')!=self.signature:raise ValueError('foreign pressure grid/source history')
        if state.step<0 or state.step>=len(self.times):raise ValueError('invalid pressure grid index')
        expected=self.times[state.step]
        if abs(state.time-expected)>time_tolerance(expected,state.time):raise ValueError('checkpoint time differs from frozen index')

    def step(self,dt=None,*,inject=None,external_force=None):
        self.validate(self.state);index=self.state.step
        if index>=len(self.times)-1:raise ValueError('pressure grid exhausted')
        h=float(self.times[index+1]-self.times[index])
        if dt is not None and (not np.isfinite(dt) or abs(dt-h)>time_tolerance(h,dt)):raise ValueError('dt differs from actual frozen interval')
        def check(stage,state):
            if inject is not None:inject(stage,state)
            self.validate(state)
        return self.core.step(h,source_m3_s=self.source,external_force=external_force,inject=check)
