"""Actual RT0 quasi-Newton equilibration; original residual/energy/transactions retained."""
import copy
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_cost_phase_next.pressure import MultiCellAVF
from engine.aniso_phase1.research_local_span_next.rt0 import TensorCellGeometry,MOBILITY
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_phase_stress_next.coupled import ExplicitGridCoupling

def balanced_solve(A,b,*,atol=1e-13,diagnose=False):
    A=np.asarray(A,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if A.ndim!=2 or A.shape[0]!=A.shape[1] or b.shape!=(A.shape[0],) or not np.isfinite(A).all() or not np.isfinite(b).all():raise ValueError('finite square mixed system required')
    rows=np.max(abs(A),axis=1)
    if np.any(rows<=0):raise ValueError('zero mixed-system row')
    left=1/rows;RA=left[:,None]*A;cols=np.max(abs(RA),axis=0)
    if np.any(cols<=0):raise ValueError('zero mixed-system column')
    right=1/cols;balanced=RA*right[None,:]
    if not np.isfinite(balanced).all():raise ValueError('invalid equilibration scale')
    lu,piv=la.lu_factor(balanced);y=la.lu_solve((lu,piv),left*b);x=right*y
    residual=A@x-b;budget=np.asarray(atol)+1e-9*(abs(A)@abs(x)+abs(b))
    fraction=float(np.max(abs(residual)/budget))
    if not np.isfinite(x).all() or not np.isfinite(fraction) or fraction>1:raise ValueError('scaled solve failed original-system residual')
    info=dict(strategy='row-column-max-general-LU',original_linear_residual_budget_fraction=fraction,row_scale_range=[float(left.min()),float(left.max())],column_scale_range=[float(right.min()),float(right.max())])
    if diagnose:
        gecon=la.get_lapack_funcs('gecon',(balanced,));raw_lu,_=la.lu_factor(A)
        info['raw_rcond_1']=float(gecon(raw_lu,la.norm(A,1))[0]);info['scaled_rcond_1']=float(gecon(lu,la.norm(balanced,1))[0])
    return x,info

class ScaledTensorCellAVF(MultiCellAVF):
    """Reuse the general residual/transaction with a new authenticated RT0 model."""
    def __init__(self,model,config,*,alpha=.8,storage=.0002,mobility=MOBILITY,fixed_solid=False,pressure0=.01,reservoir=.002,state=None):
        if model.boundary.hold!=0:raise ValueError('fixed zero grips required')
        if not all(np.isfinite(x) for x in (alpha,storage,reservoir)) or not 0<=alpha<=1 or storage<=0:raise ValueError('invalid material parameters')
        self.model=model;self.config=copy.deepcopy(config);self.alpha=float(alpha);self.storage=float(storage);self.drained=True;self.fixed_solid=bool(fixed_solid);self.reservoir=float(reservoir)
        self.geometry=TensorCellGeometry(model,mobility);self.cells=2;self.capacity=storage*self.geometry.V0;self.active=np.arange(11);self.B=self.geometry.B;self.nflux=11;self.gb=self.geometry.topology.boundary_term(reservoir)
        self.identity=dict(schema='phase-reference-scaled-two-cell-RT0-AVF-v1',linear_solver='positive row/column max equilibration, general LU, original residual',solid=model.identity,mass=digest(model.M.tolist()),geometry=self.geometry.identity,alpha=alpha,storage=storage,
            fixed_solid=fixed_solid,boundary_pressure_Pa=reservoir,time='unchanged solid AVF, midpoint p/z, exact volume discrete gradient',H_update='midpoint current F, full tensor, true residual with general quasi-Newton solve')
        self.signature=digest(self.identity)
        if state is None:
            state=model.rest();p=np.broadcast_to(pressure0,(2,)).astype(float).copy();content=alpha*(self.geometry.evaluate(state.q)['volume']-self.geometry.V0)+self.capacity*p
            state.child_states['fluid']=dict(model=self.signature,pressure_Pa=p.tolist(),flux_interval_m3_s=np.zeros(11).tolist(),content_m3=content.tolist(),cumulative_boundary_m3=0.,cumulative_source_m3=np.zeros(2).tolist())
        self.validate(state);self._transaction=StateTransaction(state,validator=self.validate);self.avf=ValidatedAVF(model,config,state)

    def _compute(self,s,h,source,external):
        linear=[];m=self.model;n=len(m.ids);c=self.cells;p0=np.array(s.child_states['fluid']['pressure_Pa']);q0=s.q;v0=s.velocity
        geo0=self.geometry.evaluate(q0);initial=m.evaluate(q0);ext=np.zeros_like(q0) if external is None else np.array(external,copy=True)
        if ext.shape!=q0.shape or not np.isfinite(ext).all():raise ValueError('finite generalized external force required')
        x=np.zeros(n+c+self.nflux);x[:n]=0 if self.fixed_solid else v0[m.free].ravel();x[n:n+c]=p0+h*source/self.capacity
        tolerances=np.r_[np.full(n,1e-10),np.full(c,1e-12),np.full(self.nflux,1e-10)]
        def residual(x):
            W=np.zeros_like(q0);W[m.free]=x[:n].reshape(-1,3);p1=x[n:n+c];z=x[n+c:];pbar=.5*(p0+p1);q1=q0+h*W
            g1=self.geometry.evaluate(q1);gb=self.geometry.discrete(q0,q1);mid=self.geometry.evaluate((q0+q1)/2);H=mid['H'][np.ix_(self.active,self.active)]
            path=self.avf.path(q0,W,h);pressure_force=-self.alpha*np.einsum('k,kij->ij',pbar,gb)
            solid=2*m.M@(W-v0)+h*(path['force']+pressure_force-ext)
            mass=self.alpha*(g1['volume']-geo0['volume'])+self.capacity*(p1-p0)+h*self.B@z-h*source
            darcy=H@z-self.B.T@pbar+self.gb
            return np.r_[W[m.free].ravel() if self.fixed_solid else solid[m.free].ravel(),mass,darcy],dict(W=W,q1=q1,p1=p1,pbar=pbar,z=z,g1=g1,gb=gb,H=H,path=path,solid=solid)
        for it in range(12):
            res,w=residual(x);norm=float(np.max(abs(res)/tolerances))
            if norm<=1:break
            if it==11:raise ValueError('multicell true residual did not converge')
            A=self.matrix(h,w['gb'],w['g1']['gradient'],w['H']);dx,diagnostic=balanced_solve(A,-res,atol=.01*tolerances,diagnose=not linear);linear.append(diagnostic)
            for ls in range(10):
                trial=x+2.**(-ls)*dx
                try:new,_=residual(trial)
                except ValueError:continue
                if np.max(abs(new)/tolerances)<norm:x=trial;break
            else:raise ValueError('multicell line search failed')
        q1=w['q1'];W=w['W'];v1=2*W-v0;p1=w['p1'];pbar=w['pbar'];z=w['z'];final=m.evaluate(q1)
        dv=w['g1']['volume']-geo0['volume'];dq=q1-q0
        pressure_solid=-self.alpha*pbar*np.einsum('kij,ij->k',w['gb'],dq);pressure_fluid=self.alpha*pbar*dv
        E0=m.kinetic(v0)+initial['U']+.5*np.sum(self.capacity*p0*p0);Ef=.5*np.sum(self.capacity*p1*p1);E1=m.kinetic(v1)+final['U']+Ef
        D=h*float(z@w['H']@z);external_work=h*float(np.sum(ext*W));source_work=h*float(pbar@source);boundary_work=-h*float(self.gb@z)
        path_error=final['U']-initial['U']-h*float(np.sum(w['path']['force']*W))
        solve_work=float(np.sum(w['solid']*W)+pbar@res[n:n+c]+h*z@res[n+c:])
        balance=E1-E0+D-external_work-source_work-boundary_work;closure=balance-path_error-solve_work
        scale=max(abs(E0),abs(E1),abs(source_work),abs(boundary_work),1e-8)
        if abs(closure)>1e-10+.01*scale or abs(balance)>1e-9+.01*scale:raise ValueError('multicell energy budget failed')
        if D<0 or np.max(abs(pressure_solid+pressure_fluid))>1e-10:raise ValueError('local pressure work or dissipation failed')
        candidate=s.clone();candidate.q=q1;candidate.velocity=v1;candidate.predictor=W.copy();candidate.time+=h;candidate.step+=1
        content=self.alpha*(w['g1']['volume']-self.geometry.V0)+self.capacity*p1
        row=dict(linear_scaling=linear,step=candidate.step,time=candidate.time,dt=h,iterations=it,true_scaled_residual=norm,
            pressure_Pa=p1.tolist(),flux_interval_m3_s=z.tolist(),content_m3=content.tolist(),mass_defect_m3=res[n:n+c].tolist(),darcy_residual_Pa=res[n+c:].tolist(),
            pressure_solid_work_J=pressure_solid.tolist(),pressure_fluid_work_J=pressure_fluid.tolist(),pressure_work_defect_J=(pressure_solid+pressure_fluid).tolist(),
            solid_material_J=final['material_U'],stabilization_J=final['stabilization_U'],kinetic_J=m.kinetic(v1),fluid_storage_J=float(Ef),total_energy_J=float(E1),
            darcy_dissipation_J=D,external_work_J=external_work,source_work_J=source_work,reservoir_work_J=boundary_work,energy_balance_J=float(balance),
            path_error_J=float(path_error),solve_work_J=solve_work,ledger_closure_J=float(closure),min_detF=min(final['min_detF'],w['path']['min_detF'],w['g1']['min_detF']))
        f=s.child_states['fluid'];candidate.child_states['fluid']=dict(model=self.signature,pressure_Pa=p1.tolist(),flux_interval_m3_s=z.tolist(),content_m3=content.tolist(),
            cumulative_boundary_m3=f['cumulative_boundary_m3']+h*float(np.sum(self.B@z)),cumulative_source_m3=(np.array(f['cumulative_source_m3'])+h*source).tolist(),last_ledger=copy.deepcopy(row))
        return candidate,row


class ScaledTensorGridCoupling(ExplicitGridCoupling):
    def __init__(self,model,config,times,*,source_m3_s=0.,state=None,**kwargs):
        ts=np.asarray(times,dtype=float)
        if ts.ndim!=1 or len(ts)<2 or not np.isfinite(ts).all() or ts[0]!=0 or np.any(np.diff(ts)<=0):raise ValueError('explicit positive time grid required')
        self.times=ts.copy();self.times.setflags(write=False);self.core=ScaledTensorCellAVF(model,config,state=state,**kwargs)
        self.source=np.broadcast_to(source_m3_s,(2,)).astype(float).copy()
        if not np.isfinite(self.source).all():raise ValueError('finite source required')
        self.source.setflags(write=False);self.identity=dict(schema='phase-reference-scaled-RT0-explicit-grid-v1',core=self.core.identity,times_s=ts.tolist(),source_m3_s=self.source.tolist());self.signature=digest(self.identity)
        if state is None:
            s=self.core.state;s.child_states['explicit_pressure_grid']=self.signature;self.core._transaction=StateTransaction(s,validator=self.validate)
        self.validate(self.core.state)
