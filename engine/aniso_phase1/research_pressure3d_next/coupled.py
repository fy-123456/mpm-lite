"""Inject bounded geometry before validation; inherit the full theta/AVF equations."""
import copy
import numpy as np
from engine.aniso_phase1.research_pressure_startup_next.coupled import StartupAVF, StartupCoupling
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from .geometry import BlockGeometry

class BlockAVF(StartupAVF):
    def __init__(self,model,config,*,cuts,window,alpha=.8,storage=.0002,mobility=MOBILITY,fixed_solid=False,pressure0=.01,reservoir=.002,state=None,theta_schedule=()):
        if model.boundary.hold!=0:raise ValueError('zero fixed grips required')
        if not all(np.isfinite(x) for x in (alpha,storage,reservoir,window)) or alpha!=.8 or storage<=0 or window<=0:raise ValueError('invalid registered fluid parameters')
        if fixed_solid:raise ValueError('this research adapter validates actual two-way equations')
        self.thetas=np.array(theta_schedule,dtype=float,copy=True)
        if self.thetas.ndim!=1 or not len(self.thetas) or not np.isin(self.thetas,[.5,1.]).all():raise ValueError('registered pressure theta schedule required')
        self.thetas.setflags(write=False)
        self.model=model;self.config=copy.deepcopy(config);self.alpha=float(alpha);self.storage=float(storage);self.drained=True;self.fixed_solid=False;self.reservoir=float(reservoir);self.window=float(window)
        self.geometry=BlockGeometry(model,cuts,mobility);self.cells=self.geometry.cells;self.capacity=storage*self.geometry.V0;self.active=np.arange(self.geometry.topology.nflux);self.B=self.geometry.B;self.nflux=len(self.active);self.gb=self.geometry.topology.boundary_term(reservoir)
        self.identity=dict(schema='pressure3d-injected-AVF-v1',theta_schedule=self.thetas.tolist(),solid=model.identity,mass=digest(model.M.tolist()),geometry=self.geometry.identity,alpha=alpha,storage=storage,boundary_pressure_Pa=reservoir,window_s=window,linear_solver='row-column general LU, original system residual',residual_policy='h*1e-7N + roundoff; time/volume allocated mass; 1e-10Pa Darcy',time='unchanged AVF solid, registered pressure theta, current full tensor H, discrete volume gradient')
        self.signature=digest(self.identity)
        if state is None:
            state=model.rest();p=np.broadcast_to(pressure0,(self.cells,)).astype(float).copy();content=alpha*(self.geometry.evaluate(state.q)['volume']-self.geometry.V0)+self.capacity*p
            state.child_states['fluid']=dict(model=self.signature,pressure_Pa=p.tolist(),flux_interval_m3_s=np.zeros(self.nflux).tolist(),content_m3=content.tolist(),cumulative_boundary_m3=0.,cumulative_source_m3=np.zeros(self.cells).tolist(),cumulative_numerical_dissipation_J=0.)
        self.validate(state);self._transaction=StateTransaction(state,validator=self.validate);self.avf=ValidatedAVF(model,config,state)


class BlockCoupling(StartupCoupling):
    def __init__(self,model,config,times,*,source_m3_s=0.,state=None,method="backward-euler",**kwargs):
        from engine.aniso_phase1.research_zero_source_next.coupled import explicit_zero
        source_m3_s=explicit_zero(source_m3_s)
        ts=np.array(times,dtype=float,copy=True)
        if ts.ndim!=1 or len(ts)<2 or ts[0]!=0 or not np.isfinite(ts).all() or np.any(np.diff(ts)<=0):raise ValueError('increasing finite zero-origin times required')
        from engine.aniso_phase1.research_pressure_startup_next.theta import schedule
        thetas=schedule(ts,method)
        self.times=ts;self.times.setflags(write=False);self.core=BlockAVF(model,config,window=float(ts[-1]),state=state,theta_schedule=thetas,**kwargs)
        self.source=self.core.geometry.topology.source() if source_m3_s is None else np.broadcast_to(source_m3_s,(self.core.cells,)).astype(float).copy()
        if not np.isfinite(self.source).all():raise ValueError('finite source required')
        self.source.setflags(write=False);self.identity=dict(schema='pressure3d-zero-source-exact-grid-v1',method=method,core=self.core.identity,times_s=ts.tolist(),source_m3_s=self.source.tolist(),time_validation='8 ULP and exact frozen index');self.signature=digest(self.identity)
        if state is None:
            owned=self.core.state;owned.child_states['explicit_pressure_grid']=self.signature
            self.core._transaction=StateTransaction(owned,validator=self.validate)
        self.validate(self.core.state)

