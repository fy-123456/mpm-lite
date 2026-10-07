"""Equivalent RT0 tensor assembly using scalar axis basis and batched products.

Only H assembly changes. Volume gradients, current F, mixed AVF, scaling,
owned state, geometry cache lifetime and physical identities are unchanged.
"""
import copy
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_cross_direction_next.rt0 import BoundedTopology,BoundedGeometry,BoundedAVF,BoundedGridCoupling,MOBILITY
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_phase_stress_next.coupled import ExplicitGridCoupling

def det3(F):
    """Algebraic 3x3 determinant at current F; no state cache or approximation."""
    return F[:,0,0]*(F[:,1,1]*F[:,2,2]-F[:,1,2]*F[:,2,1])-F[:,0,1]*(F[:,1,0]*F[:,2,2]-F[:,1,2]*F[:,2,0])+F[:,0,2]*(F[:,1,0]*F[:,2,1]-F[:,1,1]*F[:,2,0])

class FastTopology(BoundedTopology):
    def assemble(self,X,weights,cell_ids,F,mobility=MOBILITY):
        X=np.asarray(X);weights=np.asarray(weights);cell_ids=np.asarray(cell_ids);F=np.asarray(F);mobility=np.asarray(mobility)
        if mobility.shape!=(3,3) or not np.isfinite(mobility).all() or not np.allclose(mobility,mobility.T) or la.eigvalsh(mobility)[0]<=0:raise ValueError('SPD full mobility required')
        if X.shape!=(len(weights),3) or F.shape!=(len(weights),3,3) or cell_ids.shape!=weights.shape or not np.isfinite(X).all() or not np.isfinite(F).all() or not np.isfinite(weights).all() or np.any(weights<0) or not np.issubdtype(cell_ids.dtype,np.integer) or np.any((cell_ids<0)|(cell_ids>=self.cells)):raise ValueError('invalid RT0 integration arrays')
        invk=la.solve(mobility,np.eye(3),assume_a='pos');H=np.zeros((self.nflux,self.nflux));minJ=float('inf');axes=np.repeat(np.arange(3),2)
        for cell in range(self.cells):
            indices=np.flatnonzero(cell_ids==cell);local=np.zeros((6,6));bounds=self.cell_bounds[cell]
            if not len(indices):raise ValueError('each pressure cell requires integration points')
            ends=bounds[axes,np.tile([1,0],3)]
            for start in range(0,len(indices),16384):
                ix=indices[start:start+16384];f=F[ix];J=det3(f);minJ=min(minJ,float(J.min()))
                if np.any(J<=.1):raise ValueError('RT0 geometry leaves positive-J range')
                # Exact same F^T mobility^-1 F / J, evaluated as two batched products.
                invK=(np.swapaxes(f,1,2)@(invk@f))/J[:,None,None]
                # Each RT0 face basis has exactly one nonzero Cartesian component.
                phi=(X[ix][:,axes]-ends)/self.V0[cell]*self.signs
                w=weights[ix]
                for a in range(6):
                    wa=w*phi[:,a]
                    for b in range(a,6):
                        value=float(np.dot(wa*phi[:,b],invK[:,axes[a],axes[b]]));local[a,b]+=value
                        if a!=b:local[b,a]+=value
            face=self.faces[cell];H[np.ix_(face,face)]+=local
        return H,minJ

class FastGeometry(BoundedGeometry):
    def __init__(self,model,cells=2,mobility=MOBILITY,order=None):
        super().__init__(model,cells,mobility,order)
        self.topology=FastTopology(self.topology.bounds,cells)

class FastAVF(BoundedAVF):
    """Reuse the general residual/transaction with a new authenticated RT0 model."""
    def __init__(self,model,config,*,cells=2,alpha=.8,storage=.0002,mobility=MOBILITY,fixed_solid=False,pressure0=.01,reservoir=.002,state=None):
        if model.boundary.hold!=0:raise ValueError('fixed zero grips required')
        if not all(np.isfinite(x) for x in (alpha,storage,reservoir)) or not 0<=alpha<=1 or storage<=0:raise ValueError('invalid material parameters')
        self.model=model;self.config=copy.deepcopy(config);self.alpha=float(alpha);self.storage=float(storage);self.drained=True;self.fixed_solid=bool(fixed_solid);self.reservoir=float(reservoir)
        self.geometry=FastGeometry(model,cells,mobility);self.cells=cells;self.capacity=storage*self.geometry.V0;self.active=np.arange(self.geometry.topology.nflux);self.B=self.geometry.B;self.nflux=self.geometry.topology.nflux;self.gb=self.geometry.topology.boundary_term(reservoir)
        self.identity=dict(schema='observable-bounded-scaled-RT0-AVF-v1',cells=cells,linear_solver='positive row/column max equilibration, general LU, original residual',solid=model.identity,mass=digest(model.M.tolist()),geometry=self.geometry.identity,alpha=alpha,storage=storage,
            fixed_solid=fixed_solid,boundary_pressure_Pa=reservoir,time='unchanged solid AVF, midpoint p/z, exact volume discrete gradient',H_update='midpoint current F, full tensor, true residual with general quasi-Newton solve')
        self.signature=digest(self.identity)
        if state is None:
            state=model.rest();p=np.broadcast_to(pressure0,(cells,)).astype(float).copy();content=alpha*(self.geometry.evaluate(state.q)['volume']-self.geometry.V0)+self.capacity*p
            state.child_states['fluid']=dict(model=self.signature,pressure_Pa=p.tolist(),flux_interval_m3_s=np.zeros(self.nflux).tolist(),content_m3=content.tolist(),cumulative_boundary_m3=0.,cumulative_source_m3=np.zeros(cells).tolist())
        self.validate(state);self._transaction=StateTransaction(state,validator=self.validate);self.avf=ValidatedAVF(model,config,state)

class FastGridCoupling(ExplicitGridCoupling):
    def __init__(self,model,config,times,*,source_m3_s=0.,state=None,**kwargs):
        ts=np.asarray(times,dtype=float)
        if ts.ndim!=1 or len(ts)<2 or not np.isfinite(ts).all() or ts[0]!=0 or np.any(np.diff(ts)<=0):raise ValueError('explicit positive time grid required')
        self.times=ts.copy();self.times.setflags(write=False);self.core=FastAVF(model,config,state=state,**kwargs)
        self.source=np.broadcast_to(source_m3_s,(self.core.cells,)).astype(float).copy()
        if not np.isfinite(self.source).all():raise ValueError('finite source required')
        self.source.setflags(write=False);self.identity=dict(schema='observable-bounded-RT0-explicit-grid-v1',core=self.core.identity,times_s=ts.tolist(),source_m3_s=self.source.tolist());self.signature=digest(self.identity)
        if state is None:
            s=self.core.state;s.child_states['explicit_pressure_grid']=self.signature;self.core._transaction=StateTransaction(s,validator=self.validate)
        self.validate(self.core.state)
