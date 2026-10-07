"""Bounded 2/4-cell full-tensor RT0; unchanged scaled mixed AVF equations."""
import copy
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_cost_phase_next.pressure import CellGeometry
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import ScaledTensorCellAVF as ParentAVF
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_phase_stress_next.coupled import ExplicitGridCoupling

class BoundedTopology:
    """All global normals point along the positive coordinate axis."""
    def __init__(self,bounds,cells=2):
        if isinstance(cells,bool) or not isinstance(cells,(int,np.integer)) or cells not in (2,4):raise ValueError("only 2/4 aligned pressure cells supported")
        self.bounds=np.array(bounds,dtype=float,copy=True)
        if self.bounds.shape!=(3,2) or not np.isfinite(self.bounds).all() or np.any(np.diff(self.bounds,axis=1)<=0):raise ValueError('positive rectangular reference bounds required')
        self.cuts=np.linspace(*self.bounds[0],cells+1);self.cells=cells;self.nflux=5*cells+1
        self.faces=np.array([[k,k+1,*range(cells+1+4*k,cells+5+4*k)] for k in range(cells)])
        self.signs=np.array([-1.,1.,-1.,1.,-1.,1.]);self.B=np.zeros((cells,self.nflux));self.centres=np.zeros((self.nflux,3));self.axes=np.zeros(self.nflux,dtype=int);self.areas=np.zeros(self.nflux)
        self.V0=np.zeros(cells);self.cell_bounds=[]
        for cell in range(self.cells):
            bounds=self.bounds.copy();bounds[0]=self.cuts[cell:cell+2];self.cell_bounds.append(bounds);width=np.diff(bounds,axis=1).ravel();self.V0[cell]=np.prod(width)
            for local,face in enumerate(self.faces[cell]):
                axis=local//2;centre=bounds.mean(axis=1);centre[axis]=bounds[axis,local%2]
                self.centres[face]=centre;self.axes[face]=axis;self.areas[face]=np.prod(np.delete(width,axis));self.B[cell,face]=self.signs[local]
        self.boundary=np.where(np.count_nonzero(self.B,axis=0)==1)[0];self.internal=np.where(np.count_nonzero(self.B,axis=0)==2)[0]
        self.boundary_sign=self.B.sum(axis=0)
        for value in (self.bounds,self.cuts,self.faces,self.signs,self.B,self.centres,self.axes,self.areas,self.V0,self.boundary,self.internal,self.boundary_sign):value.setflags(write=False)

    def basis(self,cell,X):
        X=np.asarray(X);b=self.cell_bounds[cell];phi=np.zeros((len(X),6,3))
        for axis in range(3):
            phi[:,2*axis,axis]=(X[:,axis]-b[axis,1])/self.V0[cell]
            phi[:,2*axis+1,axis]=(X[:,axis]-b[axis,0])/self.V0[cell]
        return phi

    def assemble(self,X,weights,cell_ids,F,mobility=MOBILITY):
        X=np.asarray(X);weights=np.asarray(weights);cell_ids=np.asarray(cell_ids);F=np.asarray(F);mobility=np.asarray(mobility)
        if mobility.shape!=(3,3) or not np.isfinite(mobility).all() or not np.allclose(mobility,mobility.T) or la.eigvalsh(mobility)[0]<=0:raise ValueError('SPD full mobility required')
        if X.shape!=(len(weights),3) or F.shape!=(len(weights),3,3) or cell_ids.shape!=weights.shape or not np.isfinite(X).all() or not np.isfinite(F).all() or not np.isfinite(weights).all() or np.any(weights<0) or np.any((cell_ids<0)|(cell_ids>=self.cells)):raise ValueError('invalid RT0 integration arrays')
        invk=la.solve(mobility,np.eye(3),assume_a='pos');H=np.zeros((self.nflux,self.nflux));minJ=float('inf')
        for cell in range(self.cells):
            indices=np.flatnonzero(cell_ids==cell);local=np.zeros((6,6))
            for start in range(0,len(indices),16384):
                ix=indices[start:start+16384];f=F[ix];J=np.linalg.det(f);minJ=min(minJ,float(J.min()))
                if np.any(J<=.1):raise ValueError('RT0 geometry leaves positive-J range')
                invK=np.einsum('nki,kl,nlj->nij',f,invk,f)/J[:,None,None];phi=self.basis(cell,X[ix])
                local+=np.einsum('n,nai,nij,nbj->ab',weights[ix],phi,invK,phi,optimize=True)
            face=self.faces[cell];H[np.ix_(face,face)]+=self.signs[:,None]*local*self.signs[None,:]
        return .5*(H+H.T),minJ

    def boundary_term(self,pressure):
        p=np.broadcast_to(pressure,(self.nflux,)).astype(float)
        if not np.isfinite(p).all():raise ValueError('finite boundary pressure required')
        return self.boundary_sign*p

class BoundedGeometry(CellGeometry):
    def __init__(self,model,cells=2,mobility=MOBILITY,order=None):
        super().__init__(model,cells,.1,order)
        self.topology=BoundedTopology([[e[0],e[-1]] for e in model.parent.edges],cells);self.mobility=np.array(mobility,dtype=float,copy=True);self.mobility.setflags(write=False)
        self.B=self.topology.B;self.V0=self.topology.V0
        self.X=np.stack(np.meshgrid(*self.points,indexing='ij'),axis=-1).reshape(-1,3);self.total_weights=np.sum(self.weights,axis=0)
        self.identity=dict(schema='observable-bounded-full-tensor-RT0-v1',cells=cells,solid=model.reduction.signature,bounds=self.topology.bounds.tolist(),faces=self.topology.faces.tolist(),B=self.B.tolist(),
            global_orientation='positive x,y,z; local outward signs -,+,-,+,-,+',mobility=self.mobility.tolist(),geometry_order=self.order,
            integration_points_sha256=digest([x.tolist() for x in self.points]),volume_gradient='inherited exact detF volume / original independent coordinates',H='integral phi^T (F^T mobility^-1 F / J) phi')

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        value=super().evaluate(q)
        try:
            H,minJ=self.topology.assemble(self.X,self.total_weights,self.cell_ids,self.field(q).numpy(),self.mobility)
            value=dict(value,H=H,min_detF=min(value['min_detF'],minJ));self.cache[key]=value;return value
        except Exception:
            self.cache.pop(key,None);raise

class BoundedAVF(ParentAVF):
    """Reuse the general residual/transaction with a new authenticated RT0 model."""
    def __init__(self,model,config,*,cells=2,alpha=.8,storage=.0002,mobility=MOBILITY,fixed_solid=False,pressure0=.01,reservoir=.002,state=None):
        if model.boundary.hold!=0:raise ValueError('fixed zero grips required')
        if not all(np.isfinite(x) for x in (alpha,storage,reservoir)) or not 0<=alpha<=1 or storage<=0:raise ValueError('invalid material parameters')
        self.model=model;self.config=copy.deepcopy(config);self.alpha=float(alpha);self.storage=float(storage);self.drained=True;self.fixed_solid=bool(fixed_solid);self.reservoir=float(reservoir)
        self.geometry=BoundedGeometry(model,cells,mobility);self.cells=cells;self.capacity=storage*self.geometry.V0;self.active=np.arange(self.geometry.topology.nflux);self.B=self.geometry.B;self.nflux=self.geometry.topology.nflux;self.gb=self.geometry.topology.boundary_term(reservoir)
        self.identity=dict(schema='observable-bounded-scaled-RT0-AVF-v1',cells=cells,linear_solver='positive row/column max equilibration, general LU, original residual',solid=model.identity,mass=digest(model.M.tolist()),geometry=self.geometry.identity,alpha=alpha,storage=storage,
            fixed_solid=fixed_solid,boundary_pressure_Pa=reservoir,time='unchanged solid AVF, midpoint p/z, exact volume discrete gradient',H_update='midpoint current F, full tensor, true residual with general quasi-Newton solve')
        self.signature=digest(self.identity)
        if state is None:
            state=model.rest();p=np.broadcast_to(pressure0,(cells,)).astype(float).copy();content=alpha*(self.geometry.evaluate(state.q)['volume']-self.geometry.V0)+self.capacity*p
            state.child_states['fluid']=dict(model=self.signature,pressure_Pa=p.tolist(),flux_interval_m3_s=np.zeros(self.nflux).tolist(),content_m3=content.tolist(),cumulative_boundary_m3=0.,cumulative_source_m3=np.zeros(cells).tolist())
        self.validate(state);self._transaction=StateTransaction(state,validator=self.validate);self.avf=ValidatedAVF(model,config,state)

class BoundedGridCoupling(ExplicitGridCoupling):
    def __init__(self,model,config,times,*,source_m3_s=0.,state=None,**kwargs):
        ts=np.asarray(times,dtype=float)
        if ts.ndim!=1 or len(ts)<2 or not np.isfinite(ts).all() or ts[0]!=0 or np.any(np.diff(ts)<=0):raise ValueError('explicit positive time grid required')
        self.times=ts.copy();self.times.setflags(write=False);self.core=BoundedAVF(model,config,state=state,**kwargs)
        self.source=np.broadcast_to(source_m3_s,(self.core.cells,)).astype(float).copy()
        if not np.isfinite(self.source).all():raise ValueError('finite source required')
        self.source.setflags(write=False);self.identity=dict(schema='observable-bounded-RT0-explicit-grid-v1',core=self.core.identity,times_s=ts.tolist(),source_m3_s=self.source.tolist());self.signature=digest(self.identity)
        if state is None:
            s=self.core.state;s.child_states['explicit_pressure_grid']=self.signature;self.core._transaction=StateTransaction(s,validator=self.validate)
        self.validate(self.core.state)
