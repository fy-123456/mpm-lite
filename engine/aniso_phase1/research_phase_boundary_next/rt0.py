"""Conservative Cartesian RT0 on explicit cuts, with the inherited mixed AVF."""
from collections import OrderedDict
import copy
import numpy as np
import scipy.linalg as la
import warp as wp
from engine.aniso_phase1.research_cross_direction_next.det_rt0 import FastTopology as ParentTopology
from engine.aniso_phase1.research_cross_direction_next.geometry import CellGeometry,cell_geometry_kernel
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import ScaledTensorCellAVF
from engine.aniso_phase1.research_phase_stress_next.coupled import ExplicitGridCoupling
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.tensor_metrics import quadrature_axis

class CartesianTopology(ParentTopology):
    def __init__(self,cuts):
        if len(cuts)!=3:raise ValueError('three explicit Cartesian cut arrays required')
        self.cuts=tuple(np.array(v,dtype=float,copy=True) for v in cuts)
        if any(v.ndim!=1 or len(v)<2 or not np.isfinite(v).all() or np.any(np.diff(v)<=0) for v in self.cuts):raise ValueError('finite strictly increasing cuts required')
        self.shape=tuple(len(v)-1 for v in self.cuts);self.cells=int(np.prod(self.shape))
        if not 1<=self.cells<=16:raise ValueError('bounded research grid supports at most sixteen cells')
        self.bounds=np.array([[x[0],x[-1]] for x in self.cuts]);self.signs=np.array([-1.,1.,-1.,1.,-1.,1.]);self.cell_bounds=[]
        self.faces=[];centres=[];axes=[];areas=[];face_ids={};volumes=[]
        for idx in np.ndindex(self.shape):
            b=np.array([self.cuts[a][idx[a]:idx[a]+2] for a in range(3)]);self.cell_bounds.append(b);width=np.diff(b,axis=1).ravel();volumes.append(float(np.prod(width)));row=[]
            for a in range(3):
                for side in (0,1):
                    key=(a,idx[a]+side,*[idx[k] for k in range(3) if k!=a])
                    if key not in face_ids:
                        face_ids[key]=len(centres);c=b.mean(axis=1);c[a]=b[a,side];centres.append(c);axes.append(a);areas.append(float(np.prod(np.delete(width,a))))
                    row.append(face_ids[key])
            self.faces.append(row)
        self.faces=np.array(self.faces,dtype=int);self.nflux=len(centres);self.centres=np.array(centres);self.axes=np.array(axes);self.areas=np.array(areas);self.V0=np.array(volumes)
        self.B=np.zeros((self.cells,self.nflux))
        for c,faces in enumerate(self.faces):self.B[c,faces]=self.signs
        self.boundary=np.where(np.count_nonzero(self.B,axis=0)==1)[0];self.internal=np.where(np.count_nonzero(self.B,axis=0)==2)[0];self.boundary_sign=self.B.sum(axis=0)
        if np.any(self.B[:,self.internal].sum(axis=0)!=0):raise ValueError('internal face must have opposite ownership')
        for v in (*self.cuts,self.bounds,self.signs,self.faces,self.centres,self.axes,self.areas,self.V0,self.B,self.boundary,self.internal,self.boundary_sign):v.setflags(write=False)

    def source(self,density=.001):
        mid=self.bounds[0].mean();out=[]
        for b in self.cell_bounds:
            width=np.diff(b,axis=1).ravel();overlap=max(0.,min(b[0,1],mid)-b[0,0]);out.append(density*overlap*width[1]*width[2])
        return np.array(out)

    def locate(self,X):
        ids=[np.clip(np.searchsorted(e,X[:,a],side='right')-1,0,len(e)-2) for a,e in enumerate(self.cuts)]
        return np.ravel_multi_index(ids,self.shape)

class CartesianGeometry(CellGeometry):
    def __init__(self,model,cuts,mobility=MOBILITY,order=7):
        self.model=model;self.op=model.operator;self.topology=CartesianTopology(cuts);self.cells=self.topology.cells;self.order=order;self.cache=OrderedDict()
        self.mobility=np.array(mobility,dtype=float,copy=True)
        if self.mobility.shape!=(3,3) or not np.isfinite(self.mobility).all() or not np.allclose(self.mobility,self.mobility.T) or la.eigvalsh(self.mobility)[0]<=0:raise ValueError('full SPD mobility required')
        self.mobility.setflags(write=False);self.V0=self.topology.V0;self.B=self.topology.B
        edges=model.parent.edges
        for a,e in enumerate(edges):
            if not np.allclose([e[0],e[-1]],self.topology.bounds[a],atol=1e-13,rtol=0):raise ValueError('pressure grid must cover the actual solid')
        # Exact intersections include every solid/pressure cell boundary on all axes.
        integration_edges=tuple(np.unique(np.r_[e,self.topology.cuts[a]]) for a,e in enumerate(edges))
        quadrature=[quadrature_axis(e,order) for e in integration_edges];self.points=tuple(v[0] for v in quadrature);self.shape=tuple(map(len,self.points));self.count=int(np.prod(self.shape))
        self.layout=self.op.maps.layout(self.points);self.X=np.stack(np.meshgrid(*self.points,indexing='ij'),axis=-1).reshape(-1,3)
        self.total_weights=(quadrature[0][1][:,None,None]*quadrature[1][1][None,:,None]*quadrature[2][1][None,None,:]).ravel();self.cell_ids=self.topology.locate(self.X)
        self.weights=[self.total_weights*(self.cell_ids==c) for c in range(self.cells)];self.gpu_weights=[wp.array(w,dtype=wp.float64,device=self.op.device) for w in self.weights]
        if not np.allclose([w.sum() for w in self.weights],self.V0,atol=1e-13,rtol=1e-12):raise ValueError('intersection quadrature does not cover cell volumes')
        self.identity=dict(schema='phase-boundary-cartesian-RT0-v1',solid=model.reduction.signature,cuts=[v.tolist() for v in self.topology.cuts],faces=self.topology.faces.tolist(),B=self.B.tolist(),mobility=self.mobility.tolist(),geometry_order=order,integration_points_sha256=digest([v.tolist() for v in self.points]),volume_gradient='original independent coordinates, exact detF discrete gradient',H='full F^T mobility^-1 F/J',orientation='positive axes, local signs -,+,-,+,-,+')

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        F=self.field(q);cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device)
        wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,unused,bad,.1],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('cell geometry leaves positive-J range')
        jj=J.numpy();V=[];G=[]
        for w,gw in zip(self.weights,self.gpu_weights):
            grad=self.op.maps.gradient_adjoint(cof,self.layout,gw).numpy().reshape(self.model.parent.ndof,3);G.append(self.model.reduction.P.T@grad);V.append(float(w@jj))
        H,minJ=self.topology.assemble(self.X,self.total_weights,self.cell_ids,F.numpy(),self.mobility)
        value=dict(volume=np.array(V),gradient=np.array(G),H=H,min_detF=min(float(jj.min()),minJ));self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return value

class CartesianAVF(ScaledTensorCellAVF):
    def __init__(self,model,config,*,cuts,alpha=.8,storage=.0002,mobility=MOBILITY,fixed_solid=False,pressure0=.01,reservoir=.002,state=None):
        if model.boundary.hold!=0:raise ValueError('fixed zero grips required')
        if not all(np.isfinite(x) for x in (alpha,storage,reservoir)) or not 0<=alpha<=1 or storage<=0:raise ValueError('invalid fluid parameters')
        self.model=model;self.config=copy.deepcopy(config);self.alpha=float(alpha);self.storage=float(storage);self.drained=True;self.fixed_solid=bool(fixed_solid);self.reservoir=float(reservoir)
        self.geometry=CartesianGeometry(model,cuts,mobility);self.cells=self.geometry.cells;self.capacity=storage*self.geometry.V0;self.active=np.arange(self.geometry.topology.nflux);self.B=self.geometry.B;self.nflux=self.geometry.topology.nflux;self.gb=self.geometry.topology.boundary_term(reservoir)
        self.identity=dict(schema='phase-boundary-cartesian-AVF-v1',linear_solver='row-column scaling, general LU, original residual',solid=model.identity,mass=digest(model.M.tolist()),geometry=self.geometry.identity,alpha=alpha,storage=storage,fixed_solid=fixed_solid,boundary_pressure_Pa=reservoir,time='inherited solid AVF and midpoint p/z; exact volume work',H_update='current midpoint F and full mobility')
        self.signature=digest(self.identity)
        if state is None:
            state=model.rest();p=np.broadcast_to(pressure0,(self.cells,)).astype(float).copy();content=alpha*(self.geometry.evaluate(state.q)['volume']-self.geometry.V0)+self.capacity*p
            state.child_states['fluid']=dict(model=self.signature,pressure_Pa=p.tolist(),flux_interval_m3_s=np.zeros(self.nflux).tolist(),content_m3=content.tolist(),cumulative_boundary_m3=0.,cumulative_source_m3=np.zeros(self.cells).tolist())
        self.validate(state);self._transaction=StateTransaction(state,validator=self.validate);self.avf=ValidatedAVF(model,config,state)

class CartesianCoupling(ExplicitGridCoupling):
    def __init__(self,model,config,times,*,source_m3_s=None,state=None,**kwargs):
        ts=np.asarray(times,dtype=float)
        if ts.ndim!=1 or len(ts)<2 or not np.isfinite(ts).all() or ts[0]!=0 or np.any(np.diff(ts)<=0):raise ValueError('explicit increasing pressure times required')
        self.times=ts.copy();self.times.setflags(write=False);self.core=CartesianAVF(model,config,state=state,**kwargs)
        self.source=self.core.geometry.topology.source() if source_m3_s is None else np.broadcast_to(source_m3_s,(self.core.cells,)).astype(float).copy()
        if not np.isfinite(self.source).all():raise ValueError('finite sources required')
        self.source.setflags(write=False);self.identity=dict(schema='phase-boundary-cartesian-time-v1',core=self.core.identity,times_s=ts.tolist(),source_m3_s=self.source.tolist());self.signature=digest(self.identity)
        if state is None:
            self.core.state.child_states['explicit_pressure_grid']=self.signature;self.core._transaction=StateTransaction(self.core.state,validator=self.validate)
        self.validate(self.core.state)
