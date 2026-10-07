"""Cell-centred conservative face flow on the bounded 2/4/8 aligned pressure research fixture.

New spatial discretization: integrate inverse mobility over each half-cell.
The solid AVF, volume work, flux orientation and state transaction are inherited.
This is not an equivalent replacement for a general 3D RT0 operator.
"""
import copy
from collections import OrderedDict
import numpy as np
import scipy.linalg as la
import warp as wp
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_reference_next.coupling import cofactor_action
from engine.aniso_phase1.tensor_metrics import quadrature_axis


@wp.kernel
def cell_geometry_kernel(F:wp.array(dtype=wp.mat33d),cof:wp.array(dtype=wp.mat33d),vol:wp.array(dtype=wp.float64),
                         inv_xx:wp.array(dtype=wp.float64),bad:wp.array(dtype=wp.int32),mobility:wp.float64):
    i=wp.tid();f=F[i];j=wp.determinant(f)
    if not wp.isfinite(j) or j<=wp.float64(.1):
        wp.atomic_add(bad,0,1);return
    cof[i]=j*wp.transpose(wp.inverse(f));vol[i]=j
    # A^-1 = F^T F / ((k/mu) J). Includes shear, not 1/A_xx.
    inv_xx[i]=(f[0,0]*f[0,0]+f[1,0]*f[1,0]+f[2,0]*f[2,0])/(mobility*j)


class CellGeometry:
    def __init__(self,model,cells=2,k_mu=.1,order=None):
        if isinstance(cells,bool) or not isinstance(cells,(int,np.integer)) or cells not in (2,4,8) or not np.isfinite(k_mu) or k_mu<=0:raise ValueError('bounded cells and positive mobility required')
        self.model=model;self.op=model.operator;self.cells=cells;self.k_mu=float(k_mu);self.order=order or model.rule.orders[0];self.cache=OrderedDict()
        edges=model.parent.edges;self.cuts=np.linspace(edges[0][0],edges[0][-1],cells+1)
        for j,x in enumerate(self.cuts):
            i=int(np.argmin(abs(edges[0]-x)))
            if abs(edges[0][i]-x)<1e-12:self.cuts[j]=edges[0][i]
        # Integrate the intersections even when a pressure cut is inside a solid cell.
        integration_edges=(np.unique(np.r_[edges[0],self.cuts,.5*(self.cuts[:-1]+self.cuts[1:])]),edges[1],edges[2]);axes=[quadrature_axis(e,self.order) for e in integration_edges]
        self.points=tuple(x[0] for x in axes);self.shape=tuple(map(len,self.points));self.count=int(np.prod(self.shape))
        self.layout=self.op.maps.layout(self.points)
        weights=(axes[0][1][:,None,None]*axes[1][1][None,:,None]*axes[2][1][None,None,:]).ravel()
        self.x=np.broadcast_to(self.points[0][:,None,None],self.shape).ravel().copy()
        self.cell_ids=np.clip(np.searchsorted(self.cuts,self.x,side='right')-1,0,cells-1)
        self.area=float((edges[1][-1]-edges[1][0])*(edges[2][-1]-edges[2][0]));self.V0=np.diff(self.cuts)*self.area
        self.weights=[weights*(self.cell_ids==k) for k in range(cells)]
        self.gpu_weights=[wp.array(w,dtype=wp.float64,device=self.op.device) for w in self.weights]
        if not np.allclose([w.sum() for w in self.weights],self.V0,atol=1e-13,rtol=1e-12):raise ValueError('pressure intersections do not cover reference volume')
        self.B=np.zeros((cells,cells+1))
        for k in range(cells):self.B[k,k]=-1.;self.B[k,k+1]=1.
        self.identity=dict(cuts=self.cuts.tolist(),B=self.B.tolist(),orientation='all x faces +x',geometry_order=self.order,
            mobility='A=J F^-1 (k/mu I) F^-T; Hff=half-cell integral (A^-1)xx / area^2; zero transverse flux',k_mu=self.k_mu,
            integration_edges=[e.tolist() for e in integration_edges],closed_transverse_faces=True)

    def field(self,q,direction=False):
        r=self.model.reduction;maps=self.op.maps;full=r.velocity(q) if direction else r.expand(q)
        return maps.gradient(maps.nodes(maps.expand(full,direction=direction)),self.layout,identity=not direction)

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        F=self.field(q);cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);invxx=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device)
        wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,invxx,bad,self.k_mu],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('cell geometry leaves positive-J range')
        jj=J.numpy();aa=invxx.numpy();V=[];G=[];H=np.zeros((self.cells+1,self.cells+1))
        for k,(w,gw) in enumerate(zip(self.weights,self.gpu_weights)):
            grad=self.op.maps.gradient_adjoint(cof,self.layout,gw).numpy().reshape(self.model.parent.ndof,3)
            G.append(self.model.reduction.P.T@grad);V.append(float(w@jj))
            midpoint=.5*(self.cuts[k]+self.cuts[k+1])
            left=self.x<midpoint
            H[k,k]+=float(np.sum(w*aa*left))/self.area**2
            H[k+1,k+1]+=float(np.sum(w*aa*(~left)))/self.area**2
        value=dict(volume=np.array(V),gradient=np.array(G),H=H,min_detF=float(jj.min()))
        self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return value

    def action(self,q,d):
        F=self.field(q);dF=self.field(d,True);dc=wp.empty_like(F)
        wp.launch(cofactor_action,dim=self.count,inputs=[F,dF,dc],device=self.op.device)
        return np.array([self.model.reduction.P.T@self.op.maps.gradient_adjoint(dc,self.layout,w).numpy().reshape(self.model.parent.ndof,3) for w in self.gpu_weights])

    def discrete(self,q0,q1):
        a=.5-.5/np.sqrt(3);b=.5+.5/np.sqrt(3)
        return .5*(self.evaluate(q0+a*(q1-q0))['gradient']+self.evaluate(q0+b*(q1-q0))['gradient'])

