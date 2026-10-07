"""Cell-centred conservative face flow on the bounded x-only solid fixture.

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
        if cells not in (2,4) or not np.isfinite(k_mu) or k_mu<=0:raise ValueError('bounded cells and positive mobility required')
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


class MultiCellAVF:
    def __init__(self,model,config,*,cells=2,alpha=.8,storage=.2,k_mu=.1,drained=False,fixed_solid=False,pressure0=.01,reservoir=0.,state=None):
        if model.boundary.hold!=0:raise ValueError('fixed zero grip lift required')
        if not all(np.isfinite(x) for x in (alpha,storage,reservoir)) or not 0<=alpha<=1 or storage<=0:raise ValueError('invalid coupling parameters')
        self.model=model;self.config=copy.deepcopy(config);self.alpha=float(alpha);self.storage=float(storage);self.drained=bool(drained);self.fixed_solid=bool(fixed_solid);self.reservoir=float(reservoir)
        self.geometry=CellGeometry(model,cells,k_mu);self.cells=cells;self.capacity=storage*self.geometry.V0
        self.active=np.arange(1,cells+int(drained));self.B=self.geometry.B[:,self.active];self.nflux=len(self.active)
        self.gb=np.zeros(self.nflux)
        if drained:self.gb[-1]=reservoir
        self.identity=dict(schema='cost-phase-half-cell-pressure-v1',solid=model.identity,mass=digest(model.M.tolist()),alpha=alpha,storage=storage,
            geometry=self.geometry.identity,drained=drained,fixed_solid=fixed_solid,reservoir_Pa=reservoir,active_faces=self.active.tolist(),
            time='original solid AVF; midpoint p/z; exact two-Gauss volume gradient',parameters='numerical fixture, not calibrated')
        self.signature=digest(self.identity)
        if state is None:
            state=model.rest();p=np.broadcast_to(pressure0,(cells,)).astype(float).copy()
            content=alpha*(self.geometry.evaluate(state.q)['volume']-self.geometry.V0)+self.capacity*p
            state.child_states['fluid']=dict(model=self.signature,pressure_Pa=p.tolist(),flux_interval_m3_s=np.zeros(self.nflux).tolist(),
                content_m3=content.tolist(),cumulative_boundary_m3=0.,cumulative_source_m3=np.zeros(cells).tolist())
        self.validate(state);self._transaction=StateTransaction(state,validator=self.validate);self.avf=ValidatedAVF(model,config,state)

    @property
    def state(self):return self._transaction.snapshot()

    def validate(self,state):
        self.model.validate(state);f=state.child_states.get('fluid',{})
        if f.get('model')!=self.signature:raise ValueError('foreign fluid model/grid/face history')
        for name,shape in [('pressure_Pa',(self.cells,)),('content_m3',(self.cells,)),('flux_interval_m3_s',(self.nflux,)),('cumulative_source_m3',(self.cells,))]:
            a=np.asarray(f.get(name,[]))
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid owned fluid field '+name)
        if not np.isfinite(f.get('cumulative_boundary_m3',np.nan)):raise ValueError('invalid boundary history')
        if self.fixed_solid and max(np.max(abs(state.q)),np.max(abs(state.velocity)))>1e-12:raise ValueError('fixed solid moved')
        expected=self.alpha*(self.geometry.evaluate(state.q)['volume']-self.geometry.V0)+self.capacity*np.array(f['pressure_Pa'])
        if np.max(abs(expected-f['content_m3']))>1e-10:raise ValueError('fluid content inconsistent with q/p')

    def matrix(self,h,G0,G1,H,storage=None):
        m=self.model;n=len(m.ids);c=self.cells;f=self.nflux;A=np.zeros((n+c+f,n+c+f))
        A[:n,:n]=np.eye(n) if self.fixed_solid else 2*m.M3ff+.5*h*h*m.rest_K[np.ix_(m.ids,m.ids)]
        if not self.fixed_solid:
            A[:n,n:n+c]=-.5*h*self.alpha*G0[:,m.free].reshape(c,n).T
            A[n:n+c,:n]=h*self.alpha*G1[:,m.free].reshape(c,n)
        A[n:n+c,n:n+c]=np.diag(self.capacity if storage is None else storage*self.geometry.V0)
        A[n:n+c,n+c:]=h*self.B;A[n+c:,n:n+c]=-.5*self.B.T;A[n+c:,n+c:]=H
        return A

    def rest_matrix(self,h,storage=None):
        g=self.geometry.evaluate(self.model.rest().q)
        return self.matrix(h,g['gradient'],g['gradient'],g['H'][np.ix_(self.active,self.active)],storage)

    def step(self,dt,*,source_m3_s=0.,external_force=None,inject=None):
        source=np.broadcast_to(source_m3_s,(self.cells,)).astype(float).copy()
        if not np.isfinite(dt) or dt<=0 or not np.isfinite(source).all():raise ValueError('finite source and positive dt required')
        trial=self._transaction.begin_trial()
        try:
            state,row=self._compute(trial.state,dt,source,external_force)
            if inject is not None:inject('before_commit',state)
            self.validate(state)
            for key in state.__dataclass_fields__:setattr(trial.state,key,copy.deepcopy(getattr(state,key)))
            self._transaction.commit(trial);return row
        except Exception:
            self._transaction.rollback(trial);self.geometry.cache.clear();raise

    def _compute(self,s,h,source,external):
        m=self.model;n=len(m.ids);c=self.cells;p0=np.array(s.child_states['fluid']['pressure_Pa']);q0=s.q;v0=s.velocity
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
            A=self.matrix(h,w['gb'],w['g1']['gradient'],w['H']);dx=la.solve(A,-res,assume_a='gen')
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
        row=dict(step=candidate.step,time=candidate.time,dt=h,iterations=it,true_scaled_residual=norm,
            pressure_Pa=p1.tolist(),flux_interval_m3_s=z.tolist(),content_m3=content.tolist(),mass_defect_m3=res[n:n+c].tolist(),darcy_residual_Pa=res[n+c:].tolist(),
            pressure_solid_work_J=pressure_solid.tolist(),pressure_fluid_work_J=pressure_fluid.tolist(),pressure_work_defect_J=(pressure_solid+pressure_fluid).tolist(),
            solid_material_J=final['material_U'],stabilization_J=final['stabilization_U'],kinetic_J=m.kinetic(v1),fluid_storage_J=float(Ef),total_energy_J=float(E1),
            darcy_dissipation_J=D,external_work_J=external_work,source_work_J=source_work,reservoir_work_J=boundary_work,energy_balance_J=float(balance),
            path_error_J=float(path_error),solve_work_J=solve_work,ledger_closure_J=float(closure),min_detF=min(final['min_detF'],w['path']['min_detF'],w['g1']['min_detF']))
        f=s.child_states['fluid'];candidate.child_states['fluid']=dict(model=self.signature,pressure_Pa=p1.tolist(),flux_interval_m3_s=z.tolist(),content_m3=content.tolist(),
            cumulative_boundary_m3=f['cumulative_boundary_m3']+h*float(np.sum(self.B@z)),cumulative_source_m3=(np.array(f['cumulative_source_m3'])+h*source).tolist(),last_ledger=copy.deepcopy(row))
        return candidate,row

    def frame(self,cache):
        frame=cache.frame(self.state);X=frame['X'];ids=np.clip(np.searchsorted(self.geometry.cuts,X[...,0],side='right')-1,0,self.cells-1)
        p=np.asarray(self.state.child_states['fluid']['pressure_Pa']);F=frame['F'];cof=np.linalg.det(F)[...,None,None]*np.swapaxes(np.linalg.inv(F),-1,-2)
        frame['pressure_Pa']=p[ids];frame['pressure_cell_id']=ids;frame['solid_PK1']=frame['PK1'].copy();frame['total_PK1']=frame['PK1']-self.alpha*p[ids,None,None]*cof
        return frame
