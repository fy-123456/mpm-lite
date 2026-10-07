"""Minimal finite-strain 3D solid / one pressure-cell Darcy coupling.

Uses the actual original144 solid, full M5 and unchanged material/Ks. The fluid
space is explicitly one constant pressure cell and one right-drain face flux.
It certifies a minimal exchange interface, not spatial pore-pressure accuracy.
"""
import copy
from collections import OrderedDict
import numpy as np
import scipy.linalg as la
import warp as wp
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_d.stage2.gpu_operator import reduce_kernel


@wp.kernel
def geometry_kernel(F:wp.array(dtype=wp.mat33d),cof:wp.array(dtype=wp.mat33d),vol:wp.array(dtype=wp.float64),
                    mobility:wp.array(dtype=wp.float64),bad:wp.array(dtype=wp.int32),k_mu:wp.float64):
    i=wp.tid();f=F[i];j=wp.determinant(f)
    if not wp.isfinite(j) or j<=wp.float64(.1):
        wp.atomic_add(bad,0,1);cof[i]=wp.mat33d();vol[i]=wp.float64(0);mobility[i]=wp.float64(0);return
    inv=wp.inverse(f);cof[i]=j*wp.transpose(inv);vol[i]=j
    mobility[i]=k_mu*j*(inv[0,0]*inv[0,0]+inv[0,1]*inv[0,1]+inv[0,2]*inv[0,2])


@wp.kernel
def cofactor_action(F:wp.array(dtype=wp.mat33d),dF:wp.array(dtype=wp.mat33d),dc:wp.array(dtype=wp.mat33d)):
    i=wp.tid();inv=wp.inverse(F[i]);it=wp.transpose(inv);j=wp.determinant(F[i])
    dc[i]=j*(wp.trace(inv@dF[i])*it-it@wp.transpose(dF[i])@it)


class VolumeGeometry:
    def __init__(self,model,k_mu=.1):
        self.model=model;self.op=model.operator;self.k_mu=float(k_mu);self.cache=OrderedDict()
        if not np.isfinite(k_mu) or k_mu<=0:raise ValueError('positive finite permeability/viscosity required')
        lengths=np.array([e[-1]-e[0] for e in model.parent.edges]);self.V0=float(np.prod(lengths));self.factor=float(2*lengths[1]*lengths[2]/lengths[0])

    def field(self,q,direction=False):
        m=self.op.maps;r=self.model.reduction
        full=r.velocity(q) if direction else r.expand(q)
        return m.gradient(m.nodes(m.expand(full,direction=direction)),self.op.layout,identity=not direction)

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:self.cache.move_to_end(key);return self.cache[key]
        op=self.op;F=self.field(q);cof=wp.empty_like(F);v=wp.empty(op.count,dtype=wp.float64,device=op.device);mob=wp.empty_like(v);bad=wp.zeros(1,dtype=wp.int32,device=op.device)
        wp.launch(geometry_kernel,dim=op.count,inputs=[F,cof,v,mob,bad,self.k_mu],device=op.device)
        if bad.numpy()[0]:raise ValueError('coupled geometry outside positive-J range')
        grad=op.maps.gradient_adjoint(cof,op.layout,op.weights).numpy().reshape(self.model.parent.ndof,3);grad=self.model.reduction.P.T@grad
        n=(op.count+255)//256;total=wp.empty(n,dtype=wp.float64,device=op.device);minimum=wp.empty_like(total)
        wp.launch(reduce_kernel,dim=n,inputs=[v,op.weights,v,total,minimum,op.count],device=op.device);V=float(total.numpy().sum());minJ=float(minimum.numpy().min())
        wp.launch(reduce_kernel,dim=n,inputs=[mob,op.weights,v,total,minimum,op.count],device=op.device);T=self.factor*float(total.numpy().sum())/self.V0
        result=dict(volume=V,gradient=grad,conductance=T,min_detF=minJ)
        self.cache[key]=result
        while len(self.cache)>6:self.cache.popitem(last=False)
        return result

    def action(self,q,d):
        op=self.op;F=self.field(q);df=self.field(d,True);dc=wp.empty_like(F)
        wp.launch(cofactor_action,dim=op.count,inputs=[F,df,dc],device=op.device)
        grad=op.maps.gradient_adjoint(dc,op.layout,op.weights).numpy().reshape(self.model.parent.ndof,3)
        return self.model.reduction.P.T@grad

    def discrete(self,q0,q1):
        # det(F) is cubic along a coefficient segment; its gradient is quadratic.
        # Two-point Gauss therefore gives an exact discrete volume gradient.
        a=.5-.5/np.sqrt(3);b=.5+.5/np.sqrt(3)
        return .5*(self.evaluate(q0+a*(q1-q0))['gradient']+self.evaluate(q0+b*(q1-q0))['gradient'])


class CoupledAVF:
    def __init__(self,model,config,*,alpha=.8,storage=.2,k_mu=.1,drained=False,fixed_solid=False,pressure0=.01,state=None):
        if model.boundary.hold!=0:raise ValueError('minimal coupled patch has fixed zero grip lifts')
        if not all(np.isfinite(x) for x in (alpha,storage,k_mu,pressure0)) or not 0<=alpha<=1 or storage<=0:raise ValueError('invalid fluid parameters')
        self.model=model;self.config=copy.deepcopy(config);self.alpha=float(alpha);self.storage=float(storage);self.drained=bool(drained);self.fixed_solid=bool(fixed_solid)
        self.geometry=VolumeGeometry(model,k_mu);self.capacity=storage*self.geometry.V0
        self.identity=dict(schema='reference-next-single-pressure-cell-v1',solid=model.identity,mass=digest(model.M),alpha=alpha,storage_Pa_inverse=storage,
            mobility_m2_Pa_s=k_mu,drained=drained,fixed_solid=fixed_solid,pressure_space='one constant 3D cell',flux_space='one right-drain face volume rate',
            fluid_geometry_rule=model.rule.signature,solid_time='original AVF path order2',fluid_time='midpoint pressure/flux; exact path volume gradient')
        self.signature=digest(self.identity)
        if state is None:
            state=model.rest();state.child_states['fluid']=dict(model=self.signature,pressure_Pa=float(pressure0),flux_interval_m3_s=0.,content_m3=self.capacity*pressure0)
        self.validate(state);self._transaction=StateTransaction(state,validator=self.validate);self.avf=ValidatedAVF(model,config,state)

    @property
    def state(self):return self._transaction.snapshot()

    def validate(self,state):
        self.model.validate(state);f=state.child_states.get('fluid',{})
        if f.get('model')!=self.signature:raise ValueError('foreign fluid history')
        if not all(np.isfinite(f.get(k,np.nan)) for k in ('pressure_Pa','flux_interval_m3_s','content_m3')):raise ValueError('nonfinite fluid state')
        if self.fixed_solid and max(np.max(abs(state.q)),np.max(abs(state.velocity)))>1e-12:raise ValueError('fixed solid has moved')

    def rest_matrix(self,dt,storage=None):
        m=self.model;g=self.geometry.evaluate(np.zeros_like(m.rest().q))['gradient'][m.free].ravel();n=len(g)
        A=np.zeros((n+2,n+2));A[:n,:n]=np.eye(n) if self.fixed_solid else 2*m.M3ff+.5*dt**2*m.rest_K[np.ix_(m.ids,m.ids)]
        if not self.fixed_solid:A[:n,n]=-.5*dt*self.alpha*g
        A[n,:n]=0 if self.fixed_solid else dt*self.alpha*g;A[n,n]=(self.storage if storage is None else storage)*self.geometry.V0;A[n,n+1]=dt
        A[n+1,n]=-.5 if self.drained else 0.;A[n+1,n+1]=1/self.geometry.evaluate(m.rest().q)['conductance'] if self.drained else 1.
        return A

    def step(self,dt,*,source_m3_s=0.,external_force=None,inject=None):
        if not np.isfinite(dt) or dt<=0 or not np.isfinite(source_m3_s):raise ValueError('finite positive step and finite volume source required')
        trial=self._transaction.begin_trial();base=trial.state
        try:
            state,row=self._compute(base,dt,source_m3_s,external_force)
            if inject is not None:inject('before_commit',state)
            self.validate(state)
            for key in state.__dataclass_fields__:setattr(trial.state,key,copy.deepcopy(getattr(state,key)))
            self._transaction.commit(trial);return row
        except Exception:
            self._transaction.rollback(trial);self.geometry.cache.clear();raise

    def _compute(self,s,h,source,external):
        m=self.model;n=len(m.ids);p0=s.child_states['fluid']['pressure_Pa'];q0=s.q;v0=s.velocity
        geo0=self.geometry.evaluate(q0);initial=m.evaluate(q0);ext=np.zeros_like(q0) if external is None else np.array(external,copy=True)
        if ext.shape!=q0.shape or not np.isfinite(ext).all():raise ValueError('finite generalized load required')
        x=np.zeros(n+2);x[:n]=0 if self.fixed_solid else v0[m.free].ravel();x[n]=p0+h*source/self.capacity
        Hsolid=2*m.M3ff+.5*h*h*m.rest_K[np.ix_(m.ids,m.ids)]
        tolerances=np.r_[np.full(n,1e-10),1e-12,1e-10]
        def residual(x):
            W=np.zeros_like(q0);W[m.free]=x[:n].reshape(-1,3);p1,z=x[-2:];pbar=.5*(p0+p1);q1=q0+h*W
            geo1=self.geometry.evaluate(q1);gbar=self.geometry.discrete(q0,q1);mid=self.geometry.evaluate((q0+q1)/2)
            path=self.avf.path(q0,W,h);solid=2*m.M@(W-v0)+h*(path['force']-self.alpha*pbar*gbar-ext)
            mass=self.alpha*(geo1['volume']-geo0['volume'])+self.capacity*(p1-p0)+h*z-h*source
            resistance=1/mid['conductance'];darcy=resistance*z-pbar if self.drained else z
            res=np.r_[W[m.free].ravel() if self.fixed_solid else solid[m.free].ravel(),mass,darcy]
            return res,dict(W=W,q1=q1,p1=p1,pbar=pbar,z=z,geo1=geo1,gbar=gbar,path=path,resistance=resistance,solid=solid)
        for it in range(12):
            res,work=residual(x);norm=float(np.max(abs(res)/tolerances))
            if norm<=1:break
            if it==11:raise ValueError('coupled true residual did not converge')
            A=np.zeros((n+2,n+2));A[:n,:n]=np.eye(n) if self.fixed_solid else Hsolid
            if not self.fixed_solid:A[:n,n]=-.5*h*self.alpha*work['gbar'][m.free].ravel()
            A[n,:n]=0 if self.fixed_solid else h*self.alpha*work['geo1']['gradient'][m.free].ravel()
            A[n,n]=self.capacity;A[n,n+1]=h;A[n+1,n]=-.5 if self.drained else 0.;A[n+1,n+1]=work['resistance'] if self.drained else 1.
            update=la.solve(A,-res,assume_a='gen')
            for ls in range(10):
                candidate=x+2.**(-ls)*update
                try:new,_=residual(candidate)
                except ValueError:continue
                if np.max(abs(new)/tolerances)<norm:x=candidate;break
            else:raise ValueError('coupled line search failed')
        W=work['W'];q1=work['q1'];p1=float(work['p1']);pbar=float(work['pbar']);z=float(work['z']);v1=2*W-v0
        final=m.evaluate(q1);delta_volume=work['geo1']['volume']-geo0['volume'];dq=q1-q0
        pressure_solid=-self.alpha*pbar*float(np.sum(work['gbar']*dq));pressure_fluid=self.alpha*pbar*delta_volume
        energy0=m.kinetic(v0)+initial['U']+.5*self.capacity*p0*p0;energy1=m.kinetic(v1)+final['U']+.5*self.capacity*p1*p1
        darcy=h*work['resistance']*z*z if self.drained else 0.;external_work=h*float(np.sum(ext*W));source_work=h*pbar*source
        path_error=final['U']-initial['U']-h*float(np.sum(work['path']['force']*W))
        solve_work=float(np.sum(work['solid']*W))+pbar*res[-2]+h*z*res[-1]
        balance=energy1-energy0+darcy-external_work-source_work;closure=balance-path_error-solve_work
        scale=max(abs(energy0),abs(energy1),abs(source_work),1e-8)
        if abs(closure)>1e-10+.01*scale or abs(balance)>1e-9+.01*scale:raise ValueError('coupled energy budget failed')
        if darcy<0 or abs(pressure_solid+pressure_fluid)>1e-10:raise ValueError('pressure exchange or dissipation failed')
        candidate=s.clone();candidate.q=q1;candidate.velocity=v1;candidate.predictor=W.copy();candidate.time+=h;candidate.step+=1
        content=self.alpha*(work['geo1']['volume']-self.geometry.V0)+self.capacity*p1
        row=dict(step=candidate.step,time=candidate.time,dt=h,iterations=it,true_scaled_residual=norm,
            pressure_Pa=p1,flux_interval_m3_s=z,content_m3=content,mass_defect_m3=float(res[-2]),darcy_residual_Pa=float(res[-1]),
            pressure_solid_work_J=pressure_solid,pressure_fluid_work_J=pressure_fluid,pressure_work_defect_J=pressure_solid+pressure_fluid,
            total_energy_J=energy1,fluid_storage_J=.5*self.capacity*p1*p1,darcy_dissipation_J=darcy,
            external_work_J=external_work,source_work_J=source_work,energy_balance_J=balance,path_error_J=path_error,solve_work_J=solve_work,
            ledger_closure_J=closure,min_detF=min(final['min_detF'],work['path']['min_detF'],work['geo1']['min_detF']))
        candidate.child_states['fluid']=dict(model=self.signature,pressure_Pa=p1,flux_interval_m3_s=z,content_m3=content,last_ledger=copy.deepcopy(row))
        return candidate,row
