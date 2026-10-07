"""Material-only retry with bounded, continuously advanced reserve histories.

Scoped to the common 10-scalar research space. Geometry/mass always use q4.
No reconstruction of reserve F from particles, and no catch-all error fallback.
"""
from dataclasses import dataclass,field
import copy,json,time
import numpy as np
from engine.aniso_phase1.research_common_kinematics_next.model import (
    CommonOperator,CommonBridge,State as CommonState,Space,Particles,frozen,sha_arrays,advance)
from engine.aniso_phase1.research_common_kinematics_next.mapping import basis,push
from engine.aniso_phase1.research_eulerian_poro_next.materials import make_operator,validate_moments

SCHEMA='bounded-reserve-material-only-v1'
BUDGET=dict(rtol=.05,force_atol=1e-7,energy_atol=1e-10)

class SeparatedOperator(CommonOperator):
    def __init__(self,space,gx,gF,mx,mF,A2,A4,order):
        # Geometry and inertia are independent of selected material quadrature.
        ng=len(gx);a=np.tile(np.diag([1.,0.,0.]),(ng,1,1));v=a.reshape(ng,9)
        super().__init__(space,gx,gF,a,v[:,:,None]*v[:,None,:],4)
        X,w,_=space.rule(order);N,D=basis(space,X,mx,mF)
        validate_moments(A2,A4)
        if len(A2)!=len(X):raise ValueError('material moments/rule mismatch')
        self.order=order
        self.points,self.weights,self.N,self.D,self.mx,self.mF,self.A2,self.A4=map(frozen,(X,w,N,D,mx,mF,A2,A4))
        self.identity.update(schema=SCHEMA,order=order,material_order=order,geometry_order=4,mass_order=4,
            state_sha=sha_arrays(gx,gF,mx,mF,N,D,A2,A4))
    def field(self,d,D=None):
        if D is None:return self.mF+np.einsum('qnd,ni->qid',self.D,d)
        if D is self.gD:return self.F0+np.einsum('qnd,ni->qid',self.gD,d)
        raise ValueError('unknown field derivative layout')
    def save(self,path):
        np.savez_compressed(path,gx=self.x0,gF=self.F0,mx=self.mx,mF=self.mF,A2=self.A2,A4=self.A4,
            metadata=np.array(json.dumps(self.identity,sort_keys=True)))
    @classmethod
    def load_package(cls,path):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']));s=m['space']
            if m['schema']!=SCHEMA:raise ValueError('foreign frozen operator')
            op=cls(Space(tuple(s['shape']),s['bubbles'],s['lengths']),*[d[k] for k in ('gx','gF','mx','mF','A2','A4')],m['order'])
            if op.identity!=m:raise ValueError('frozen operator identity mismatch')
            return op

@dataclass
class State(CommonState):
    reserve_x:np.ndarray=field(default_factory=lambda:np.empty((0,3)))
    reserve_F:np.ndarray=field(default_factory=lambda:np.empty((0,3,3)))
    low_x:np.ndarray=field(default_factory=lambda:np.empty((0,3)))
    low_F:np.ndarray=field(default_factory=lambda:np.empty((0,3,3)))
    active_order:int=4
    fallback_count:int=0
    rule_work:float=0.
    def digest(self):
        return sha_arrays(np.frombuffer(super().digest().encode(),dtype=np.uint8),
            self.reserve_x,self.reserve_F,self.low_x,self.low_F,
            np.array([self.active_order,self.fallback_count,self.rule_work]))

def compare(primary,reserve,d):
    """Compare same trial configuration, assembled force and cell weak PK1 moments."""
    e,f,p=primary.material(d);er,fr,pr=reserve.material(d)
    def weak(op,P):
        _,_,cells=op.space.rule(op.order)
        return np.stack([np.einsum('q,qij->ij',op.weights[cells==c],P[cells==c]) for c in range(op.space.cells)])
    a,b=weak(primary,p),weak(reserve,pr)
    r={}
    for name,x,y,atol in [('energy',e,er,BUDGET['energy_atol']),('force',f,fr,BUDGET['force_atol']),('weak_stress',a,b,BUDGET['force_atol'])]:
        err=float(np.linalg.norm(x-y));scale=float(np.linalg.norm(y));limit=atol+BUDGET['rtol']*scale
        r[name]=dict(error=err,scale=scale,limit=limit,passed=err<=limit)
    r['passed']=all(r[k]['passed'] for k in ('energy','force','weak_stress'))
    return r

class BoundedBridge(CommonBridge):
    def __init__(self,space=None,ppc=4,rule='director-log',primary_order=4):
        if primary_order not in (2,4,6):raise ValueError('supported primary orders: diagnostic 2, default 4, reserve 6')
        super().__init__(space,ppc,rule);self.primary_order=primary_order
        X,_,_=self.space.rule(6);L=self.space.rule(2)[0] if primary_order==2 else np.empty((0,3))
        self.state=State(**vars(self.state),reserve_x=X.copy(),reserve_F=np.tile(np.eye(3),(len(X),1,1)),
            low_x=L.copy(),low_F=np.tile(np.eye(3),(len(L),1,1)),active_order=primary_order)
    def history(self,order):
        s=self.state
        if order==4:return s.qx,s.qF
        if order==6:return s.reserve_x,s.reserve_F
        if order==2 and len(s.low_x):return s.low_x,s.low_F
        raise ValueError('requested material rule has no continuous history')
    def operator(self,order):
        x,F=self.history(order);material,_=make_operator(self.space,self.state.particles,self.rule,order)
        return SeparatedOperator(self.space,self.state.qx,self.state.qF,x,F,material.A2,material.A4,order)
    def prepare_pair(self):
        tick=time.perf_counter()
        # Existing q4 Hermite fit: material-rule changes cannot affect velocity or M.
        _,v,fit=super().prepare()
        active=self.operator(self.state.active_order)
        reserve=active if active.order==6 else self.operator(6)
        if not np.array_equal(active.M,reserve.M) or not np.array_equal(active.gD,reserve.gD):
            raise ValueError('material retry changed geometry or inertia')
        fit['common_prepare_seconds']=fit['seconds'];fit['seconds']=time.perf_counter()-tick
        return active,reserve,v,fit
    def step(self,h=.0025,*,amplitude=1.,fault=None):
        self.attempts+=1;old=self.state;digest=old.digest()
        def fail(stage):
            if fault==stage:
                assert self.state.digest()==digest
                raise RuntimeError('injected '+stage)
        op,reserve,v0,fit=self.prepare_pair();first=op;fail('prepare');tick=time.perf_counter()
        zero=np.zeros_like(v0)
        d,v,p,z,m=advance(op,zero,v0,old.p,h,old.time,amplitude)
        verdict=compare(op,reserve,d);retry=not verdict['passed'];first_order=op.order
        jump=0.
        if retry:
            fail('retry')
            # No mutation occurred: discard the whole candidate, repeat the same step.
            assert self.state.digest()==digest
            jump=reserve.material(zero)[0]-op.material(zero)[0]
            op=reserve;d,v,p,z,m=advance(op,zero,v0,old.p,h,old.time,amplitude)
        m['solve_seconds']=time.perf_counter()-tick;fail('solve');load_tick=time.perf_counter()
        part=old.particles.copy();N,D=basis(self.space,part.X,part.x,part.F)
        part.x+=N@d;part.F+=np.einsum('qnd,ni->qid',D,d)
        part.v=N@v;part.grad_v=np.einsum('qnd,ni->qid',D,v)
        qx=old.qx+op.gN@d;qF=op.field(d,op.gD);qv=op.gN@v
        rx,rF=push(self.space,self.space.rule(6)[0],old.reserve_x,old.reserve_F,d)
        lx,lF=(push(self.space,self.space.rule(2)[0],old.low_x,old.low_F,d) if len(old.low_x) else (old.low_x.copy(),old.low_F.copy()))
        for xx,FF in ((part.x,part.F),(qx,qF),(rx,rF),(lx,lF)):
            if not np.isfinite(xx).all() or not np.isfinite(FF).all() or (len(FF) and np.linalg.det(FF).min()<=.1):
                raise ValueError('invalid accepted point history')
        V=op.geometry(d,False)[0];content=op.alpha*(V-op.top.V0)+op.capacity*p
        defect=content-old.content+h*op.top.B@z
        if abs(defect).max()>1e-9:raise ValueError('actual fluid content mismatch')
        fail('load')
        trial=State(part,qx,qF,qv,p,content,d,old.time+h,old.step+1,
            old.boundary_volume+m['boundary_volume'],old.dissipation+m['darcy_dissipation'],
            old.projection_work+fit['kinetic_projection'],old.external_work+m['external_work'],
            rx,rF,lx,lF,op.order,old.fallback_count+int(retry),old.rule_work+jump)
        m.update(time=trial.time,step=trial.step,prepare=fit,actual_content_defect=float(abs(defect).max()),
            Nq=len(op.points),particles=len(part.X),inner_particle_reads=0,material_order=op.order,
            load_seconds=time.perf_counter()-load_tick,
            material_calls=sum(o.material_calls for o in ({id(first):first,id(reserve):reserve}).values()),
            material_evaluations=sum(o.material_evaluations for o in ({id(first):first,id(reserve):reserve}).values()),
            first_order=first_order,retried=retry,rule_check=verdict,rule_energy_jump=jump,
            reserve_points=len(rx),total_history_points=len(qx)+len(rx)+len(lx),
            min_particle_detF=float(np.linalg.det(part.F).min()),
            max_particle_displacement=float(np.linalg.norm(part.x-part.X,axis=1).max()))
        fail('commit');self.state=trial;self.accepted+=1
        return m
