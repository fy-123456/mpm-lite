"""Bounded updated-map material-control-volume coupled research model.

Only the local step increment is represented in the modal space. Particle and
fixed quadrature histories x,F remain authoritative point values; they are never
reset by fitting a global displacement. Rebuilding projects velocity only, with
its kinetic energy and momentum changes explicitly reported. This prototype
uses PIC/Hermite velocity fitting, not the production APIC implicit solve.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import copy,json,time
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_unified_lite_poro.model import (
    FrozenOperator,Particles,seed,frozen,sha_arrays,PARAMS,MOBILITY)
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from engine.aniso_phase1.research_unified_lite_poro.solve import advance
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from engine.aniso_phase1.research_eulerian_poro_next.materials import make_operator,validate_moments
from .mapping import basis,SCHEMA


class CommonOperator(FrozenOperator):
    """Owns bounded quadrature/geometry data, no particle arrays or callbacks."""
    def __init__(self,space,x,F,A2,A4,order=4):
        self.space=copy.deepcopy(space);self.order=order
        X,w,cells=space.rule(order);N,D=basis(space,X,x,F)
        validate_moments(A2,A4)
        if len(A2)!=len(X):raise ValueError('moment and quadrature count mismatch')
        self.points,self.weights,self.A2,self.A4=map(frozen,(X,w,A2,A4))
        self.gX,self.gw,self.cells,self.N,self.D,self.F0,self.x0=map(frozen,(X,w,cells,N,D,F,x))
        self.gN=self.N;self.gD=self.D
        self.params=SimpleNamespace(**PARAMS);self.top=ReferenceTopology(space.cuts)
        self.M=frozen(N.T@(w[:,None]*N));self.M3=frozen(np.kron(self.M,np.eye(3)))
        ev=la.eigvalsh(self.M)
        if ev[0]<=1e-12*ev[-1]:raise ValueError('full common-map mass is rank deficient')
        self.capacity=frozen(.01*self.top.V0);self.alpha=.8
        self.mobility=frozen(MOBILITY);self.gb=frozen(self.top.boundary_term(0.))
        self.material_calls=0;self.material_evaluations=0;self.geometry_calls=0
        self.identity=dict(schema=SCHEMA,space=space.identity(),order=order,params=PARAMS,
            storage=.01,alpha=self.alpha,mobility=MOBILITY.tolist(),
            state_sha=sha_arrays(x,F,N,D,A2,A4),time_layer='step-initial')

    def field(self,d,D=None):
        return self.F0+np.einsum('qnd,ni->qid',self.D if D is None else D,d)

    def save(self,path):
        np.savez_compressed(path,x=self.x0,F=self.F0,A2=self.A2,A4=self.A4,
            metadata=np.array(json.dumps(self.identity,sort_keys=True)))

    @classmethod
    def load_package(cls,path):
        with np.load(path,allow_pickle=False) as data:
            m=json.loads(str(data['metadata']));s=m['space']
            if m['schema']!=SCHEMA or m['time_layer']!='step-initial':raise ValueError('foreign mapping/time layer')
            op=cls(Space(tuple(s['shape']),s['bubbles'],s['lengths']),data['x'],data['F'],data['A2'],data['A4'],m['order'])
            if op.identity!=m:raise ValueError('frozen package identity mismatch')
            return op


@dataclass
class State:
    particles:Particles
    qx:np.ndarray
    qF:np.ndarray
    qv:np.ndarray
    p:np.ndarray
    content:np.ndarray
    last_increment:np.ndarray
    time:float=0.
    step:int=0
    boundary_volume:float=0.
    dissipation:float=0.
    projection_work:float=0.
    external_work:float=0.
    def digest(self):
        arrays=[getattr(self,n) for n in ('qx','qF','qv','p','content','last_increment')]
        arrays+=[np.array([self.time,self.step,self.boundary_volume,self.dissipation,self.projection_work,self.external_work]),
                 np.frombuffer(self.particles.digest().encode(),dtype=np.uint8)]
        return sha_arrays(*arrays)


class CommonBridge:
    def __init__(self,space=None,ppc=4,rule='director-log'):
        self.space=Space() if space is None else space;self.rule=rule;self.order=4
        p=seed(self.space,ppc);X,w,c=self.space.rule(self.order);n=8+len(self.space.bubbles)
        self.state=State(p,X.copy(),np.tile(np.eye(3),(len(X),1,1)),np.zeros_like(X),
            np.full(self.space.cells,.2),.01*np.bincount(c,weights=w)*.2,np.zeros((n,3)))
        self.attempts=0;self.accepted=0

    def prepare(self):
        t=time.perf_counter();s=self.state;p=s.particles
        if any(not np.isfinite(getattr(p,k)).all() for k in p.__dataclass_fields__) or np.any(p.weight<=0):
            raise ValueError('invalid particle state')
        if np.any(np.linalg.det(p.F)<=.1):raise ValueError('invalid particle deformation')
        material,info=make_operator(self.space,p,self.rule,self.order)
        op=CommonOperator(self.space,s.qx,s.qF,material.A2,material.A4,self.order)
        N,D=basis(self.space,p.X,p.x,p.F);scale=float(min(self.space.h));W=np.sqrt(p.weight)
        L=np.vstack([N*W[:,None],*(D[:,:,a]*W[:,None]*scale for a in range(3))])
        rhs=np.vstack([p.v*W[:,None],*(p.grad_v[:,:,a]*W[:,None]*scale for a in range(3))])
        v,_,rank,sv=la.lstsq(L,rhs,lapack_driver='gelsd')
        if rank!=len(v):raise ValueError('velocity reconstruction is rank deficient')
        projected=op.N@v
        before=.5*np.sum(op.gw[:,None]*s.qv**2);after=.5*np.sum(op.gw[:,None]*projected**2)
        fit=dict(rank=int(rank),condition=float(sv[0]/sv[-1]),
            weighted_velocity_residual=float(la.norm(L@v-rhs)),
            relative_velocity_projection=float(la.norm(projected-s.qv)/max(la.norm(s.qv),1e-5)),
            kinetic_projection=after-before,
            momentum_projection=(op.gw@(projected-s.qv)).tolist(),
            material_rule=info,seconds=time.perf_counter()-t,particle_reads=len(p.X))
        return op,v,fit

    def step(self,h=.00125,*,amplitude=1.,fault=None):
        self.attempts+=1;old=self.state;digest=old.digest()
        def fail(stage):
            if fault==stage:
                assert self.state.digest()==digest
                raise RuntimeError('injected '+stage)
        op,v0,fit=self.prepare();fail('prepare')
        t=time.perf_counter();d,v,p,z,m=advance(op,np.zeros_like(v0),v0,old.p,h,old.time,amplitude)
        m['solve_seconds']=time.perf_counter()-t;fail('solve')
        t=time.perf_counter();part=old.particles.copy();N,D=basis(self.space,part.X,part.x,part.F)
        part.x+=N@d;part.F+=np.einsum('qnd,ni->qid',D,d)
        part.v=N@v;part.grad_v=np.einsum('qnd,ni->qid',D,v)
        qx=old.qx+op.N@d;qF=op.field(d);qv=op.N@v
        if not np.isfinite(part.x).all() or not np.isfinite(part.F).all() or np.linalg.det(part.F).min()<=.1:
            raise ValueError('particle load invalid')
        V=op.geometry(d,False)[0];content=op.alpha*(V-op.top.V0)+op.capacity*p
        defect=content-old.content+h*op.top.B@z
        if np.max(abs(defect))>1e-9:raise ValueError('actual fluid content mismatch')
        fail('load')
        trial=State(part,qx,qF,qv,p,content,d,old.time+h,old.step+1,
            old.boundary_volume+m['boundary_volume'],old.dissipation+m['darcy_dissipation'],
            old.projection_work+fit['kinetic_projection'],old.external_work+m['external_work'])
        m.update(time=trial.time,step=trial.step,prepare=fit,load_seconds=time.perf_counter()-t,
            actual_content_defect=float(abs(defect).max()),Nq=len(op.points),particles=len(part.X),
            material_calls=op.material_calls,material_evaluations=op.material_evaluations,
            inner_particle_reads=0,full_mass_cross_norm=float(la.norm(op.M[:8,8:])),
            min_particle_detF=float(np.linalg.det(part.F).min()),
            max_particle_displacement=float(np.linalg.norm(part.x-part.X,axis=1).max()))
        fail('commit');self.state=trial;self.accepted+=1
        return m
