"""Particle unload -> owned frozen operator -> load, with explicit limitations.

Reuses the existing Hencky response, direction partition/moments and RT0
Piola-transformed full-tensor Darcy assembly. Inner objects own no particles.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import copy
import hashlib
import time
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.material_snapshot import moments, moment_points, response
from engine.aniso_phase1.joint_sampling import partition_material
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology

PARAMS=dict(mu=10.,lam=20.,k_f=200.)
MOBILITY=np.array([[.002,.0003,.0001],[.0003,.0015,-.0002],[.0001,-.0002,.001]])

def frozen(a):
    a=np.array(a,copy=True);a.setflags(write=False);return a

def sha_arrays(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        v=np.ascontiguousarray(a);h.update(str((v.shape,v.dtype.str)).encode());h.update(v.tobytes())
    return h.hexdigest()

@dataclass
class Particles:
    X:np.ndarray
    x:np.ndarray
    v:np.ndarray
    F:np.ndarray
    grad_v:np.ndarray
    A:np.ndarray
    weight:np.ndarray
    def copy(self):return copy.deepcopy(self)
    def digest(self):return sha_arrays(*(getattr(self,k) for k in self.__dataclass_fields__))

def seed(space, ppc=3, angle_shift=0.):
    X,w,_=space.rule(ppc,True)
    a=np.deg2rad(20+50*X[:,1]/space.lengths[1]+15*X[:,0]/space.lengths[0]+angle_shift)
    direction=np.column_stack([np.cos(a),np.sin(a),np.zeros(len(a))])
    return Particles(X.copy(),X.copy(),np.zeros_like(X),np.tile(np.eye(3),(len(X),1,1)),
                     np.zeros((len(X),3,3)),direction[:,:,None]*direction[:,None,:],w)

def unload(space, particles):
    """Hermite reconstruction from actual particle displacement/F and v/grad_v.

    Gradients distinguish cell bubbles even when PPC=2 position samples alone
    are rank deficient. No old generalized coefficients enter this fit.
    """
    p=particles;N,D=space.basis(p.X);scale=float(np.min(space.h))
    if any(not np.isfinite(getattr(p,k)).all() for k in p.__dataclass_fields__):raise ValueError('invalid particle state')
    if np.any(p.weight<=0) or np.any(np.linalg.det(p.F)<=0):raise ValueError('invalid particle mass or F')
    W=np.sqrt(p.weight)
    L=np.vstack([N*W[:,None],*(D[:,:,a]*W[:,None]*scale for a in range(3))])
    values=np.vstack([(p.x-p.X)*W[:,None],*((p.F-np.eye(3))[:,:,a]*W[:,None]*scale for a in range(3))])
    speeds=np.vstack([p.v*W[:,None],*(p.grad_v[:,:,a]*W[:,None]*scale for a in range(3))])
    coeff,_,rank,s=la.lstsq(L,np.column_stack([values,speeds]),lapack_driver='gelsd')
    if rank != L.shape[1]:raise ValueError('particle Hermite history cannot resolve motion space')
    mismatch=float(la.norm(L@coeff-np.column_stack([values,speeds])))
    return coeff[:,:3],coeff[:,3:],dict(rank=int(rank),condition=float(s[0]/s[-1]),weighted_history_residual=mismatch,particle_reads=len(p.X))

class FrozenOperator:
    def __init__(self,space,points,weights,A2,A4,*,order=4,params=None):
        points=np.asarray(points,float);weights=np.asarray(weights,float)
        A2=np.asarray(A2,float);A4=np.asarray(A4,float)
        count=len(weights)
        if (points.shape!=(count,3) or A2.shape!=(count,3,3) or A4.shape!=(count,9,9) or count==0
            or not all(np.isfinite(x).all() for x in (points,weights,A2,A4)) or np.any(weights<=0)):
            raise ValueError('finite positive bounded quadrature package required')
        if not np.allclose(A2,A2.swapaxes(1,2)) or not np.allclose(np.trace(A2,axis1=1,axis2=2),1.):
            raise ValueError('normalized symmetric direction moments required')
        if not np.allclose(A4,A4.swapaxes(1,2)) or np.linalg.eigvalsh(A4).min() < -1e-10:
            raise ValueError('positive fourth direction moment required')
        self.space=copy.deepcopy(space)
        self.params=SimpleNamespace(**(PARAMS if params is None else params))
        if self.params.mu<=0 or self.params.lam<0 or self.params.k_f<0 or not np.isfinite(list(vars(self.params).values())).all():
            raise ValueError('invalid material parameters')
        self.points=frozen(points);self.weights=frozen(weights);self.A2=frozen(A2);self.A4=frozen(A4)
        self.N,self.D=map(frozen,space.basis(points))
        X,w,cells=space.rule(order);N,D=space.basis(X)
        self.gX,self.gw,self.cells,self.gN,self.gD=map(frozen,(X,w,cells,N,D))
        self.M=frozen(N.T@(w[:,None]*N));self.M3=frozen(np.kron(self.M,np.eye(3)))
        eig=la.eigvalsh(self.M)
        if eig[0]<=1e-12*eig[-1]:raise ValueError('motion mass is not positive definite')
        self.top=ReferenceTopology(space.cuts);self.capacity=frozen(.01*self.top.V0)
        self.alpha=.8;self.mobility=frozen(MOBILITY);self.gb=frozen(self.top.boundary_term(0.))
        self.material_calls=0;self.material_evaluations=0;self.geometry_calls=0
        self.order=order
        self.identity=dict(space=space.identity(),params=vars(self.params),alpha=self.alpha,
             storage=.01,mobility=self.mobility.tolist(),order=order,rule_sha=sha_arrays(points,weights,A2,A4),
             model='material-coordinate particle bridge, not Eulerian production Lite')

    @classmethod
    def from_particles(cls,space,p,groups=2,order=4,direction='moments'):
        owner=np.ravel_multi_index(np.minimum((p.X/space.h).astype(int),np.array(space.shape)-1).T,space.shape)
        points=[];weights=[];a2=[];a4=[]
        for c in range(space.cells):
            ids=np.flatnonzero(owner==c)
            if len(ids)==0:raise ValueError('empty material region')
            leaves=partition_material(p.X[ids],p.A[ids],p.weight[ids],float(min(space.h)),groups,False)
            for leaf in leaves:
                ix=ids[leaf];a,b=moments(p.A[ix],p.weight[ix])
                if direction=='mean':b=np.outer(a.ravel(),a.ravel())
                elif direction!='moments':raise ValueError('unknown direction representation')
                pts=moment_points(p.X[ix],p.weight[ix])
                points.extend(pts);weights.extend([p.weight[ix].sum()/8]*8);a2.extend([a]*8);a4.extend([b]*8)
        return cls(space,points,weights,a2,a4,order=order)

    @classmethod
    def fixed_positive(cls,space,p,order=4):
        """Fixed Gauss sites with positive trilinear interpolation of A2/A4.

        Only complete Cartesian reference particle templates are supported.
        Boundary values use constant extension; no negative extrapolation weights.
        Positions remain associated with local direction moments. This is a new
        scoped rule, not the old group4x8 or pure-direction vector averaging.
        """
        from scipy.interpolate import RegularGridInterpolator
        axes=tuple(np.unique(p.X[:,a]) for a in range(3))
        if np.prod([len(x) for x in axes])!=len(p.X):
            raise ValueError('fixed-positive rule requires Cartesian reference particles')
        ids=np.lexsort((p.X[:,2],p.X[:,1],p.X[:,0]))
        expected=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        if not np.allclose(p.X[ids],expected,rtol=0,atol=1e-12):
            raise ValueError('particle template has duplicates or missing sites')
        if not np.allclose(p.weight,p.weight[0]) or not np.isclose(p.weight.sum(),np.prod(space.lengths)):
            raise ValueError('uniform positive full-domain particle weights required')
        X,w,_=space.rule(order);query=np.column_stack([np.clip(X[:,a],axis[0],axis[-1]) for a,axis in enumerate(axes)])
        A=p.A[ids];a=A.reshape(-1,9);M=a[:,:,None]*a[:,None,:]
        shape=tuple(len(x) for x in axes)
        A2=RegularGridInterpolator(axes,A.reshape(*shape,3,3))(query)
        A4=RegularGridInterpolator(axes,M.reshape(*shape,9,9))(query)
        return cls(space,X,w,A2,A4,order=order)

    def field(self,q,D=None):
        return np.eye(3)+np.einsum('qnd,ni->qid',self.D if D is None else D,q)

    def material(self,q):
        F=self.field(q);iso=SimpleNamespace(mu=self.params.mu,lam=self.params.lam,k_f=0.)
        E,P,_=response(F,self.A2,self.A4,iso)
        d=(F.swapaxes(1,2)@F-np.eye(3)).reshape(-1,9)
        S=np.einsum('qij,qj->qi',self.A4,d).reshape(-1,3,3)
        E+=.5*self.params.k_f*np.einsum('qi,qi->q',d,S.reshape(-1,9))
        P+=2*self.params.k_f*F@S
        self.material_calls+=1;self.material_evaluations+=len(F)
        return float(self.weights@E),np.einsum('q,qid,qnd->ni',self.weights,P,self.D),P

    def geometry(self,q,need_H=True):
        F=self.field(q,self.gD);J=np.linalg.det(F)
        if not np.isfinite(F).all() or np.any(J<=.1):raise ValueError('inadmissible geometry')
        co=J[:,None,None]*np.linalg.inv(F).swapaxes(1,2)
        V=np.bincount(self.cells,weights=self.gw*J,minlength=self.top.cells)
        G=np.zeros((self.top.cells,len(q),3))
        for c in range(self.top.cells):
            ix=self.cells==c;G[c]=np.einsum('q,qid,qnd->ni',self.gw[ix],co[ix],self.gD[ix])
        H=self.top.assemble(self.gX,self.gw,self.cells,F,self.mobility)[0] if need_H else None
        self.geometry_calls+=1
        return V,G,H,float(J.min())

    def discrete_G(self,q0,q1):
        return sum(w*self.geometry(q,False)[1] for w,q in [(1/6,q0),(4/6,(q0+q1)/2),(1/6,q1)])

    def avf_force(self,q0,q1):
        a,w=np.polynomial.legendre.leggauss(3)
        return sum(ww/2*self.material(q0+(aa+1)/2*(q1-q0))[1] for aa,ww in zip(a,w))

    def load(self,time,amplitude=1.):
        # One smooth load/hold/unload cycle of 0.06 s, same physical quadrature.
        t=time/.06
        ramp=max(0.,min(t/.25,1.,(1-t)/.25))
        body=np.zeros_like(self.gX)
        body[:,1]=-amplitude*ramp*np.exp(-((self.gX[:,0]-.7)/.2)**2)
        return self.gN.T@(self.gw[:,None]*body)

    def save(self,path):
        import json
        np.savez_compressed(path,points=self.points,weights=self.weights,A2=self.A2,A4=self.A4,
           metadata=np.array(json.dumps(dict(space=self.space.identity(),params=vars(self.params),order=self.order))))

    @classmethod
    def load_package(cls,path):
        import json
        from .space import Space
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']));s=m['space']
            return cls(Space(tuple(s['shape']),s['bubbles'],s['lengths']),d['points'],d['weights'],d['A2'],d['A4'],order=m['order'],params=m['params'])

@dataclass
class State:
    particles:Particles
    q:np.ndarray
    v:np.ndarray
    p:np.ndarray
    content:np.ndarray
    time:float=0.
    step:int=0
    boundary_volume:float=0.
    dissipation:float=0.
    def digest(self):return sha_arrays(self.q,self.v,self.p,self.content,np.array([self.time,self.step,self.boundary_volume,self.dissipation]),np.frombuffer(self.particles.digest().encode(),dtype=np.uint8))

class Bridge:
    def __init__(self,space,ppc=3,groups=2,order=4,angle_shift=0.,rule="moments"):
        self.space=space;self.groups=groups;self.order=order
        if rule not in ("moments","fixed-positive"):raise ValueError("unknown rule")
        self.rule=rule
        p=seed(space,ppc,angle_shift);q,v,_=unload(space,p);op=FrozenOperator.from_particles(space,p,groups,order)
        pressure=np.full(space.cells,.2)
        self.state=State(p,q,v,pressure,op.capacity*pressure)
        self.attempts=0;self.accepted=0

    def prepare(self):
        start=time.perf_counter();s=self.state
        q,v,fit=unload(self.space,s.particles)
        fit['q_recovery']=float(np.max(abs(q-s.q)));fit['v_recovery']=float(np.max(abs(v-s.v)))
        if max(fit['q_recovery'],fit['v_recovery'],fit['weighted_history_residual'])>1e-7:
            raise ValueError('incompatible particle history; no silent projection/reset')
        op=(FrozenOperator.from_particles(self.space,s.particles,self.groups,self.order) if self.rule=="moments"
            else FrozenOperator.fixed_positive(self.space,s.particles,self.order))
        fit['seconds']=time.perf_counter()-start
        return op,q,v,fit

    def step(self,h=.005,*,fault=False,amplitude=1.):
        """Trial objects are private; commit only after all physical/Load checks."""
        self.attempts+=1;before=self.state.digest();old=self.state
        op,q0,v0,fit=self.prepare();tick=time.perf_counter()
        from .solve import advance
        q1,v1,p1,z,metrics=advance(op,q0,v0,old.p,h,old.time,amplitude)
        metrics['solve_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();p=old.particles.copy();N,D=self.space.basis(p.X)
        p.x+=N@(q1-q0);p.F+=np.einsum('qnd,ni->qid',D,q1-q0)
        p.v=N@v1;p.grad_v=np.einsum('qnd,ni->qid',D,v1)
        qcheck,vcheck,check=unload(self.space,p)
        if np.any(np.linalg.det(p.F)<=.1) or max(np.max(abs(qcheck-q1)),np.max(abs(vcheck-v1)))>1e-7:
            raise ValueError('Load/history compatibility failed')
        V=op.geometry(q1,False)[0];content=op.alpha*(V-op.top.V0)+op.capacity*p1
        defect=content-old.content+h*op.top.B@z
        if np.max(abs(defect))>1e-9:raise ValueError('committed fluid content mismatch')
        metrics.update(load_seconds=time.perf_counter()-tick,prepare=fit,load_fit=check,
          material_evaluations=op.material_evaluations,material_calls=op.material_calls,
          geometry_calls=op.geometry_calls,inner_particle_reads=0,particles=len(p.X),Nq=len(op.points),
          max_particle_displacement=float(np.max(np.linalg.norm(p.x-p.X,axis=1))),
          actual_content_defect=float(np.max(abs(defect))))
        trial=State(p,q1,v1,p1,content,old.time+h,old.step+1,
                    old.boundary_volume+h*float(np.sum(op.top.B@z)),old.dissipation+metrics['darcy_dissipation'])
        if fault:
            if self.state.digest()!=before:raise AssertionError('trial leaked into committed state')
            raise RuntimeError('injected before commit')
        self.state=trial;self.accepted+=1
        return metrics
