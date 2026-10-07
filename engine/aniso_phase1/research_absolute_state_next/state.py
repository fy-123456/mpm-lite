"""Fixed material coordinates own the absolute configuration.

Route A retains the qualified original formal space and Ks. SH is an auxiliary
transfer followed by an explicit residual correction, NOT the previously
proposed C1-masked updated physical space. No production kernel is replaced.
"""
from dataclasses import dataclass
import json
import numpy as np
from engine.aniso_phase1.research_constrained_history_next.formal import FormalCandidate,paired,sh_axis,sampling,PACKAGE_SHA
from engine.aniso_phase1.research_unified_lite_poro.model import sha_arrays

SCHEMA='formal-material-absolute-with-corrected-SH-v1'

@dataclass
class State:
    q:np.ndarray
    velocity:np.ndarray
    time:float=0.
    step:int=0
    def digest(self):return sha_arrays(self.q,self.velocity,np.array([self.time,self.step]))

class MaterialStateModel:
    def __init__(self):
        self.candidate=FormalCandidate();self.r=self.candidate.reduction;self.space=self.r.parent
        self.identity=dict(schema=SCHEMA,package_sha256=PACKAGE_SHA,reduction_sha256=self.r.signature,
            physical_space='original fixed material carrier plus 144 original local functions',
            stabilization='original absolute Y quadratic',transfer='SH carrier plus exact material residual; local evaluated in material coordinates',
            density=1.,time_layer='committed-step-end',state_owner='independent material coordinates',production=False)
        P=self.r.P[:self.space.n];self.Ks=P.T@self.space.Ks@P
        self.Ks=(self.Ks+self.Ks.T)/2
    def rest(self):
        q=np.zeros((self.r.P.shape[1],3));return State(q,q.copy())
    def validate(self,s):
        self.r.expand(s.q);self.r.velocity(s.velocity)
        if not np.isfinite(s.time) or s.time<0 or s.step<0 or int(s.step)!=s.step:raise ValueError('invalid time layer')
    def basis(self,X):
        X=np.asarray(X,float);s=self.space
        if X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():raise ValueError('finite material points required')
        if any(np.any((X[:,a]<e[0])|(X[:,a]>e[-1])) for a,e in enumerate(s.edges)):raise ValueError('material point outside declared domain')
        Q=[sampling(e,2,X[:,a]) for a,e in enumerate(s.oldedges)]
        C=paired(Q)@s.oldA
        DC=np.stack([paired([sampling(s.oldedges[j],2,X[:,j],True) if a==j else Q[j] for j in range(3)])@s.oldA for a in range(3)],axis=-1)
        L,DL=self.candidate.local_points(X)
        return np.c_[C,L],np.concatenate([DC,DL],axis=1)
    def fields(self,state,X):
        self.validate(state);B,D=self.basis(X);full=self.r.expand(state.q)
        x=X+B@full;F=np.eye(3)+np.einsum('qna,ni->qia',D,full)
        if np.any(np.linalg.det(F)<=.1):raise ValueError('inadmissible material configuration')
        v=B@self.r.velocity(state.velocity)
        return x,F,v
    def absolute_Y(self,q):return self.space.reference[:self.space.n]+self.r.expand(q)[:self.space.n]
    def stabilization(self,q,direction=None):
        Y=self.absolute_Y(q);KY=self.space.Ks@Y
        return .5*float(np.sum(Y*KY)),self.r.P[:self.space.n].T@KY,(None if direction is None else self.Ks@direction)
    def corrected_transfer(self,state,X,increment):
        """CPU transfer contract, retaining raw/corrected fields separately.

        The residual is evaluated at Load. It is not a free improvement in SH
        accuracy and its particle-dependent evaluation cost must be counted.
        """
        x,F,_=self.fields(state,X);B,D=self.basis(X);s=self.space;full=self.r.velocity(increment)
        Q=[sh_axis(e,x[:,a]) for a,e in enumerate(s.oldedges)]
        H=paired(Q)@s.oldA
        Dx=np.stack([paired([sh_axis(s.oldedges[j],x[:,j],derivative=True) if a==j else Q[j] for j in range(3)])@s.oldA for a in range(3)],axis=-1)
        DH=np.einsum('qni,qij->qnj',Dx,F)
        raw=H@full[:s.n]+B[:,s.n:]@full[s.n:]
        raw_D=np.einsum('qna,ni->qia',DH,full[:s.n])+np.einsum('qna,ni->qia',D[:,s.n:],full[s.n:])
        correction=(B[:,:s.n]-H)@full[:s.n]
        correction_D=np.einsum('qna,ni->qia',D[:,:s.n]-DH,full[:s.n])
        return dict(raw=raw,raw_material_gradient=raw_D,correction=correction,correction_material_gradient=correction_D,
            displacement=raw+correction,material_gradient=raw_D+correction_D,
            spatial_gradient=(raw_D+correction_D)@np.linalg.inv(F),exact=B@full,
            exact_material_gradient=np.einsum('qna,ni->qia',D,full))
    def trial(self,state,increment,velocity,dt,*,fault=None):
        self.validate(state)
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive time increment required')
        self.r.velocity(increment);self.r.velocity(velocity)
        trial=State(state.q+increment,np.array(velocity,copy=True),state.time+dt,state.step+1);self.validate(trial)
        if fault:raise RuntimeError('injected before commit')
        return trial
    def save(self,state,path):
        self.validate(state)
        np.savez_compressed(path,q=state.q,velocity=state.velocity,metadata=np.array(json.dumps(dict(identity=self.identity,time=state.time,step=state.step,digest=state.digest()),sort_keys=True)))
    def load(self,path):
        with np.load(path,allow_pickle=False) as d:
            m=json.loads(str(d['metadata']))
            if m.get('identity')!=self.identity:raise ValueError('foreign absolute-state identity')
            s=State(d['q'].copy(),d['velocity'].copy(),m['time'],m['step'])
        self.validate(s)
        if s.digest()!=m['digest']:raise ValueError('absolute state digest mismatch')
        return s
