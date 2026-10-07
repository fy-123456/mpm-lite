"""CPU directions share one spectral preparation per bounded material slab."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import uuid
import numpy as np
from ..tensor_metrics import quadrature_axis,evaluate_gradient
from ..high_order_space import transpose_gradient
from ..research_d.stage2.contracts import array_digest
from ..research_d.identity import digest


def spectral_prepare(F,params):
    if not np.isfinite(F).all() or np.any(np.linalg.det(F)<=0):raise ValueError('invalid spectral state')
    c,Q=np.linalg.eigh(F.transpose(0,2,1)@F)
    if np.any(c<=0):raise ValueError('nonpositive material spectrum')
    logs=np.log(c);g=params.mu*logs+.5*params.lam*logs.sum(axis=1)[:,None];s=g/c
    ratio=(c[:,:,None]-c[:,None,:])/c[:,None,:]
    logdiv=np.ones_like(ratio);np.divide(np.log1p(ratio),ratio,out=logdiv,where=ratio!=0);logdiv/=c[:,None,:]
    divided=(params.mu*logdiv-s[:,None,:])/c[:,:,None]
    return F,c,Q,s,divided


def spectral_action(prepared,A,dF,params):
    F,c,Q,s,divided=prepared
    dC=F.transpose(0,2,1)@dF+dF.transpose(0,2,1)@F
    local=Q.transpose(0,2,1)@dC@Q;dS=divided*local
    trace=(np.diagonal(local,axis1=1,axis2=2)/c).sum(axis=1)
    idx=np.arange(3);dS[:,idx,idx]+=.5*params.lam*trace[:,None]/c
    S=(Q*s[:,None,:])@Q.transpose(0,2,1)
    out=dF@S+F@Q@dS@Q.transpose(0,2,1)
    FA=F@A;strain=np.einsum('qij,qij->q',FA,F)-1;contraction=np.einsum('qij,qij->q',FA,dF)
    return out+2*params.k_f*strain[:,None,None]*(dF@A)+4*params.k_f*contraction[:,None,None]*FA


@dataclass(frozen=True)
class CPUPrepared:
    owner:str
    key:str
    serial:int


class CPUStreamingLinearizations:
    """One nodal state, bounded direction batch, and one slab resident at a time.

    This CPU utility is explicitly opt-in for multiple directions. It does not
    replace the ordinary CPU material response or its AVF execution path.
    """
    def __init__(self,operator,*,max_bytes=512<<20,max_directions=4):
        if max_bytes<1 or max_directions<1:raise ValueError('positive CPU slice limits required')
        self.operator=operator;self.space=operator.space;self.max_bytes=max_bytes;self.max_directions=max_directions
        self.owner=uuid.uuid4().hex;self.serial=0;self.token=None;self.q=None;self.u=None
        self.signature=digest(dict(operator=operator.signature,source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dtype='float64',device='cpu'))
        self.stats=dict(prepared_states=0,spectral_slab_prepares=0,tangent_directions=0,peak_estimated_bytes=0)

    def _key(self,q):return digest(dict(operator=self.signature,state=array_digest(q)))

    def prepare(self,q):
        q=np.asarray(q,dtype=np.float64)
        if q.shape!=(self.space.ndof,3) or not np.isfinite(q).all():raise ValueError('invalid CPU full state')
        key=self._key(q)
        if self.token is not None and self.token.key==key:return self.token
        state_bytes=int(np.prod(self.space.shape))*3*8+q.nbytes
        if state_bytes>self.max_bytes:raise MemoryError('CPU prepared state exceeds byte limit')
        self.q=q.copy();self.q.setflags(write=False);self.u=self.space.nodes(q);self.u.setflags(write=False)
        self.serial+=1;self.token=CPUPrepared(self.owner,key,self.serial);self.stats['prepared_states']+=1
        return self.token

    def apply_many(self,token,q,directions):
        if token!=self.token or token.owner!=self.owner or token.key!=self._key(q):raise ValueError('foreign, expired or wrong-state CPU linearization')
        ds=np.asarray(directions,dtype=np.float64)
        if ds.ndim!=3 or ds.shape[1:]!=self.q.shape or not 1<=len(ds)<=self.max_directions or not np.isfinite(ds).all():
            raise ValueError('bounded finite batch of full directions required')
        s=self.space;maxorder=max(self.operator.rule.orders)
        maxpoints=maxorder**3*(len(s.edges[1])-1)*(len(s.edges[2])-1)
        # Conservative live-array allowance, including gradient/adjoint work.
        estimate=self.u.nbytes*(4+3*len(ds))+maxpoints*9*8*40+ds.nbytes
        if estimate>self.max_bytes:raise MemoryError('CPU direction/slab workspace estimate exceeds limit')
        self.stats['peak_estimated_bytes']=max(self.stats['peak_estimated_bytes'],estimate)
        du=[s.nodes(d) for d in ds];actions=[np.zeros_like(self.u) for _ in ds]
        for cell,order in enumerate(self.operator.rule.orders):
            qs=[quadrature_axis(e,order) for e in [s.edges[0][cell:cell+2],s.edges[1],s.edges[2]]]
            points,weights=[x[0] for x in qs],[x[1] for x in qs]
            F=np.eye(3)+evaluate_gradient((s.edges,s.p,self.u),points);flat=F.reshape(-1,3,3)
            prepared=spectral_prepare(flat,s.params);self.stats['spectral_slab_prepares']+=1
            A=np.broadcast_to(s.A,flat.shape)
            for i,nodal in enumerate(du):
                dF=evaluate_gradient((s.edges,s.p,nodal),points).reshape(-1,3,3)
                dP=spectral_action(prepared,A,dF,s.params)
                actions[i]+=transpose_gradient(dP.reshape(F.shape),s.edges,s.p,points,weights)
        result=[]
        for d,action in zip(ds,actions):
            value=s.adjoint(action);value[:s.n]+=s.Ks@d[:s.n]
            if not np.isfinite(value).all():raise ValueError('nonfinite CPU cached tangent')
            result.append(value)
        self.stats['tangent_directions']+=len(ds)
        return np.asarray(result)

    def report(self):return dict(self.stats,max_bytes=self.max_bytes,max_directions=self.max_directions,resident_state_bytes=0 if self.u is None else self.u.nbytes+self.q.nbytes)
