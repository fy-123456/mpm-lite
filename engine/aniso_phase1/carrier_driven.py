"""Bounded v18 AVF experiment: moving grips and equivalent dense fast paths.

Existing v16/v17 solver stays untouched. The same elastic energy and kinetic
metric are used. Square support uses exact LU inversion instead of SVD; other
supports fall back to Geometry. Chord iterations solve the ORIGINAL residual,
with the unmodified exact tangent available for refresh. No shifts or damping.
"""
import itertools
import numpy as np
import scipy.linalg as la
from .carrier_joint import State,Geometry,gradient
from .carrier_avf import SITES
from .consistent_transfer import material_response
from .material_patch import carrier_map
from .unresolved_velocity import pack,unpack

CORNERS=np.array(list(itertools.product((0,1),repeat=3)))

def kinetic_metric(x,m,h):
    f=x/h-.5;f-=np.floor(f);D=h*h*(f*(1-f)+.25)
    return np.concatenate([m]+[m*D[:,j] for j in range(3)])


def position_map(x,m,h):
    q=x/h-.5;base=np.floor(q).astype(int);f=q-base
    centers=base[:,None,:]+CORNERS;weights=np.prod(np.where(CORNERS[None,:,:],f[:,None,:],1-f[:,None,:]),axis=2)
    nodes=(centers[:,:,None,:]+CORNERS[None,None,:,:]).reshape(-1,3)
    weight=np.repeat(weights,8,axis=1).ravel()/8;particles=np.repeat(np.arange(len(x)),64);keep=weight>0
    nodes=nodes[keep];weight=weight[keep];particles=particles[keep]
    lo=nodes.min(0);shape=nodes.max(0)-lo+1
    code=np.ravel_multi_index((nodes-lo).T,tuple(shape));unique,inv=np.unique(code,return_inverse=True)
    nodes=np.array(np.unravel_index(unique,tuple(shape))).T+lo
    if len(nodes)>512 or len(x)>2048:raise ValueError('bounded dense support exceeded')
    T=np.zeros((len(x),len(nodes)));np.add.at(T,(particles,inv),weight)
    return nodes,T


class EquivalentEnergy:
    """Batched BLAS patch products, exactly the existing row energy/force."""
    def __init__(self,original):
        self.original=original;self.__dict__.update(original.__dict__)
        ids,first,inverse=np.unique(self.ids,axis=0,return_index=True,return_inverse=True)
        if np.array_equal(self.P,self.P[first][inverse]):
            weights=np.bincount(inverse,weights=self.weights)
            self.ids=ids;self.P=self.P[first];self.weights=weights
    def evaluate(self,Y):
        F=gradient(self.B,Y);psi,Pm=material_response(F,self.A,self.params)
        r=self.P@Y[self.ids];Us=.5*float(np.sum(self.weights[:,None,None]*r*r))
        local=self.weights[:,None,None]*(self.P.swapaxes(1,2)@r);cf=np.zeros_like(Y)
        np.add.at(cf,self.ids.ravel(),local.reshape(-1,3))
        force=cf+sum(b.T@(self.V[:,None]*Pm[:,:,j]) for j,b in enumerate(self.B));Um=float(self.V@psi)
        return dict(U=Um+Us,Um=Um,Us=Us,F=F,P=Pm,force=force)
    def tangent(self,Y,Q=None):return self.original.tangent(Y,Q)


class FastGeometry:
    def __init__(self,s,e,m,h,clamped=True):
        nodes,Tg=position_map(s.x,m,h);lo=nodes.min(0);hi=nodes.max(0);shape=hi-lo+1
        if int(np.prod(shape))==len(nodes):
            local=s.Y/h
            if np.any(local<lo-1) or np.any(local>hi+1):raise ValueError('carriers exceed support extension')
            base=np.maximum(lo,np.minimum(np.floor(local).astype(int),hi-1));f=local-base
            ids=np.ravel_multi_index((base[:,None,:]+CORNERS-lo).reshape(-1,3).T,tuple(shape)).reshape(-1,8)
            weights=np.prod(np.where(CORNERS[None,:,:],f[:,None,:],1-f[:,None,:]),axis=2)
            N=np.zeros((len(s.Y),len(nodes)));N[np.arange(len(s.Y))[:,None],ids]=weights
        else:
            _,_,Ns=carrier_map(s.Y,nodes,h);N=Ns.toarray()
        if N.shape[0]!=N.shape[1]:
            g=Geometry(s,e,m,h,clamped);self.__dict__.update(g.__dict__);self.fast_square=False
            return
        lu,piv=la.lu_factor(N);rcond,info=la.lapack.dgecon(lu,la.norm(N,1))
        if info or rcond<1e-8:raise ValueError('square support too ill-conditioned for fast path')
        E=la.lu_solve((lu,piv),np.eye(len(N)));fixed=((nodes[:,0]*h<=.25)|(nodes[:,0]*h>=.75)) if clamped else np.zeros(len(nodes),bool)
        # Q need not be orthonormal. E Q is exactly the free-grid injection.
        Q=N[:,~fixed];self.fast_square=True;self.nodes=nodes;self.N=N;self.E=E;self.Q=Q;self.metric=kinetic_metric(s.x,m,h)
        self.T=Tg@E;invF=np.linalg.inv(gradient(e.B,s.Y));L=[sum(b*invF[:,k,j,None] for k,b in enumerate(e.B)) for j in range(3)]
        self.J=np.vstack([self.T]+L)
        ri=float(np.max(abs(N@E-np.eye(len(N)))));ce=float(np.max(abs(E[fixed]@Q))) if fixed.any() else 0.
        aff=max(float(np.max(abs(self.T@s.Y-s.x))),float(np.max(abs(gradient(L,s.Y)-np.eye(3)))))
        if max(ri,ce,aff)>1e-9:raise ValueError('fast support identity failure')
        self.info=dict(carriers=len(N),grid_nodes=len(nodes),free_scalar_dofs=Q.shape[1],right_inverse_error=ri,constraint_error=ce,affine_error=aff,rcond1=float(rcond))


def projection(J,q,z):
    Q,R=la.qr(np.sqrt(q)[:,None]*J,mode="economic");U,sv,_=la.svd(R,full_matrices=False);keep=sv>1e-12*sv[0];U=Q@U[:,keep]
    return U@(U.T@(np.sqrt(q)[:,None]*z))/np.sqrt(q)[:,None],int(keep.sum())


def displacement(t):
    if t<=0 or t>=1.1:return 0.
    if t<.5:return .0025*(1-np.cos(np.pi*t/.5))
    if t<=.6:return .005
    return .0025*(1+np.cos(np.pi*(t-.6)/.5))


class DrivenAVF:
    """hold preserves admissible residual (v16); driven reflects full residual.

For driven boundaries, z_mid=z_perp(full J)+J W, z1=2*z_mid-z0.
The reaction impulse is a=2 J.T Mz(JW-z0)+dt*fbar; Q.T a=0.
External work equals <a, W_bc>, explicitly separate from metric changes.
"""
    def __init__(self,s,e,m,h,moving=True,mode='hold',clamped=True,chord=True):
        if mode not in ('hold','driven'):raise ValueError(mode)
        self.state=s.clone();self.energy=EquivalentEnergy(e);self.m=m;self.h=h;self.moving=moving;self.mode=mode;self.clamped=clamped;self.chord=chord
        self.geometry=FastGeometry(s,e,m,h,clamped);self.steps=0;self.previous=None;self.Kcache=None;self.cache_time=None

    def boundary(self,g,t,dt):
        if self.mode=='hold':return np.zeros_like(self.state.Y),0.
        speed=(displacement(t+dt)-displacement(t))/dt;grid=np.zeros((len(g.nodes),3));grid[g.nodes[:,0]*self.h>=.75,0]=speed
        fixed=(g.nodes[:,0]*self.h<=.25)|(g.nodes[:,0]*self.h>=.75)
        if g.fast_square:Wb=g.N@grid
        else:Wb=la.lstsq(g.E[fixed],grid[fixed],cond=1e-11)[0]
        err=float(np.max(abs(g.E[fixed]@Wb-grid[fixed])))
        if err>1e-9:raise ValueError(f'unrepresentable moving grip: {err}')
        return Wb,speed

    def step(self,dt,max_iters=20):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        s=self.state;e=self.energy;g=FastGeometry(s,e,self.m,self.h,self.clamped) if self.moving else self.geometry
        Q=g.Q;J=g.J;Jr=J@Q;q=g.metric;z=pack(s.v,s.C);Wb,speed=self.boundary(g,s.time,dt);initial=e.evaluate(s.Y);n=Q.shape[1]
        projected,rank=projection(Jr if self.mode=='hold' else J,q,z)
        # Previous physical carrier velocity is merely a Newton predictor.
        u=la.lstsq(Q,self.previous-Wb,cond=1e-12)[0] if self.previous is not None else np.zeros((n,3))
        Mr=Jr.T@(q[:,None]*Jr);mass=2*la.block_diag(Mr,Mr,Mr)
        def trial(u):
            W=Wb+Q@u;d=J@W-z;val=float(np.sum(q[:,None]*d*d));force=np.zeros_like(s.Y)
            for a in SITES:
                out=e.evaluate(s.Y+a*dt*W);val+=.5/a*out['U'];force+=.5*out['force']
            residual=2*Jr.T@(q[:,None]*d)+dt*Q.T@force
            return val,residual,force,W
        if self.chord:
            if self.Kcache is None or s.time-self.cache_time>.01:
                self.Kcache=e.tangent(s.Y);self.cache_time=s.time
            blocks=[[Q.T@self.Kcache[a*e.n:(a+1)*e.n,b*e.n:(b+1)*e.n]@Q for b in range(3)] for a in range(3)]
            H=mass+.5*dt*dt*np.block(blocks);factor=la.cho_factor(H)
        accepted=False;backs=0
        for it in range(max_iters+1):
            val,r,force,W=trial(u);rn=float(la.norm(r))
            if rn<=1e-17:accepted=True;break
            if it==max_iters:break
            if not self.chord or it>=4:
                H=mass+sum(.5*a*dt*dt*e.tangent(s.Y+a*dt*W,Q) for a in SITES)
                du=la.solve(H,-r.T.ravel(),assume_a='sym').reshape(3,n).T
            else:du=la.cho_solve(factor,-r.T.ravel()).reshape(3,n).T
            descent=float(np.sum(r*du))
            if descent>=0:raise ValueError('non-descent unmodified AVF residual')
            for ls in range(25):
                alpha=2.**(-ls)
                try:nval=trial(u+alpha*du)[0]
                except ValueError:continue
                if nval<=val+1e-4*alpha*descent+1e-18:u+=alpha*du;backs+=ls;break
            else:raise RuntimeError('AVF line search failed')
        if not accepted:raise RuntimeError(f'AVF residual failure: {rn}')
        Y=s.Y+dt*W;out=e.evaluate(Y);zp=z+2*(J@W-projected);v,C=unpack(zp)
        x=s.x+dt*g.T@W if self.moving else s.x.copy();q1=kinetic_metric(x,self.m,self.h) if self.moving else q
        K0=.5*float(np.sum(q[:,None]*z*z));Kf=.5*float(np.sum(q[:,None]*zp*zp));K1=.5*float(np.sum(q1[:,None]*zp*zp))
        work=float(np.sum(force*(Y-s.Y)));impulse=2*J.T@(q[:,None]*(J@W-z))+dt*force
        boundary_work=float(np.sum(impulse*Wb)) if self.mode=='driven' else 0.
        defect=Kf-K0+work-boundary_work;quad=out['U']-initial['U']-work
        # Generalized force dual to the unit right-grip velocity, independent
        # of current loading speed, so force is measured during holds too.
        grid=np.zeros((len(g.nodes),3));right=g.nodes[:,0]*self.h>=.75;grid[right,0]=1.
        fixed=(g.nodes[:,0]*self.h<=.25)|right
        lift=g.N@grid if g.fast_square else la.lstsq(g.E[fixed],grid[fixed],cond=1e-11)[0]
        reaction=float(np.sum(impulse*lift)/dt)
        bc=g.E@W;expected=np.zeros_like(bc);expected[right,0]=speed
        bcerr=float(np.max(abs(bc[fixed]-expected[fixed]))) if self.clamped else 0.
        row=dict(step=self.steps+1,time=s.time+dt,mode=self.mode,moving=self.moving,dt=dt,
            material_J=out['Um'],stabilization_J=out['Us'],kinetic_J=K1,total_J=K1+out['U'],delta_total_J=K1-K0+out['U']-initial['U'],
            metric_change_J=K1-Kf,boundary_work_J=boundary_work,reaction_N=reaction,loading_speed=speed,command_displacement=displacement(s.time+dt),
            kinetic_force_work_defect_J=defect,potential_quadrature_error_J=quad,newton_residual=rn,newton_iterations=it,line_search_backtracks=backs,
            history_commit_max=float(np.max(abs(out['F']-initial['F']-gradient(e.B,Y-s.Y)))),grip_velocity_error=bcerr,
            stress_rms_Pa=float(np.sqrt(np.mean(np.sum(out['P']**2,axis=(1,2))))),min_det_F=float(np.linalg.det(out['F']).min()),kinetic_projection_rank=rank,**g.info)
        if bcerr>1e-8:raise ValueError('grip velocity constraint failure')
        self.state=State(x,Y,v,C,s.time+dt);self.geometry=g;self.previous=W;self.steps+=1;self.last=dict(old=s,force=force,W=W,impulse=impulse,projected=projected)
        return row
