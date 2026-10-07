"""CPU prototype: kinetic-orthogonal dissipation of weak P2G velocity modes.

State z=(v,C_x,C_y,C_z) is weighted by the exact composite-APIC kinetic
metric. Complete affine velocity fields are protected. No elastic quantity
is changed. The dense SVD is intentionally limited to small verification grids.
"""
import itertools
import numpy as np
import scipy.sparse as sp

CORNERS=np.array(list(itertools.product((0,1),repeat=3)))
VISIBILITY=.1  # modes transmitting >=10% of their kinetic energy are untouched
RANK_TOL=1e-10
MAX_NODES=512
MAX_PARTICLES=2048


def pack(v,C):return np.concatenate([v]+[C[:,:,k] for k in range(3)])
def unpack(z):
    n=len(z)//4
    return z[:n].copy(),np.stack([z[(k+1)*n:(k+2)*n] for k in range(3)],axis=2)


def maps(x,m,h,grid_size=None):
    x=np.asarray(x,dtype=float);m=np.asarray(m,dtype=float)
    if x.ndim!=2 or x.shape[1]!=3 or m.shape!=(len(x),) or not len(x) or len(x)>MAX_PARTICLES:
        raise ValueError('small nonempty (N,3) particle state required (N<=2048)')
    if not np.isfinite(x).all() or not np.isfinite(m).all() or np.any(m<=0) or not np.isfinite(h) or h<=0:
        raise ValueError('finite positions and positive masses/dx required')
    q=x/h-.5;base=np.floor(q).astype(int);f=q-base
    centers=base[:,None,:]+CORNERS
    weights=np.prod(np.where(CORNERS[None,:,:],f[:,None,:],1-f[:,None,:]),axis=2)
    nodes=(centers[:,:,None,:]+CORNERS[None,None,:,:]).reshape(-1,3)
    w=np.repeat(weights,8,axis=1).ravel()/8
    particles=np.repeat(np.arange(len(x)),64);keep=w>0
    nodes=nodes[keep];w=w[keep];particles=particles[keep]
    if grid_size is not None and (np.any(nodes<0) or np.any(nodes>=np.asarray(tuple(grid_size)))):
        raise ValueError('complete in-domain particle-to-center-to-grid support required')
    nodes,inverse=np.unique(nodes,axis=0,return_inverse=True)
    if len(nodes)>MAX_NODES:raise ValueError('dense velocity dissipation prototype supports at most 512 active nodes')
    T=sp.coo_matrix((w,(inverse,particles)),shape=(len(nodes),len(x))).toarray()
    xn=nodes*h;mn=T@m;D=h*h*(f*(1-f)+.25)
    metric=np.concatenate([m]+[m*D[:,k] for k in range(3)])
    A=np.concatenate([T*m[None,:]]+[T*m[None,:]*(xn[:,k,None]-x[None,:,k]) for k in range(3)],axis=1)
    W=A/np.sqrt(mn[:,None]*metric[None,:])
    return dict(nodes=nodes,T=T,mass=mn,D=D,metric=metric,momentum_map=A,normalized=W)


class VelocityFilter:
    def __init__(self,x,m,h,dt,rate,mode='weak',grid_size=None):
        if mode not in ('null','weak') or not np.isfinite(dt) or dt<=0 or not np.isfinite(rate) or rate<=0:
            raise ValueError('null/weak mode and positive finite dt/rate required')
        self.x=np.array(x,copy=True);self.m=np.array(m,copy=True);self.h=h;self.mode=mode;self.dt=dt;self.rate=rate
        data=maps(x,m,h,grid_size);self.data=data;self.sqrt_metric=np.sqrt(data['metric'])
        n=len(x);Z=np.zeros((4*n,4));Z[:n,0]=1.;Z[:n,1:]=(x-np.average(x,axis=0,weights=m))/h
        for k in range(3):Z[(k+1)*n:(k+2)*n,k+1]=1/h
        self.affine=np.linalg.qr(self.sqrt_metric[:,None]*Z,mode='reduced')[0]
        W=data['normalized'];restricted=W-(W@self.affine)@self.affine.T
        _,singular,right=np.linalg.svd(restricted,full_matrices=False)
        keep=singular>RANK_TOL
        self.singular=singular[keep];self.right=right[keep]
        if mode=='null':self.alpha=np.ones(len(self.singular));self.alpha_null=0.
        else:
            attenuation=np.maximum(0.,1-self.singular**2/VISIBILITY)**2
            self.alpha=np.exp(-rate*dt*attenuation);self.alpha_null=float(np.exp(-rate*dt))

    def apply(self,v,C):
        y=self.sqrt_metric[:,None]*pack(v,C)
        if not np.isfinite(y).all():raise ValueError('finite velocity state required')
        affine=self.affine@(self.affine.T@y);other=y-affine
        coefficients=self.right@other;null=other-self.right.T@coefficients
        out=affine+self.right.T@(self.alpha[:,None]*coefficients)+self.alpha_null*null
        vp,Cp=unpack(out/self.sqrt_metric[:,None])
        K0=.5*float(np.sum(y*y));K1=.5*float(np.sum(out*out))
        expected=-.5*float(np.sum((1-self.alpha**2)[:,None]*coefficients**2)+(1-self.alpha_null**2)*np.sum(null**2))
        j0=self.data['normalized']@y;j1=self.data['normalized']@out
        visible=self.singular**2>=VISIBILITY
        info=dict(dissipation_mode=self.mode,dissipation_delta=K1-K0,dissipation_formula=expected,
            dissipation_identity_error=abs(K1-K0-expected),kinetic_before_dissipation=K0,kinetic_after_dissipation=K1,
            strict_null_energy=.5*float(np.sum(null**2)),weak_energy=.5*float(np.sum(coefficients[~visible]**2)+np.sum(null**2)),
            protected_affine_energy=.5*float(np.sum(affine**2)),dissipation_rate=self.rate,visibility_threshold=VISIBILITY,
            transfer_rank=len(self.singular),protected_affine_rank=4,raw_grid_kinetic_change=.5*float(np.sum(j1*j1)-np.sum(j0*j0)),
            momentum_change_norm=float(np.linalg.norm(self.m@(vp-v))),
            protected_affine_change_max=float(np.max(abs(self.affine.T@(out-y)))),
            preserved_visible_mode_error=float(np.max(abs(self.right[visible]@(out-y)))) if np.any(visible) else 0.)
        return vp,Cp,info


def prepare(s):
    """Perform every fallible decomposition BEFORE any particle commit."""
    from .tensile import grid_values
    cv=grid_values(s,s.center_v,s.mapped_centers)
    x=s.ptc_x.numpy()+s.dt*(s.mapped_S@cv)
    density=float(s.ptc_m.numpy().sum()/s.ptc_vol0.numpy().sum())
    rate=float(np.sqrt(s.aniso_params.mu/density)/s.dx)
    result=VelocityFilter(x,s.ptc_m.numpy(),s.dx,s.dt,rate,s.velocity_dissipation,s.grid_size)
    # Check the predicted return before commit as well; actual Warp-returned
    # velocities, not this prediction, are filtered by commit().
    dv=grid_values(s,s.center_dv,s.mapped_centers);G=grid_values(s,s.center_G,s.mapped_centers)
    L=(s.mapped_S@G.reshape(-1,9)).reshape(-1,3,3)
    v=s.flip_ratio*(s.ptc_v.numpy()+s.mapped_S@dv)+(1-s.flip_ratio)*(s.mapped_S@cv)
    beta=s.flip_ratio if s.affine_flip_ratio is None else s.affine_flip_ratio
    C=L+beta*s._apic_difference.numpy()
    result.apply(v,C)
    return result


def commit(s,plan):
    v=s.ptc_v.numpy().copy();C=s.ptc_C.numpy().copy()
    vp,Cp,info=plan.apply(v,C)
    s.ptc_v.assign(vp);s.ptc_C.assign(Cp)
    s.dissipation_stats=info
    s._dissipation_unfiltered=(v,C)
