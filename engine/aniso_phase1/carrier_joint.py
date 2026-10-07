"""Small CPU research solver: material-DOF support lift and joint kinetic solve.

The elastic potential is exactly the v15 carried particle/patch potential.
Eulerian nodes are a representation, not additional independent elastic DOFs.
A right inverse of N preserves every grid polynomial through degree two.
No stiffness, mass shift, damping, energy offset, or history reset is added.

This is a bounded dense prototype, not a production Lite solver. N must have
full row rank; complete support and <=512 nodes are required. Moving geometry
changes the APIC kinetic metric; that energy change is reported separately.
"""
import itertools
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from .material_patch import carrier_map
from .selective_patch import polynomial
from .unresolved_velocity import maps as apic_maps, pack, unpack
from .consistent_transfer import material_response
from .history_increment import material_tangent
from .types import AnisotropicMaterialParams

CORNERS=np.array(list(itertools.product((0,1),repeat=3)))
MODES=('legacy_split','common_split','adjoint_split','joint_projected','joint')


def support_lift(N, nodes, h):
    """N E=I; E N S=S for all degree<=2 grid fields S, when feasible."""
    N=np.asarray(N.toarray() if sp.issparse(N) else N,float)
    if max(N.shape)>512 or N.shape[0]>N.shape[1]:
        raise ValueError('carrier lift needs at most 512 nodes and full row support')
    U,s,Vh=la.svd(N,full_matrices=False)
    if s[-1]<=1e-11*s[0]:raise ValueError('carrier interpolation lost row rank')
    pinv=(Vh.T/s)@U.T
    X=np.asarray(nodes,float)*h;S=polynomial((X-X.mean(0))/h)
    NS=N@S;ns=la.svdvals(NS)
    if ns[-1]<=1e-11*ns[0]:raise ValueError('support cannot preserve quadratic fields')
    E=pinv+(S-pinv@NS)@la.pinv(NS,rtol=1e-12)
    err=max(float(np.max(abs(N@E-np.eye(len(N))))),float(np.max(abs(E@NS-S))))
    if err>1e-9:raise ValueError(f'inaccurate support lift: {err}')
    return E,dict(carriers=N.shape[0],grid_nodes=N.shape[1],row_rank=len(s),
                  right_inverse_error=float(np.max(abs(N@E-np.eye(len(N))))),
                  quadratic_reproduction_error=float(np.max(abs(E@NS-S))),
                  smallest_N_singular=float(s[-1]),lift_norm=float(la.norm(E,2)))


def current_gradient(x,nodes,h):
    """Independent two-level Lite center gradient, on the active APIC nodes."""
    lookup={tuple(n):i for i,n in enumerate(nodes)}
    q=x/h-.5;base=np.floor(q).astype(int);f=q-base
    G=[np.zeros((len(x),len(nodes))) for _ in range(3)]
    for a in CORNERS:
        w=np.prod(np.where(a,f,1-f),axis=1)
        for b in CORNERS:
            keep=np.flatnonzero(w!=0)
            ids=np.array([lookup[tuple(n)] for n in base[keep]+a+b])
            for k in range(3):G[k][keep,ids]+=w[keep]*(2*b[k]-1)/(4*h)
    return G


def gradient(B,Y):return np.stack([b@Y for b in B],axis=2)


class CarrierEnergy:
    def __init__(self,G0,R,V,A,ids,P,weights,params=None):
        self.V=np.array(V,copy=True);self.A=np.array(A,copy=True)
        self.params=params or AnisotropicMaterialParams(10.,20.,200.)
        G=[g.toarray() if sp.issparse(g) else np.asarray(g) for g in G0]
        self.B=tuple(sum(g*R[:,j,k,None] for j,g in enumerate(G)) for k in range(3))
        self.ids=np.array(ids,copy=True);self.P=np.array(P,copy=True);self.weights=np.array(weights,copy=True)
        self.n=self.B[0].shape[1];m=ids.shape[1]
        self.Ks=sp.csr_matrix(((weights[:,None,None]*(P.swapaxes(1,2)@P)).ravel(),
                 (np.repeat(ids,m,axis=1).ravel(),np.tile(ids,(1,m)).ravel())),shape=(self.n,self.n)).toarray()

    def evaluate(self,Y):
        F=gradient(self.B,Y);psi,Pm=material_response(F,self.A,self.params)
        r=np.einsum('cij,cja->cia',self.P,Y[self.ids]);Us=.5*float(np.sum(self.weights[:,None,None]*r*r))
        # Residual evaluation avoids subtracting nearly equal O(1) terms.
        cf=np.zeros_like(Y);local=self.weights[:,None,None]*np.einsum('cji,cja->cia',self.P,r)
        np.add.at(cf,self.ids.ravel(),local.reshape(-1,3))
        force=cf+sum(b.T@(self.V[:,None]*Pm[:,:,k]) for k,b in enumerate(self.B))
        Um=float(self.V@psi)
        return dict(U=Um+Us,Um=Um,Us=Us,F=F,P=Pm,force=force)

    def tangent(self,Y,Q=None):
        Q=np.eye(self.n) if Q is None else Q
        F=gradient(self.B,Y);B=[b@Q for b in self.B];H=np.empty((len(F),9,9))
        for k in range(9):
            d=np.zeros_like(F);d.reshape(-1,9)[:,k]=1
            H[:,:,k]=material_tangent(F,self.A,d,self.params).reshape(-1,9)
        blocks=[[sum(B[b].T@((self.V*H[:,3*a+b,3*c+d])[:,None]*B[d]) for b in range(3) for d in range(3)) for c in range(3)] for a in range(3)]
        for a in range(3):blocks[a][a]+=Q.T@self.Ks@Q
        K=np.block(blocks)
        if np.max(abs(K-K.T))>1e-8*max(np.max(abs(K)),1.):raise ValueError('asymmetric material tangent')
        return (K+K.T)/2


@dataclass
class State:
    x: np.ndarray
    Y: np.ndarray
    v: np.ndarray
    C: np.ndarray
    time: float=0.
    def clone(self):return State(*(np.array(getattr(self,k),copy=True) for k in ('x','Y','v','C')),self.time)


class Geometry:
    def __init__(self,state,energy,m,h,clamped=True):
        data=apic_maps(state.x,m,h);self.data=data;self.nodes=data['nodes'];self.metric=data['metric']
        _,_,Ns=carrier_map(state.Y,self.nodes,h);self.N=Ns.toarray()
        self.E,self.info=support_lift(self.N,self.nodes,h)
        fixed=((self.nodes[:,0]*h<=.25)|(self.nodes[:,0]*h>=.75)) if clamped else np.zeros(len(self.nodes),bool)
        C=self.E[fixed]
        self.Q=la.null_space(C,rcond=1e-11) if len(C) else np.eye(energy.n)
        if not self.Q.shape[1]:raise ValueError('no free material coordinates')
        self.info['constraint_error']=float(np.max(abs(C@self.Q))) if len(C) else 0.
        self.T=data['T'].T@self.E
        G=current_gradient(state.x,self.nodes,h)
        self.Jold=np.vstack([self.T]+[g@self.E for g in G])
        invF=np.linalg.inv(gradient(energy.B,state.Y))
        L=[sum(b*invF[:,k,j,None] for k,b in enumerate(energy.B)) for j in range(3)]
        self.J=np.vstack([self.T]+L)
        self.Mgrid=self.E.T@(data['mass'][:,None]*self.E)
        self.M=self.J.T@(self.metric[:,None]*self.J)
        U,singular,Vh=la.svd(np.sqrt(self.metric)[:,None]*self.J,full_matrices=False)
        keep=singular>1e-12*singular[0]
        self.kinetic_U=U[:,keep];self.kinetic_s=singular[keep];self.kinetic_V=Vh[keep].T
        self.info.update(kinetic_rank=int(keep.sum()),kinetic_null_modes=int((~keep).sum()),
                         smallest_retained_kinetic_singular=float(singular[keep][-1]),
                         independent_scalar_dofs=energy.n,free_scalar_dofs=self.Q.shape[1],
                         affine_position_error=float(np.max(abs(self.T@state.Y-state.x))),
                         affine_gradient_error=float(np.max(abs(gradient(L,state.Y)-np.eye(3)))))
        if max(self.info['affine_position_error'],self.info['affine_gradient_error'])>1e-9:
            raise ValueError('joint maps failed affine reproduction')


class CarrierJointSolver:
    """Transactional steps. All fallible work precedes physical state commit."""
    def __init__(self,state,energy,m,h,mode='joint',moving=False,clamped=True):
        if mode not in MODES:raise ValueError(mode)
        self.state=state.clone();self.energy=energy;self.m=np.array(m,copy=True);self.h=h
        self.mode=mode;self.moving=moving;self.clamped=clamped
        self.geometry=Geometry(self.state,energy,m,h,clamped);self.steps=0

    def step(self,dt,max_iters=15):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        s=self.state;e=self.energy;g=Geometry(s,e,self.m,self.h,self.clamped) if self.moving else self.geometry
        Q=g.Q;z=pack(s.v,s.C);q=g.metric;initial=e.evaluate(s.Y)
        if self.mode in ('legacy_split','common_split'):
            rhs=g.E.T@g.data['momentum_map']@z
            w0=la.solve(g.Mgrid,rhs,assume_a='pos')
        else:
            coeff=g.kinetic_U.T@(np.sqrt(q)[:,None]*z)
            w0=g.kinetic_V@(coeff/g.kinetic_s[:,None])
        zperp=z-(g.kinetic_U@(g.kinetic_U.T@(np.sqrt(q)[:,None]*z)))/np.sqrt(q)[:,None]
        J=g.Jold if self.mode=='legacy_split' else g.J
        M=g.M if self.mode in ('joint','joint_projected') else g.Mgrid
        Mr=Q.T@M@Q
        br=Q.T@(g.J.T@(q[:,None]*z)) if self.mode in ('joint','joint_projected') else Q.T@(M@w0)
        n=Q.shape[1]
        if self.mode=='joint':
            # Preserve ALL old velocity history orthogonal to the admissible
            # (clamped) joint motion, avoiding an initial boundary projection
            # that would dissipate finite energy as dt -> 0.
            Uj,sj,_=la.svd(np.sqrt(q)[:,None]*(g.J@Q),full_matrices=False)
            Uj=Uj[:,sj>1e-12*sj[0]]
            zperp=z-(Uj@(Uj.T@(np.sqrt(q)[:,None]*z)))/np.sqrt(q)[:,None]
        # Start from a constrained least-squares projection; singular kinetic
        # directions remain unconstrained until the ORIGINAL elastic energy acts.
        if self.mode in ('joint','joint_projected'):
            # A well-conditioned affine predictor avoids normal-equation
            # roundoff in weak inertial directions. Only the starting iterate
            # changes; the full unmodified Newton system is still solved.
            Paff=np.column_stack((np.ones(e.n),(s.Y-s.Y.mean(0))/self.h))
            Aaff=np.sqrt(q)[:,None]*(g.J@Paff)
            coeff=la.lstsq(Aaff,np.sqrt(q)[:,None]*z,cond=1e-12)[0]
            u=Q.T@(Paff@coeff)
        else:u=la.lstsq(Mr,br,cond=1e-12)[0]
        def trial(u):
            W=Q@u;Y=s.Y+dt*W;v=e.evaluate(Y);d=W-w0
            potential=(.5*float(np.sum(q[:,None]*(J@W-z)**2)) if self.mode in ('joint','joint_projected') else .5*float(np.sum(d*(M@d))))+v['U']
            residual=Mr@u-br+dt*Q.T@v['force']
            return potential,residual,v,Y,W
        accepted=False;backtracks=0
        for iteration in range(max_iters+1):
            val,r,new,Y,W=trial(u);rn=float(la.norm(r));tol=max(1e-14,dt*1e-9)
            if rn<=tol:accepted=True;break
            if iteration==max_iters:break
            H=la.block_diag(Mr,Mr,Mr)+dt*dt*e.tangent(Y,Q)
            du=la.solve(H,-r.T.ravel(),assume_a='sym').reshape(3,n).T
            descent=float(np.sum(r*du))
            if not descent<0:raise ValueError('non-descent unmodified tangent')
            for ls in range(25):
                alpha=2.**(-ls)
                try:next_val=trial(u+alpha*du)[0]
                except ValueError:continue
                if next_val<=val+1e-4*alpha*descent+1e-18:
                    u=u+alpha*du;backtracks+=ls;break
            else:raise RuntimeError('joint line search failed')
        if not accepted:raise RuntimeError(f'joint Newton did not converge: {rn}')
        zp=zperp+J@W if self.mode in ('joint','joint_projected') else z+J@(W-w0)
        delta_z=zp-z;vp,Cp=unpack(zp)
        xp=s.x+dt*g.T@W if self.moving else s.x.copy()
        qnext=apic_maps(xp,self.m,self.h)['metric'] if self.moving else q
        K0=.5*float(np.sum(q[:,None]*z*z));Kf=.5*float(np.sum(q[:,None]*zp*zp))
        K1=.5*float(np.sum(qnext[:,None]*zp*zp));dy=Y-s.Y
        work=float(np.sum(new['force']*dy));kinetic_loss=.5*float(np.sum(q[:,None]*delta_z*delta_z))
        defect=(Kf-K0)+kinetic_loss+work
        row=dict(step=self.steps+1,time=s.time+dt,mode=self.mode,moving=self.moving,
            material_J=new['Um'],stabilization_J=new['Us'],kinetic_J=K1,total_J=K1+new['U'],
            delta_total_J=K1-K0+new['U']-initial['U'],metric_change_J=K1-Kf,
            kinetic_increment_norm_J=kinetic_loss,potential_backward_euler_loss_J=work-(new['U']-initial['U']),
            kinetic_force_work_defect_J=defect,newton_residual=rn,newton_iterations=iteration,line_search_backtracks=backtracks,
            momentum_change_norm=float(la.norm(self.m@(vp-s.v))),
            reference_rebuild_delta_J=0.,history_commit_max=float(np.max(abs(new['F']-(initial['F']+gradient(e.B,dy))))),
            stress_rms_Pa=float(np.sqrt(np.mean(np.sum(new['P']**2,axis=(1,2))))),
            min_det_F=float(np.linalg.det(new['F']).min()),max_carrier_speed=float(np.max(la.norm(W,axis=1))),**g.info)
        # Last fallible calculations completed: no partial state on errors.
        self.state=State(xp,Y,vp,Cp,s.time+dt);self.geometry=g;self.steps+=1
        return row
