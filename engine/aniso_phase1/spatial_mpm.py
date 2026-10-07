"""Small implicit updated-Lagrangian MPM with rebuilt spatial Q1 grids.

Particles are the material integration sites and authoritative x/v/F history.
The full consistent mass is solved without lumping, clipping or regularization.
Every step rebuilds weights from current spatial particle positions. This is
an executable MPM baseline, not center-quadrature Lite or a fixed material mesh.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.linalg import cholesky, solve_triangular, cho_factor, cho_solve
from .consistent_transfer import material_response
from .spatial_basis import sample_basis, extend_weak_nodes, affine_weights, residual_modes
from .history_increment import frozen, material_tangent
from .types import AnisotropicMaterialParams


@dataclass(frozen=True)
class ParticleHistory:
    X: np.ndarray
    x: np.ndarray
    v: np.ndarray
    F: np.ndarray
    mass: np.ndarray
    volume: np.ndarray
    A: np.ndarray

    def __post_init__(self):
        n=len(self.x)
        for key,shape in [('X',(n,3)),('x',(n,3)),('v',(n,3)),('F',(n,3,3)),
                          ('mass',(n,)),('volume',(n,)),('A',(n,3,3))]:
            a=np.asarray(getattr(self,key),dtype=float)
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid particle '+key)
            object.__setattr__(self,key,frozen(a))
        if n==0 or np.any(self.mass<=0) or np.any(self.volume<=0) or np.any(np.linalg.det(self.F)<=0):
            raise ValueError('positive particle mass, reference volume and Jacobian required')


class SpatialGrid:
    """Freeze N and spatial gradients at x_p^n for exactly one implicit step."""
    def __init__(self,x,mass,h,origin=(0.,0.,0.),kernel='q1',boundary='raw',support_threshold=.01):
        x=np.asarray(x,dtype=float);mass=np.asarray(mass,dtype=float);origin=np.asarray(origin,dtype=float)
        if not np.isfinite(h) or h<=0 or origin.shape!=(3,) or not np.isfinite(origin).all():
            raise ValueError('finite positive cell width and 3-vector origin required')
        if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.isfinite(x).all() or mass.shape!=(len(x),) or not np.isfinite(mass).all() or np.any(mass<=0):
            raise ValueError('finite particle positions and positive masses required')
        self.h,self.origin=float(h),origin.copy()
        self.cells=np.floor((x-origin)/h).astype(np.int64)
        coordinates,N,D=sample_basis(x,h,origin,kernel)
        diagonal=np.asarray(N.power(2).T@mass).ravel()
        counts=np.asarray((abs(N)>1e-12).sum(axis=0)).ravel()
        self.support_diagnostics=dict(raw_nodes=len(coordinates),raw_support_min=int(counts.min()),
            raw_mass_diagonal_min=float(diagonal.min()),raw_mass_diagonal_max=float(diagonal.max()),
            weak_nodes=int(np.sum(diagonal<support_threshold*diagonal.max())),constrained_nodes=0)
        if boundary=='affine_extension':
            N,D,coordinates,constrained=extend_weak_nodes(N,D,coordinates,h,origin,mass,support_threshold)
            self.support_diagnostics['constrained_nodes']=constrained
        elif boundary!='raw':raise ValueError('boundary must be raw or affine_extension')
        self.coordinates=coordinates;self.x=origin+h*coordinates
        self.N,self.D=N,D;self.mass=mass.copy();self.enrichment_modes=0;self.enrichment_radius=0.
        self._factor_mass()

    def _factor_mass(self):
        self.M=(self.N.T@self.N.multiply(self.mass[:,None])).toarray()
        eigen=np.linalg.eigvalsh(self.M)
        if eigen[0]<=1e-12*eigen[-1]:
            raise ValueError('spatial consistent mass is rank deficient; no regularization')
        self.minimum_eigenvalue=float(eigen[0]);self.mass_condition=float(eigen[-1]/eigen[0])
        self.L=cholesky(self.M,lower=True)

    def enrich_velocity(self,x,vp):
        residual=vp-self.N@self.project(vp)
        # Remove accumulated roundoff in the full-mass projection.
        residual-=self.N@self.project(residual)
        Q,Dq,s,radius=residual_modes(x,self.mass,residual)
        self.enrichment_modes=Q.shape[1];self.enrichment_radius=radius
        if not Q.shape[1]:return
        anchors=np.stack([(self.mass*q*q)@x/np.sum(self.mass*q*q) for q in Q.T])
        W=np.stack([affine_weights(self.x,anchor,self.h) for anchor in anchors])
        # N_aug=[N-QW,Q], x_aug=[x_nodes,anchors]. W1=1 and Wx=anchors,
        # so partition of unity and linear reproduction remain exact.
        self.N=sparse.hstack([self.N-sparse.csr_matrix(Q@W),sparse.csr_matrix(Q)],format='csr')
        self.D=tuple(sparse.hstack([d-sparse.csr_matrix(dq@W),sparse.csr_matrix(dq)],format='csr') for d,dq in zip(self.D,Dq))
        self.x=np.vstack([self.x,anchors]);self._factor_mass()

    def to_y(self,v):return self.L.T@v
    def from_y(self,y):return solve_triangular(self.L.T,y,lower=False,check_finite=False)
    def force_to_y(self,f):return solve_triangular(self.L,f,lower=True,check_finite=False)
    def project(self,vp):
        vp=np.asarray(vp,dtype=float)
        if vp.shape!=(len(self.mass),3) or not np.isfinite(vp).all():raise ValueError('finite particle velocities required')
        return self.from_y(self.force_to_y(self.N.T@(self.mass[:,None]*vp)))
    def gradient(self,v):return np.stack([D@v for D in self.D],axis=2)


class FrozenSpatialStep:
    def __init__(self,state,h,params,origin=(0.,0.,0.),response=None,kernel='q1',boundary='raw',support_threshold=.01,velocity_enrichment=False):
        self.state,self.params=state,params
        self.grid=SpatialGrid(state.x,state.mass,h,origin,kernel,boundary,support_threshold)
        if velocity_enrichment:self.grid.enrich_velocity(state.x,state.v)
        # B_J = sum_d dN/dx_d F^n_dJ, so Ftrial = F^n + dt B v.
        self.B=tuple(sum((D.multiply(state.F[:,d,J,None]) for d,D in enumerate(self.grid.D)),sparse.csr_matrix(self.grid.N.shape)).tocsr() for J in range(3))
        self.response=response or (lambda F:material_response(F,state.A,params))

    def trial_F(self,v,dt):
        return self.state.F+dt*np.stack([B@v for B in self.B],axis=2)

    def elastic(self,v,dt):
        F=self.trial_F(v,dt);psi,P=self.response(F)
        f=sum((B.T@(self.state.volume[:,None]*P[:,:,J]) for J,B in enumerate(self.B)),np.zeros_like(v))
        return float(self.state.volume@psi),f,F

    def potential(self,v,predictor,dt):
        U,f,F=self.elastic(v,dt);dv=v-predictor
        return U+.5*float(np.sum(dv*(self.grid.M@dv))),self.grid.M@dv+dt*f

    def tangent(self,v,direction,dt):
        F=self.trial_F(v,dt);dF=np.stack([B@direction for B in self.B],axis=2)
        dP=material_tangent(F,self.state.A,dF,self.params)
        action=sum((B.T@(self.state.volume[:,None]*dP[:,:,J]) for J,B in enumerate(self.B)),np.zeros_like(direction))
        return self.grid.M@direction+dt*dt*action


    def whitened_hessian(self,v,dt):
        """Exact 3x3 block assembly; mass whitening changes only coordinates."""
        F=self.trial_F(v,dt);n=len(v)
        C=np.empty((len(F),3,3,3,3))
        for b in range(3):
            for K in range(3):
                dF=np.zeros_like(F);dF[:,b,K]=1.
                C[:,:,:,b,K]=material_tangent(F,self.state.A,dF,self.params)
        H=np.eye(3*n)
        for a in range(3):
            for b in range(3):
                block=sum((BJ.T@BK.multiply((self.state.volume*C[:,a,J,b,K])[:,None])
                    for J,BJ in enumerate(self.B) for K,BK in enumerate(self.B)),sparse.csr_matrix((n,n))).toarray()
                left=self.grid.force_to_y(block)
                white=self.grid.force_to_y(left.T).T
                H[a::3,b::3]+=dt*dt*white
        return .5*(H+H.T)


def minimize_newton(objective,hessian,start,tolerance=1e-10,max_iterations=40):
    """Small dense Newton, Armijo on the original potential, no mass changes."""
    y=start.copy();rejected=0
    for iteration in range(max_iterations+1):
        value,gradient=objective(y)
        if np.isfinite(value) and np.max(abs(gradient))<=tolerance:return y,iteration,rejected
        if not np.isfinite(value) or iteration==max_iterations:raise RuntimeError('MPM Newton residual did not converge')
        H=hessian(y)
        try:factor=cho_factor(H,lower=True)
        except np.linalg.LinAlgError:
            # A search-direction modification, not physical mass/stiffness replacement.
            shift=max(0.,1e-6-float(np.linalg.eigvalsh(H)[0]))
            factor=cho_factor(H+shift*np.eye(len(H)),lower=True)
        direction=-cho_solve(factor,gradient)
        slope=float(gradient@direction)
        if slope>=0:direction=-gradient;slope=-float(gradient@gradient)
        alpha=1.
        for _ in range(40):
            trial=y+alpha*direction;energy,_=objective(trial)
            if np.isfinite(energy) and energy<=value+1e-4*alpha*slope:
                y=trial;break
            rejected+=1;alpha*=.5
        else:raise RuntimeError('MPM Newton line search failed')
    raise RuntimeError('unreachable Newton termination')


class SpatialMPM:
    def __init__(self,h=1/16,params=None,device='cpu',kernel='q1',boundary='raw',support_threshold=.01,velocity_enrichment=False):
        self.h=h;self.params=params or AnisotropicMaterialParams(10.,20.,200.)
        self.device=device;self._material=None;self._material_A=None
        self.last_grid=None
        self.basis_options=dict(kernel=kernel,boundary=boundary,support_threshold=support_threshold,velocity_enrichment=velocity_enrichment)

    def frozen_step(self,state,origin=(0.,0.,0.)):
        response=None
        if self.device!='cpu':
            from .refinement_gpu import DeviceMaterial
            if self._material is None or not np.array_equal(self._material_A,state.A):
                self._material=DeviceMaterial(state.A,self.params,self.device);self._material_A=state.A.copy()
            response=self._material
        return FrozenSpatialStep(state,self.h,self.params,origin,response,**self.basis_options)

    def step(self,state,dt,origin=(0.,0.,0.),tolerance=1e-10):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive finite dt required')
        saved_F=state.F.copy()
        system=self.frozen_step(state,origin);g=system.grid
        history_rebuild=float(np.max(abs(saved_F-state.F)))
        predictor=g.project(state.v);yhat=g.to_y(predictor)
        Uold=float(state.volume@system.response(state.F)[0])
        Kold=.5*float(np.sum(state.mass[:,None]*state.v**2))
        def objective(y):
            v=g.from_y(y.reshape(-1,3))
            try:U,f,_=system.elastic(v,dt)
            except ValueError:return np.inf,np.zeros_like(y)
            dy=y.reshape(-1,3)-yhat
            return U+.5*float(np.sum(dy*dy)),(dy+dt*g.force_to_y(f)).ravel()
        y,iters,backtracks=minimize_newton(objective,
            lambda y:system.whitened_hessian(g.from_y(y.reshape(-1,3)),dt),yhat.ravel(),tolerance=tolerance)
        value,grad=objective(y)
        if not np.isfinite(value) or np.max(abs(grad))>1.01*tolerance:raise RuntimeError('MPM step did not converge')
        v=g.from_y(y.reshape(-1,3));U,force,predicted_F=system.elastic(v,dt)
        vp=g.N@v;xp=state.x+dt*vp
        # Independent spatial-gradient commit, rather than copying predicted F.
        committed_F=(np.eye(3)+dt*g.gradient(v))@state.F
        next_state=ParticleHistory(state.X,xp,vp,committed_F,state.mass,state.volume,state.A)
        Ferror=float(np.max(abs(committed_F-predicted_F)))
        history_energy=float(state.volume@system.response(committed_F)[0])
        Kg=.5*float(np.sum(v*(g.M@v)))
        Kp=.5*float(np.sum(state.mass[:,None]*vp**2))
        Kproj=.5*float(np.sum(predictor*(g.M@predictor)))
        dv=v-predictor
        budget=dict(projection_delta=Kproj-Kold,inertia_remainder=-.5*float(np.sum(dv*(g.M@dv))),
            elastic_remainder=U-Uold-dt*float(np.sum(force*v)),
            solver_work=float(np.sum(v*(g.M@dv+dt*force))),kinetic_readback_delta=Kp-Kg,
            history_commit_delta=history_energy-U)
        residual=(history_energy+Kp-Uold-Kold)-sum(budget.values())
        bulk=(state.mass[:,None]*state.v).sum(axis=0)/state.mass.sum()
        fluctuation=.5*float(np.sum(state.mass[:,None]*(state.v-bulk)**2))
        momentum_before=(state.mass[:,None]*state.v).sum(axis=0)
        momentum_after=(state.mass[:,None]*vp).sum(axis=0)
        grid_velocity_roundtrip=g.project(vp)
        out=dict(nodes=len(g.x),particles=len(state.x),iterations=iters,backtracks=backtracks,
            scaled_residual=float(np.max(abs(grad))),min_det=float(np.linalg.det(committed_F).min()),
            elastic=history_energy,kinetic=Kp,mechanical=history_energy+Kp,
            prediction_commit_F_absolute=Ferror,history_rebuild_F_absolute=history_rebuild,
            kinetic_relative=abs(Kp-Kg)/max(Kp,1e-30),
            roundtrip_relative=float(np.linalg.norm(grid_velocity_roundtrip-v)/max(np.linalg.norm(v),1e-30)),
            mass_min=g.minimum_eigenvalue,mass_condition=g.mass_condition,
            enrichment_modes=g.enrichment_modes,enrichment_radius=g.enrichment_radius,**g.support_diagnostics,
            partition_error=float(np.max(abs(np.asarray(g.N.sum(axis=1)).ravel()-1))),
            affine_position_error=float(np.max(abs(g.N@g.x-state.x))),
            affine_gradient_error=float(np.max(abs(g.gradient(g.x)-np.eye(3)))),
            mass_error=abs(float(g.M.sum())-float(state.mass.sum())),
            projection_loss_over_fluctuation=max(Kold-Kproj,0)/max(fluctuation,1e-30),
            momentum_absolute=float(np.linalg.norm(momentum_after-momentum_before)),
            energy_budget_residual=residual,**budget)
        self.last_grid=g
        return next_state,out
