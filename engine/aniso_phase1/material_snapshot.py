"""Read-only material quadrature diagnostics on immutable particle snapshots.

No solver step, stabilization energy, velocity transfer or force correction.
All comparisons use reference volumes and the same Hencky/quadratic-fiber law.
"""
from dataclasses import dataclass
import itertools
import numpy as np
from scipy.spatial import cKDTree
from .quadratic import polynomial,history_polynomial,GAUSS


@dataclass(frozen=True)
class MaterialSnapshot:
    x: np.ndarray
    X: np.ndarray
    F: np.ndarray
    A: np.ndarray
    volume: np.ndarray

    def __post_init__(self):
        n=len(self.x)
        for name,shape in (('x',(n,3)),('X',(n,3)),('F',(n,3,3)),('A',(n,3,3)),('volume',(n,))):
            a=np.array(getattr(self,name),dtype=float,copy=True)
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('invalid snapshot '+name)
            a.setflags(write=False);object.__setattr__(self,name,a)
        if not n or np.any(self.volume<=0) or np.any(np.linalg.det(self.F)<=0):raise ValueError('snapshot requires positive volume and det F')
        if not np.allclose(self.A,self.A.transpose(0,2,1)) or np.linalg.eigvalsh(self.A).min()<-1e-10 or not np.allclose(np.trace(self.A,axis1=1,axis2=2),1):
            raise ValueError('snapshot requires normalized positive structure tensors')

    def save(self,path):np.savez_compressed(path,**{k:getattr(self,k) for k in ('x','X','F','A','volume')})

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as data:return cls(**{k:data[k] for k in ('x','X','F','A','volume')})


def center_support(snapshot,h):
    """Same trilinear particle-to-center reference-volume partition as production."""
    if not np.isfinite(h) or h<=0:raise ValueError('positive finite dx required')
    z=snapshot.x/h-.5;base=np.floor(z).astype(int);f=z-base;groups={}
    for corner in itertools.product((0,1),repeat=3):
        weight=np.prod(np.where(corner,f,1-f),axis=1)*snapshot.volume
        for p in np.flatnonzero(weight>0):
            key=tuple(base[p]+corner)
            if key not in groups:groups[key]=([],[])
            groups[key][0].append(p);groups[key][1].append(weight[p])
    return [(np.array(k),np.array(ids),np.array(w)) for k,(ids,w) in sorted(groups.items())]


def moments(A,w):
    w=np.asarray(w)/np.sum(w);a=A.reshape(-1,9)
    return np.einsum('p,pij->ij',w,A),np.einsum('p,pi,pj->ij',w,a,a)


def response(F,A2,A4,params):
    """Batch energy, first Piola stress, and Kirchhoff stress; exact common-F M4."""
    F=np.asarray(F)
    if not np.isfinite(F).all() or np.any(np.linalg.det(F)<=0):raise ValueError('nonpositive/nonfinite diagnostic F')
    U,s,Vt=np.linalg.svd(F);logs=np.log(np.maximum(s,1e-12));trace=logs.sum(axis=-1)
    E=params.mu*np.sum(logs*logs,axis=-1)+.5*params.lam*trace**2
    P=(U*((2*params.mu*logs+params.lam*trace[...,None])/s)[...,None,:])@Vt
    C=F.swapaxes(-1,-2)@F;c=C.reshape(-1,9)
    MC=np.einsum('pij,pj->pi',np.broadcast_to(A4,(len(c),9,9)),c).reshape(F.shape)
    E+=.5*params.k_f*(np.sum(C*MC,axis=(-2,-1))-2*np.sum(C*A2,axis=(-2,-1))+1)
    P+=2*params.k_f*F@(MC-A2)
    return E,P,P@F.swapaxes(-1,-2)


def particle_response(snapshot,params):
    a=snapshot.A.reshape(-1,9)
    # Per-particle fourth moment (pure directions or a supplied single tensor).
    return response(snapshot.F,snapshot.A,np.einsum('pi,pj->pij',a,a),params)


def reconstructed_F(coef,points,center,h):
    _,g=polynomial((points-center)/h)
    J=np.eye(3)+np.einsum('qdk,km->qmd',g,coef)/h
    if not np.isfinite(J).all() or np.any(np.linalg.det(J)<=0):raise ValueError('invalid reconstructed reference map')
    if np.linalg.svd(J,compute_uv=False).min()<1e-8:raise ValueError('singular reconstructed reference map')
    return np.linalg.inv(J)


def moment_points(x,w):
    """Eight positive equal-weight samples matching the actual mean/covariance.

    These are diagnostic cubature points, not a new production discretization.
    No energy-dependent weight, rescaling, or fitting to the reference answer.
    """
    w=w/w.sum();mean=w@x;d=x-mean;cov=np.einsum('p,pi,pj->ij',w,d,d)
    eig,U=np.linalg.eigh(cov);L=U*np.sqrt(np.maximum(eig,0))[None,:]
    corners=np.array(list(itertools.product((-1.,1.),repeat=3)))
    return mean+corners@L.T


def compare_snapshot(snapshot,h,params,reference_F_at=None):
    groups=center_support(snapshot,h);tree=cKDTree(snapshot.x);ep,pp,tp=particle_response(snapshot,params)
    names=('particle','center','grid8','moment8','reconstructed_particles','decorrelated_particles')
    if reference_F_at is not None:names+=('analytic_grid8','analytic_moment8')
    records={name:[] for name in names};details=[];fit_error=0.;stress_site_error=0.;center_site_error=0.;condition=0.
    for key,ids,w in groups:
        V=w.sum();wn=w/V;center=(key+.5)*h;A2,A4=moments(snapshot.A[ids],w)
        Fc=np.einsum('p,pij->ij',wn,snapshot.F[ids])
        coef,cond=history_polynomial(snapshot.x,snapshot.X,snapshot.F,snapshot.volume,center,h,tree);condition=max(condition,cond)
        Fsites=reconstructed_F(coef,snapshot.x[ids],center,h)
        fit_error+=float(np.einsum('p,pij,pij->',w,Fsites-snapshot.F[ids],Fsites-snapshot.F[ids]))
        a=snapshot.A[ids].reshape(-1,9);single4=np.einsum('pi,pj->pij',a,a)
        site=response(Fsites,snapshot.A[ids],single4,params)
        stress_site_error+=float(np.einsum('p,pij,pij->',w,site[1]-pp[ids],site[1]-pp[ids]))
        grid_points=center+h*GAUSS[1:];cloud_points=moment_points(snapshot.x[ids],w)
        batches=dict(particle=(ep[ids],pp[ids],tp[ids]),center=response(Fc[None],A2,A4,params),
            grid8=response(reconstructed_F(coef,grid_points,center,h),A2,A4,params),
            moment8=response(reconstructed_F(coef,cloud_points,center,h),A2,A4,params),
            reconstructed_particles=site,decorrelated_particles=response(snapshot.F[ids],A2,A4,params))
        if reference_F_at is not None:
            batches.update(analytic_grid8=response(reference_F_at(grid_points),A2,A4,params),
                analytic_moment8=response(reference_F_at(cloud_points),A2,A4,params))
        constant_error=batches['center'][1][0]-pp[ids]
        center_site_error+=float(np.einsum('p,pij,pij->',w,constant_error,constant_error))
        for name,batch in batches.items():
            weight=wn if name in ('particle','reconstructed_particles','decorrelated_particles') else np.ones(len(batch[0]))/len(batch[0])
            E=float(weight@batch[0]);P=np.einsum('p,pij->ij',weight,batch[1]);tau=np.einsum('p,pij->ij',weight,batch[2])
            records[name].append((V,E,P,tau))
            detail=dict(method=name,i=int(key[0]),j=int(key[1]),k=int(key[2]),volume=V,energy_density=E)
            detail.update({f'P_{i}{j}':float(P[i,j]) for i in range(3) for j in range(3)})
            details.append(detail)
    V=np.array([r[0] for r in records['particle']]);refP=np.array([r[2] for r in records['particle']]);refT=np.array([r[3] for r in records['particle']])
    Uref=float(snapshot.volume@ep);total=float(snapshot.volume.sum());out={}
    Pden=float(np.einsum('p,pij,pij->',V,refP,refP));Tden=float(np.einsum('p,pij,pij->',V,refT,refT))
    for name,rows in records.items():
        E=np.array([r[1] for r in rows]);P=np.array([r[2] for r in rows]);tau=np.array([r[3] for r in rows]);energy=float(V@E)
        p_error=float(np.einsum('p,pij,pij->',V,P-refP,P-refP));t_error=float(np.einsum('p,pij,pij->',V,tau-refT,tau-refT))
        out[name]=dict(energy=energy,energy_relative_error=(energy-Uref)/max(abs(Uref),1e-20),
            center_P_rms_relative_error=float(np.sqrt(p_error/max(Pden,1e-30))),center_tau_rms_relative_error=float(np.sqrt(t_error/max(Tden,1e-30))),
            integrated_P=np.einsum('p,pij->ij',V,P).tolist(),integrated_tau=np.einsum('p,pij->ij',V,tau).tolist())
    return dict(particles=len(snapshot.x),centers=len(groups),total_reference_volume=total,
        partition_volume_error=float(abs(V.sum()-total)),reference_energy=Uref,methods=out,
        reconstruction_F_rms=float(np.sqrt(fit_error/total)),reconstruction_particle_P_relative_error=float(np.sqrt(stress_site_error/max(float(np.einsum('p,pij,pij->',snapshot.volume,pp,pp)),1e-30))),
        center_constant_particle_P_relative_error=float(np.sqrt(center_site_error/max(float(np.einsum('p,pij,pij->',snapshot.volume,pp,pp)),1e-30))),
        history_fit_max_condition=condition,center_evaluations=len(groups),eight_point_evaluations=8*len(groups),particle_evaluations=len(snapshot.x)),details
