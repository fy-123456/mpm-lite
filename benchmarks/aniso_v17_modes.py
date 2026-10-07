"""Stress-ranked modal diagnostics, including genuine zero-inertia constraints.

No engine mutation. Massless coordinates are statically condensed only for the
analysis of singular rest states; no mass/stiffness regularization is used.
"""
import itertools
from pathlib import Path
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_carrier_joint import load_case, spectrum
from benchmarks.aniso_compatible_diagnosis import zones, LO, HI
from benchmarks.aniso_apic_frequency import Oracle
from engine.aniso_phase1.carrier_joint import CarrierEnergy, State, Geometry, current_gradient, gradient
from engine.aniso_phase1.selective_patch import scalar_matrix
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.history_increment import material_tangent

DURATION=.05

def weighted_rms(P,V):
    return float(np.sqrt(np.mean(np.sum(P*P,axis=(-2,-1))@(V/V.sum()))))


def oscillator(omega,eq,v0,time,dt=None):
    time=np.asarray(time)
    rate=omega if dt is None else 2*np.arctan(omega*dt/2)/dt
    return eq[:,None]-eq[:,None]*np.cos(rate[:,None]*time)+v0[:,None]/omega[:,None]*np.sin(rate[:,None]*time)


def trig_average(omega,eq,v0,duration):
    """Exact finite-window mean square of the oscillation about equilibrium."""
    a=-eq;b=v0/omega;t=duration;w=omega
    return a*a*(.5+np.sin(2*w*t)/(4*w*t))+b*b*(.5-np.sin(2*w*t)/(4*w*t))+a*b*np.sin(w*t)**2/(w*t)


def analytic_field(X,amplitude):
    """Smooth clamp-compatible common physical displacement/velocity field."""
    q=(X[:,0]-.25)/.5;inside=(q>0)&(q<1);q=np.clip(q,0,1)
    f=np.sin(np.pi*q)**2;df=2*np.pi*np.sin(2*np.pi*q)
    yy=2*np.pi*(X[:,1]-.375)/.25
    u=np.zeros_like(X);D=np.zeros((len(X),3,3))
    u[:,0]=amplitude*f*np.sin(yy);u[:,1]=.3*amplitude*f
    D[:,0,0]=amplitude*df*np.sin(yy);D[:,0,1]=amplitude*f*np.cos(yy)*(2*np.pi/.25);D[:,1,0]=.3*amplitude*df
    u[~inside]=0;D[~inside]=0
    return u,D


def controlled_case(h=1/8,ns=4,amplitude=0.):
    nn=(3*ns,ns,ns)
    xp=np.array(list(itertools.product(*[a+(np.arange(n)+.5)*(b-a)/n for a,b,n in zip(LO,HI,nn)])))
    V=np.full(len(xp),np.prod(HI-LO)/len(xp));o=Oracle(xp,V,h);X=o.nodes*h
    _,ids,P=scalar_matrix(X,o.c,o.S.T@V,h)
    a=np.array([1,1,0])/np.sqrt(2);A=np.broadcast_to(np.outer(a,a),(len(xp),3,3)).copy()
    e=CarrierEnergy(current_gradient(xp,o.nodes,h),np.tile(np.eye(3),(len(xp),1,1)),V,A,ids,P,10*(o.S.T@V)/(h*h*ids.shape[1]))
    u,_=analytic_field(xp,amplitude);uy,_=analytic_field(X,amplitude);v,C=analytic_field(xp,.001)
    s=State(xp+u,X+uy,v,C)
    return s,e,V,h,dict(kind='analytic_common_field',h=h,ns=ns,amplitude=amplitude,particle_reference=xp,carrier_reference=X)


class ModalModel:
    def __init__(self,s,e,m,h,particle_reference=None,carrier_reference=None):
        self.s=s;self.e=e;self.g=g=Geometry(s,e,m,h);Q=g.Q;self.Q=Q
        self.K=e.tangent(s.Y,Q);self.M=la.block_diag(*([Q.T@g.M@Q]*3))
        self.Ks=la.block_diag(*([Q.T@e.Ks@Q]*3));self.Km=self.K-self.Ks
        self.gate=spectrum(self.K)
        if not self.gate['passed']:raise ValueError(('massless stiffness failed',self.gate))
        Jr=g.J@Q;_,sigma,Vh=la.svd(np.sqrt(g.metric)[:,None]*Jr,full_matrices=False)
        keep=sigma>1e-12*sigma[0];self.sigma=sigma;self.null_count=3*int((~keep).sum())
        W=la.block_diag(*([Vh[keep].T/sigma[keep]]*3))
        Z=la.block_diag(*([Vh[~keep].T]*3))
        f=(Q.T@e.evaluate(s.Y)['force']).T.ravel()
        b=(Jr.T@(g.metric[:,None]*pack(s.v,s.C))).T.ravel()
        self.null_basis=Z;self.static_shift=np.zeros(len(f))
        if self.null_count:
            K00=Z.T@self.K@Z
            W=W-Z@la.solve(K00,Z.T@self.K@W,assume_a='pos')
            self.static_shift=-Z@la.solve(K00,Z.T@f,assume_a='pos')
        H=W.T@self.K@W;lam,V=la.eigh((H+H.T)/2)
        if lam.min()<=0:raise ValueError('nonpositive finite modal stiffness')
        self.phi=W@V;self.omega=np.sqrt(lam);self.eq=-(self.phi.T@f)/lam;self.v0=self.phi.T@b
        self.F=gradient(e.B,s.Y);self.P0=e.evaluate(s.Y)['P'];self.B=[bb@Q for bb in e.B]
        self.D=np.array([material_tangent(self.F,e.A,gradient(self.B,p.reshape(3,Q.shape[1]).T),e.params) for p in self.phi.T])
        self.dPshift=material_tangent(self.F,e.A,gradient(self.B,self.static_shift.reshape(3,Q.shape[1]).T),e.params)
        self.particle_reference=s.x if particle_reference is None else particle_reference
        self.carrier_reference=s.Y if carrier_reference is None else carrier_reference
        self.gram=np.einsum('ipab,jpab,p->ij',self.D,self.D,e.V/e.V.sum(),optimize=True)
        # SVD whitening avoids squaring the condition number in eigensolves.
        self.eigen_residual=float(la.norm(self.K@self.phi-(self.M@self.phi)*lam)/la.norm(self.K@self.phi))
        self.orthogonality=float(la.norm(self.phi.T@self.M@self.phi-np.eye(len(lam))))
        self.score=np.diag(self.gram)*trig_average(self.omega,self.eq,self.v0,DURATION)
        self.order=np.argsort(-self.score)

    def coordinates(self,t,dt=None):return oscillator(self.omega,self.eq,self.v0,t,dt)

    def stress(self,t,dt=None):
        q=self.coordinates(t,dt)
        return self.P0[None]+self.dPshift[None]+np.einsum('it,ipab->tpab',q,self.D,optimize=True)

    def project(self,Y,z):
        displacement=(self.Q.T@(Y-self.s.Y)).T.ravel()-self.static_shift
        # Weighted-J evaluation avoids ill-conditioned normal mass products.
        dq=displacement.reshape(3,self.Q.shape[1]).T
        dj=self.g.J@self.Q@dq
        alpha=self.phi.T@(self.Q.T@self.g.J.T@(self.g.metric[:,None]*dj)).T.ravel()
        speed=self.phi.T@(self.Q.T@self.g.J.T@(self.g.metric[:,None]*z)).T.ravel()
        return alpha,speed

    def record(self):
        e=self.e;g=self.g;Q=self.Q;np_=len(e.V);zp=zones(self.particle_reference);zc=zones(self.carrier_reference)
        rows=[];rank=np.empty(len(self.order),int);rank[self.order]=np.arange(len(rank))+1
        for i,p in enumerate(self.phi.T):
            dy=Q@p.reshape(3,Q.shape[1]).T;dF=gradient(e.B,dy);dP=self.D[i]
            mat=e.V*np.sum(dF*dP,axis=(1,2));r=np.einsum('cij,cja->cia',e.P,dy[e.ids])
            stab=e.weights[:,None]*np.sum(r*r,axis=2);kn=np.zeros(e.n);np.add.at(kn,e.ids.ravel(),stab.ravel())
            vel=g.J@dy;kin=g.metric[:,None]*vel*vel
            kp=np.sum(kin.reshape(4,np_,3),axis=(0,2));st=e.V*np.sum(dP*dP,axis=(1,2))
            mm=float(p@self.M@p);km=float(mat.sum());ks=float(stab.sum());eff=mm/float(p@p)
            assert abs(km+ks-self.omega[i]**2*mm)<1e-6*max(km+ks,1)
            def partition(values,masks):return {k:float(values[v].sum()/max(abs(values.sum()),1e-300)) for k,v in masks.items()}
            rows.append(dict(index=i,stress_rank=int(rank[i]),omega_rad_s=float(self.omega[i]),period_s=float(2*np.pi/self.omega[i]),
                stress_self_power_Pa2=float(self.score[i]),stress_self_power_fraction=float(self.score[i]/self.score.sum()),
                material_stiffness=km,stabilization_stiffness=ks,stabilization_fraction=ks/(km+ks),
                modal_mass=mm,unit_euclidean_effective_mass=eff,
                initial_modal_kinetic_J=.5*float(self.v0[i]**2),local_release_J=.5*float(self.omega[i]**2*self.eq[i]**2),
                kinetic_translation_fraction=float(kin[:np_].sum()/kin.sum()),kinetic_C_fraction=float(kin[np_:].sum()/kin.sum()),
                material_zones=partition(mat,zp),stress_zones=partition(st,zp),kinetic_zones=partition(kp,zp),
                stabilization_row_zones=partition(kn,zc)))
        # Distinct displacement modes need not be stress orthogonal. Account for
        # cross terms using a resolved reference-time grid; report its refinement.
        group=[];q=self.coordinates(np.linspace(0,DURATION,50001))-self.eq[:,None]
        total=float(np.mean(np.einsum('it,ij,jt->t',q,self.gram,q,optimize=True)))
        for dt in (.000125,.00003125,.0000078125):
            ids=np.flatnonzero(self.omega*dt>1);qg=q[ids];gg=self.gram[np.ix_(ids,ids)]
            deleted=float(np.mean(np.einsum('it,ij,jt->t',qg,gg,qg,optimize=True))) if len(ids) else 0.
            group.append(dict(dt=dt,modes=int(len(ids)),self_power_fraction=float(self.score[ids].sum()/self.score.sum()),
                stress_of_deleted_group_over_total_rms=float(np.sqrt(max(deleted,0)/total))))
        nulls=[]
        for p in self.null_basis.T:
            dy=Q@p.reshape(3,Q.shape[1]).T
            nulls.append(dict(material_stiffness=float(p@self.Km@p),stabilization_stiffness=float(p@self.Ks@p),
                kinetic_norm=float(la.norm(np.sqrt(g.metric)[:,None]*(g.J@dy)))))
        return dict(modes=rows,stress_ranking=[int(i) for i in self.order],fastest_mode_index=int(np.argmax(self.omega)),
            gate=self.gate,finite_modes=len(rows),zero_inertia_modes=self.null_count,zero_inertia_details=nulls,
            kinetic_singular_ratio=float(self.sigma[-1]/self.sigma[0]),
            rank_sensitivity={str(r):3*int(np.sum(self.sigma<=r*self.sigma[0])) for r in (1e-14,1e-12,1e-10,1e-8)},
            eigen_residual=self.eigen_residual,mass_orthogonality=self.orthogonality,
            static_constraint_initial_displacement_norm=float(la.norm(self.static_shift)),
            total_oscillatory_stress_power_Pa2=total,sum_self_power_Pa2=float(self.score.sum()),fast_groups=group,
            zone_convention='material/kinetic/stress on physical particles; stabilization allocated by projector residual rows to reference carriers, not physical exterior energy density',
            ranking='finite-window self stress power about local equilibrium, actual initial modal displacement and velocity; individual self powers do not add to total stress power')


def snapshot_model(t):
    s,e,m,h,src=load_case(t)
    with np.load(Path(__file__).resolve().parents[1]/src['path']) as z:
        xp=z['particle_reference_x'].copy();X=z['patch_X'].copy()
    return ModalModel(s,e,m,h,xp,X)
