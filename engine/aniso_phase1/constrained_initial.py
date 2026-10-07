"""Independent compatible initial controls with explicit integral constraints."""
import numpy as np
from scipy.linalg import cholesky,cho_solve,solve_triangular
from scipy.optimize import minimize

from .history_increment import HistoryField,frozen,material_tangent
from .initial_controls import identity_coefficients
from .consistent_transfer import material_response
from .convergence_reference import directions


def features(basis,X):
    active=basis.powers[:,0]>0
    powers=basis.powers[active];s=(X-basis.origin)/basis.scale
    N=np.prod(s[:,None,:]**powers[None,:,:],axis=2)
    D=[]
    for d in range(3):
        p=powers.copy();p[:,d]=np.maximum(p[:,d]-1,0)
        D.append(np.prod(s[:,None,:]**p[None,:,:],axis=2)*powers[None,:,d]/basis.scale[d])
    return active,N,np.stack(D,axis=2)


def momentum_energy_velocity(original,smooth,X,w):
    """Closest constrained velocity along the fitted momentum-free component.

    The minimum-energy momentum carrier and zero-momentum remainder are
    mass-orthogonal. Scaling only the remainder preserves all three momenta.
    Left-face zero velocity is built into the scalar polynomial space.
    """
    if len(smooth.terms)!=1:raise ValueError('single polynomial velocity required')
    basis,c=smooth.terms[0];active,N,_=features(basis,X)
    if np.max(abs(c[~active]))>1e-12:raise ValueError('velocity must vanish on left face')
    M=N.T @ (w[:,None]*N);b=N.T @ w
    L=cholesky(M,lower=True);carrier=cho_solve((L,True),b)
    carrier/=b @ carrier
    v0=original.evaluate(X)[0]
    momentum=np.sum(w[:,None]*v0,axis=0);K=.5*float(np.sum(w[:,None]*v0*v0))
    mean=carrier[:,None]*momentum
    remainder=c[active]-carrier[:,None]*(b @ c[active])
    Kmin=.5*float(np.sum(mean*(M @ mean)))
    Kr=.5*float(np.sum(remainder*(M @ remainder)))
    if Kmin>K+1e-11*max(K,1e-30):raise ValueError('momentum/energy target infeasible in this smooth space')
    if K==0:coefficients=np.zeros_like(c);gamma=0.
    else:
        if Kr<=0:raise ValueError('no momentum-free shape to match kinetic energy')
        gamma=np.sqrt(max(K-Kmin,0)/Kr)
        coefficients=np.zeros_like(c);coefficients[active]=mean+gamma*remainder
    result=HistoryField(((basis,frozen(coefficients)),))
    v=result.evaluate(X)[0]
    return result,dict(target_kinetic=K,target_momentum=momentum.tolist(),minimum_kinetic=Kmin,
        remainder_scale=float(gamma),kinetic=.5*float(np.sum(w[:,None]*v*v)),
        momentum=np.sum(w[:,None]*v,axis=0).tolist(),
        mass_condition=float(np.linalg.cond(M)),momentum_constraints=3,
        angular_momentum_constrained=False)


class StressFit:
    """Stress L2 objective with energy equality and displacement/gradient caps."""
    def __init__(self,original,initial,X,w,params):
        if len(initial.terms)!=1:raise ValueError('single polynomial initial history required')
        self.basis,c=initial.terms[0];self.X=X;self.w=w/w.sum();self.volume=w.sum();self.params=params
        self.active,self.N,self.D=features(self.basis,X)
        self.identity=identity_coefficients(self.basis)
        if np.max(abs((c-self.identity)[~self.active]))>1e-12:raise ValueError('left clamp required')
        K=np.einsum('q,qkd,qld->kl',self.w,self.D,self.D,optimize=True)
        self.L=cholesky(K,lower=True)
        self.x0,F0=original.evaluate(X);self.F0=F0;self.A=directions(X)
        psi,self.P0=material_response(F0,self.A,params)
        self.U0=float(self.w @ psi);self.Pscale=float(np.sum(self.w[:,None,None]*self.P0**2))
        self.xscale=float(np.sum(self.w[:,None]*(self.x0-X)**2))
        self.Fscale=float(np.sum(self.w[:,None,None]*(F0-np.eye(3))**2))
        offset=c[self.active]-self.identity[self.active]
        self.scale=float(np.linalg.norm(self.L.T @ offset))
        if min(self.U0,self.Pscale,self.xscale,self.scale)<=0:raise ValueError('nonzero deformed state required')
        self.start=(self.L.T @ offset/self.scale).ravel()
        x=X+self.N @ offset
        self.xcap=float(np.sum(self.w[:,None]*(x-self.x0)**2)/self.xscale)*(1+1e-10)+1e-14
        F=np.eye(3)+np.einsum('qkd,ki->qid',self.D,offset,optimize=True)
        self.Fcap=float(np.sum(self.w[:,None,None]*(F-F0)**2)/self.Fscale)*(1+1e-10)+1e-14
        self.cache=None

    def coefficients(self,y):
        return self.scale*solve_triangular(self.L.T,y.reshape(-1,3),lower=False)

    def gradient_coordinates(self,g):
        return (self.scale*solve_triangular(self.L,g,lower=True)).ravel()

    def evaluate(self,y):
        if self.cache is not None and np.array_equal(y,self.cache[0]):return self.cache[1]
        c=self.coefficients(y);x=self.X+self.N @ c
        F=np.eye(3)+np.einsum('qkd,ki->qid',self.D,c,optimize=True)
        psi,P=material_response(F,self.A,self.params)
        dP=P-self.P0;dx=x-self.x0
        objective=.5*float(np.sum(self.w[:,None,None]*dP*dP))/self.Pscale
        adjoint=material_tangent(F,self.A,dP,self.params)/self.Pscale
        gradient=self.gradient_coordinates(np.einsum('q,qkd,qid->ki',self.w,self.D,adjoint,optimize=True))
        constraint=float(self.w @ psi)/self.U0-1
        energy_gradient=self.gradient_coordinates(np.einsum('q,qkd,qid->ki',self.w,self.D,P,optimize=True)/self.U0)
        displacement=float(np.sum(self.w[:,None]*dx*dx))/self.xscale
        shape_gradient=self.gradient_coordinates(-2*self.N.T @ (self.w[:,None]*dx)/self.xscale)
        dF=F-self.F0
        deformation=float(np.sum(self.w[:,None,None]*dF*dF))/self.Fscale
        deformation_gradient=self.gradient_coordinates(-2*np.einsum('q,qkd,qid->ki',self.w,self.D,dF,optimize=True)/self.Fscale)
        result=dict(objective=objective,gradient=gradient,energy=constraint,energy_gradient=energy_gradient,
            shape=self.xcap-displacement,shape_gradient=shape_gradient,displacement_relative=np.sqrt(displacement),
            deformation=self.Fcap-deformation,deformation_gradient=deformation_gradient,deformation_relative=np.sqrt(deformation),
            stress_relative=np.sqrt(2*objective),minimum_det=float(np.linalg.det(F).min()))
        self.cache=(y.copy(),result);return result

    def solve(self,max_iterations=500):
        start=self.evaluate(self.start);trace=[]
        def safe(y):
            try:return self.evaluate(y)
            except ValueError:
                return dict(objective=1e20+float(y @ y),gradient=2*y,
                    energy=1e10,energy_gradient=np.zeros_like(y),shape=-1e10,shape_gradient=np.zeros_like(y),
                    deformation=-1e10,deformation_gradient=np.zeros_like(y))
        def callback(y):
            r=self.evaluate(y);trace.append({k:r[k] for k in ('objective','energy','shape','deformation','minimum_det')})
        answer=minimize(lambda y:(safe(y)['objective'],safe(y)['gradient']),self.start,jac=True,method='SLSQP',
            constraints=[dict(type='eq',fun=lambda y:safe(y)['energy'],jac=lambda y:safe(y)['energy_gradient']),
                         dict(type='ineq',fun=lambda y:safe(y)['shape'],jac=lambda y:safe(y)['shape_gradient']),
                         dict(type='ineq',fun=lambda y:safe(y)['deformation'],jac=lambda y:safe(y)['deformation_gradient'])],
            callback=callback,options=dict(ftol=1e-12,maxiter=max_iterations))
        final=self.evaluate(answer.x)
        if (not answer.success or abs(final['energy'])>1e-8 or final['shape'] < -1e-9 or final['deformation'] < -1e-9
                or final['stress_relative']>=start['stress_relative']):
            raise RuntimeError(f'constrained stress fit failed: {answer.message}, energy={final["energy"]}, shape={final["shape"]}')
        c=self.identity.copy();c[self.active]+=self.coefficients(answer.x)
        info=dict(success=bool(answer.success),message=str(answer.message),iterations=int(answer.nit),
            initial_stress_relative=start['stress_relative'],final_stress_relative=final['stress_relative'],
            energy_relative=final['energy'],displacement_relative=final['displacement_relative'],
            displacement_cap=float(np.sqrt(self.xcap)),minimum_det=final['minimum_det'],
            deformation_relative=final['deformation_relative'],deformation_cap=float(np.sqrt(self.Fcap)),
            gradient_condition=float(np.linalg.cond(self.L)**2),trace=trace)
        return HistoryField(((self.basis,frozen(c)),)),info
