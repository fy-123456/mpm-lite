"""Separate compatible initial-state and dead-traction controls."""
import itertools
import numpy as np
from scipy.optimize import brentq

from .smooth_history import PolynomialHistoryBasis,fit_smooth_history
from .history_increment import HistoryField,ReferenceBasis,frozen
from .convergence_reference import geometry,knots,tensor_rule,directions,sum_fields
from .consistent_transfer import material_response


def identity_coefficients(basis):
    c=np.zeros((len(basis.powers),3))
    for d in range(3):
        c[np.all(basis.powers==0,axis=1),d]=basis.origin[d]
        exponent=np.eye(3,dtype=int)[d]
        c[np.all(basis.powers==exponent,axis=1),d]=basis.scale[d]
    return c


def scaled_position(field,alpha):
    if len(field.terms)!=1 or not isinstance(field.terms[0][0],PolynomialHistoryBasis):
        raise ValueError('single analytic polynomial position history required')
    basis,c=field.terms[0]; identity=identity_coefficients(basis)
    return HistoryField(((basis,frozen(identity+alpha*(c-identity))),))


def energy(field,X,w,params):
    F=field.evaluate(X)[1]
    return float(w @ material_response(F,directions(X),params)[0])


def match_elastic(original,smooth,X,w,params):
    target=energy(original,X,w,params)
    if target<=0: raise ValueError('positive target elastic energy required')
    def error(alpha): return energy(scaled_position(smooth,alpha),X,w,params)/target-1
    hi=1.
    for _ in range(20):
        if error(hi)>=0: break
        hi*=1.2
    else: raise ValueError('unable to bracket elastic-energy match')
    alpha=brentq(error,0.,hi,xtol=1e-13,rtol=1e-13)
    result=scaled_position(smooth,alpha)
    return result,dict(alpha=float(alpha),target=target,matched=energy(result,X,w,params),
                       relative_mismatch=abs(error(alpha)))


def smooth_velocity(velocity,X,w):
    source=geometry(17)
    identity=HistoryField(((ReferenceBasis(source),frozen(source.X)),))
    fitted,_=fit_smooth_history(sum_fields(identity,velocity),X,w)
    b,c=fitted.terms[0]
    raw=HistoryField(((b,frozen(c-identity_coefficients(b))),))
    v=velocity.evaluate(X)[0]; vs=raw.evaluate(X)[0]
    target=.5*float(np.sum(w[:,None]*v*v))
    fitted_energy=.5*float(np.sum(w[:,None]*vs*vs))
    if target<=0 or fitted_energy<=0: raise ValueError('positive kinetic energy required')
    beta=np.sqrt(target/fitted_energy)
    result=HistoryField(((b,frozen((c-identity_coefficients(b))*beta)),))
    return result,dict(beta=float(beta),target=target,matched=fitted_energy*beta*beta,
                       original_momentum=np.sum(w[:,None]*v,axis=0).tolist(),
                       smooth_momentum=np.sum(w[:,None]*result.evaluate(X)[0],axis=0).tolist())


def velocity_moments(source,velocity,order=7):
    """Integrate analytic initial velocity accurately; the Q1 mass is unchanged."""
    basis=ReferenceBasis(source);X,w=tensor_rule(knots(source),order)
    kinetic=0.;rhs=np.zeros_like(source.X)
    for start in range(0,len(X),32768):
        xx,ww=X[start:start+32768],w[start:start+32768]
        v=velocity.evaluate(xx)[0]
        kinetic+=.5*float(np.sum(ww[:,None]*v*v))
        rhs+=basis.sample(xx,ww).N.T @ (ww[:,None]*v)
    return kinetic,rhs


def surface_rule(axes,order=5):
    """Five free reference faces, excluding the fixed left face."""
    z,w=np.polynomial.legendre.leggauss(order)
    for normal_axis in range(3):
        for side in (0,1):
            if normal_axis==0 and side==0: continue
            tangent=[d for d in range(3) if d!=normal_axis]
            points=[];weights=[]
            for d in tangent:
                a=axes[d]; h=np.diff(a)
                points.append(((a[:-1,None]+a[1:,None])/2+h[:,None]*z/2).ravel())
                weights.append((h[:,None]*w/2).ravel())
            mesh=np.array(list(itertools.product(*points)))
            X=np.empty((len(mesh),3));X[:,tangent]=mesh
            X[:,normal_axis]=axes[normal_axis][-1 if side else 0]
            normal=np.eye(3)[normal_axis]*(1 if side else -1)
            yield X,np.outer(*weights).ravel(),normal


def initial_dead_traction(source,field,order=5):
    basis=ReferenceBasis(source); force=np.zeros_like(source.X)
    for X,w,n in surface_rule(knots(source),order):
        F=field.evaluate(X)[1]
        P=material_response(F,directions(X),source.params)[1]
        force+=basis.sample(X,w).N.T @ (w[:,None]*(P @ n))
    return force


def release_factor(t,duration=.00025):
    s=np.clip(t/duration,0.,1.)
    return float(1-3*s*s+2*s*s*s)
