"""Host-side quadrature audit of second versus fourth direction moments.

The fourth moment is an exact compression only for the current quadratic fiber
energy at a common F. This module does not change production constitutive laws.
"""
import numpy as np


def fiber_response(F,A,kf):
    FA=F@A;I4=np.sum(FA*F)
    P=2*kf*(I4-1)*FA
    H=2*kf*(I4-1)*np.kron(np.eye(3),A)+4*kf*np.outer(FA.ravel(),FA.ravel())
    return .5*kf*(I4-1)**2,P,H,I4


def fourth_moment_response(F,A,weights,kf):
    weights=np.asarray(weights,dtype=float);weights=weights/weights.sum()
    A2=np.einsum('p,pij->ij',weights,A)
    A4=np.einsum('p,pij,pkl->ijkl',weights,A,A)
    C=F.T@F
    S=np.einsum('ijkl,kl->ij',A4,C)-A2
    psi=.5*kf*(np.einsum('ij,ijkl,kl',C,A4,C)-2*np.sum(A2*C)+1)
    P=2*kf*F@S
    H=np.empty((9,9))
    for j,dF in enumerate(np.eye(9).reshape(9,3,3)):
        dC=dF.T@F+F.T@dF
        H[:,j]=(2*kf*(dF@S+F@np.einsum('ijkl,kl->ij',A4,dC))).ravel()
    return psi,P,H


def audit_direction_mixture(F,directions,weights=None,kf=200.):
    a=np.asarray(directions,dtype=float);a=a/np.linalg.norm(a,axis=1,keepdims=True)
    w=np.ones(len(a))/len(a) if weights is None else np.asarray(weights,dtype=float)/np.sum(weights)
    A=np.einsum('pi,pj->pij',a,a);Amean=np.einsum('p,pij->ij',w,A)
    responses=[fiber_response(F,t,kf) for t in A]
    exact=tuple(sum(wi*r[j] for wi,r in zip(w,responses)) for j in range(3))
    mean=fiber_response(F,Amean,kf)
    fourth=fourth_moment_response(F,A,w,kf)
    I4=np.array([r[3] for r in responses]);variance=float(np.sum(w*(I4-np.sum(w*I4))**2))
    result=dict(fiber_energy_exact=float(exact[0]),fiber_energy_mean_A=float(mean[0]),
                missing_energy=float(exact[0]-mean[0]),predicted_missing_energy=.5*kf*variance,
                invariant_variance=variance)
    for j,key in enumerate(('energy','stress','tangent')):
        denominator=max(float(np.linalg.norm(exact[j])),1e-12)
        result[key+'_mean_A_relative_error']=float(np.linalg.norm(mean[j]-exact[j])/denominator)
        result[key+'_fourth_moment_relative_error']=float(np.linalg.norm(fourth[j]-exact[j])/denominator)
    return result
