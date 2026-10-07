"""Explicit reference/current-gradient contract, not a full Eulerian bridge."""
import numpy as np


def reference_gradient(L,F):
    L,F=np.asarray(L,float),np.asarray(F,float)
    if L.shape != F.shape or L.shape[-2:] != (3,3) or not np.isfinite(L).all() or not np.isfinite(F).all():
        raise ValueError('matched finite velocity gradients and deformation required')
    if np.any(np.linalg.det(F)<=0):raise ValueError('positive deformation required')
    return L@F


def current_gradient(grad_X,F):
    reference_gradient(grad_X,F)  # validate shapes and deformation
    return np.asarray(grad_X)@np.linalg.inv(F)


def capture_lite_kinematics(solver, *, gradient_reference_F=None):
    if not hasattr(solver,'ptc_L'):
        raise ValueError('explicit physical L required; APIC C cannot substitute')
    result={name:getattr(solver,field).numpy().copy() for name,field in
            [('x','ptc_x'),('v','ptc_v'),('F','ptc_F'),('C','ptc_C'),('L','ptc_L'),('A0','ptc_A0')]}
    # finish_transfer stores L sampled BEFORE advection while F has already
    # advanced. Its pullback must use pre-step F, not the committed new F.
    result['L_time_layer']='pre-advection sample from the last transfer'
    if gradient_reference_F is not None:
        result['grad_X_v']=reference_gradient(result['L'],gradient_reference_F)
    return result


def transfer_compatibility(space,X,x,F,v,L):
    """Return Hermite residual of supplied history, never silently reset it."""
    from engine.aniso_phase1.research_unified_lite_poro.model import Particles,unload
    n=len(X);p=Particles(np.asarray(X),np.asarray(x),np.asarray(v),np.asarray(F),
                         reference_gradient(L,F),np.tile(np.diag([1.,0.,0.]),(n,1,1)),
                         np.full(n,np.prod(space.lengths)/n))
    q,vel,fit=unload(space,p)
    return q,vel,fit
