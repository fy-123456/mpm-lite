"""Backward Euler two-phase exchange with actual applied impulse accounting."""
from dataclasses import dataclass
import numpy as np
from .flow import positive_tensor


@dataclass
class DragResult:
    solid_velocity: np.ndarray
    fluid_velocity: np.ndarray
    solid_impulse: np.ndarray
    fluid_impulse: np.ndarray
    physical_dissipation: float
    numerical_loss: float
    energy_residual: float
    momentum_defect: np.ndarray


def drag_tensor(permeability, viscosity, porosity, volume):
    """D=V*n^2*mu*K^-1, so exchange power = V*mu*q.K^-1.q."""
    if not np.isfinite([viscosity, porosity, volume]).all() or viscosity <= 0 or volume <= 0 or not 0 < porosity < 1:
        raise ValueError('positive viscosity/volume and 0<n<1 required')
    K = positive_tensor(permeability, np.shape(permeability)[-1])
    return volume*porosity**2*viscosity*np.linalg.inv(K)


def implicit_drag(vs, vf, ms, mf, D, dt):
    vs, vf, D = np.asarray(vs, float), np.asarray(vf, float), np.asarray(D, float)
    if vs.ndim != 1 or vf.shape != vs.shape or D.shape != (vs.size, vs.size):
        raise ValueError('matching vector and drag tensor shapes required')
    if not np.isfinite(vs).all() or not np.isfinite(vf).all() or not np.isfinite(D).all():
        raise ValueError('finite velocities and drag required')
    if not np.isfinite([ms, mf, dt]).all() or ms <= 0 or mf < 0 or dt <= 0:
        raise ValueError('ms>0, mf>=0, dt>0 required')
    scale = max(float(np.max(np.abs(D))), np.finfo(float).tiny)
    if np.max(np.abs(D-D.T)) > 1e-12*scale or np.min(np.linalg.eigvalsh(D)) < 0:
        raise ValueError('symmetric nonnegative drag required')
    D = .5*(D+D.T)  # remove only validated roundoff antisymmetry
    if mf == 0:
        if np.any(D != 0): raise ValueError('zero-fluid limit requires zero drag')
        return DragResult(vs.copy(), vf.copy(), np.zeros_like(vs), np.zeros_like(vf), 0., 0., 0., np.zeros_like(vs))
    relative = np.linalg.solve(np.eye(vs.size)+dt*(1/ms+1/mf)*D, vf-vs)
    impulse = -dt*D @ relative
    sn, fn = vs-impulse/ms, vf+impulse/mf
    # Compute from velocities actually delivered to each phase.
    js, jf = ms*(sn-vs), mf*(fn-vf)
    new_relative = fn-sn
    loss = dt*float(new_relative @ D @ new_relative)
    numerical = .5*(ms*np.sum((sn-vs)**2)+mf*np.sum((fn-vf)**2))
    e0 = .5*(ms*(vs @ vs)+mf*(vf @ vf)); e1 = .5*(ms*(sn @ sn)+mf*(fn @ fn))
    if loss < -1e-12 or not np.isfinite(sn).all() or not np.isfinite(fn).all():
        raise RuntimeError('invalid drag trial')
    return DragResult(sn, fn, js, jf, loss, float(numerical), float(e1-e0+loss+numerical), js+jf)


def exact_relative(relative, ms, mf, D, time):
    eig, U = np.linalg.eigh(D)
    return U @ (np.exp(-time*(1/ms+1/mf)*eig)*(U.T @ relative))
