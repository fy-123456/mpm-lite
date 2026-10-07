"""Constant-capacity full-tensor pressure theta steps with explicit energy ledger."""
import numpy as np
import scipy.linalg as la

def schedule(times,method):
    t=np.asarray(times,dtype=float)
    if t.ndim!=1 or len(t)<2 or t[0]!=0 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('finite increasing zero-origin time grid required')
    if method not in ('midpoint','backward-euler','startup'):raise ValueError('unknown pressure method')
    if method=='startup' and not np.any(np.isclose(t,2.5e-5,rtol=0,atol=1e-16)):
        raise ValueError('startup switching time must be a grid endpoint')
    return np.full(len(t)-1,.5 if method=='midpoint' else 1.) if method!='startup' else np.where(t[1:]<=2.5e-5+1e-16,1.,.5)

def step(a,p,h,theta):
    p=np.asarray(p,dtype=float)
    if not np.isfinite(h) or h<=0 or theta not in (.5,1.) or p.shape!=a['C'].shape or not np.isfinite(p).all():
        raise ValueError('invalid theta step')
    eye=np.eye(len(p));b=(eye-(1-theta)*h*a['A'])@(a['D']*p)+h*a['rhs']/a['D']
    new=la.solve(eye+theta*h*a['A'],b,assume_a='pos')/a['D'];pt=(1-theta)*p+theta*new
    z=a['Z']@pt+a['z0'];delta=new-p;mass=a['C']*delta+h*(a['L']@pt-a['rhs'])
    delta_E=.5*np.sum(a['C']*(new*new-p*p));Dnum=(theta-.5)*np.sum(a['C']*delta*delta)
    Dphysical=h*float(z@a['H']@z)
    # Z and z0 bind the same full-tensor boundary/source algebra as the state update.
    source=a['rhs']+a['top'].B@a['z0'];gb=-a['H']@a['z0']
    Wsource=h*float(pt@source);Wreservoir=-h*float(gb@z)
    ledger=dict(storage_change_J=float(delta_E),darcy_dissipation_J=Dphysical,numerical_dissipation_J=float(Dnum),source_work_J=Wsource,reservoir_work_J=Wreservoir,energy_balance_J=float(delta_E+Dphysical+Dnum-Wsource-Wreservoir),mass_defect_m3=float(np.max(abs(mass))))
    return new,z,ledger

def integrate(a,params,times,method):
    times=np.asarray(times);thetas=schedule(times,method);p=np.full(len(a['C']),params['pressure0_Pa']);Q=np.zeros(a['top'].nflux)
    ps=[p.copy()];qs=[Q.copy()];zs=[];rows=[]
    for h,theta in zip(np.diff(times),thetas):
        p,z,row=step(a,p,h,float(theta));Q+=h*z;ps.append(p.copy());qs.append(Q.copy());zs.append(z);rows.append(row)
    return dict(pressure=np.array(ps),cumulative=np.array(qs),flux=np.array(zs),ledger=rows,theta=thetas)
