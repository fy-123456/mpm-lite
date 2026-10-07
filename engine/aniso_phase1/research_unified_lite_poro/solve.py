"""Small general Newton solve of original mixed residual, no SPD assumption.

Solid: midpoint inertia and 3-point AVF material force. Pressure exchange uses
Simpson's exact determinant chain for fixed affine-in-q F. Darcy H uses current
midpoint deformation. Dense numerical Jacobian is deliberate CPU reference cost.
"""
import numpy as np
import scipy.linalg as la


def advance(op,q0,v0,p0,h,time,amplitude=1.):
    if not np.isfinite(h) or h<=0:raise ValueError('positive finite step required')
    shape=q0.shape;n=q0.size;V0=op.geometry(q0,False)[0]
    load=op.load(time+h/2,amplitude)
    evaluations=0
    def residual(y,extra=False):
        nonlocal evaluations
        evaluations+=1
        q=y[:n].reshape(shape);p=y[n:];dq=q-q0
        v=2*dq/h-v0;mid=(q+q0)/2;pm=(p+p0)/2
        V,G,H,minJ=op.geometry(q,False);_,_,Hm,_=op.geometry(mid)
        Gb=op.discrete_G(q0,q)
        z=la.solve(Hm,op.top.B.T@pm-op.gb,assume_a='pos')
        fi=op.avf_force(q0,q)
        mech=op.M@(v-v0)/h+fi-op.alpha*np.einsum('cni,c->ni',Gb,pm)-load
        mass=op.capacity*(p-p0)+op.alpha*(V-V0)+h*op.top.B@z
        # Row scaling improves units only. Raw equations are checked below.
        r=np.r_[mech.ravel(),mass/h]
        if extra:return dict(q=q,v=v,p=p,z=z,mech=mech,mass=mass,fi=fi,Gb=Gb,H=Hm,minJ=minJ,pm=pm,V=V)
        return r
    y=np.r_[(q0+h*v0).ravel(),p0]
    last=float('inf')
    for it in range(16):
        r=residual(y);norm=float(np.max(abs(r)))
        if norm<2e-9:break
        J=np.empty((len(y),len(y)))
        for j in range(len(y)):
            eps=2e-7*max(1.,abs(y[j]));d=np.zeros_like(y);d[j]=eps
            J[:,j]=(residual(y+d)-residual(y-d))/(2*eps)
        delta=la.solve(J,-r)
        for k in range(12):
            candidate=y+delta*(.5**k)
            try:rn=residual(candidate);new=float(np.max(abs(rn)))
            except ValueError:continue
            if new<norm:
                y=candidate;last=new;break
        else:raise RuntimeError('mixed Newton backtracking failed')
    else:raise RuntimeError('mixed Newton iteration budget exhausted')
    t=residual(y,True)
    if max(np.max(abs(t['mech'])),np.max(abs(t['mass']))/h)>1e-8:
        raise RuntimeError('true coupled residual failed')
    q,v,p,z=t['q'],t['v'],t['p'],t['z'];dq=q-q0
    e0=op.material(q0)[0]+.5*np.sum(v0*(op.M@v0))+.5*np.dot(op.capacity*p0,p0)
    e1=op.material(q)[0]+.5*np.sum(v*(op.M@v))+.5*np.dot(op.capacity*p,p)
    diss=h*float(z@t['H']@z);work=float(np.sum(load*dq)-h*z@op.gb)
    chain=t['V']-V0-np.einsum('cni,ni->c',t['Gb'],dq)
    metrics=dict(iterations=it,residual_calls=evaluations,raw_mechanical_residual=float(np.max(abs(t['mech']))),
      local_mass_defect=float(np.max(abs(t['mass']))),volume_chain_defect=float(np.max(abs(chain))),
      exchange_work_defect=float(op.alpha*t['pm']@chain),darcy_dissipation=diss,external_work=work,
      energy_initial=e0,energy_final=e1,energy_defect=e1-e0+diss-work,min_detF=t['minJ'],
      p_min=float(p.min()),p_max=float(p.max()),boundary_volume=h*float(np.sum(op.top.B@z)))
    if diss < -1e-14 or not np.isfinite(e1):raise ValueError('invalid energy/dissipation')
    return q.copy(),v.copy(),p.copy(),z.copy(),metrics
