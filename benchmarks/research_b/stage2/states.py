"""Preregistered, whole-family-separated material states and five probes."""
import numpy as np
from engine.aniso_phase1.tensor_metrics import evaluate_gradient, quadrature_axis
from engine.aniso_phase1.research_b.stage2.material import array_digest

SEED = 93022026


def normalize(space, full, amplitude=1.):
    """Deterministic kinematic scaling only; no material/candidate errors used."""
    u=space.nodes(full)
    points=[quadrature_axis(e,2)[0] for e in space.edges]
    grad=evaluate_gradient((space.edges,space.p,u),points)
    scale=float(np.max(np.linalg.norm(grad,axis=(-2,-1))))
    if not np.isfinite(scale) or scale<=1e-14:raise ValueError('empty kinematic mode')
    return full*(amplitude/scale)


def carrier_mode(s, family):
    x,y,z=s.carrier_X.T;t=np.clip((x-.25)/.5,0,1);Y=(y-.5)/.125;Z=(z-.5)/.125
    a=np.zeros((s.ndof,3));inside=(x>.25)&(x<.75)
    if family=='carrier':a[:s.n,0]=np.sin(np.pi*t)
    elif family=='shear':a[:s.n,1]=np.sin(np.pi*t)*(1+.2*Y)
    elif family=='clamp':a[:s.n,2]=np.sin(np.pi*t)*np.exp(-((t-.07)/.09)**2)
    elif family=='hidden_twist':
        a[:s.n,1]=np.sin(3*np.pi*t)*Z;a[:s.n,2]=-np.sin(3*np.pi*t)*Y
    elif family=='hidden_cross':
        a[:s.n,0]=np.sin(2*np.pi*t)*Y*Z;a[:s.n,1]=np.sin(4*np.pi*t)*(Z*Z-.3)
    elif family=='hidden_bend':
        a[:s.n,1]=t*(1-t)*(2*t-1);a[:s.n,0]=np.sin(3*np.pi*t)*Y
    else:raise ValueError('unknown mode family')
    a[np.flatnonzero(~inside)]=0
    return normalize(s,a)


def local_mode(s, seed, sensitive=False):
    a=np.zeros((s.ndof,3));rng=np.random.default_rng(seed)
    if sensitive:
        # Largest nodal support-amplitude column, chosen without quadrature
        # errors. This probes one concentrated local mode rather than a mean.
        strengths=np.asarray((s.raw @ s.transform)**2).sum(axis=0)
        k=int(np.argmax(strengths));a[s.n+k]=[.3,.7,-.2]
    else:a[s.n:]=rng.normal(size=(s.ndof-s.n,3))
    return normalize(s,a)


def build(s):
    carrier=carrier_mode(s,'carrier');shear=carrier_mode(s,'shear')
    local=local_mode(s,SEED);sensitive=local_mode(s,SEED+1,True)
    directions=dict(carrier=carrier,local=local,mixed=normalize(s,carrier+.7*local),shear=shear,sensitive=sensitive)
    q0=s.expand(s.q0);zero=np.zeros_like(q0)
    rows=[]
    def add(split,family,label,q,amplitude):
        rows.append(dict(split=split,family=family,name=label,q=q,amplitude=amplitude,
                         q_sha256=array_digest(q),directions={k:array_digest(v) for k,v in directions.items()}))
    add('train','zero','zero',zero,0.)
    add('train','archive','archive',q0,0.005)
    add('train','carrier_sine','carrier_tension',q0+.07*carrier,.07)
    add('train','local_random','local_compression',q0-.09*local,.09)
    add('train','shear_mix','shear_mix',q0+.11*normalize(s,shear+.4*local),.11)
    add('development','clamp_local','clamp_local',q0+.18*normalize(s,carrier_mode(s,'clamp')+.6*sensitive),.18)
    add('development','compression_mix','compression_mix',-2*q0+.22*normalize(s,carrier-.8*shear+.3*sensitive),.22)
    add('development','near_domain','near_domain',3*q0+.32*normalize(s,local_mode(s,SEED+19)+.3*carrier_mode(s,'clamp')),.32)
    # No amplitude/rotation neighbor of train/development: different carrier
    # functions and independent local seeds. The final set is evaluated once.
    for j,family in enumerate(('hidden_twist','hidden_cross','hidden_bend')):
        c=carrier_mode(s,family)
        for i,amp in enumerate((.045,.11,.21,.31)):
            mode=normalize(s,c+.45*local_mode(s,SEED+100+11*j+i))
            q=(1+.2*i)*q0 + ((-1)**(i+j))*amp*mode
            add('hidden',family,f'{family}_{i}',q,amp)
    return rows,directions
