"""Fixed engineering observations over conservative, fully coupled microsteps."""
import numpy as np
from engine.aniso_phase1.research_pressure_startup_next.coupled import time_tolerance
from engine.aniso_phase1.research_pressure_startup_next.theta import schedule as theta_schedule

OBSERVATIONS=np.arange(17,dtype=float)*1.25e-5
OBSERVATIONS.setflags(write=False)

def validate_times(times):
    t=np.array(times,dtype=float,copy=True)
    if t.ndim!=1 or len(t)<2 or t[0]!=0 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('finite increasing zero-origin times required')
    return t

def indices(times,observations):
    t=validate_times(times);o=validate_times(observations);out=[]
    for x in o:
        i=int(np.argmin(abs(t-x)))
        if abs(t[i]-x)>time_tolerance(t[i],x):raise ValueError('observation endpoint missing')
        out.append(i)
    if out[0]!=0 or np.any(np.diff(out)<=0):raise ValueError('invalid observation coverage')
    return np.array(out,dtype=int)

def make_times(candidate,fine=False):
    if candidate not in ('U2','U4'):raise ValueError('unregistered startup candidate')
    m=int(candidate[1]);t=[0.]
    for k,(left,right) in enumerate(zip(OBSERVATIONS[:-1],OBSERVATIONS[1:])):
        t.extend(np.linspace(left,right,m+1 if k<4 else 2)[1:])
    t=np.array(t)
    if fine:t=np.sort(np.r_[t,.5*(t[:-1]+t[1:])])
    indices(t,OBSERVATIONS);theta_schedule(t,'startup')
    return t

def aggregate(values,times,observations):
    """Endpoint states and integrated interval flux; never average endpoint states."""
    t=validate_times(times);o=validate_times(observations);ix=indices(t,o)
    z=np.asarray(values['flux'],dtype=float);Q=np.asarray(values['cumulative'],dtype=float)
    if z.ndim<1 or len(z)!=len(t)-1 or Q.shape!=(len(t),*z.shape[1:]):raise ValueError('flux history shape mismatch')
    if not np.isfinite(z).all() or not np.isfinite(Q).all():raise ValueError('nonfinite history')
    h=np.diff(t).reshape((-1,)+(1,)*(z.ndim-1));expected=np.cumsum(h*z,axis=0)+Q[0]
    if not np.allclose(expected,Q[1:],atol=1e-18,rtol=2e-12):raise ValueError('flux and cumulative history disagree')
    out={k:np.asarray(values[k])[ix].copy() for k in ('pressure','content') if k in values}
    if any(len(values[k])!=len(t) or not np.isfinite(values[k]).all() for k in out):raise ValueError('invalid endpoint field')
    out['cumulative']=Q[ix].copy();out['flux']=np.diff(Q[ix],axis=0)/np.diff(o).reshape((-1,)+(1,)*(z.ndim-1))
    return out

def interval_rows(rows,observations,*,average=(),summed=()):
    """Only expose fully committed observations; reject gaps or duplicate rows."""
    o=validate_times(observations);t=[0.]
    for row in rows:
        end=float(row['time']);h=float(row['dt']);left=end-h
        if not np.isfinite([end,h]).all() or h<=0 or abs(left-t[-1])>time_tolerance(left,t[-1]):raise ValueError('gap/duplicate row')
        t.append(end)
    if not rows:return []
    t=validate_times(t);complete=o[o<=t[-1]+time_tolerance(t[-1],t[-1])]
    if len(complete)<2:return []
    ix=indices(t,complete);out=[]
    for a,b,left,right in zip(ix[:-1],ix[1:],complete[:-1],complete[1:]):
        part=rows[a:b];record=dict(time=float(right),dt=float(right-left),raw_start=int(a),raw_stop=int(b))
        for key in average:record[key]=sum(float(r['dt'])*float(r[key]) for r in part)/(right-left)
        for key in summed:record[key]=sum(float(r[key]) for r in part)
        if not all(np.isfinite(record[k]) for k in (*average,*summed)):raise ValueError('nonfinite interval row')
        out.append(record)
    return out
