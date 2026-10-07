"""Early-time slab cell means and integrated boundary flow, no t=0 flux sampling."""
import numpy as np
from scipy.special import erfc

def slab_reference(cuts,times,*,mobility,storage,p0,reservoir,area):
    x=np.asarray(cuts,dtype=float);t=np.asarray(times,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or np.any(np.diff(x)<=0):raise ValueError('increasing finite cell edges required')
    if t.ndim!=1 or len(t)<2 or t[0]!=0 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('finite zero-origin time grid required')
    if not all(np.isfinite(v) for v in (mobility,storage,p0,reservoir,area)) or min(mobility,storage,area)<=0 or not p0>=reservoir>=0:raise ValueError('invalid physical reference parameters')
    L=x[-1]-x[0];D=mobility/storage
    if 8*np.sqrt(D*t[-1])>=L:raise ValueError('early half-space superposition outside isolated-layer scope')
    means=[np.full(len(x)-1,p0)];drop=p0-reservoir
    for tt in t[1:]:
        a=2*np.sqrt(D*tt)
        def integral(y):
            u=y/a;return y*erfc(u)-a/np.sqrt(np.pi)*np.exp(-u*u)
        left=np.diff(integral(x-x[0]));right=-np.diff(integral(x[-1]-x))
        means.append(p0-drop*(left+right)/np.diff(x))
    Q=2*area*drop*np.sqrt(mobility*storage*t/np.pi)
    return dict(pressure=np.array(means),boundary_cumulative_per_end_m3=Q,boundary_interval_per_end_m3_s=np.diff(Q)/np.diff(t),diffusion_m2_s=D,cell_capacity=storage*area*np.diff(x),scope='fixed skeleton, constant scalar mobility, insulated sides, isolated two end layers')

def boundary_grid(base,first_m,growth=2.2):
    bounds=np.array(base[0],float);lo,hi=bounds[0],bounds[-1];width=first_m*growth**np.arange(7)
    if first_m<=0 or growth<=1 or width.sum()>=(hi-lo)/2:raise ValueError('boundary layers do not fit domain')
    left=lo+np.r_[0.,np.cumsum(width),(hi-lo)/2]
    cuts=np.r_[left,lo+hi-left[-2::-1]]
    if len(cuts)!=17 or np.any(np.diff(cuts)<=0):raise ValueError('invalid symmetric16-cell mother grid')
    return [cuts.tolist(),*base[1:]]
