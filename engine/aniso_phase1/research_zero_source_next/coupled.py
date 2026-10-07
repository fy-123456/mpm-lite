"""Require a declared zero source before constructing the inherited theta solver."""
import numpy as np
from engine.aniso_phase1.research_pressure_startup_next.coupled import StartupCoupling

_MISSING=object()

def explicit_zero(source=_MISSING):
    if source is _MISSING or source is None:raise ValueError('explicit zero source required; no topology default')
    try:a=np.asarray(source,dtype=float)
    except (TypeError,ValueError) as e:raise ValueError('finite zero source required') from e
    if a.size==0 or not np.isfinite(a).all() or np.any(a!=0):raise ValueError('only registered zero source supported')
    return a.copy()

class ZeroSourceCoupling(StartupCoupling):
    def __init__(self,*args,source_m3_s=_MISSING,**kwargs):
        source=explicit_zero(source_m3_s)
        super().__init__(*args,source_m3_s=source,**kwargs)
        if np.any(self.source!=0):raise ValueError('constructed source disagrees with zero-source protocol')
