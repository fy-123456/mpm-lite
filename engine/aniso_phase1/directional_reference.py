"""Explicit Cartesian axes for controlled directional reference refinement."""
import itertools
import numpy as np

from .convergence_reference import geometry
from .history_increment import frozen
from .template_remap import RemappedQ1


def cartesian_geometry(axes):
    """Own nonuniform axes on the unchanged beam material domain."""
    base=geometry(17)
    if len(axes)!=3:raise ValueError('three Cartesian axes required')
    axes=tuple(np.asarray(a,dtype=float) for a in axes)
    lengths=np.array([.5,.125,.125])
    for d,a in enumerate(axes):
        if (a.ndim!=1 or len(a)<2 or not np.isfinite(a).all() or np.any(np.diff(a)<=0)
                or a[0]!=base.lo[d] or a[-1]!=base.lo[d]+lengths[d]):
            raise ValueError('finite increasing axes with exact beam endpoints required')
    source=RemappedQ1.__new__(RemappedQ1)
    source.lo=base.lo.copy();source.field=base.field;source.params=base.params
    source.axes=tuple(frozen(a) for a in axes)
    source.counts=np.array([len(a)-1 for a in axes]);source.shape=tuple(source.counts+1)
    source.X=np.array(list(itertools.product(*source.axes)))
    source.fixed=source.X[:,0]==source.lo[0];source.free=~source.fixed
    # Reporting scale only: mappings and mass use the actual axis widths.
    source.h=float(min(np.min(np.diff(a)) for a in axes))
    return source


def directional_geometry(counts):
    counts=np.asarray(counts)
    if counts.shape!=(3,) or not np.isfinite(counts).all() or np.any(counts<1) or np.any(counts!=counts.astype(int)):
        raise ValueError('three positive integer cell counts required')
    base=geometry(17);lengths=np.array([.5,.125,.125])
    return cartesian_geometry(tuple(np.linspace(lo,lo+length,int(n)+1)
        for lo,length,n in zip(base.lo,lengths,counts)))
