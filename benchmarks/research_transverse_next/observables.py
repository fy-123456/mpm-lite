"""Volume-weighted transverse modes and signed RT0 face observables."""
import numpy as np

def modes(topology,p):
    V=topology.V0;p=np.asarray(p);centres=np.mean(np.asarray(topology.cell_bounds),axis=2)
    mean=float(V@p/V.sum());out=dict(mean_Pa=mean)
    for a,name in ((1,'Ay'),(2,'Az')):
        eta=2*(centres[:,a]-.5)/.25;den=float(V@(eta*eta))
        out[name]=dict(resolved=den>0,value_Pa=float((V*eta)@(p-mean)/den) if den>0 else None,denominator_m3=den)
    return out

def face_signals(topology,z):
    z=np.asarray(z);out={}
    for axis,name in enumerate('xyz'):
        ids=[int(i) for i in topology.internal if topology.axes[i]==axis]
        values=z[ids]
        out[name]=dict(indices=ids,values_m3_s=values.tolist(),signed_sum_m3_s=float(values.sum()),max_abs_m3_s=float(np.max(abs(values))) if len(ids) else 0.)
    sides={}
    for axis,name in enumerate('xyz'):
        for sign,label in ((-1,'-'),(1,'+')):
            ids=[int(i) for i in topology.boundary if topology.axes[i]==axis and topology.boundary_sign[i]==sign]
            sides[name+label]=float(sign*np.sum(z[ids]))
    return dict(internal=out,boundary_outward_m3_s=sides)

def spatial_statistics(X,values,weights):
    values=np.asarray(values);weights=np.asarray(weights)
    if values.shape[:weights.ndim]!=weights.shape or X.shape[:-1]!=weights.shape:raise ValueError('probe shape differs')
    axes=tuple(range(weights.ndim));w=weights/weights.sum()
    mean=np.tensordot(w,values,axes=(axes,axes))
    rms=np.sqrt(np.tensordot(w,values**2,axes=(axes,axes)))
    return dict(mean=np.asarray(mean).tolist(),rms=np.asarray(rms).tolist(),max_abs=float(np.max(abs(values))))
