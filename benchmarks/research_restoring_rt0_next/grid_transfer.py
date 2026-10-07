"""Conservative Cartesian pressure/content and oriented-face restriction."""
import numpy as np

def restriction(coarse,fine):
    a=np.asarray(coarse.cell_bounds);b=np.asarray(fine.cell_bounds)
    width=np.maximum(0.,np.minimum(a[:,None,:,1],b[None,:,:,1])-np.maximum(a[:,None,:,0],b[None,:,:,0]));volume=np.prod(width,axis=2)
    if not np.allclose(volume.sum(axis=1),coarse.V0,atol=1e-13,rtol=1e-12) or not np.allclose(volume.sum(axis=0),fine.V0,atol=1e-13,rtol=1e-12):raise ValueError('pressure domains do not cover one another')
    pressure=volume/coarse.V0[:,None];content=volume/fine.V0[None,:];flux=np.zeros((coarse.nflux,fine.nflux))
    for i,(center,axis) in enumerate(zip(coarse.centres,coarse.axes)):
        cells=np.flatnonzero(np.any(coarse.faces==i,axis=1));bounds=a[cells[0]];cross=[j for j in range(3) if j!=axis]
        for j,(p,k) in enumerate(zip(fine.centres,fine.axes)):
            if k!=axis or abs(p[axis]-center[axis])>1e-13:continue
            owners=np.flatnonzero(np.any(fine.faces==j,axis=1));bb=b[owners[0]]
            overlap=float(np.prod(np.maximum(0.,np.minimum(bounds[cross,1],bb[cross,1])-np.maximum(bounds[cross,0],bb[cross,0]))))
            flux[i,j]=overlap/fine.areas[j]
    defect=coarse.B@flux-content@fine.B
    if np.max(abs(defect))>1e-10:raise ValueError('nonconservative flux restriction')
    return pressure,content,flux
