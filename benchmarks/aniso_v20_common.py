"""Shared v20 input provenance and separated material/kinetic state lifts."""
import json,hashlib,tempfile
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from benchmarks.aniso_v19_runs import ROOT,BASE,load,write,sha
from benchmarks.aniso_v17_modes import controlled_case
from engine.aniso_phase1.carrier_joint import State
from engine.aniso_phase1.compatible_carrier import lite_gradient
from benchmarks.aniso_v18_space import gauss_sites
OUT=BASE/'v20'

def snapshot(t):
    path=BASE/'v19/cases/cycle-L3'/f'audit-{round(t/.0000625):06d}.npz';s,e,m,h,meta=controlled_case()
    with np.load(path) as a:s=State(a['x'].copy(),a['Y'].copy(),a['v'].copy(),a['C'].copy(),float(a['time']))
    return s,e,m,h,dict(path=str(path.relative_to(ROOT)),sha256=sha(path),**meta)

class SnapshotField:
    def __init__(self,s,X):
        self.axes=[np.unique(X[:,j]) for j in range(3)];shape=tuple(map(len,self.axes));assert np.prod(shape)==len(X)
        self.maps={n:RegularGridInterpolator(self.axes,getattr(s,n).reshape(*shape,*getattr(s,n).shape[1:]),bounds_error=False,fill_value=None) for n in ('x','v','C')}
    def at(self,X):return {n:f(X) for n,f in self.maps.items()}

def lift_state(s,e,m,h,meta,X,V):
    f=SnapshotField(s,meta['particle_reference']);values=f.at(X);state=State(values['x'],s.Y.copy(),values['v'],values['C'],s.time)
    # Elastic energy/force/tangent stay at the ORIGINAL 192 material points.
    import copy
    energy=copy.copy(e);energy.kinetic_B=tuple(lite_gradient(X,meta['carrier_reference'],h));energy.kinetic_reference=np.array(X);energy.material_reference=meta['particle_reference']
    return state,energy,np.array(V),h

def freeze():
    from utils.resource_guard import inspect_storage
    assert not (OUT/'protocol.json').exists();old=load(BASE/'v19/artifact-sha256.json')
    for n,d in old.items():assert sha(ROOT/n)==d,n
    scratch=Path(tempfile.mkdtemp(prefix='mpm-lite-v20-',dir='/dev/shm'));scratch.chmod(0o700);assert not inspect_storage(scratch).paused
    write(OUT/'protocol.json',dict(prior_v19_artifacts_verified=len(old),scratch=str(scratch),snapshots=[.85,1.1,1.4,1.6],reaction_time_target=.02,stress_time_target=.02,
        model='Original 192-point material/patch energy fixed; kinetic quadrature varied independently. Snapshot x,v,C lifted by continuous piecewise trilinear interpolation, exactly retaining old values at old sites; no energy/momentum renormalization.',
        geometry='same physical box, F45, h=.125',default_changed=False,reference_accuracy_certified=False))
if __name__=='__main__':freeze()
