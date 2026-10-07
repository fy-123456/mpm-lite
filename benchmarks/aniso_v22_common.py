"""Frozen v22 spatial study, preserving all delivered v21 evidence."""
import json,tempfile
from pathlib import Path
import numpy as np
from benchmarks.aniso_v21_common import ROOT,BASE,read_field,hessian,controlled_case,load,write,sha,FIBER
from engine.aniso_phase1.tensor_metrics import compare_fields
OUT=BASE/'v22'

def compare(a,b,label='F45',**kw):
    theta={'ISO':0.,'F0':0.,'F45':np.pi/4,'F90':np.pi/2}[label]
    return compare_fields(a,b,hessian(label),np.array([np.cos(theta),np.sin(theta),0.]),**kw)

def freeze():
    assert not OUT.exists();OUT.mkdir()
    old=load(BASE/'v21/artifact-sha256.json')
    for n,d in old.items():assert sha(ROOT/n)==d,n
    scratch=Path(tempfile.mkdtemp(prefix='mpm-lite-v22-',dir='/dev/shm'));scratch.chmod(0o700)
    from utils.resource_guard import inspect_storage
    assert not inspect_storage(scratch).paused
    write(OUT/'protocol.json',dict(prior_v21_verified=len(old),scratch=str(scratch),scope='Linear static F45 spatial accuracy. Same physical geometry, exact hard grips, material, original carrier stabilization. No mass, stiffness shifts or time/dynamic claims.',gates=dict(reaction=.01,stress=.02,fiber_strain=.02),reference_plan='Two further levels splitting the first/last TWO free-span and transverse intervals. Q3 and Q4 at each level; h and p checks. Rigid volumes represented exactly by one interval each, unchanged physical region.',candidate_plan='Compare Q2/Q3/Q4 component-closed local residual corrections on the same physical mesh. Original potential determines equilibrium. Reference only ranks correctable stress error; held-out final reference never defines bases.',source_sha256={n:sha(ROOT/n) for n in ['benchmarks/aniso_v22_common.py','benchmarks/aniso_v22_reference.py','engine/aniso_phase1/tensor_metrics.py','engine/aniso_phase1/tensor_reference.py','tests/test_aniso_v22_metrics.py']}))
if __name__=='__main__':freeze()
