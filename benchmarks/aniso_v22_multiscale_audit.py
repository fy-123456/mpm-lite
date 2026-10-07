"""Actual multiscale candidate audit, without changing the control experiments."""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v22_common import *
from engine.aniso_phase1.tensor_reference import interpolate
from engine.aniso_phase1.high_order_space import HighOrderPotential

def main():
    folder=OUT/'multiscale/space/q4'
    while not (folder/'summary.json').exists():time.sleep(10)
    summary=load(folder/'summary.json');assert summary['completed'];records={}
    for name,row in summary['records'].items():
        assert row['quadratic_polynomial_error']<1e-10 and row['local_fixed_grip_value']<1e-12 and row['local_fixed_grip_displacement']<1e-12
        for label,r in row['materials'].items():
            assert r['static_passed'] and r['static_min']>r['static_tolerance'] and r['zero_modes']==r['negative_modes']==0 and r['stiffness_shift']==0. and not r['mass_included'];assert r['local_stiffness_min']>0 and r['schur_min']>0;records[name+'/'+label]=r
    s,e,_,_,_=controlled_case();source=folder/'round6.npz';edges,p,_=read_field(source)
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:A=interpolate([z[f'axis{k}'] for k in range(3)],2,z['A'],edges,p)
    with np.load(source) as z:y=z['y'];alpha=z['local_coefficients']
    W=sp.load_npz(folder/'basis-raw.npz')@np.load(folder/'basis-transform.npz')['transform'];pot=HighOrderPotential(edges,p,A,W,e.Ks,e.params,e.A[0]);Y=np.vstack((s.Y+y,alpha));d=np.random.default_rng(221).normal(size=Y.shape);d/=la.norm(d);eps=1e-6;a=pot.evaluate(Y,d);b=pot.evaluate(Y+eps*d);c=pot.evaluate(Y-eps*d);ee=abs((b['U']-c['U'])/(2*eps)-np.sum(a['force']*d));te=float(la.norm((b['force']-c['force'])/(2*eps)-a['tangent_action'])/la.norm(a['tangent_action']));R=la.expm(np.array([[0,-.6,.2],[.6,0,-.1],[-.2,.1,0]]));rot=Y@R.T;rot[:len(s.Y)]+=[.013,-.021,.008];b=pot.evaluate(rot);oe=abs(b['U']-a['U']);of=float(la.norm(b['force']-a['force']@R.T)/la.norm(a['force']));assert ee<1e-8 and te<2e-5 and oe<1e-11 and of<1e-8
    nonlinear=dict(passed=True,source_sha256=sha(source),basis_raw_sha256=sha(folder/'basis-raw.npz'),basis_transform_sha256=sha(folder/'basis-transform.npz'),energy_directional_derivative_absolute_error=ee,tangent_relative_error=te,rigid_rotation_translation_energy_error_J=oe,force_rotation_relative_error=of)
    assert len(records)==24;write(OUT/'multiscale-audit.json',dict(completed=True,all_static_passed=True,static_candidates=24,records=records,nonlinear=nonlinear));print('MULTISCALE AUDIT',nonlinear,flush=True)
if __name__=='__main__':main()
