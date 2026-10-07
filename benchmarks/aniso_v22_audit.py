"""Audit actual high-order candidates, original force/tangent, and all statics."""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v22_common import *
from engine.aniso_phase1.tensor_reference import interpolate
from engine.aniso_phase1.high_order_space import HighOrderPotential

def main():
    s,e,_,_,_=controlled_case();records=[];statics={}
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:oldedges=[z[f'axis{k}'] for k in range(3)];oldA=z['A']
    for p in (2,3,4):
        folder=OUT/f'space/q{p}'
        while not (folder/'summary.json').exists():time.sleep(10)
        summary=load(folder/'summary.json');assert summary['completed']
        for name,record in summary['records'].items():
            assert record['quadratic_polynomial_error']<1e-10 and record['local_fixed_grip_value']<1e-12 and record['local_fixed_grip_displacement']<1e-12
            for label,v in record['materials'].items():
                assert v['static_passed'] and v['static_min']>v['static_tolerance'] and v['negative_modes']==0 and v['zero_modes']==0 and not v['mass_included'] and v['stiffness_shift']==0.;assert v['local_stiffness_min']>0 and v['schur_min']>0;statics[f'q{p}/{name}/{label}']=v
        source=folder/'round6.npz';edges,_,_=read_field(source)
        with np.load(source) as z:y=z['y'];alpha=z['local_coefficients']
        raw=sp.load_npz(folder/'basis-raw.npz');T=np.load(folder/'basis-transform.npz')['transform'];W=raw@T;A=interpolate(oldedges,2,oldA,edges,p);pot=HighOrderPotential(edges,p,A,W,e.Ks,e.params,e.A[0]);Y=np.vstack((s.Y+y,alpha));rng=np.random.default_rng(220);d=rng.normal(size=Y.shape);d/=la.norm(d);eps=1e-6
        a=pot.evaluate(Y,d);b=pot.evaluate(Y+eps*d);c=pot.evaluate(Y-eps*d);energy_error=abs((b['U']-c['U'])/(2*eps)-np.sum(a['force']*d));tangent_error=float(la.norm((b['force']-c['force'])/(2*eps)-a['tangent_action'])/la.norm(a['tangent_action']));R=la.expm(np.array([[0,-.6,.2],[.6,0,-.1],[-.2,.1,0]]));rot=Y@R.T;rot[:len(s.Y)]+=[.013,-.021,.008];b=pot.evaluate(rot);objectivity_energy=abs(b['U']-a['U']);objectivity_force=float(la.norm(b['force']-a['force']@R.T)/la.norm(a['force']));assert energy_error<1e-8 and tangent_error<2e-5 and objectivity_energy<1e-11 and objectivity_force<1e-8
        record=dict(degree=p,source_sha256=sha(source),basis_raw_sha256=sha(folder/'basis-raw.npz'),basis_transform_sha256=sha(folder/'basis-transform.npz'),scalar_local_dofs=W.shape[1],energy_directional_derivative_absolute_error=energy_error,tangent_relative_error=tangent_error,rigid_rotation_translation_energy_error_J=objectivity_energy,force_rotation_relative_error=objectivity_force,passed=True);records.append(record);write(OUT/'nonlinear-audits.json',dict(completed=len(records)==3,records=records));print('NONLINEAR',record,flush=True)
    assert len(statics)==72;write(OUT/'static-audits.json',dict(completed=True,candidates=len(statics),all_passed=True,minimum_unshifted_stiffness=min(v['static_min'] for v in statics.values()),max_work_identity_relative=max(v['work_identity_relative'] for v in statics.values()),max_free_residual=max(v['free_residual'] for v in statics.values()),directions=['ISO','F0','F45','F90'],records=statics,mass_included=False,stiffness_shift=0.,normal_bending='Original exact quadratic polynomial subspace is retained. This is a consistency check, not a continuum accuracy certificate.'))
if __name__=='__main__':main()
