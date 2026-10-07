"""Native nonlinear energy/force/tangent checks on final real adaptive spaces."""
import time
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1.tensor_reference import interpolate
from engine.aniso_phase1.stress_local_space import LocalPotential

def main():
    while not all((OUT/'adaptive'/n/'summary.json').exists() for n in ('stress','geometric')):time.sleep(10)
    s,e,_,_,_=controlled_case()
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:oldedges=[z[f'axis{k}'] for k in range(3)];A=z['A']
    records=[]
    for name in ('stress','geometric'):
        folder=OUT/'adaptive'/name;source=folder/'round6.npz';edges,_,_=read_field(source)
        with np.load(source) as z:y=z['y'];alpha=z['local_coefficients']
        W=np.load(folder/'round6-basis.npz')['W'];Af=interpolate(oldedges,2,A,edges,2);pot=LocalPotential(edges,Af,W,e.Ks,e.params,e.A[0]);Y=np.vstack((s.Y+y,alpha));rng=np.random.default_rng(210);direction=rng.normal(size=Y.shape);direction/=la.norm(direction);eps=1e-6
        a=pot.evaluate(Y,direction);p=pot.evaluate(Y+eps*direction);m=pot.evaluate(Y-eps*direction);energy_error=abs((p['U']-m['U'])/(2*eps)-np.sum(a['force']*direction));tangent_error=float(la.norm((p['force']-m['force'])/(2*eps)-a['tangent_action'])/la.norm(a['tangent_action']))
        R=la.expm(np.array([[0,-.6,.2],[.6,0,-.1],[-.2,.1,0]]));rot=Y@R.T;rot[:len(s.Y)]+=[.013,-.021,.008];b=pot.evaluate(rot);objectivity_energy=abs(b['U']-a['U']);objectivity_force=float(la.norm(b['force']-a['force']@R.T)/la.norm(a['force']));record=dict(name=name,source_sha256=sha(source),basis_sha256=sha(folder/'round6-basis.npz'),scalar_local_dofs=W.shape[1],nonlinear_energy_J=a['U'],energy_directional_derivative_absolute_error=energy_error,tangent_relative_error=tangent_error,rigid_rotation_translation_energy_error_J=objectivity_energy,force_rotation_relative_error=objectivity_force)
        assert energy_error<1e-8 and tangent_error<2e-5 and objectivity_energy<1e-11 and objectivity_force<1e-8;record['passed']=True;records.append(record);write(OUT/'candidate-nonlinear-audits.json',dict(completed=len(records)==2,records=records));print(record,flush=True)
if __name__=='__main__':main()
