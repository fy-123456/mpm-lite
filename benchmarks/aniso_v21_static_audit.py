"""Consolidate every static candidate and verify shared polynomial containment."""
import numpy as np
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1.tensor_reference import interpolate,coordinates

def main():
    s,e,_,_,_=controlled_case();records={};basis_checks=[]
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:oldedges=[z[f'axis{k}'] for k in range(3)];oldA=z['A']
    for mesh in ('coarse','graded'):
        folder=OUT/'space'/mesh;a=load(folder/'summary.json');assert a['completed']
        for n,r in a['records'].items():records[mesh+'/'+n]=r
        edges=oldedges if mesh=='coarse' else read_field(BASE/'v11-reference-q3/cases/local2.npz',3)[0];A=oldA if mesh=='coarse' else interpolate(oldedges,2,oldA,edges,2);x=s.Y;X=np.array(np.meshgrid(*coordinates(edges,2),indexing='ij')).reshape(3,-1).T;poly=lambda x:np.column_stack((np.ones(len(x)),x,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]));error=float(np.max(abs(A@poly(x)-poly(X))));W=np.load(folder/'corrections.npz')['W'];fixed=(X[:,0]<=.25)|(X[:,0]>=.75);grip=float(np.max(abs(W[fixed])));assert error<1e-10 and grip==0.;basis_checks.append(dict(space=mesh,quadratic_polynomial_error=error,local_fixed_grip_value=grip,scalar_component_closure=True))
    for ordering in ('stress','geometric','gain'):
        folder=OUT/'adaptive'/ordering;a=load(folder/'summary.json');assert a['completed']
        for n,row in a['records'].items():
            for label,r in row['materials'].items():records[ordering+'/'+n+'/'+label]=r
            W=np.load(folder/(n+'-basis.npz'))['W'];error=float(np.linalg.norm(W.T@W-np.eye(W.shape[1])));grip=float(np.max(abs(W[fixed])));physical=np.load(folder/(n+'.npz'))['local_coefficients'];trace=float(np.max(abs(W[fixed]@physical)));assert error<1e-8 and grip<1e-12 and trace<1e-12;basis_checks.append(dict(space=ordering+'/'+n,orthogonality_error=error,local_fixed_grip_value=grip,local_fixed_grip_displacement=trace,grip_tolerance=1e-12,scalar_component_closure=True,polynomial_containment='Unchanged proven graded base A is a subspace, all local coefficients can be zero.'))
    for n,r in records.items():
        assert r['static_passed'] and r['static_min']>r['static_tolerance'] and r['zero_modes']==0 and r['negative_modes']==0 and not r['mass_included'] and r['stiffness_shift']==0.,n
        if r['scalar_local_dofs']:assert r['local_stiffness_min']>0 and r['schur_min']>0,n
        assert r['work_identity_relative']<1e-8 and r['free_residual']<1e-8,n
    assert len(records)==176;write(OUT/'static-audits.json',dict(completed=True,candidates=len(records),directions=['ISO','F0','F45','F90'],all_passed=True,basis_checks=basis_checks,minimum_unshifted_stiffness=min(r['static_min'] for r in records.values()),max_work_identity_relative=max(r['work_identity_relative'] for r in records.values()),max_free_residual=max(r['free_residual'] for r in records.values()),mass_included=False,stiffness_shift=0.,normal_bending='The scalar reconstruction exactly retains quadratic fields; enrichment adds freedom and introduces no new kinematic constraints. This does not certify continuum bending stiffness.'))
if __name__=='__main__':main()
