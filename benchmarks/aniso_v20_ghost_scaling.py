"""Same material-null carrier directions: inertia versus geometry perturbation.

Fixed-direction Rayleigh quotients are not claimed to be continuum modes or
exact coupled eigenbranches. No velocity/history projection is performed.
"""
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v20_common import *
from engine.aniso_phase1.integrated_avf import material_null_condensation
from engine.aniso_phase1.separate_kinetic import KineticGeometry
from engine.aniso_phase1.carrier_joint import State

def main():
    old,e,m,h,meta=snapshot(1.4);ref,_,_,_,_=controlled_case();H,R,Z,C=material_null_condensation(e,meta['carrier_reference'],h);K=Z.T@e.Ks@Z;rows=[]
    for kind in ('sampled','gauss3'):
        base=(old,e,m,h) if kind=='sampled' else lift_state(old,e,m,h,meta,*gauss_sites(h,3));target,ee,mm,hh=base;X=meta['particle_reference'] if kind=='sampled' else ee.kinetic_reference;ee.carrier_reference=meta['carrier_reference'];previous=None
        for eps in (1.,.5,.25,.125,.0625,0.):
            s=State(X+eps*(target.x-X),meta['carrier_reference']+eps*(old.Y-meta['carrier_reference']),target.v.copy(),target.C.copy(),old.time);g=KineticGeometry(s,ee,mm,hh);JZ=g.J@Z;M=JZ.T@(g.metric[:,None]*JZ);rayleigh=np.diag(K)/np.diag(M);r=dict(kind=kind,geometry_scale=eps,kinetic_trace=float(np.trace(M)),fixed_direction_inertia=np.diag(M).tolist(),fixed_direction_stiffness=np.diag(K).tolist(),fixed_direction_rayleigh_rad_s=np.sqrt(rayleigh).tolist() if eps else None,material_null_norm=float(la.norm(np.vstack(ee.B)@Z)))
            if previous is not None and eps:r['trace_ratio_to_previous']=r['kinetic_trace']/previous
            previous=r['kinetic_trace'];rows.append(r);print(r,flush=True)
    for kind in ('sampled','gauss3'):
        a=[r for r in rows if r['kind']==kind and r['geometry_scale']>0];assert all(abs(r['trace_ratio_to_previous']-.25)<.002 for r in a[1:])
    write(OUT/'ghost-inertia-scaling.json',dict(completed=True,records=rows,geometry_source_sha256=meta['sha256'],fixed_potential=True,scope='Controlled scaling of one real deformed geometry toward rest, using fixed material-null directions. Reveals representation-induced vanishing inertia; fixed-field Rayleigh quotients are not exact coupled eigenfrequencies.'))
if __name__=='__main__':main()
