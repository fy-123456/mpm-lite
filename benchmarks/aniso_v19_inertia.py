"""Deformed-state quadrature cross-check, without the runtime particle cap."""
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v19_runs import OUT,write
from benchmarks.aniso_v17_modes import controlled_case
from benchmarks.aniso_apic_frequency import Oracle
from engine.aniso_phase1.compatible_carrier import CompatibleReconstruction,make_case
from engine.aniso_phase1.carrier_joint import gradient
from engine.aniso_phase1.carrier_driven import kinetic_metric
from engine.aniso_phase1.material_patch import carrier_map

def kinetic(case,displacement):
    s,e,m,h,_=case
    if e.position_basis is None:o=Oracle(s.x,m,h);T=o.S@o.H
    else:T=e.position_basis
    s.x+=T@displacement;s.Y+=displacement
    if e.position_basis is None:
        o=Oracle(s.x,m,h);_,_,N=carrier_map(s.Y,o.nodes,h);T=o.S@o.H@la.inv(N.toarray())
    inv=np.linalg.inv(gradient(e.B,s.Y));L=[sum(b*inv[:,k,j,None] for k,b in enumerate(e.B)) for j in range(3)];J=np.vstack([T]+L);q=kinetic_metric(s.x,m,h)
    return J.T@(q[:,None]*J),float(np.max(abs(T@s.Y-s.x)))

def main():
    s,e,m,h,_=controlled_case();rec=CompatibleReconstruction(s.Y,h,16);records=[]
    for name,r in [('gauss3',None),('compatible16',rec)]:
        u=np.load(OUT/'space'/f'F45-{name}.npz')['u'];M=[];errors=[]
        for order in (3,4,5):
            mat,err=kinetic(make_case(r,order),u);M.append(mat);errors.append(err)
        records.append(dict(name=name,position_identity_max=max(errors),M3_vs_M4_relative=float(la.norm(M[0]-M[1])/la.norm(M[1])),M4_vs_M5_relative=float(la.norm(M[1]-M[2])/la.norm(M[2])),scope='Same loaded static carrier field, moved quadrature geometry, no time step.'))
    write(OUT/'deformed-inertia.json',dict(completed=True,records=records,exact_after_motion=False,scope='Nonlinear inverse F and moved grid-kernel breakpoints prevent automatic exactness of rest-state Gauss3.'))
    print(records,flush=True)
if __name__=='__main__':main()
