"""Locate F45 static stiffness changes without changing any coefficients."""
import numpy as np
from benchmarks.aniso_v19_runs import OUT,write
from benchmarks.aniso_v17_modes import controlled_case
from benchmarks.aniso_boundary_reference import hessian
from engine.aniso_phase1.compatible_carrier import CompatibleReconstruction,make_case
from engine.aniso_phase1.carrier_joint import gradient

def main():
    s,e,m,h,_=controlled_case();r16=CompatibleReconstruction(s.Y,h,16);r32=CompatibleReconstruction(s.Y,h,32)
    cases=dict(sampled=controlled_case(),gauss3=make_case(None,3),compatible16=make_case(r16,3),compatible32=make_case(r32,3));out={};H=hessian('F45')
    for name,(s,e,m,h,meta) in cases.items():
        u=np.load(OUT/'space'/f'F45-{name}.npz')['u'];F=gradient(e.B,u);P=(F.reshape(-1,9)@H.T).reshape(-1,3,3);Um=.5*e.V*np.sum(F*P,axis=(1,2));Us=.5*np.sum(u*(e.Ks@u));fm=sum(b.T@(e.V[:,None]*P[:,:,j]) for j,b in enumerate(e.B));fs=e.Ks@u;right=s.Y[:,0]>=.75;grip=(s.x[:,0]<=.25)|(s.x[:,0]>=.75)
        out[name]=dict(material_energy_J=float(Um.sum()),stabilization_energy_J=float(Us),physical_grip_material_energy_J=float(Um[grip].sum()),interior_material_energy_J=float(Um[~grip].sum()),material_reaction_N=float(fm[right,0].sum()),stabilization_reaction_N=float(fs[right,0].sum()),stabilization_fraction=float(Us/(Us+Um.sum())),max_grip_increment_F=float(np.max(abs(F[grip]))))
    write(OUT/'static-parts.json',dict(completed=True,cases=out,scope='Linear F45 equilibrium. Component forces depend on the same solved displacement; totals are the reaction. No coefficient or mass changes.'));print(out,flush=True)
if __name__=='__main__':main()
