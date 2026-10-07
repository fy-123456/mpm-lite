"""Material-only Q2 limit on the exact candidate mesh, plus patch floor."""
import numpy as np
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1.tensor_reference import solve

def main():
    folder=OUT/'space-limit';folder.mkdir(exist_ok=False);edges=read_field(BASE/'v11-reference-q3/cases/local2.npz',3)[0];initial=read_field(OUT/'space/graded/F45-stress-00.npz');u,r=solve(edges,2,hessian('F45'),initial=initial,rtol=2e-12);assert r['passed'];np.savez_compressed(folder/'graded-q2.npz',u=u,degree=2,**{f'axis{k}':e for k,e in enumerate(edges)});write(folder/'graded-q2.json',r);old=read_field(BASE/'v11-reference-q3/cases/local2.npz',3);validation=read_field(OUT/'reference/local3-q4.npz');a=compare((edges,2,u),old);b=compare((edges,2,u),validation);floor=load(BASE/'v20/local-relaxation.json')['patch_only_minimum_reaction_N'];write(OUT/'space-limit.json',dict(completed=True,case=r,training_comparison=a,validation_comparison=b,unchanged_patch_minimum_reaction_N=floor,full_relaxation_reaction_N=r['reaction_N']+floor,scope='Mesh-limited material-only optimum; exact full interior variational limit adds original carrier patch minimum. This is not a reduced candidate.'))
if __name__=='__main__':main()
