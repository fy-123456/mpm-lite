"""Independently solve full candidate Q3/Q4 spaces to expose mesh limitations."""
import numpy as np
from benchmarks.aniso_v22_common import *
from engine.aniso_phase1.tensor_reference import solve

def main():
    folder=OUT/'space-limit';folder.mkdir(exist_ok=False);edges=read_field(BASE/'v11-reference-q3/cases/local2.npz',3)[0];cases={};initial=None
    for p in (3,4):
        u,r=solve(edges,p,hessian('F45'),initial=initial,rtol=2e-12);assert r['passed'] and r['work_identity_relative']<1e-8;np.savez_compressed(folder/f'q{p}.npz',u=u,degree=p,**{f'axis{k}':e for k,e in enumerate(edges)});write(folder/f'q{p}.json',r);cases[f'q{p}']=r;initial=(edges,p,u);print('LIMIT',p,r,flush=True)
    floor=load(BASE/'v20/local-relaxation.json')['patch_only_minimum_reaction_N'];write(OUT/'space-limit.json',dict(completed=True,cases=cases,unchanged_patch_minimum_reaction_N=floor,scope='Material-only full FE space, plus original carrier patch minimum for variational-limit reaction. Not a reduced candidate.'))
if __name__=='__main__':main()
