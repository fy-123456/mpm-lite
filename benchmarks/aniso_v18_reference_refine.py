"""Continue the sealed modal projection solves with local FE refinement."""
import shutil
import numpy as np
from benchmarks.aniso_v18_runs import OUT,ROOT,sha,load,write
from benchmarks.aniso_v18_compatible_reference import Targets,solve_projection,compare_cases
from engine.aniso_phase1 import local_reference as ref

def main():
    assert not (OUT/'compatible-reference.json').exists();dest=OUT/'compatible-reference';targets=Targets();cases={}
    for name in ('q1-n32','q2-n16','q1-n64','q2-n32','q2-n48'):
        with np.load(dest/f'{name}.npz') as z:cases[name]=([z[f'axis{k}'] for k in range(3)],int(name[1]),z['u'])
    base=[np.linspace(a,b,round((b-a)*48)+1) for a,b in zip(ref.LO,ref.HI)];levels=[]
    for level in (1,2):
        edges=[]
        for k,e in enumerate(base):
            mid=(e[:-1]+e[1:])/2
            ids=np.flatnonzero(np.minimum(abs(mid-.25),abs(mid-.75))<1/48) if k==0 else np.array([0,len(e)-2])
            extra=np.concatenate([np.linspace(e[i],e[i+1],2**level+1)[1:-1] for i in ids]);edges.append(np.union1d(e,extra))
        levels.append(edges)
    write(OUT/'compatible-reference-protocol.json',dict(sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'benchmarks/aniso_v18_reference_refine.py',ROOT/'benchmarks/aniso_v18_compatible_reference.py',ROOT/'tests/test_aniso_v18_space.py',ROOT/'engine/aniso_phase1/local_reference.py']},
        reused_solves={str(p.relative_to(OUT)):sha(p) for p in dest.iterdir()},initial_protocol='reference-comparison-quadrature-attempt/compatible-reference-protocol.json',
        levels=[[e.tolist() for e in edges] for edges in levels],marking='n48 base: two x intervals adjacent to each grip transition, outermost y/z intervals; subdivide by 2 then 4',
        reference_target=.02,comparison_integration='Gauss3 on union of every mesh interface, including Q1 n64',tests='five independent spatial/reference tests passed',
        scope='Physical compatible-strain projection, no mass or stabilization in FE solve; not a natural-frequency reference.'))
    for level,edges in enumerate(levels,1):
        name=f'q2-local{level}';x,u,r=solve_projection(edges,2,targets);np.savez_compressed(dest/f'{name}.npz',nodes=x,u=u,**{f'axis{k}':v for k,v in enumerate(edges)});write(dest/f'{name}.json',r);cases[name]=(edges,2,u);print(name,r,flush=True)
    comparisons=compare_cases(cases,targets,'q2-local2')
    passed=all(max(r['stress_vs_finest_regions'].values())<.02 for r in comparisons['q2-local1'])
    write(OUT/'compatible-reference.json',dict(completed=True,comparisons=comparisons,finest='q2-local2',finite_element_refinement_passed=passed,
        physical_mode_accuracy_accepted=False,scope='Reference refinement and physical compatibility are separate gates. Projected-field Rayleigh quotients are not natural frequencies.'))
    print('REFERENCE COMPLETE',passed,flush=True)
if __name__=='__main__':main()
