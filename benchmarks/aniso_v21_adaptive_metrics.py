"""Evaluate all adaptive rounds without using validation fields for selection."""
import time
from benchmarks.aniso_v21_common import *
from benchmarks.aniso_v21_metrics import compare_many

def main():
    while not all((OUT/'adaptive'/n/'summary.json').exists() for n in ('stress','geometric')):time.sleep(10)
    paths={ordering+'/'+f'round{k}':OUT/'adaptive'/ordering/f'round{k}.npz' for ordering in ('stress','geometric') for k in range(1,7)};fields={n:read_field(p) for n,p in paths.items()}
    for stage,reference,p in [('training',BASE/'v11-reference-q3/cases/local2.npz',3),('validation',OUT/'reference/local3-q4.npz',None)]:
        result=compare_many(fields,read_field(reference,p));R=load(reference.with_suffix('.json'))['reaction_N'];cases={}
        for name,regions_ in result.items():
            data=load(paths[name].with_suffix('.json'))['materials']['F45'];cases[name]=dict(regions=regions_,reaction_relative=abs(data['reaction_N']-R)/abs(R),scalar_local_dofs=data['scalar_local_dofs'],source_sha256=sha(paths[name]));print(stage,name,cases[name],flush=True)
        write(OUT/f'adaptive-{stage}.json',dict(completed=True,cases=cases,reference=str(reference.relative_to(ROOT)),reference_sha256=sha(reference),held_out=stage=='validation',reference_continuum_certified=False))
if __name__=='__main__':main()
