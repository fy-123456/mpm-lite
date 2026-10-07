"""Repeat same-space frozen-material solve costs; not a full MPM benchmark."""
import argparse
from pathlib import Path
import numpy as np
import warp as wp
from engine.aniso_phase1 import select_lowest_memory_device,query_gpu_memory
from engine.aniso_phase1.quadrature_probe import MemoryMonitor
from benchmarks.aniso_quadrature_dynamics import run,save,guard


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='auto');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--steps',type=int,default=12);p.add_argument('--dt',type=float,default=.005)
    p.add_argument('--out',type=Path,default=Path('docs/results/quadrature-validation/solve-cost-rerun'));args=p.parse_args()
    if args.repeats<1 or args.steps<1 or not np.isfinite(args.dt) or args.dt<=0:p.error('positive repeats/steps/dt required')
    guard();wp.init();device=select_lowest_memory_device(args.device);args.out.mkdir(parents=True,exist_ok=True)
    save(args.out/'environment.json',dict(device=device,gpus=query_gpu_memory(),scope='same frozen material field, mass, load, preconditioner; no MPM transfers'))
    results=[]
    with MemoryMonitor() as monitor:
        for repeat in range(args.repeats):
            folder=args.out/f'repeat{repeat}';folder.mkdir(exist_ok=True)
            rules=['gauss3','particle4','particle8']
            if repeat%2:rules=rules[::-1]
            results.append(run(device,folder,args.steps,args.dt,scenes=('release','load_unload'),rules=rules,time_pair=False))
            save(args.out/'repeats.json',results)
    save(args.out/'memory-observed.json',monitor.result())
    rows=[]
    for name in sorted({r['name'] for trial in results for r in trial}):
        group=[r for trial in results for r in trial if r['name']==name]
        row=dict(name=name,repeats=len(group))
        for key in ('build_seconds','whole_run_seconds','whole_case_seconds','shared_setup_seconds'):
            values=[r[key] for r in group]
            row[key]=dict(median=float(np.median(values)),minimum=min(values),maximum=max(values))
        row['linear_iterations_median']=float(np.median([sum(s['pcg_iterations'] for s in r['rows']) for r in group]))
        rows.append(row)
    save(args.out/'cost-summary.json',rows)


if __name__=='__main__':main()
