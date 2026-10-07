from pathlib import Path
import argparse
import numpy as np
from .provenance import read,write,serial_lock
from benchmarks.research_sequential_next.checkpoint import GenerationStore

def history(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()

def solid(run):
    run=Path(run);p=read(run/'S6/final-protocol.json');records={}
    for name in set((p['full_case'],p['default_case'])):
        folder=run/'cases'/name;h=history(folder);cfg=read(folder/'execution-protocol.json');rows=h[-1]['rows'];budget=cfg['acceptance']['energy_fraction']*cfg['acceptance']['energy_scale_J']
        def finite(value):
            if isinstance(value,dict):return all(finite(v) for v in value.values())
            if isinstance(value,list):return all(finite(v) for v in value)
            return bool(np.isfinite(value)) if isinstance(value,(float,int)) else True
        r=dict(steps=len(rows),all_finite=finite(rows),all_accepted=all(x['accepted'] for x in rows),max_true_residual_fraction=max(x['true_residual']/x['residual_tolerance'] for x in rows),
            boundary_max=max(max(x['displacement_constraint'],x['velocity_constraint']) for x in rows),min_detF=min(x['min_detF'] for x in rows),max_energy_balance_J=max(abs(x['energy_balance_J']) for x in rows),
            max_ledger_closure_J=max(abs(x['budget_defect_J']) for x in rows),path_budget_fraction=rows[-1]['cumulative_abs_path_quadrature_error_J']/budget,solve_work_budget_fraction=rows[-1]['cumulative_abs_solve_work_error_J']/budget)
        if not r['all_finite'] or not r['all_accepted'] or r['max_true_residual_fraction']>1 or r['boundary_max']>1e-8 or r['min_detF']<=.1 or max(r['path_budget_fraction'],r['solve_work_budget_fraction'])>1:raise ValueError('raw scene stability budget failed')
        records[name]=r
    write(run/'S6/raw-physical-check.json',dict(status='passed_scoped',cases=records,scope='all committed raw steps, inherited physical budgets, no extra integration'))
    print('RAW_SCENE_PHYSICS',records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):solid(a.run)
