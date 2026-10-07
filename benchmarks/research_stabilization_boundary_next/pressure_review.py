"""Explain the screened boundary startup layer without adding coupled steps."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .pressure import algebra,bisect

def review(run):
    run=Path(run);verify(run);design=read(run/'S2/grid-design.json');report=read(run/'S2/fixed-skeleton-reference.json');records=[]
    for ratio,cuts0 in design['coarse_cuts'].items():
        for n,cuts in [(16,cuts0),(32,bisect(cuts0))]:
            a=algebra(cuts,design['parameters']);lam=a['lam'];h=2e-4/16;hc=2e-4/16**3
            records.append(dict(ratio=int(ratio),cells=n,minimum_width_m=float(np.diff(cuts[0]).min()),fastest_pressure_time_s=float(1/lam[-1]),slowest_pressure_time_s=float(1/lam[0]),uniform_h_lambda_max=float(h*lam[-1]),first_cubic_h_lambda_max=float(hc*lam[-1]),midpoint_highest_mode_uniform_amplification=float((1-.5*h*lam[-1])/(1+.5*h*lam[-1])),exact_highest_mode_uniform_amplification=float(np.exp(-h*lam[-1])),H_condition_number=float(np.linalg.cond(a['H'])),capacity_scaled_condition_number=float(lam[-1]/lam[0])))
    failures=[]
    for r in report['records']:
        worst=[]
        for category,cmp in [('reference',r['reference']),('space',r['space']),*r['time'].items()]:
            row=max(cmp['records'],key=lambda x:max(x['budget_ratios'].values()));quantity=max(row['budget_ratios'],key=row['budget_ratios'].get);worst.append(dict(category=category,time_s=row['time_s'],quantity=quantity,budget_ratio=row['budget_ratios'][quantity],minimum_pressure_Pa=cmp.get('minimum_pressure_Pa')))
        failures.append(dict(ratio=r['ratio'],schedule=r['schedule'],worst=worst))
    write(run/'S2/startup-layer-analysis.json',dict(status='limited',spectra=records,worst_intervals=failures,reason='For C dp/dt + L p = b, exact modal decay is exp(-h lambda); midpoint amplification (1-h lambda/2)/(1+h lambda/2) approaches -1 for stiff modes. Thin cells increase lambda; small pressure error and total mass closure do not certify startup face flux.',equation_bug_found=False,no_pressure_clipping=True,no_boundary_or_mobility_change=True,new_coupled_steps=0,no_third_design=True,CPU_algebra_assemblies_this_review=4))
    print('STARTUP_LAYER',records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
