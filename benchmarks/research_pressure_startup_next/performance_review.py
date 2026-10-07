"""Use every registered measurement, including the noisy first state-1 pair."""
from pathlib import Path
import numpy as np
from .provenance import *
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric

def review(run):
    run=Path(run);verify(run);protocol=read(run/'S4/repeat-protocol.json');records=[]
    for i in (0,1):
        folders=[run/'S4'] if i==0 else [run/'S4/first-state1',run/'S4']
        pairs=[]
        for folder in folders:
            A=read(folder/f'A{i}/measurement.json');B=read(folder/f'B{i}/measurement.json');a=history(folder/f'A{i}')[-1]['state'];b=history(folder/f'B{i}')[-1]['state']
            checks={k:metric(getattr(a,k),getattr(b,k),1e-8,2e-5) for k in ('q','velocity','predictor')}
            for key,tol in [('pressure_Pa',1e-6),('flux_interval_m3_s',1e-10),('content_m3',1e-10),('cumulative_source_m3',1e-10),('cumulative_boundary_m3',1e-10)]:checks[key]=metric(a.child_states['fluid'][key],b.child_states['fluid'][key],tol,2e-5)
            pairs.append(dict(folder=str(folder),A=A,B=B,checks=checks,passed=all(v['passed'] for v in checks.values())))
        at=float(np.mean([v['A']['advance_s'] for v in pairs]));bt=float(np.mean([v['B']['advance_s'] for v in pairs]));setup=max(v['B']['additional_setup_s'] for v in pairs)
        records.append(dict(index=i,pairs=pairs,A_mean_s=at,B_mean_s=bt,gain=1-bt/at,setup_break_even_steps=setup/max(at-bt,1e-30),passed=all(v['passed'] for v in pairs)))
    gains=[r['gain'] for r in records];median=float(np.median(gains));spread=float(np.ptp(gains));exclusive=all(p[k][s]['exclusive'] for r in records for p in r['pairs'] for k in ('A','B') for s in ('sharing_before','sharing_after'))
    selected=all(r['passed'] for r in records) and exclusive and min(gains)>=0 and median>=max(.05,spread) and max(r['setup_break_even_steps'] for r in records)<=16
    attempts=sum(read(p)['attempts'] for p in (run/'S4').rglob('attempt.json'))
    if attempts>8:raise ValueError('performance budget exceeded')
    write(run/'S4/paired-performance.json',dict(status='passed_scoped' if selected else 'limited',records=records,median_gain=median,between_state_gain_range=spread,all_registered_pairs_included=True,repeat_protocol=protocol,statistical_confidence_interval=False,exclusive=exclusive))
    write(run/'S4/performance-decision.json',dict(status='research_bounded_raw_qualified' if selected else 'retain_BASE',selected=bool(selected),backend='bounded-raw-column-cell-gradient' if selected else 'BASE-local-cell-gradient',attempts=attempts,formal_solid_changed=False,production_default_changed=False,scope='original16-cell midpoint research fixture only',new_pressure_scheme_uses_BASE_geometry=True))
    print('FINAL_PERFORMANCE',selected,median,spread,attempts,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
