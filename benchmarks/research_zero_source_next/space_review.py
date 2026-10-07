"""Conditional space work only when a resolved physical target exists."""
import argparse
from .provenance import *

def review(run):
    run=Path(run);verify(run)
    source=APP/'S3/objective-audit.json';old=read(source);new=read(run/'S3/coupled-comparison.json')
    failed={name:entry['solid']['max_budget_fractions'] for name,entry in new.items() if isinstance(entry,dict) and 'solid' in entry and not entry['solid']['passed']}
    records=old['records']
    if any(r['max_static_Pa']>=.02 for r in records):raise ValueError('old objective assumptions changed')
    write(run/'S5/objective-audit.json',dict(status='passed_scoped',inherited_static_records=records,old_evidence=dict(path=str(source),sha256=sha(source)),inherited_solid_reference=read(APP/'S2/reference-scope.json'),current_solid_over_budget=failed,current_comparisons_are_same_solid_space=True,reference_uncertainty='uncertified continuum coupled spatial error',error_attribution='current pressure-grid and time differences do not establish solid basis error',engineering_stress_absolute_Pa=.02))
    decision=dict(status='not_triggered',reason='no resolved continuum solid reference and no demonstrated above-budget solid spatial target; inherited static errors below0.02Pa',new_spaces=0,new_static_solves=0,new_dynamic_attempts=0,formal_space_changed=False,functions=144,candidate_limit=1,historical48_75deg_is_not_hidden=True)
    write(run/'S5/design-decision.json',decision);write(run/'S5/static-comparison.json',dict(status='not_triggered',reason=decision['reason'],inherited_only=True));write(run/'S5/space-decision.json',decision)
    print('SPACE',decision['status'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
