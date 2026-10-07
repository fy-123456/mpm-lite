"""Engineering-benefit gate: reuse static evidence before designing another space."""
from pathlib import Path
import argparse
from .provenance import *

def review(run):
    run=Path(run);verify(run);ref=read(run/'S2/reference-scope.json')
    write(run/'S2/stage-window-decision.json',dict(status='not_triggered',reason='available authenticated candidate reference ends during unloading at 1.078125; no holding-stage candidate seed certified, no long run solely to produce one',extended_unload_passed=ref['local_prefix_time_passed'],new_steps=0,raw_events='unobserved',global_temporal_accuracy=False))
    rows=read(run/'cases/candidate-extended-half/ledger.json')
    physical=dict(status='passed_scoped',total_steps=len(rows),new_steps=16,min_detF=min(r['min_detF'] for r in rows),max_true_residual_fraction=max(r['true_residual']/r['residual_tolerance'] for r in rows),max_constraint=max(max(r['displacement_constraint'],r['velocity_constraint']) for r in rows),max_energy_closure_J=max(abs(r['budget_defect_J']) for r in rows))
    if physical['min_detF']<=.1 or physical['max_true_residual_fraction']>1 or physical['max_constraint']>1e-8:raise ValueError('reference physical gate')
    write(run/'S2/physical-check.json',physical)
    source=APP/'S4/static-comparison.json';data=read(source);records=[]
    for angle,row in data['records'].items():
        records.append(dict(angle=angle,regional_old_errors=row['old'],max_static_Pa=max(v['absolute'] for region in row['old'].values() for v in region.values()),reference_sha256=row['reference_sha256']))
    if not all(r['max_static_Pa']<.02 for r in records):raise ValueError('benefit review assumption changed')
    write(run/'S3/objective-audit.json',dict(status='passed_scoped',records=records,source=dict(path=str(source),sha256=sha(source)),engineering_absolute_scale_Pa=.02,restoring_difference_is_formal_fidelity_not_continuum_error=True,restoring_source=dict(path=str(APP/'S1/projection-vs-history.json'),sha256=sha(APP/'S1/projection-vs-history.json')),mass_coordinate_method='existing complete mass Schur whitening; no new artificial diagonal or fresh condition solve',inherited_rank_evidence=read(APP/'S4/operator-check.json')['rank'],rank_evidence_scope='prior rejected origin-restoring candidate only; not new current-space conditioning evidence',current_mass_validation='inherited full-rank complete M7, no numerical mass edit',reference_scope=ref))
    decision=dict(status='not_triggered',reason='no_material_benefit: all inherited static regional stress errors are below 0.02 Pa; the extended same-space time window passes but does not certify continuum dynamic spatial error or supply a resolved retraining objective',new_spaces=0,new_static_solves=0,new_dynamic_attempts=0,formal_space_changed=False,global_spatial_accuracy=False,dynamic_eligibility=False,reviewed_existing_138_plus_6_selection=True,historical_48_75_not_hidden=True)
    write(run/'S3/design-decision.json',decision);write(run/'S3/space-decision.json',decision)
    print('REVIEW',physical,'SPACE',decision['status'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
