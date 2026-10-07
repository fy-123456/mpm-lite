"""192 new steps extending two distinct authenticated baseline histories."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .run import create_config,load_model
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_reference_next.time_study import compare
from benchmarks.research_observable_pressure_next.time_study import grid,H,modal_signal
from .events import difference

def prepare(run):
    run=Path(run);verify(run);base=read(run/'baseline-space.json');hist={k:history(APP/'cases'/k) for k in ('candidate-chain','reference-0','final-full')}
    origin={k:next(x for x in h if abs(x['state'].time-1.075)<1e-11) for k,h in hist.items()}
    for k in ('q','velocity','predictor'):
        if any(not np.array_equal(getattr(x['state'],k),getattr(origin['reference-0']['state'],k)) for x in origin.values()):raise ValueError('common 1.075 initial mismatch')
    second=history(APP/'cases/reference-1')[0]['state']
    for label in ('candidate-chain','final-full'):
        actual=next(x['state'] for x in hist[label] if abs(x['state'].time-1.225)<1e-11)
        for key in ('q','velocity','predictor'):
            if not np.array_equal(getattr(actual,key),getattr(second,key)):raise ValueError('parent 1.225 window initial mismatch')
    initial={k:next(x for x in hist[k] if abs(x['state'].time-1.1)<1e-11) for k in ('candidate-chain','reference-0')}
    write(run/'S2/baseline-output-review.json',dict(status='passed_scoped',parent=APP_SHA,prior=read(APP/'S1/local-reference-check.json'),older_raw_discrepancies=read(APP/'S1/output-baseline.json'),origin_common=True,second_origin_common=True,raw_high_frequency_events_retained=True,scope='BASELINE only'))
    register(run,'S2/extension-protocol.json',dict(start=1.1,end=1.125,common_ancestor_s=1.075,ancestor_sha256=sha(origin['reference-0']['folder']/'state.json'),initials={k:dict(path=str(v['folder']/'state.json'),sha256=sha(v['folder']/'state.json'),digest=v['state'].digest()) for k,v in initial.items()},steps=[64,128],h_s=H,distinct_initials_preserved=True,baseline=base,frames_per_case=3,no_damping=True))
    for name,key,h in [('propagation-h','candidate-chain',H),('propagation-half','reference-0',H/2)]:
        create_config(run,name,start=1.1,end=1.125,times=grid(1.1,1.125,h),initial=initial[key]['folder']/'state.json',space=base['package'],field_cache=True,display_frames=3)

def analyze(run):
    run=Path(run);cfg=read(run/'cases/propagation-h/execution-protocol.json');m,_=load_model(run,cfg)
    with np.load(APP/'S1/modal-basis.npz') as z:V=z['vectors'].copy()
    if read(APP/'S1/modal-observable-map.json')['space']!=m.reduction.signature:raise ValueError('wrong modal space')
    candidate=history(run/'cases/propagation-h');ref=history(run/'cases/propagation-half');base=[x for x in history(APP/'cases/final-full') if 1.1-1e-11<=x['state'].time<=1.125+1e-11]
    fine=compare(candidate,ref,m,V[:,72],1.1,1.125);coarse=compare(base,ref,m,V[:,72],1.1,1.125)
    lookup={round(x['time_s'],10):x for x in fine['field_records']};gains=[]
    for row in coarse['field_records'][1:]:
        for reg,fields in row['regions'].items():
            for key,old in fields.items():
                new=lookup[round(row['time_s'],10)]['regions'][reg][key];gains.append(dict(time=row['time_s'],region=reg,field=key,old=old['absolute'],new=new['absolute'],budget=old['budget'],gain=1-new['absolute']/max(old['absolute'],1e-30),important=old['absolute']>.1*old['budget']))
    events=[]
    for j in (4,72,523):
        a,b=modal_signal(candidate,m,V[:,j]),modal_signal(ref,m,V[:,j]);events.append(dict(mode=j,candidate=a,reference=b,comparison=difference(a,b,.003125)))
    useful=any(g['important'] and g['gain']>=.2 for g in gains)
    write(run/'S2/propagation-comparison.json',dict(status='passed_scoped' if fine['all_engineering_fields_passed'] else 'reference_limited',candidate_vs_half=fine,coarse_vs_half=coarse,gains=gains,events=events,new_steps=192,reference_is_empirical=True,no_alignment_or_filtering=True))
    times=read(APP/'S1/time-decision.json')['times']
    write(run/'S2/time-scope-decision.json',dict(status='retain_formal_grid',times=times,steps=len(times)-1,extension_research_useful=useful,fields_resolved=fine['all_engineering_fields_passed'],temporal_accuracy=False,scope='BASELINE local only; do not transfer to changed solid space',linear_diagnostic='theta(h)=2*atan(omega*h/2), not proof of nonlinear output accuracy'))
    print('PHASE_PROPAGATION',fine['all_engineering_fields_passed'],useful,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','analyze']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
