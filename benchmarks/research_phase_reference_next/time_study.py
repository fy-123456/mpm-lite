"""Narrow independent references, exact inherited initial history, no long prefix."""
from pathlib import Path
import argparse,shutil
import numpy as np
import scipy.linalg as la
from .provenance import APP,read,write,sha,register,serial_lock,verify
from .run import create_config,load_model
from .events import events,difference
from .base_config import time_grid
from benchmarks.research_phase_stress_next.time_study import history,leaf_failures
from benchmarks.research_reference_next.time_study import compare
from engine.aniso_phase1.endpoint_boundary import prescribed_speed
WINDOWS=((1.0875,1.1),(1.2375,1.25))
LEVELS=(('bridge',.00078125),('fine',.000390625),('half',.0001953125))

def prepare(run):
    run=Path(run);verify(run);source=APP/'cases/final-full';lookup={round(x['state'].time,10):x for x in history(source)}
    create_config(run,'phase-model',end=.025,field_cache=True);m,_=load_model(run,read(run/'cases/phase-model/execution-protocol.json'))
    mapping=read(APP/'S2/modal-observable-map.json')
    if mapping['space']!=m.reduction.signature:raise ValueError('foreign modal basis')
    with np.load(APP/'S2/modal-basis.npz') as z:V=z['vectors'];lam=z['values'];M=m.M3ff;K=m.rest_K[np.ix_(m.ids,m.ids)];orth=float(la.norm(V.T@M@V-np.eye(len(lam))));res=float(la.norm(K@V-(M@V)*lam)/la.norm(K@V))
    if orth>1e-6 or res>1e-7:raise ValueError('invalid inherited modes')
    shutil.copyfile(APP/'S2/modal-basis.npz',run/'S1/modal-basis.npz');write(run/'S1/modal-observable-map.json',dict(mapping,validation=dict(mass_orthogonality=orth,eigen_residual=res),parent_sha256=sha(APP/'S2/modal-observable-map.json')))
    register(run,'S1/time-protocol.json',dict(windows=WINDOWS,levels=LEVELS,steps=[16,32,64],max_new_reference_steps=224,source_identity_sha256=sha(source/'identity.json'),reference_fraction=.25,reference_events_s=.003125,candidate_events_s=.00625,candidate='fine level, reused exact actual trajectory',global_temporal_accuracy=False,improvement='dominant-mode endpoint error at common endpoint; no unequal-grid RMS comparison'))
    initials=[]
    for i,(a,b) in enumerate(WINDOWS):
        path=lookup[a]['folder']/'state.json'
        for label,h in LEVELS:create_config(run,f'time-{i}-{label}',dt=h,start=a,end=b,initial=path,field_cache=True,display_frames=3)
        initials.append(dict(window=[a,b],path=str(path),sha256=sha(path),full_state_digest=lookup[a]['state'].digest()))
    write(run/'S1/window-initials.json',dict(status='authenticated_common_initials',records=initials));print('TIME_READY',initials,flush=True)

def analyze(run):
    run=Path(run);m,_=load_model(run,read(run/'cases/phase-model/execution-protocol.json'));mapping=read(run/'S1/modal-observable-map.json');base=history(APP/'cases/final-full')
    with np.load(run/'S1/modal-basis.npz') as z:V=z['vectors'].copy()
    chosen=mapping['selected_modes'];dominant=max(chosen,key=lambda x:x['energy_peak_J'])['mode'];records=[]
    def signal(h,j):return events([x['state'].time for x in h],[float(V[:,j]@m.M3ff@(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel()) for x in h])
    for i,(a,b) in enumerate(WINDOWS):
        hs={k:history(run/'cases'/f'time-{i}-{k}') for k,_ in LEVELS};old=[x for x in base if a-1e-12<=x['state'].time<=b+1e-12]
        for h in [old,*hs.values()]:
            for key in ('q','velocity','predictor'):
                if not np.array_equal(getattr(h[0]['state'],key),getattr(old[0]['state'],key)):raise ValueError('window initial differs')
        modes=[]
        for row in chosen:
            j=row['mode'];raw={k:signal(h,j) for k,h in hs.items()};d=difference(raw['fine'],raw['half'],.003125)
            d['no_observed_events']=not raw['fine']['events'] and not raw['half']['events']
            modes.append(dict(mode=j,important=row['important'],reference=d,raw=raw))
        reference=compare(hs['fine'],hs['half'],m,V[:,dominant],a,b);baseline=compare(old,hs['half'],m,V[:,dominant],a,b)
        metrics=[v for x in reference['field_records'] for reg in x['regions'].values() for v in reg.values()]+reference['reaction_intervals'];fraction=max(x['absolute']/x['budget'] for x in metrics)
        visible=[x for x in modes if x['important'] and not x['reference']['no_observed_events']];evt=bool(visible) and all(x['reference']['status']=='passed_scoped' for x in visible)
        def end_error(h):return abs(float(V[:,dominant]@m.M3ff@(h[-1]['state'].velocity-hs['half'][-1]['state'].velocity)[m.free].ravel()))
        err0,err1=end_error(old),end_error(hs['fine']);gain=1-err1/max(err0,1e-30)
        good=fraction<=.25 and evt and reference['all_engineering_fields_passed'] and gain>=.2
        records.append(dict(window=[a,b],reference_max_budget_fraction=fraction,reference_events_resolved=evt,reference_resolved=fraction<=.25 and evt,
            reference=reference,baseline=baseline,modes=modes,dominant_mode=dominant,endpoint_modal_errors=[err0,err1],endpoint_modal_improvement=gain,eligible=good))
        print('TIME_WINDOW',i,fraction,evt,gain,good,flush=True)
    times=time_grid(.0125);adopt=all(x['eligible'] for x in records)
    if adopt:
        for a,b in WINDOWS:times=sorted(set(times+[round(a+k*.000390625,12) for k in range(33)]))
    if len(times)-1>256:raise ValueError('time budget exceeded')
    write(run/'S1/reference-resolution.json',dict(status='passed_scoped' if all(x['reference_resolved'] for x in records) else 'reference_limited',records=records,full_cycle_certified=False))
    write(run/'S1/event-resolution.json',dict(records=[dict(window=x['window'],modes=x['modes']) for x in records],no_alignment=True))
    write(run/'S1/impulse-check.json',dict(records=[dict(window=x['window'],reference=x['reference']['reaction_intervals'],baseline=x['baseline']['reaction_intervals']) for x in records]))
    write(run/'S1/candidate-times.json',dict(status='adopted_scoped' if adopt else 'condition_not_triggered',new_schedules=int(adopt),times=times))
    write(run/'S1/time-decision.json',dict(status='scoped_segmented' if adopt else 'retain_dt_reference_limits',dt_s=.0125,times=times,steps=len(times)-1,temporal_accuracy=False,windows=[{k:v for k,v in x.items() if k not in ['reference','baseline','modes']} for x in records]))
    write(run/'S1/model-selection.json',dict(space=read(run/'selected-space.json'),time=read(run/'S1/time-decision.json'),material='full until final certificate'))
    write(run/'S1/dependency-invalidation.json',dict(space_changed=False,time_changed=adopt,new_main_certificate_required=True,old_states_not_rewritten=True,full_cycle_temporal_accuracy=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','analyze']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'prepare':prepare,'analyze':analyze}[a.phase](a.run)
