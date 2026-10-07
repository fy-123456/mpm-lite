"""One output-relevant quarter-step reference; no global schedule change."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .run import create_config,load_model
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_observable_boundary_next.events import events,difference
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_sequential_next.compare import metric,impulse_average
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.endpoint_boundary import prescribed_speed


def prepare(run):
    run=Path(run);verify(run);settings=read(OLD/'S1/time-reference-protocol.json');old=read(APP/'S4/observable-phase-review.json');attribution=read(OLD/'S1/modal-observable-map.json');windows=[]
    for j,interval in enumerate(settings['windows']):
        h=history(OLD/f'cases/window{j}-h');fine=history(OLD/f'cases/window{j}-half');rows=h[-1]['rows'];score=max(abs(r['reaction_N']) for r in rows)
        windows.append(dict(window=j,interval=interval,reaction_amplitude_N=score,original_initial_digest=h[0]['state'].digest(),origins_equal=h[0]['state'].digest()==fine[0]['state'].digest(),old_event_review=old['events'][j]))
    # Select before the new trial using the largest actually measured raw reaction.
    chosen=max(windows,key=lambda x:x['reaction_amplitude_N']);n=chosen['window'];a,b=chosen['interval'];source=history(OLD/f'cases/window{n}-h')[0]['folder']/'state.json'
    register(run,'S4/output-phase-attribution.json',dict(status='diagnostic',inherited_attribution_sha256=sha(OLD/'S1/modal-observable-map.json'),inherited_nonlinear_remainder=old['sampled_sensitivity'],records=windows,all_important_modes=[0,1,615],inertia_stabilization_endpoint_included=True))
    register(run,'S4/window-selection.json',dict(status='registered',selected=n,interval=[a,b],reason='largest old short-window complete reaction amplitude; selected before new trajectory',initial_path=str(source),initial_sha256=sha(source),steps=256,dt_s=(b-a)/256,common_origin=True,event_budgets_s=settings['event_budgets_s'],reaction_event_budget_s=settings['reaction_event_budget_s'],new_full_cycle=False))
    create_config(run,'phase-quarter',start=a,end=b,times=np.linspace(a,b,257).tolist(),initial=source,field_cache=True,display_frames=3)
    print('PHASE_SELECTED',n,a,b,flush=True)


def analyze(run):
    run=Path(run);s=read(run/'S4/window-selection.json');n=s['selected'];old=history(OLD/f'cases/window{n}-h');a=history(OLD/f'cases/window{n}-half');b=history(run/'cases/phase-quarter');cfg=read(run/'cases/phase-quarter/execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m)
    if a[0]['state'].digest()!=b[0]['state'].digest():raise ValueError('phase origins differ')
    with np.load(OLD/'S1/modal-basis.npz') as z:V=z['vectors'].copy()
    raw=[]
    for j in (0,1,615):
        signals=[events([x['state'].time for x in h],[float(V[:,j]@m.M3ff@(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel()) for x in h]) for h in (a,b)]
        cmp=difference(*signals,s['event_budgets_s'][str(j)]);observed=bool(signals[0]['events'] or signals[1]['events']);raw.append(dict(mode=j,status=cmp['status'] if observed else 'unobserved',comparison=cmp,signals=signals))
    signals=[events([r['time']-.5*r['dt'] for r in h[-1]['rows']],[r['reaction_N'] for r in h[-1]['rows']]) for h in (a,b)];rc=difference(*signals,s['reaction_event_budget_s']);records=[]
    for i,item in enumerate(a[1:],1):
        witness=b[2*i];row=item['rows'][-1]
        if abs(item['state'].time-witness['state'].time)>1e-12:raise ValueError('phase nodes differ')
        f=fields(cache.frame(item['state']),cache.frame(witness['state']),m.parent.params.fiber_direction);R=metric(row['reaction_N'],impulse_average(b[-1]['rows'],row['time']-row['dt'],row['time']),1e-4,.05);records.append(dict(time_s=item['state'].time,fields=f,reaction=R,passed=good(f) and R['passed']))
    fieldgood=all(x['passed'] for x in records);egood=all(x['status']=='passed_scoped' for x in raw if x['status']!='unobserved') and rc['status']=='passed_scoped'
    inherited=read(OLD/'S1/time-comparison.json')['records'][n]
    write(run/'S4/local-phase-check.json',dict(status='passed_scoped' if fieldgood and egood else 'limited',common_origin=True,new_steps=256,fields_passed=fieldgood,observed_events_passed=egood,modal_events=raw,reaction_events=rc,field_records=records,coarse_to_half_inherited=inherited,inherited_comparison_sha256=sha(OLD/'S1/time-comparison.json'),no_phase_shift=True,unobserved_not_passed=True))
    write(run/'S4/time-decision.json',dict(status='retain_252_scoped',formal_times=read(APP/'S6/final-protocol.json')['times'],formal_protocol_sha256=sha(APP/'S6/final-protocol.json'),global_temporal_accuracy=False,new_short_steps=256,fields_passed=fieldgood,observed_events_passed=egood,window=n,reason='one local reference does not authorize a whole-cycle schedule change',conditional_next_segment=dict(interval=s['interval'],fine_reference_steps=256,scope='research sensitivity/reference only; would need new full-prefix response and source-bound q5 certification before any formal schedule adoption')))
    print('LOCAL_PHASE',fieldgood,egood,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','analyze']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
