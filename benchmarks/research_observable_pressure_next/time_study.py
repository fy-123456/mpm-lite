"""Bounded physical-output phase diagnostics; no filtering of accepted states."""
from pathlib import Path
import argparse,shutil
import numpy as np
from .provenance import APP,read,write,sha,register,verify,serial_lock
from .run import create_config,load_model
from .base_config import time_grid
from .events import events,difference
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_reference_next.time_study import compare
from benchmarks.research_sequential_next.compare import metric,regions,impulse_average
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.carrier_driven import displacement
from engine.aniso_phase1.endpoint_boundary import prescribed_speed
WINDOWS=((1.075,1.1),(1.225,1.25))
H=.000390625

def grid(a,b,h):return [round(a+k*h,12) for k in range(round((b-a)/h)+1)]
def model(run):return load_model(run,read(Path(run)/'cases/phase-model/execution-protocol.json'))[0]
def modal_signal(hist,m,v):return events([x['state'].time for x in hist],[float(v@m.M3ff@(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel()) for x in hist])

def baseline(run):
    run=Path(run);verify(run);(run/'S1').mkdir(exist_ok=True)
    if not (run/'cases/phase-model').exists():create_config(run,'phase-model',end=.025,field_cache=True,display_frames=3)
    m=model(run);cache=CachedProbes(m)
    mapping=read(APP/'S1/modal-observable-map.json')
    if mapping['space']!=m.reduction.signature:raise ValueError('modal basis belongs to another physical space')
    shutil.copyfile(APP/'S1/modal-basis.npz',run/'S1/modal-basis.npz');write(run/'S1/modal-observable-map.json',mapping)
    register(run,'S1/importance-protocol.json',dict(modes=[4,72,523],times_s=[1.1,1.25],small_output_fraction=.1,
        observation='mass-projected free displacement for PK1 diagnostic; mass-projected velocity for events',
        nonlinear_check='actual diagnostic state with one modal displacement removed; not an accepted dynamic trajectory',reference_budget_fraction=.25,
        important_output_minimum_gain=.2,phase_certification=False,no_filtering=True))
    with np.load(run/'S1/modal-basis.npz') as z:V=z['vectors'].copy()
    h=history(APP/'cases/final-full');lookup={round(x['state'].time,10):x for x in h};records=[]
    for t in (1.1,1.25):
        s=lookup[t]['state'];f=cache.frame(s);w=regions(f['X']);dvec=m.parent.params.fiber_direction
        for j in (4,72,523):
            a=float(V[:,j]@m.M3ff@(s.q-m.boundary.unit*displacement(t))[m.free].ravel());d=np.zeros_like(s.q);d[m.free]=V[:,j].reshape(-1,3)
            eps=1e-7/max(1.,np.linalg.norm(d));plus=s.clone();minus=s.clone();without=s.clone();plus.q+=eps*d;minus.q-=eps*d;without.q-=a*d
            derivative=(cache.frame(plus)['PK1']-cache.frame(minus)['PK1'])/(2*eps);actual=f['PK1']-cache.frame(without)['PK1'];linear=a*derivative;data={}
            for region,weights in w.items():
                impact=metric(actual,np.zeros_like(actual),.02,.05,weights);budget=.02+.05*metric(f['PK1'],np.zeros_like(actual),0,0,weights)['absolute']
                remainder=metric(actual,linear,.02,.05,weights)['absolute'];linear_norm=metric(linear,np.zeros_like(linear),0,0,weights)['absolute']
                data[region]=dict(actual_Pa=impact['absolute'],linear_Pa=linear_norm,remainder_Pa=remainder,budget_Pa=budget,
                    small_at_this_state=max(impact['absolute'],linear_norm)+remainder<.1*budget)
            records.append(dict(time_s=t,mode=j,displacement_coordinate=a,regions=data,diagnostic_only=True))
    raw=[]
    for i in (0,1):
        hs={label:history(APP/'cases'/f'time-{i}-{label}') for label in ('bridge','fine','half')}
        for label,hist in hs.items():
            for key in ('q','velocity','predictor'):
                if not np.array_equal(getattr(hist[0]['state'],key),getattr(hs['half'][0]['state'],key)):raise ValueError('parent window initial changed')
        raw.append(dict(window=i,modes=[dict(mode=j,signals={k:modal_signal(v,m,V[:,j]) for k,v in hs.items()}) for j in (4,72,523)]))
    counts={k:len(raw[0]['modes'][2]['signals'][k]['events']) for k in ('bridge','fine','half')}
    if counts!={'bridge':7,'fine':9,'half':8}:raise ValueError('parent raw event mismatch not reproduced')
    write(run/'S1/output-baseline.json',dict(status='passed_scoped',source=str(APP),source_release_sha256=sha(APP/'release.json'),inherited_fields=read(APP/'S1/reference-resolution.json'),
        original_window_initials=read(APP/'S1/window-initials.json'),raw_event_counts=counts,new_steps=0,raw=raw))
    write(run/'S1/observable-importance.json',dict(status='diagnostic_scoped',records=records,reaction_importance='requires actual paired interval equations; not inferred from static modal stress',mode523_retained=True))
    important=any(not x['small_at_this_state'] for r in records for x in r['regions'].values())
    register(run,'S1/candidate-protocol.json',dict(triggered=important,windows=WINDOWS,dt_s=H,reference_dt_s=H/2,steps=252,max_candidate_diagnostic_steps=144,max_reference_steps=256,
        local_scope_only=True,no_global_phase_claim=True,source=str(APP/'cases/final-full'),minimum_gain=.2))
    if important:
        times=grid(1.075,1.1,H)+grid(1.1,1.225,.0125)[1:]+grid(1.225,1.25,H)[1:]
        create_config(run,'candidate-chain',start=1.075,end=1.25,times=times,initial=lookup[1.075]['folder']/'state.json',field_cache=True,display_frames=5)
        create_config(run,'reference-0',start=1.075,end=1.1,times=grid(1.075,1.1,H/2),initial=lookup[1.075]['folder']/'state.json',field_cache=True,display_frames=3)
    else:write(run/'S1/time-decision.json',dict(status='retain_output_impact_small',times=time_grid(.0125),steps=128,temporal_accuracy=False))
    print('OUTPUT_IMPORTANCE',important,'RAW_EVENTS',counts,flush=True)

def candidate(run):
    run=Path(run);records=read(run/'S1/observable-importance.json')['records'];important=any(not x['small_at_this_state'] for r in records for x in r['regions'].values())
    register(run,'S1/candidate-protocol-amendment.json',dict(original_sha256=sha(run/'S1/candidate-protocol.json'),triggered=important,
        reason='trigger must consider all pre-registered important output modes, not only event-count mode523; mode4 and72 exceed 10% output budget; no candidate integrated before correction',
        windows=WINDOWS,dt_s=H,reference_dt_s=H/2,max_candidate_steps=144,max_reference_steps=256,steps=252,local_scope_only=True))
    if not important:raise ValueError('no important output candidate trigger')
    lookup={round(x['state'].time,10):x for x in history(APP/'cases/final-full')}
    times=grid(1.075,1.1,H)+grid(1.1,1.225,.0125)[1:]+grid(1.225,1.25,H)[1:]
    create_config(run,'candidate-chain',start=1.075,end=1.25,times=times,initial=lookup[1.075]['folder']/'state.json',field_cache=True,display_frames=5)
    create_config(run,'reference-0',start=1.075,end=1.1,times=grid(1.075,1.1,H/2),initial=lookup[1.075]['folder']/'state.json',field_cache=True,display_frames=3)
    write(run/'S1/time-decision.json',dict(status='pending_local_candidate',times=time_grid(.0125),steps=128,temporal_accuracy=False))


def second(run):
    run=Path(run);h=history(run/'cases/candidate-chain');x=next(x for x in h if abs(x['state'].time-1.225)<1e-11)
    for name,dt in [('reference-1',H/2),('coarse-witness-1',.0125)]:
        create_config(run,name,start=1.225,end=1.25,times=grid(1.225,1.25,dt),initial=x['folder']/'state.json',field_cache=True,display_frames=3)
    write(run/'S1/second-window-initial.json',dict(path=str(x['folder']/'state.json'),sha256=sha(x['folder']/'state.json'),history='candidate carries first refinement through ten coarse hold intervals'))

def analyze(run):
    run=Path(run);m=model(run)
    with np.load(run/'S1/modal-basis.npz') as z:V=z['vectors'].copy()
    chain=history(run/'cases/candidate-chain');base=history(APP/'cases/final-full');records=[];eligible=True
    for i,(a,b) in enumerate(WINDOWS):
        cand=[x for x in chain if a-1e-11<=x['state'].time<=b+1e-11];ref=history(run/'cases'/f'reference-{i}')
        coarse=[x for x in base if a-1e-11<=x['state'].time<=b+1e-11] if i==0 else history(run/'cases/coarse-witness-1')
        for h in (cand,coarse):
            for key in ('q','velocity','predictor'):
                if not np.array_equal(getattr(h[0]['state'],key),getattr(ref[0]['state'],key)):raise ValueError('paired window initial mismatch')
        new=compare(cand,ref,m,V[:,72],a,b);old=compare(coarse,ref,m,V[:,72],a,b)
        # Fair physical comparisons on exactly the coarse observation times/intervals.
        common={round(x['state'].time,10) for x in coarse};new_common=[x for x in cand if round(x['state'].time,10) in common]
        fair=compare(new_common,ref,m,V[:,72],a,b);gains=[]
        lookup={round(x['time_s'],10):x for x in fair['field_records']}
        for row in old['field_records'][1:]:
            for region,fields in row['regions'].items():
                for key,err in fields.items():
                    newer=lookup[round(row['time_s'],10)]['regions'][region][key];gain=1-newer['absolute']/max(err['absolute'],1e-30)
                    gains.append(dict(time_s=row['time_s'],region=region,field=key,old=err['absolute'],new=newer['absolute'],budget=err['budget'],gain=gain,meaningful=err['absolute']>.1*err['budget']))
        improved=any(x['meaningful'] and x['gain']>=.2 for x in gains)
        ev=[dict(mode=j,comparison=difference(modal_signal(cand,m,V[:,j]),modal_signal(ref,m,V[:,j]),.003125),raw_candidate=modal_signal(cand,m,V[:,j]),raw_reference=modal_signal(ref,m,V[:,j])) for j in (4,72,523)]
        passed=new['all_engineering_fields_passed'] and improved;eligible &=passed
        records.append(dict(window=[a,b],new=new,old=old,fair_gains=gains,important_output_improved=improved,events=ev,eligible=passed,reference_scope='one adjacent finer trajectory; event/continuum certification remains limited'))
        print('LOCAL_PHASE',i,'fields',new['all_engineering_fields_passed'],'meaningful_improvement',improved,flush=True)
    times=time_grid(.0125)
    if eligible:times=sorted(set(times+[t for a,b in WINDOWS for t in grid(a,b,H)]))
    write(run/'S1/local-reference-check.json',dict(status='passed_scoped' if eligible else 'retain_baseline',records=records,reference_steps=256,candidate_and_witness_steps=140,continuous_time_accuracy=False))
    write(run/'S1/time-decision.json',dict(status='scoped_local_output_improvement' if eligible else 'retain_reference_or_output_limits',times=times,steps=len(times)-1,temporal_accuracy=False,
        candidate_count=1,dt_s=None if eligible else .0125,scope='local unloading-end and hold outputs only; full cycle pending',no_damping=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['baseline','candidate','second','analyze']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
