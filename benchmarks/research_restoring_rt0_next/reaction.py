"""Offline raw-event and engineering assessment of interval reaction histories."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_observable_boundary_next.events import events,difference,KINDS
from benchmarks.research_sequential_next.compare import metric,impulse_average


def impulse(rows,start,end):
    """Integrate recorded interval forces, including clipped boundary intervals."""
    if end<=start:raise ValueError('empty integration interval')
    cursor=start;result=0.
    for r in rows:
        left=max(start,r['time']-r['dt']);right=min(end,r['time'])
        if right<=left:continue
        if abs(left-cursor)>1e-10:raise ValueError('gap or overlap in reaction history')
        result+=(right-left)*r['reaction_N'];cursor=right
    if abs(cursor-end)>1e-10:raise ValueError('incomplete reaction interval')
    return float(result)


def event_detail(event,signal,rows):
    t=np.asarray(signal['raw_times']);y=np.asarray(signal['raw_values']);i=int(np.argmin(abs(t-event['time_s'])))
    opposite=[e for e in signal['events'] if e['kind'] in ('maximum','minimum') and e['kind']!=event['kind']]
    before=[e for e in opposite if e['time_s']<event['time_s']];after=[e for e in opposite if e['time_s']>event['time_s']]
    neighbors=([before[-1]] if before else [])+([after[0]] if after else [])
    prominence=min((abs(float(event.get('value',y[i]))-e['value']) for e in neighbors),default=None)
    lo,hi=event['bracket_s'];mask=(t>=lo)&(t<=hi);scale=float(np.max(abs(y[mask])))
    return dict(raw_event=event,nearest_interval_reaction_N=float(y[i]),local_amplitude_N=scale,
                prominence_N=prominence,prominence_relative_local_amplitude=prominence/scale if prominence is not None and scale else None,
                prominence_definition='distance to nearest opposite extrema on available sides; not a filtering criterion',
                bracket_impulse_N_s=impulse(rows,lo,hi),below_absolute_force_scale=scale<1e-4)


def analyze(run):
    run=Path(run);verify(run)
    register(run,'S4/engineering-protocol.json',dict(status='registered',reaction_absolute_N=1e-4,relative=.05,
        reaction_event_budget_s=7.534196778150833e-5,impulse_budget='1e-4 N * window duration + .05 * abs(reference impulse)',
        primary_comparison='128/256',raw_events_unchanged=True,new_steps=0))
    paths=[OLD/'cases/window0-h',OLD/'cases/window0-half',APP/'cases/phase-quarter'];hist=[history(p) for p in paths]
    if len({h[0]['state'].digest() for h in hist})!=1:raise ValueError('different initial histories')
    start=hist[0][0]['state'].time;end=hist[0][-1]['state'].time
    if any(abs(h[-1]['state'].time-end)>1e-12 for h in hist):raise ValueError('different windows')
    outputs=[]
    for label,ha,hb in [('64/128',hist[0],hist[1]),('128/256',hist[1],hist[2])]:
        ra,rb=ha[-1]['rows'],hb[-1]['rows'];sig=[events([r['time']-.5*r['dt'] for r in rows],[r['reaction_N'] for r in rows]) for rows in (ra,rb)]
        raw=difference(*sig,7.534196778150833e-5);details=[];ix=0
        for kind in KINDS:
            aa=[e for e in sig[0]['events'] if e['kind']==kind];bb=[e for e in sig[1]['events'] if e['kind']==kind]
            for a,b in zip(aa,bb):
                pair=raw['pairs'][ix];ix+=1;lo=min(a['bracket_s'][0],b['bracket_s'][0]);hi=max(a['bracket_s'][1],b['bracket_s'][1]);ia=impulse(ra,lo,hi);ib=impulse(rb,lo,hi)
                details.append(dict(**pair,left=event_detail(a,sig[0],ra),right=event_detail(b,sig[1],rb),
                    common_event_interval_s=[lo,hi],impulse_left_N_s=ia,impulse_right_N_s=ib,impulse_difference_N_s=abs(ia-ib),
                    fraction_of_window_absolute_impulse=abs(ia-ib)/max(sum(abs(r['reaction_N'])*r['dt'] for r in rb),1e-30)))
        interval=[dict(time_s=r['time'],comparison=metric(r['reaction_N'],impulse_average(rb,r['time']-r['dt'],r['time']),1e-4,.05)) for r in ra]
        peak={k:metric(fn([r['reaction_N'] for r in ra]),fn([r['reaction_N'] for r in rb]),1e-4,.05) for k,fn in [('maximum',max),('minimum',min),('max_absolute',lambda v:max(abs(x) for x in v))]}
        ia,ib=impulse(ra,start,end),impulse(rb,start,end);im=metric(ia,ib,1e-4*(end-start),.05)
        passed=all(v['passed'] for v in peak.values()) and all(r['comparison']['passed'] for r in interval) and im['passed']
        failed=[x for x in details if not x['resolved_within_budget']]
        outputs.append(dict(comparison=label,raw_event_status=raw['status'],engineering_output_status='passed_scoped' if passed else 'limited',
            raw_event_comparison=raw,signals=sig,paired_event_details=details,failed_or_ambiguous_pairs=failed,
            peak_amplitude=peak,interval_reaction=interval,impulse=dict(left_N_s=ia,right_N_s=ib,**im),
            unmatched_event_counts={k:[sum(e['kind']==k for e in s['events']) for s in sig] for k in KINDS}))
    result=outputs[-1];old=read(APP/'S4/local-phase-check.json')['reaction_events']
    if result['raw_event_comparison']!=old:raise ValueError('raw event reanalysis changed inherited result')
    write(run/'S4/reaction-engineering.json',dict(status=result['engineering_output_status'],records=outputs,
        raw_event_status=result['raw_event_status'],engineering_output_status=result['engineering_output_status'],
        inherited_raw_events_exactly_reproduced=True,modal_evidence=dict(path=str(APP/'S4/local-phase-check.json'),sha256=sha(APP/'S4/local-phase-check.json')),
        interval_force_semantics=True,time_shift=False,smoothing=False,removed_events=0,new_steps=0,
        sources=[dict(path=str(p),identity_sha256=sha(p/'identity.json'),final_state_sha256=sha(h[-1]['folder']/'state.json')) for p,h in zip(paths,hist)]))
    bad=[r for r in result['interval_reaction'] if not r['comparison']['passed']]
    recommendation=None if result['engineering_output_status']=='passed_scoped' else dict(interval_s=[start,end],failed_interval_end_times_s=[r['time_s'] for r in bad],
        candidate_dt_s=(end-start)/256,additional_window_steps_vs_128=128,requires_full_prefix_and_q5_requalification=True,not_authorized_this_round=True)
    write(run/'S4/time-decision.json',dict(status='retain_252_scoped',formal_steps=252,new_steps=0,raw_event_status=result['raw_event_status'],
        engineering_output_status=result['engineering_output_status'],global_temporal_accuracy=False,q5_scope='original parent only',next_segment=recommendation))
    print('REACTION',result['raw_event_status'],result['engineering_output_status'],'failed pairs',len(result['failed_or_ambiguous_pairs']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):analyze(a.run)
