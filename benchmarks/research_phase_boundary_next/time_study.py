"""Two authenticated short-window trajectories on the actual current solid."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .run import create_config,load_model
from .events import events,difference
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_reference_next.time_study import compare
from engine.aniso_phase1.endpoint_boundary import prescribed_speed
H=.000390625
WINDOWS=((1.075,1.125),(1.225,1.25))

def grid(a,b,h):return [float(a+k*h) for k in range(round((b-a)/h)+1)]

def prepare(run):
    run=Path(run);verify(run);(run/'S1').mkdir(exist_ok=True);hist=history(APP/'cases/final-full');origin=next(x for x in hist if abs(x['state'].time-1.075)<1e-12)
    if read(APP/'selected-space.json')['package']['sha256']!=read(run/'selected-space.json')['package']['sha256']:raise ValueError('wrong solid space')
    cfg=read(run/'cases/compatibility/execution-protocol.json');m,_=load_model(run,cfg);m.validate(origin['state']);K=m.rest_K[np.ix_(m.ids,m.ids)];M=m.M3ff
    with np.load(APP/'S1/modal-basis.npz') as z:V=z['vectors'].copy();lam=z['values'].copy()
    orth=float(la.norm(V.T@M@V-np.eye(len(lam))));res=float(la.norm(K@V-(M@V)*lam)/la.norm(K@V))
    if orth>1e-6 or res>1e-7 or lam.min()<=0:raise ValueError('current modes do not match loaded M/K')
    velocity=np.array([(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel() for x in hist]);energy=.5*(velocity@M@V)**2
    forcing=np.abs(V.T@(m.rest_K@m.boundary.unit.ravel())[m.ids]);selected=sorted(set([0,int(np.argmax(energy.max(axis=0))),int(np.argmax(forcing))]))
    np.savez_compressed(run/'S1/modal-basis.npz',vectors=V,values=lam)
    write(run/'S1/modal-observable-map.json',dict(status='passed_scoped',space=m.reduction.signature,source_sha256=sha(APP/'S1/modal-basis.npz'),selected_modes=selected,energy_peaks_J=energy.max(axis=0)[selected].tolist(),periods_s=(2*np.pi/np.sqrt(lam[selected])).tolist(),mass_orthogonality=orth,eigen_residual=res,selection='current full-cycle kinetic peak and current forcing; no legacy modal indices'))
    source=origin['folder']/'state.json';write(run/'S1/origin-audit.json',dict(status='passed_scoped',source=str(source),sha256=sha(source),digest=origin['state'].digest(),time_s=1.075,space=read(run/'selected-space.json'),q7=True,complete_history=True))
    register(run,'S1/time-reference-protocol.json',dict(windows=WINDOWS,h_s=H,bridge_h_s=.0125,start=1.075,end=1.25,steps=[200,392],total_steps=592,common_initial_sha256=sha(source),distinct_second_window_states=True,frames_per_branch=6,no_damping=True,event_budget_s=.00625,formal_candidate_steps=314))
    for name,h in [('time-h',H),('time-half',H/2)]:
        ts=grid(1.075,1.125,h)+grid(1.125,1.225,.0125)[1:]+grid(1.225,1.25,h)[1:]
        create_config(run,name,start=ts[0],end=ts[-1],times=ts,initial=source,field_cache=True,display_frames=6)
    print('TIME_READY',selected,flush=True)

def analyze(run):
    run=Path(run);a=history(run/'cases/time-h');b=history(run/'cases/time-half');old=history(APP/'cases/final-full');m,_=load_model(run,read(run/'cases/time-h/execution-protocol.json'))
    mapping=read(run/'S1/modal-observable-map.json')
    with np.load(run/'S1/modal-basis.npz') as z:V=z['vectors'].copy()
    for key in ('q','velocity','predictor'):
        if not np.array_equal(getattr(a[0]['state'],key),getattr(b[0]['state'],key)):raise ValueError('different initial state')
    records=[];gains=[];events_ok=True
    for start,end in WINDOWS:
        aa=[x for x in a if start-1e-12<=x['state'].time<=end+1e-12];bb=[x for x in b if start-1e-12<=x['state'].time<=end+1e-12]
        fine=compare(aa,bb,m,V[:,mapping['selected_modes'][0]],start,end);baseline=compare(old,bb,m,V[:,mapping['selected_modes'][0]],start,end)
        lookup={round(x['time_s'],10):x for x in fine['field_records']}
        for row in baseline['field_records'][1:]:
            for region,fields in row['regions'].items():
                for key,v in fields.items():
                    w=lookup[round(row['time_s'],10)]['regions'][region][key];gains.append(dict(time_s=row['time_s'],region=region,field=key,old=v['absolute'],new=w['absolute'],budget=w['budget'],important=v['absolute']>.1*v['budget'],gain=1-w['absolute']/max(v['absolute'],1e-30)))
        raw=[]
        for j in mapping['selected_modes']:
            def signal(h):return events([x['state'].time for x in h],[float(V[:,j]@m.M3ff@(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel()) for x in h])
            x,y=signal(aa),signal(bb);d=difference(x,y);raw.append(dict(mode=j,a=x,b=y,comparison=d))
            if x['events'] or y['events']:events_ok &= d['status']=='passed_scoped'
        def reactions(h):
            r=[x for x in h[-1]['rows'] if start+1e-11<x['time']<=end+1e-11]
            return events([x['time']-.5*x['dt'] for x in r],[x['reaction_N'] for x in r])
        x,y=reactions(aa),reactions(bb);rd=difference(x,y)
        if x['events'] or y['events']:events_ok &= rd['status']=='passed_scoped'
        records.append(dict(window=[start,end],comparison=fine,baseline=baseline,modal_events=raw,reaction_events=dict(a=x,b=y,comparison=rd)))
    fields_ok=all(x['comparison']['all_engineering_fields_passed'] for x in records);useful=any(x['important'] and x['gain']>=.2 for x in gains);adopt=bool(fields_ok and useful and events_ok)
    times=read(APP/'S6/final-protocol.json')['times'];old_times=list(times)
    if adopt:times=sorted(set([t for t in times if not 1.075<t<1.125]+grid(1.075,1.125,H)))
    if len(times)-1 not in (252,314):raise ValueError('unexpected final step budget')
    write(run/'S1/time-comparison.json',dict(status='passed_scoped' if fields_ok else 'reference_limited',records=records,gains=gains,events_passed=events_ok,useful=useful,actual_new_steps=len(a)+len(b)-2,reference_is_empirical=True,no_alignment=True))
    write(run/'S1/time-raw-check.json',dict(status='passed_scoped',branches={k:read(run/'cases'/k/'summary.json') for k in ('time-h','time-half')},common_initial=True,second_initials_not_reset=True))
    write(run/'S1/time-decision.json',dict(status='adopt_314_scoped' if adopt else 'retain_252_scoped',times=times,steps=len(times)-1,fields_passed=fields_ok,events_passed=events_ok,meaningful_gain=useful,temporal_accuracy=False,space=read(run/'selected-space.json')['package']['sha256']))
    write(run/'S1/time-scope.json',dict(status='limited',space=read(run/'selected-space.json'),windows=WINDOWS,material='F45 .005 q7',global_temporal_accuracy=False,full_cycle_pending=True,formal_grid_changed=times!=old_times,no_new_space_transfer=True))
    print('TIME_DECISION',adopt,len(times)-1,fields_ok,events_ok,useful,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','analyze']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
