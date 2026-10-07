"""Observable attribution and two independent authenticated high-resolution windows."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .run import create_config,load_model
from .events import events,difference
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_reference_next.time_study import compare
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.endpoint_boundary import prescribed_speed
H=.000390625
h=H/8
WINDOWS=((1.075,1.078125),(1.225,1.228125))
def grid(a,b,dt):return [float(a+i*dt) for i in range(round((b-a)/dt)+1)]

def reaction(m,cfg,s0,s1):
    dt=s1.time-s0.time;W=(s1.q-s0.q)/dt
    path=ValidatedAVF(m,cfg,s0).path(s0.q,W,dt)
    midpoint=2*m.M@(W-s0.velocity)+dt*path['force']
    _,endpoint,_=m.endpoint(2*W-s0.velocity,s1.time)
    return dict(total=float(np.sum((midpoint+endpoint)*m.boundary.unit)/dt),material=float(np.sum(path['material']*m.boundary.unit)),inertia=float(np.sum((2*m.M@(W-s0.velocity))*m.boundary.unit)/dt),endpoint=float(np.sum(endpoint*m.boundary.unit)/dt))

def prepare(run):
    run=Path(run);verify(run);(run/'S1').mkdir(exist_ok=True);hist=history(APP/'cases/final-full');cfg=read(run/'cases/compatibility/execution-protocol.json');m,_=load_model(run,cfg)
    with np.load(APP/'S1/modal-basis.npz') as z:V=z['vectors'].copy();lam=z['values'].copy()
    K=m.rest_K[np.ix_(m.ids,m.ids)];M=m.M3ff
    orth=float(la.norm(V.T@M@V-np.eye(len(lam))));res=float(la.norm(K@V-(M@V)*lam)/la.norm(K@V))
    if orth>1e-6 or res>1e-7 or lam.min()<=0:raise ValueError('current modes do not match full M/K')
    coeff=np.array([(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel() for x in hist])@M@V
    energy=.5*coeff**2;selected=[0,1,615];periods=(2*np.pi/np.sqrt(lam[selected])).tolist();cache=CachedProbes(m);weights=regions(cache.X)
    indices=sorted(set([int(np.argmax(energy[:,j])) for j in selected]+[min(range(1,len(hist)),key=lambda i:abs(hist[i]['state'].time-t)) for t in (.5,1.075,1.1,1.225,1.6)]));indices=[i for i in indices if i>0];attrib=[]
    for i in indices:
        s0,s1=hist[i-1]['state'],hist[i]['state'];baseR=reaction(m,cfg,s0,s1);row=hist[i]['rows'][-1]
        if abs(baseR['total']-row['reaction_N'])>1e-7:raise ValueError('complete reaction reconstruction differs')
        base=cache.frame(s1)
        for j in selected:
            altered=[]
            for s in (s0,s1):
                out=s.clone();qfree=(s.q-m.boundary.lift(s.time))[m.free].ravel();vfree=(s.velocity-m.boundary.unit*prescribed_speed(s.time))[m.free].ravel()
                out.q[m.free]-=(V[:,j]*(V[:,j]@M@qfree)).reshape(-1,3);out.velocity[m.free]-=(V[:,j]*(V[:,j]@M@vfree)).reshape(-1,3);altered.append(out)
            f=cache.frame(altered[1]);r=reaction(m,cfg,*altered);etas={};d=m.parent.params.fiber_direction
            for region,w in weights.items():
                for field,at in [('x',5e-5),('velocity',1e-4),('PK1',.02)]:
                    a,b=f[field],base[field]
                    if field=='x':a=a-base['X'];b=b-base['X']
                    e=metric(a,b,at,.05,w);etas[region+'/'+field]=e['absolute']/e['budget']
                e=metric(np.einsum('i,...ij,j->...',d,f['PK1'],d),np.einsum('i,...ij,j->...',d,base['PK1'],d),.02,.05,w);etas[region+'/fiber']=e['absolute']/e['budget']
            e=metric(r['total'],baseR['total'],1e-4,.05);etas['complete_reaction']=e['absolute']/e['budget']
            attrib.append(dict(mode=j,time_s=s1.time,eta=etas,complete_reaction=baseR,offline_removed_reaction=r))
    np.savez_compressed(run/'S1/modal-basis.npz',vectors=V,values=lam)
    write(run/'S1/modal-observable-map.json',dict(status='passed_scoped',space=m.reduction.signature,selected_modes=selected,energy_peaks_J=energy.max(axis=0)[selected].tolist(),periods_s=periods,mass_orthogonality=orth,eigen_residual=res,attribution=attrib,all_modes_retained_important=True,attribution_scope='selected actual states; not a uniform output bound; no removal from any trajectory',unresolved_reconstruction_remainder=True))
    origins=[]
    for n,(a,b) in enumerate(WINDOWS):
        x=next(x for x in hist if abs(x['state'].time-a)<1e-12);source=x['folder']/'state.json';origins.append(dict(time_s=a,path=str(source),sha256=sha(source),digest=x['state'].digest()))
        for label,dt in [('h',h),('half',h/2)]:create_config(run,f'window{n}-{label}',start=a,end=b,times=grid(a,b,dt),initial=source,field_cache=True,display_frames=3)
    write(run/'S1/origin-audit.json',dict(status='passed_scoped',origins=origins,independent_windows=True,q7=True))
    register(run,'S1/time-reference-protocol.json',dict(status='passed_scoped',windows=WINDOWS,H=H,h=h,steps_per_window=[64,128],initial_steps=384,rebase_reserved_steps=192,max_total_steps=576,important_modes=selected,event_budgets_s={str(j):min(.00625,.1*t) for j,t in zip(selected,periods)},reaction_event_budget_s=min(.00625,.1*min(periods)),important_outputs=['regional displacement','regional velocity','regional PK1','regional fiber','complete reaction impulse'],empty_events='unobserved',no_downgrade_from_energy_only=True,no_damping_or_filtering=True,attribution_uniform_bound=False))
    print('TIME_READY',origins,periods,flush=True)

def window_report(m,V,settings,a,b,old,start,end):
    mapping=read(Path(settings['_run'])/'S1/modal-observable-map.json');fine=compare(a,b,m,V[:,0],start,end);baseline=compare(old,b,m,V[:,0],start,end);gains=[];lookup={round(x['time_s'],10):x for x in fine['field_records']}
    for row in baseline['field_records'][1:]:
        for region,fields in row['regions'].items():
            for key,v in fields.items():
                w=lookup[round(row['time_s'],10)]['regions'][region][key];gains.append(dict(time_s=row['time_s'],region=region,field=key,old=v['absolute'],new=w['absolute'],budget=w['budget'],important=v['absolute']>.1*v['budget'],gain=1-w['absolute']/max(v['absolute'],1e-30)))
    raw=[];observed=True
    for j in settings['important_modes']:
        def signal(hist):return events([x['state'].time for x in hist],[float(V[:,j]@m.M3ff@(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel()) for x in hist])
        x,y=signal(a),signal(b);cmp=difference(x,y,settings['event_budgets_s'][str(j)]);is_observed=bool(x['events'] or y['events']);observed &=is_observed
        raw.append(dict(mode=j,a=x,b=y,comparison=cmp,status=cmp['status'] if is_observed else 'unobserved'))
    def rs(hist):
        rows=[r for r in hist[-1]['rows'] if start+1e-11<r['time']<=end+1e-11]
        return events([r['time']-.5*r['dt'] for r in rows],[r['reaction_N'] for r in rows])
    x,y=rs(a),rs(b);rc=difference(x,y,settings['reaction_event_budget_s']);ro=bool(x['events'] or y['events'])
    # No-event low modes remain unobserved, never relabelled passed. A trial can
    # rely on measured output accuracy; every actually observed event must pass.
    eventgood=all(v['comparison']['status']=='passed_scoped' for v in raw if v['status']!='unobserved') and (not ro or rc['status']=='passed_scoped')
    return dict(window=[start,end],comparison=fine,baseline=baseline,gains=gains,modal_events=raw,reaction_events=dict(a=x,b=y,comparison=rc,status=rc['status'] if ro else 'unobserved'),all_events_observed=bool(observed and ro),observed_events_passed=bool(eventgood))

def analyze(run):
    run=Path(run);settings=read(run/'S1/time-reference-protocol.json');settings['_run']=str(run);old=history(APP/'cases/final-full');m,_=load_model(run,read(run/'cases/window0-h/execution-protocol.json'))
    with np.load(run/'S1/modal-basis.npz') as z:V=z['vectors'].copy()
    records=[]
    for n,(start,end) in enumerate(WINDOWS):
        a=history(run/f'cases/window{n}-h');b=history(run/f'cases/window{n}-half')
        if a[0]['state'].digest()!=b[0]['state'].digest():raise ValueError('independent branch origins differ')
        records.append(window_report(m,V,settings,a,b,old,start,end))
    fields=all(r['comparison']['all_engineering_fields_passed'] for r in records);eventgood=all(r['observed_events_passed'] for r in records);gains=[v for r in records for v in r['gains']];useful=any(v['important'] and v['gain']>=.2 for v in gains);no_regression=all(v['new']-v['old']<=.1*v['budget'] for v in gains)
    trial=bool(fields and eventgood and useful and no_regression);times=read(APP/'S6/final-protocol.json')['times']
    if trial:
        for a,b in WINDOWS:times=sorted(set([t for t in times if not a<t<b]+grid(a,b,h)))
    if len(times)-1 not in (252,364):raise ValueError('time budget mismatch')
    write(run/'S1/time-comparison.json',dict(status='passed_scoped' if fields else 'limited',records=records,initial_steps=384,observed_events_passed=eventgood,unobserved_events_claimed_pass=False,global_temporal_accuracy=False))
    write(run/'S1/time-raw-check.json',dict(status='passed_scoped',cases={f'window{n}-{s}':read(run/f'cases/window{n}-{s}/summary.json') for n in range(2) for s in ('h','half')},independent_common_origins=True))
    write(run/'S1/time-decision.json',dict(status='trial_364_pending_full_prefix' if trial else 'retain_252_scoped',times=times,steps=len(times)-1,fields_passed=fields,events_passed=eventgood,meaningful_gain=useful,no_regression=no_regression,trial=trial,global_temporal_accuracy=False,space=read(run/'selected-space.json')['package']['sha256']))
    write(run/'S1/time-scope.json',dict(status='limited',windows=WINDOWS,global_temporal_accuracy=False,unobserved_not_passed=True,trial_pending_prefix=trial,second_prefix_rebase_steps=192 if trial else 0,old_failures_retained=str(APP/'S1/time-comparison.json')))
    print('TIME_DECISION',trial,len(times)-1,fields,eventgood,useful,no_regression,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','analyze']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
