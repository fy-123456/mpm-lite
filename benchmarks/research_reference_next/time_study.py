"""Same-state phase windows on the actually selected final solid space."""
from pathlib import Path
import argparse,copy
import numpy as np
import scipy.linalg as la
from .provenance import APP,PARENT,read,write,register,verify,serial_lock
from .run import create_config,run_case,load_model
from benchmarks.research_post_release.time_study import history
from benchmarks.research_sequential_next.compare import metric,regions,impulse_average
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_sequential_next.model import PracticalModel

def events(times,values):
    times=np.asarray(times);v=np.asarray(values);out=[]
    for i in range(1,len(v)-1):
        kind='positive_peak' if v[i]>v[i-1] and v[i]>=v[i+1] else ('negative_peak' if v[i]<v[i-1] and v[i]<=v[i+1] else None)
        if kind:out.append(dict(kind=kind,time_s=float(times[i]),value=float(v[i]),bracket_s=[float(times[i-1]),float(times[i+1])]))
    for i in range(len(v)-1):
        if v[i]*v[i+1]<0:out.append(dict(kind='up_zero' if v[i+1]>v[i] else 'down_zero',
                                     bracket_s=[float(times[i]),float(times[i+1])]))
    return out


def event_difference(a,b):
    pairs=[]
    for kind in ('positive_peak','negative_peak','up_zero','down_zero'):
        aa=[x for x in a if x['kind']==kind];bb=[x for x in b if x['kind']==kind]
        for left,right in zip(aa,bb):
            lt=left.get('time_s',sum(left['bracket_s'])/2);rt=right.get('time_s',sum(right['bracket_s'])/2)
            pairs.append(dict(kind=kind,offset_s=lt-rt,a_bracket=left['bracket_s'],b_bracket=right['bracket_s']))
    return dict(pairs=pairs,max_abs_offset_s=max((abs(x['offset_s']) for x in pairs),default=None),
                same_event_counts=all(sum(x['kind']==k for x in a)==sum(x['kind']==k for x in b)
                  for k in ('positive_peak','negative_peak','up_zero','down_zero')))


def compare(a,b,model,mode,start,end):
    a=[h for h in a if start-1e-10<=h['state'].time<=end+1e-10]
    b=[h for h in b if start-1e-10<=h['state'].time<=end+1e-10]
    lookup={round(h['state'].time,10):h for h in b};points=[33,7,7]
    cache=CachedProbes(model,tuple(points))
    records=[];modal_error=[];modal_ref=[]
    for h in a:
        other=lookup[round(h['state'].time,10)]
        sa,sb=h['state'],other['state']
        fa,fb=cache.frame(sa),cache.frame(sb)
        assert np.array_equal(fa['X'],fb['X'])
        data={}
        d=np.asarray(model.parent.params.fiber_direction);d=d/la.norm(d)
        for region,w in regions(fa['X']).items():
            data[region]={key:metric(fa[key]-fa['X'] if key=='x' else fa[key],fb[key]-fb['X'] if key=='x' else fb[key],atol,.05,w)
                          for key,atol in [('x',5e-5),('velocity',1e-4),('PK1',.02)]}
            data[region]['fiber_PK1']=metric(np.einsum('i,...ij,j->...',d,fa['PK1'],d),np.einsum('i,...ij,j->...',d,fb['PK1'],d),.02,.05,w)
        records.append(dict(time_s=sa.time,regions=data))
        va=sa.velocity[model.free].ravel();vb=sb.velocity[model.free].ravel()
        modal_error.append(float(mode @ model.M3ff @ (va-vb)));modal_ref.append(float(mode @ model.M3ff @ vb))
    def signal(hist):return [float(mode @ model.M3ff @ x['state'].velocity[model.free].ravel()) for x in hist]
    ta=[x['state'].time for x in a];tb=[x['state'].time for x in b]
    ea,eb=events(ta,signal(a)),events(tb,signal(b))
    ra=[r for r in a[-1]['rows'] if start+1e-11<r['time']<=end+1e-11]
    rb=[r for r in b[-1]['rows'] if start+1e-11<r['time']<=end+1e-11]
    reaction=[dict(start=r['time']-r['dt'],end=r['time'],**metric(r['reaction_N'],impulse_average(rb,r['time']-r['dt'],r['time']),1e-4,.05)) for r in ra]
    reaction_events=event_difference(events([r['time']-.5*r['dt'] for r in ra],[r['reaction_N'] for r in ra]),
                                     events([r['time']-.5*r['dt'] for r in rb],[r['reaction_N'] for r in rb]))
    velocity_rms=float(np.sqrt(np.mean(np.square(modal_error))))
    passed=all(m['passed'] for x in records for reg in x['regions'].values() for m in reg.values()) and all(x['passed'] for x in reaction)
    return dict(field_records=records,reaction_intervals=reaction,all_engineering_fields_passed=passed,
                modal_velocity_error_rms=velocity_rms,modal_velocity_reference_rms=float(np.sqrt(np.mean(np.square(modal_ref)))),
                modal_events=event_difference(ea,eb),reaction_events=reaction_events,
                raw_modal_signals=dict(a_times=ta,a_values=signal(a),b_times=tb,b_values=signal(b)),
                phase_alignment_applied=False,reference_scope='local fine-step comparison, not an exact solution')


def windows(run):
    run=Path(run);verify(run)
    if read(run/'selected-space.json')['selected']!='original144':raise ValueError('candidate needs explicit dynamic loader and common prefix before phase study')
    register(run,'Q3/protocol.json',dict(source=str(APP/'cases/full-q7-dt0125'),windows=[[1.,1.2],[1.2,1.4]],
        dt=[.0125,.00625,.003125],minimum_improvement=.20,event_budget_s=.00625,full_schedules=['uniform .00625','segmented from .6'],
        scope='local windows from same latest committed q/v/history; no time alignment',material=7,field_cache=True))
    parent=history(APP/'cases/full-q7-dt0125');index={round(x['state'].time,10):x for x in parent}
    create_config(run,'phase-model',dt=.0125,end=.025)
    model,_=load_model(run,read(run/'cases/phase-model/execution-protocol.json'));r=model.reduction
    fields=PracticalModel(r,order=7,device='cpu');values,vectors=la.eigh(model.rest_K[np.ix_(model.ids,model.ids)],model.M3ff,subset_by_index=[0,4]);mode=vectors[:,4]
    write(run/'Q3/modal-definition.json',dict(index=4,eigenvalue=float(values[4]),period_s=float(2*np.pi/np.sqrt(values[4]))))
    reports=[]
    for i,(start,end) in enumerate([(1.,1.2),(1.2,1.4)]):
        runs=[]
        for dt,label in [(.0125,'0125'),(.00625,'00625'),(.003125,'003125')]:
            name=f'phase-{i}-{label}';create_config(run,name,dt=dt,start=start,end=end,initial=index[start]['folder']/'state.json',display_frames=3,field_cache=True)
            run_case(run,name,quiet=True);runs.append(history(run/'cases'/name))
        coarse=compare(runs[0],runs[2],fields,mode,start,end);medium=compare(runs[1],runs[2],fields,mode,start,end)
        improvement=1-medium['modal_velocity_error_rms']/max(coarse['modal_velocity_error_rms'],1e-30)
        event=medium['modal_events'];phase=improvement>=.2 and event['same_event_counts'] and event['max_abs_offset_s'] is not None and event['max_abs_offset_s']<=.00625+1e-12
        accepted=phase and medium['all_engineering_fields_passed']
        result=dict(window=[start,end],coarse_vs_fine=coarse,medium_vs_fine=medium,modal_improvement=improvement,phase_passed=phase,accepted=accepted)
        write(run/f'Q3/window{i}.json',result);reports.append(result)
        print('PHASE',i,improvement,event['max_abs_offset_s'],medium['all_engineering_fields_passed'],flush=True)
    selected=all(x['accepted'] for x in reports)
    from benchmarks.research_sequential_next.config import time_grid
    # Prefer uniform when the prefix has not been independently shown accurate;
    # no extra full-cycle screening of competing schedules.
    dt=.00625 if selected else .0125;times=time_grid(dt)
    write(run/'Q3/time-decision.json',dict(status='phase_improved_scoped' if selected else 'retain_current_time_step',dt_s=dt,times=times,steps=len(times)-1,
        windows=[{k:v for k,v in x.items() if k not in ('coarse_vs_fine','medium_vs_fine')} for x in reports],
        temporal_certified=False,reason='uniform selected step avoids unqualified coarse loading prefix' if selected else 'registered combined phase/field gates did not all pass',
        full_cycle='deferred to Q6; no damping, no change to AVF equations'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; fork before running studies')
        windows(a.run)
