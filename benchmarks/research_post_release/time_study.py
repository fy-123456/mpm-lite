"""Two bounded phase windows, raw event times and one selected full cycle."""
from pathlib import Path
import argparse
import copy
import numpy as np
import scipy.linalg as la
from .provenance import PARENT,read,write,register,serial_lock,utc,verify
from .run import create_config,run_case,load_model
from . import config
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.run import probe_frame
from benchmarks.research_sequential_next.compare import metric,regions,impulse_average


def history(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()


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
    records=[];modal_error=[];modal_ref=[]
    for h in a:
        other=lookup[round(h['state'].time,10)]
        sa,sb=h['state'],other['state']
        fa,fb=probe_frame(model,sa,points),probe_frame(model,sb,points)
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


def prepare(run):
    verify(run)
    register(run,'S1/protocol.json',dict(windows=[[1.,1.2],[1.2,1.4]],dt=[.025,.0125,.00625],
        fine_steps_max_per_window=32,minimum_meaningful_improvement=.20,phase_event_budget_s=.0125,
        events='ordered extrema by sign; zero crossing brackets; original times, no time-shift',
        observations='rest mode index 4, all common-time physical fields and interval reaction impulses',
        hypotheses='frequency dispersion after excluding previously checked tolerance/path/material causes',
        candidate_full_schedules=['uniform .0125 (128)','0-.5:.025; .5-1.6:.0125 (108)'],
        full_cycle_selection='prefer uniform .0125, do not run both full cycles',
        no_damping=True,no_artificial_mass=True))
    times=[0.,.0125,.03125,.05]
    for name in ['compat-restart','compat-uninterrupted']:
        create_config(run,name,start=0.,end=.05,times=times,display_frames=3)
    invalid=[]
    base=read(Path(run)/'cases/compat-restart/execution-protocol.json')
    for times in ([0.,.01,.01,.05],[0.,.03,.02,.05],[0.,float('nan'),.05],[0.,.55,1.6]):
        value=copy.deepcopy(base);value['times']=list(times)
        try:config.validate(value)
        except ValueError:invalid.append(True)
        else:invalid.append(False)
    assert all(invalid)
    write(Path(run)/'S1/config-check.json',dict(invalid_grids_rejected=invalid,valid_grid=base['times']))


def check_restart(run):
    a=history(Path(run)/'cases/compat-restart-v2');b=history(Path(run)/'cases/compat-uninterrupted-v2')
    for x,y in zip(a,b):
        for key in ('q','velocity','predictor'):
            assert np.array_equal(getattr(x['state'],key),getattr(y['state'],key)),key
    def numeric(rows):return [{k:v for k,v in r.items() if k!='wall_seconds'} for r in rows]
    assert numeric(a[-1]['rows'])==numeric(b[-1]['rows'])
    write(Path(run)/'S1/restart-check.json',dict(status='passed',actual_process_restart=True,steps=len(a)-1,
          state_arrays_equal=True,numerical_ledger_equal=True,wall_seconds_excluded=True))


def windows(run):
    parent=history(PARENT/'cases/seg-q7-dt0025');index={round(x['state'].time,10):x for x in parent}
    cfg=read(Path(run)/'cases/compat-restart/execution-protocol.json');model,_=load_model(run,cfg)
    from engine.aniso_phase1.research_sequential_next.model import PracticalModel
    fields=PracticalModel(model.reduction,order=7,device='cpu')
    values,vectors=la.eigh(model.rest_K[np.ix_(model.ids,model.ids)],model.M3ff,subset_by_index=[0,4])
    mode=vectors[:,4];summary=[]
    write(Path(run)/'S1/modal-definition.json',dict(index=4,eigenvalue=float(values[4]),period_s=float(2*np.pi/np.sqrt(values[4])),scope='rest diagnostic only'))
    del model
    for i,(start,end) in enumerate([(1.,1.2),(1.2,1.4)]):
        histories=[]
        for dt,tag in [(.0125,'0125'),(.00625,'00625')]:
            name=f'phase-window{i}-{tag}'
            create_config(run,name,dt=dt,start=start,end=end,initial=index[start]['folder']/'state.json',display_frames=3)
            run_case(run,name,quiet=True);histories.append(history(Path(run)/'cases'/name))
        coarse=compare(parent,histories[1],fields,mode,start,end)
        medium=compare(histories[0],histories[1],fields,mode,start,end)
        improvement=1-medium['modal_velocity_error_rms']/max(coarse['modal_velocity_error_rms'],1e-30)
        item=dict(window=[start,end],coarse_vs_fine=coarse,medium_vs_fine=medium,modal_velocity_error_reduction=improvement)
        write(Path(run)/f'S1/window{i}.json',item)
        summary.append(dict(window=[start,end],modal_velocity_error_reduction=improvement,
                            medium_engineering_passed=medium['all_engineering_fields_passed'],
                            coarse_event_error=coarse['modal_events']['max_abs_offset_s'],
                            medium_event_error=medium['modal_events']['max_abs_offset_s']))
        print('WINDOW',summary[-1],flush=True)
    selected='full-q7-dt0125'
    write(Path(run)/'S1/time-decision.json',dict(status='phase_improved_scoped',selected_case=selected,
        dt=.0125,steps=128,windows=summary,phase_improved=all(x['modal_velocity_error_reduction']>=.2 for x in summary),
        reason='uniform fine step avoids inheriting unqualified coarse loading phase; only this full candidate will run',
        temporal_certified=False,scene_stability='pending full cycle',no_damping=True))
    create_config(run,selected,dt=.0125)


def full_report(run):
    run=Path(run);decision=read(run/'S1/time-decision.json');name=decision['selected_case']
    new=history(run/'cases'/name);old=history(PARENT/'cases/seg-q7-dt0025')
    model,_=load_model(run,read(run/'cases'/name/'execution-protocol.json'))
    from engine.aniso_phase1.research_sequential_next.model import PracticalModel
    field_model=PracticalModel(model.reduction,order=7,device='cpu')
    _,vectors=la.eigh(model.rest_K[np.ix_(model.ids,model.ids)],model.M3ff,subset_by_index=[0,4])
    result=compare(old,new,field_model,vectors[:,4],0.,1.6)
    write(run/'S1/full-cycle-comparison.json',result)
    decision.update(scene_stability='passed_scoped',full_cycle_summary=read(run/'cases'/name/'summary.json'),
                    full_old_new_engineering_passed=result['all_engineering_fields_passed'],
                    temporal_certified=False,scope='local phase improvement and stable cycle; no global convergence certificate')
    write(run/'S1/time-decision.json',decision)
    print('TIME_DECISION', {k:v for k,v in decision.items() if k!='full_cycle_summary'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','restart-check','windows','full-report']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'prepare':prepare,'restart-check':check_restart,'windows':windows,'full-report':full_report}[a.phase](a.run)
