"""Independent short references with shared initial history and raw event brackets."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import APP,SPACE_PARENT,read,write,sha,register,verify,serial_lock
from .run import create_config,load_model
from .base_config import time_grid
from benchmarks.research_phase_stress_next.time_study import history,leaf_failures
from benchmarks.research_basis_allocation_next.time_study import events,difference
from benchmarks.research_reference_next.time_study import compare
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.endpoint_boundary import prescribed_speed
from engine.aniso_phase1.carrier_driven import displacement
WINDOWS=((1.05,1.1),(1.2,1.25))
LEVELS=(('base',.0125),('medium',.003125),('fine',.0015625),('half',.00078125))

def prefix(run):
    run=Path(run);verify(run);choice=read(run/'selected-space.json');old=read(APP/'selected-space.json')
    same=choice['package']['sha256']==old['package']['sha256']
    source=APP/'cases/final-q7-dt0125' if same else run/'cases/selected-prefix'
    if not same and not source.exists():create_config(run,'selected-prefix',end=1.2,field_cache=True,display_frames=3)
    write(run/'S2/prefix-source.json',dict(path=str(source),new_prefix_steps=0 if same else 96,physical_space_unchanged=same))
    return source

def prepare(run):
    run=Path(run);src=prefix(run);h=history(src)
    if h[-1]['state'].time<1.2-1e-12:raise ValueError('complete one shared prefix through 1.2 before preparing windows')
    if not (run/'cases/phase-model').exists():create_config(run,'phase-model',end=.025,field_cache=True)
    m,_=load_model(run,read(run/'cases/phase-model/execution-protocol.json'));m.validate(h[-1]['state'],material=True)
    K=m.rest_K[np.ix_(m.ids,m.ids)];M=m.M3ff;lam,V=la.eigh(K,M)
    orth=float(la.norm(V.T@M@V-np.eye(len(lam))));res=float(la.norm(K@V-(M@V)*lam)/la.norm(K@V))
    if orth>1e-6 or res>1e-7 or lam[0]<=0:raise ValueError('invalid current modal basis')
    q=np.array([(x['state'].q-m.boundary.unit*displacement(x['state'].time))[m.free].ravel() for x in h]);v=np.array([(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel() for x in h])
    a=q@M@V;ad=v@M@V;en=.5*(ad**2+a**2*lam);closure=float(np.max(abs(np.sum(ad**2,axis=1)/2-.5*np.einsum('ni,ij,nj->n',v,M,v))))
    force=(m.rest_K@m.boundary.unit.ravel()).reshape(m.boundary.unit.shape);forced=np.abs(V.T@force[m.free].ravel())
    ids=sorted(set([4,int(np.argmax(np.max(en,axis=0))),int(np.argmax(forced))]));cache=CachedProbes(m);lookup={round(x['state'].time,10):x for x in h};state=lookup[1.1]['state'];records=[]
    for j in ids:
        d=np.zeros_like(state.q);d[m.free]=V[:,j].reshape(-1,3);eps=1e-7/max(1.,la.norm(d));plus=state.clone();minus=state.clone();plus.q+=eps*d;minus.q-=eps*d
        deriv=(cache.frame(plus)['PK1']-cache.frame(minus)['PK1'])/(2*eps);rms=float(np.sqrt(np.mean(deriv**2)));amplitude=float(np.max(abs(a[:,j])))
        records.append(dict(mode=j,period_s=float(2*np.pi/np.sqrt(lam[j])),energy_peak_J=float(np.max(en[:,j])),observed_coordinate_peak=amplitude,observed_velocity_peak=float(np.max(abs(ad[:,j]))),PK1_per_modal_coordinate_Pa=rms,estimated_output_peak_Pa=rms*amplitude,important=bool(j==int(np.argmax(np.max(en,axis=0))) or rms*amplitude>.002)))
    folder=run/'S2';folder.mkdir(exist_ok=True);np.savez_compressed(folder/'modal-basis.npz',values=lam,vectors=V)
    write(folder/'modal-observable-map.json',dict(status='recomputed_for_selected_space',space=m.reduction.signature,selected_modes=records,mass_orthogonality=orth,eigen_residual=res,kinetic_closure_J=closure,source_identity_sha256=sha(src/'identity.json'),new_modal_numbers=True))
    register(run,'S2/time-protocol.json',dict(windows=WINDOWS,levels=LEVELS,max_fine_steps=[32,64],event_budget_s=.00625,reference_event_budget_s=.003125,reference_field_budget_fraction=.25,
        no_alignment=True,no_smoothing=True,fields=['displacement','velocity','PK1','fiber_PK1','interval impulse'],single_new_schedule_limit=256))
    initials=[]
    for i,(start,end) in enumerate(WINDOWS):
        # All levels deliberately use exactly this committed prefix, not different coarse histories.
        initial=lookup[round(start,10)]['folder']/'state.json'
        for label,dt in LEVELS:create_config(run,f'time-{i}-{label}',dt=dt,start=start,end=end,initial=initial,display_frames=3,field_cache=True)
        initials.append(dict(window=[start,end],state_sha256=sha(initial),path=str(initial),source_identity_sha256=sha(src/'identity.json')))
    write(folder/'window-initials.json',dict(status='authenticated_common_initials',records=initials));print('TIME_READY',records,flush=True)

def signal(h,m,vector):
    return events([x['state'].time for x in h],[float(vector@m.M3ff@(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel()) for x in h])

def final(run):
    run=Path(run);m,_=load_model(run,read(run/'cases/phase-model/execution-protocol.json'));mapping=read(run/'S2/modal-observable-map.json')
    with np.load(run/'S2/modal-basis.npz') as z:V=z['vectors'].copy()
    important=[x['mode'] for x in mapping['selected_modes'] if x['important']];selected=[x['mode'] for x in mapping['selected_modes']];records=[];eligible=True
    for i,(start,end) in enumerate(WINDOWS):
        hs={k:history(run/'cases'/f'time-{i}-{k}') for k,_ in LEVELS}
        for hh in hs.values():
            for key in ('q','velocity','predictor'):
                if not np.array_equal(getattr(hh[0]['state'],key),getattr(hs['half'][0]['state'],key)):raise ValueError('initial history mismatch')
        modes=[]
        for j in selected:
            ev={key:signal(h,m,V[:,j]) for key,h in hs.items()}
            modes.append(dict(mode=j,important=j in important,reference=difference(ev['fine'],ev['half'],budget=.003125),base=difference(ev['base'],ev['half']),medium=difference(ev['medium'],ev['half']),raw=ev))
        reference=compare(hs['fine'],hs['half'],m,V[:,important[0]],start,end);baseline=compare(hs['base'],hs['half'],m,V[:,important[0]],start,end);medium=compare(hs['medium'],hs['half'],m,V[:,important[0]],start,end)
        all_metrics=[metric for row in reference['field_records'] for reg in row['regions'].values() for metric in reg.values()]+reference['reaction_intervals']
        max_fraction=max(x['absolute']/x['budget'] for x in all_metrics)
        reference_events=all(x['reference']['status']=='passed_scoped' for x in modes if x['important']);medium_events=all(x['medium']['status']=='passed_scoped' for x in modes if x['important'])
        gain=1-medium['modal_velocity_error_rms']/max(baseline['modal_velocity_error_rms'],1e-30)
        resolved=max_fraction<=.25 and reference_events
        passed=resolved and medium['all_engineering_fields_passed'] and medium_events and gain>=.2
        eligible &=passed
        records.append(dict(window=[start,end],reference_max_budget_fraction=max_fraction,reference_events_resolved=reference_events,reference_resolved=resolved,reference=reference,baseline=baseline,medium=medium,modes=modes,modal_improvement=gain,eligible=passed))
        print('TIME_WINDOW',i,'ref',resolved,max_fraction,'gain',gain,'fields',medium['all_engineering_fields_passed'],'events',medium_events,flush=True)
    write(run/'S2/reference-resolution.json',dict(status='passed_short_windows' if all(x['reference_resolved'] for x in records) else 'reference_limited',records=records,full_window_certified=False,full_cycle_certified=False))
    write(run/'S2/raw-events.json',dict(records=[dict(window=x['window'],modes=x['modes']) for x in records],no_curve_alignment=True))
    write(run/'S2/output-error-map.json',dict(records=[dict(window=x['window'],baseline_failures=leaf_failures(x['baseline']),medium_failures=leaf_failures(x['medium'])) for x in records]))
    # Only one proposed schedule, defined by the two independently tested windows.
    points=time_grid(.0125)
    if eligible:
        points=sorted(set(points+[round(a+k*.003125,12) for a,b in WINDOWS for k in range(int(round((b-a)/.003125))+1)]))
    write(run/'S2/candidate-times.json',dict(status='selected_for_final_scene' if eligible else 'condition_not_triggered',new_candidates=int(eligible),times=points,reason='two independent fine references plus raw events and fields' if eligible else 'independent reference/event or physical-field gate unresolved; do not repeat rejected 204-step schedule'))
    write(run/'S2/window-comparison.json',dict(status='passed_scoped' if eligible else 'retain_baseline',records=[{k:v for k,v in x.items() if k not in ('reference','baseline','medium','modes')} for x in records]))
    write(run/'S2/time-decision.json',dict(status='scoped_segmented' if eligible else 'retain_dt_reference_limits',dt_s=.0125,times=points,steps=len(points)-1,temporal_accuracy=False,short_window_evidence_only=True))
    write(run/'S2/model-selection.json',dict(space=read(run/'selected-space.json'),time=read(run/'S2/time-decision.json'),material='full until final source certificate'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prefix','prepare','final']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'prefix':prefix,'prepare':prepare,'final':final}[a.phase](a.run)
