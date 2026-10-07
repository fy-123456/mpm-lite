"""Raw output events with sampling intervals; no curve alignment or smoothing."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import APP,SPACE_PARENT,read,write,sha,register,verify,serial_lock
from .run import create_config,load_model
from benchmarks.research_phase_stress_next.time_study import history,leaf_failures


def events(times,values,absolute_floor=1e-12):
    t=np.asarray(times,float);y=np.asarray(values,float)
    if t.ndim!=1 or y.shape!=t.shape or len(t)<3 or not np.isfinite(y).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('finite ordered signal required')
    floor=max(absolute_floor,float(np.max(abs(y)))*1e-8)
    if np.ptp(y)<=2*floor:return dict(resolved=False,reason='near constant or below floor',events=[],raw_times=t.tolist(),raw_values=y.tolist())
    out=[]
    for i in range(1,len(y)-1):
        if y[i]-y[i-1]>floor and y[i]-y[i+1]>floor:kind='maximum'
        elif y[i-1]-y[i]>floor and y[i+1]-y[i]>floor:kind='minimum'
        else:continue
        out.append(dict(kind=kind,time_s=float(t[i]),bracket_s=[float(t[i-1]),float(t[i+1])],value=float(y[i])))
    # A zero exactly on a node is a single event, never counted twice.
    for i in range(len(y)-1):
        if abs(y[i])<=floor and 0<i<len(y)-1 and y[i-1]*y[i+1]<0:
            out.append(dict(kind='up_zero' if y[i+1]>y[i-1] else 'down_zero',time_s=float(t[i]),bracket_s=[float(t[i-1]),float(t[i+1])]))
        elif y[i]*y[i+1]<0 and abs(y[i])>floor and abs(y[i+1])>floor:
            z=t[i]-y[i]*(t[i+1]-t[i])/(y[i+1]-y[i])
            out.append(dict(kind='up_zero' if y[i+1]>y[i] else 'down_zero',time_s=float(z),bracket_s=[float(t[i]),float(t[i+1])]))
    return dict(resolved=bool(out),events=sorted(out,key=lambda x:x['time_s']),raw_times=t.tolist(),raw_values=y.tolist(),floor=floor)


def difference(a,b,budget=.00625):
    pairs=[];counts=True;ambiguous=False
    for kind in ('maximum','minimum','up_zero','down_zero'):
        aa=[v for v in a['events'] if v['kind']==kind];bb=[v for v in b['events'] if v['kind']==kind];counts &=len(aa)==len(bb)
        for x,y in zip(aa,bb):
            lo=x['bracket_s'][0]-y['bracket_s'][1];hi=x['bracket_s'][1]-y['bracket_s'][0]
            possible=[v for v in bb if x['bracket_s'][0]-v['bracket_s'][1]<=budget and x['bracket_s'][1]-v['bracket_s'][0]>=-budget]
            amb=len(possible)>1;ambiguous |=amb
            pairs.append(dict(kind=kind,offset_estimate_s=x['time_s']-y['time_s'],offset_interval_s=[lo,hi],
                sampling_width_s=[np.ptp(x['bracket_s']).item(),np.ptp(y['bracket_s']).item()],ambiguous=amb,
                resolved_within_budget=bool(lo>=-budget and hi<=budget and not amb)))
    resolved=bool(a['resolved'] and b['resolved'] and counts and pairs and not ambiguous)
    return dict(status='passed_scoped' if resolved and all(x['resolved_within_budget'] for x in pairs) else 'unresolved_or_shifted',
        same_event_counts=bool(counts),unambiguous=not ambiguous,pairs=pairs,budget_s=budget,phase_alignment_applied=False)


def study(run):
    run=Path(run);verify(run)
    register(run,'S2/event-protocol.json',dict(windows=[[1.,1.2],[1.2,1.4]],signals=['legacy mode4','observed energy mode','stress sensitive mode','interval raw reaction'],
        peak_definition='strict raw local extrema bracketed by neighbors',zero_definition='signed crossing including exact zero nodes',
        match='same type order with count and competing interval ambiguity checks',budget_s=.00625,near_zero_abs=1e-12,no_smoothing=True,no_alignment=True))
    create_config(run,'phase-diagnostic',end=.025,field_cache=True);m,_=load_model(run,read(run/'cases/phase-diagnostic/execution-protocol.json'))
    modal=read(APP/'S1/observable-modal-contribution.json');selected=[x['mode'] for x in modal['selected_modes']]
    definition=read(SPACE_PARENT/'N2/modal-definition.json')
    with np.load(SPACE_PARENT/'N2/modal-basis.npz') as z:V=z['vectors'].copy();lam=z['values'].copy()
    if definition['space']!=m.reduction.signature:raise ValueError('modal space changed')
    K=m.rest_K[np.ix_(m.ids,m.ids)];M=m.M3ff
    orth=float(la.norm(V.T@M@V-np.eye(len(lam))));res=float(la.norm(K@V-(M@V)*lam)/la.norm(K@V))
    if orth>1e-6 or res>1e-7:raise ValueError('modal basis invalid')
    write(run/'S2/modal-observable-map.json',dict(status='inherited_verified',selected=modal['selected_modes'],mass_orthogonality=orth,eigen_residual=res,
        kinetic_closure_J=modal['kinetic_closure_J'],source_sha256=sha(APP/'S1/observable-modal-contribution.json'),space=m.reduction.signature,
        normalization='mass orthonormal; remove prescribed speed lift before projection',new_prefix_steps=0))
    controlled=[]
    for h in (.001,.002):
        t=np.arange(0,.10001,h);u=events(t,np.sin(2*np.pi*t/.04));v=events(t,np.sin(2*np.pi*(t-.001)/.04));d=difference(u,v)
        controlled.append(dict(dt=h,comparison=d))
    if events([0.,1.,2.],[1e-16,0.,-1e-16])['resolved']:raise AssertionError('near zero signal identified as resolved')
    records=[];initials=[];failures=[]
    from engine.aniso_phase1.endpoint_boundary import prescribed_speed
    for i,(start,end) in enumerate(((1.,1.2),(1.2,1.4))):
        source=SPACE_PARENT/'cases'/f'phase-{i}-003125';fine=history(source);baseline=history(SPACE_PARENT/'cases'/f'phase-{i}-0125');candidate=history(APP/'cases'/f'phase-candidate-{i}')
        for h in (baseline,candidate):
            for key in ('q','velocity','predictor'):
                if not np.array_equal(getattr(h[0]['state'],key),getattr(fine[0]['state'],key)):raise ValueError('window initial mismatch')
        initials.append(dict(window=[start,end],reference_state_sha256=sha(fine[0]['folder']/'state.json'),source_case_identity_sha256=sha(source/'identity.json')))
        signals=[]
        for j in selected:
            def series(hist):
                tt=[x['state'].time for x in hist];vv=[float(V[:,j]@M@(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel()) for x in hist]
                return events(tt,vv)
            a,b,c=series(baseline),series(fine),series(candidate)
            signals.append(dict(mode=j,period_s=float(2*np.pi/np.sqrt(lam[j])),baseline=difference(a,b),old_candidate=difference(c,b),raw=dict(baseline=a,reference=b,candidate=c)))
        def reaction(h):
            rows=[r for r in h[-1]['rows'] if start+1e-10<r['time']<=end+1e-10]
            return events([r['time']-r['dt']/2 for r in rows],[r['reaction_N'] for r in rows])
        aa,bb,cc=map(reaction,(baseline,fine,candidate))
        records.append(dict(window=[start,end],signals=signals,reaction=dict(baseline=difference(aa,bb),old_candidate=difference(cc,bb),raw=dict(baseline=aa,reference=bb,candidate=cc))))
        old=read(APP/'S1/window-comparison.json')['records'][i]
        failures.append(dict(window=[start,end],old_accepted=old['accepted'],failures=leaf_failures(old['comparison']),
            warning='event reinterpretation does not remove field/impulse failures'))
    write(run/'S2/event-diagnostic.json',dict(status='diagnostic',controlled=controlled,records=records))
    write(run/'S2/field-failure-map.json',dict(records=failures))
    write(run/'S2/window-initials.json',dict(status='verified',records=initials))
    write(run/'S2/reference-resolution.json',dict(status='inherited_local_evidence',source_sha256=sha(APP/'S1/reference-resolution.json'),
        original=read(APP/'S1/reference-resolution.json'),new_steps=0,full_window_certified=False))
    decision=read(APP/'S1/time-decision.json')
    write(run/'S2/candidate-times.json',dict(status='condition_not_triggered',new_candidates=0,
        prior_rejected_sha256=sha(APP/'S1/candidate-times.json'),reason='raw important-band events underresolved; old 204-step scheme still has field/impulse failures; no independent full-window reference to distinguish another <=256-step schedule'))
    write(run/'S2/window-comparison.json',dict(status='retained_original_field_failures',records=failures))
    write(run/'S2/time-decision.json',dict(decision,status='retain_dt_reference_and_event_limits',dt_s=.0125,temporal_accuracy=False,
        new_evidence='output-relevant raw event brackets and source checks',new_window_steps=0))
    write(run/'S2/model-selection.json',dict(space=read(run/'selected-space.json'),time=read(run/'S2/time-decision.json'),material='full until final current-source certificate'))
    print('TIME_RETAIN',selected,[len(x['failures']) for x in failures],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
