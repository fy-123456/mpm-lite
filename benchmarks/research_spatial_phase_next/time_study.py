"""Rest modal bands, unmodified excitation and bounded same-state windows."""
from pathlib import Path
import argparse,copy
import numpy as np
import scipy.linalg as la
from .provenance import APP,read,write,sha,digest,register,verify,serial_lock
from .run import create_config,run_case,load_model
from .base_config import time_grid
from benchmarks.research_post_release.time_study import history
from benchmarks.research_reference_next.time_study import compare
from engine.aniso_phase1.carrier_driven import displacement
from engine.aniso_phase1.endpoint_boundary import prescribed_speed


def bands(period):
    return {'period_gt_0.1s':period>.1,'period_0.025_to_0.1s':(period>.025)&(period<=.1),'period_le_0.025s':period<=.025}


def study(run):
    run=Path(run);verify(run);selected=read(run/'selected-space.json')
    register(run,'N2/protocol.json',dict(space=selected,windows=[[1.,1.2],[1.2,1.4]],dt=[.0125,.00625,.003125],
        joint_gain=.2,event_budget_s=.00625,latest_sufficient=str(APP/'cases/final-q7-dt0125'),no_alignment_or_damping=True))
    if selected['selected']=='original144':
        parent=history(APP/'cases/final-q7-dt0125')
    else:
        create_config(run,'common-full-prefix',dt=.0125,end=1.2,field_cache=True,display_frames=4)
        run_case(run,'common-full-prefix');parent=history(run/'cases/common-full-prefix')
    index={round(x['state'].time,10):x for x in parent}
    cfg=create_config(run,'phase-model',end=.025,field_cache=True);model,_=load_model(run,cfg)
    K=model.rest_K[np.ix_(model.ids,model.ids)];M=model.M3ff
    values,V=la.eigh(K,M);period=2*np.pi/np.sqrt(values);groups=bands(period)
    orth=float(la.norm(V.T@M@V-np.eye(len(values))))
    residual=float(la.norm(K@V-(M@V)*values)/max(la.norm(K@V),1e-30))
    if orth>1e-6 or residual>1e-7:raise ValueError('modal identity failed')
    np.savez_compressed(run/'N2/modal-basis.npz',values=values,vectors=V,period_s=period)
    write(run/'N2/modal-definition.json',dict(space=model.reduction.signature,mass_sha256=digest(model.M.tolist()),stiffness_sha256=digest(K.tolist()),
        periods_s=period.tolist(),band_counts={k:int(v.sum()) for k,v in groups.items()},mass_orthogonality=orth,eigen_residual=residual,representative_index=4,
        scope='rest linear subspaces; no nonlinear energy claim'))
    def accel(t):
        if 0<t<.5:return .01*np.pi**2*np.cos(2*np.pi*t)
        if .6<t<1.1:return -.01*np.pi**2*np.cos(2*np.pi*(t-.6))
        return 0.
    checks=[]
    for t in [.1,.3,.55,.7,.9,1.2]:
        eps=1e-6;v=(displacement(t+eps)-displacement(t-eps))/(2*eps);a=(prescribed_speed(t+eps)-prescribed_speed(t-eps))/(2*eps)
        checks.append(dict(time=t,velocity_error=abs(v-prescribed_speed(t)),acceleration_error=abs(a-accel(t))))
    if max(max(x['velocity_error'],x['acceleration_error']) for x in checks)>1e-8:raise ValueError('excitation derivative mismatch')
    forcing=[];unit=model.boundary.unit
    for t in [.1,.499999,.500001,.599999,.600001,.8,1.099999,1.100001]:
        fc=-model.M@unit*accel(t)-(model.rest_K@unit.ravel()).reshape(unit.shape)*displacement(t)
        amplitudes=V.T@fc[model.free].ravel()
        forcing.append(dict(time=t,acceleration_m_s2=accel(t),bands={k:float(la.norm(amplitudes[b])) for k,b in groups.items()}))
    breaks=[dict(time=t,left_acceleration=accel(t-1e-8),right_acceleration=accel(t+1e-8),displacement=displacement(t),speed=prescribed_speed(t)) for t in [.5,.6,1.1]]
    raw=[x for x in parent[-1]['rows'] if min(abs(x['time']-t) for t in [.5,.6,1.1])<=.025+1e-10]
    write(run/'N2/excitation-diagnostic.json',dict(status='diagnostic',derivative_checks=checks,breaks=breaks,linear_modal_forcing=forcing,raw_transition_rows=raw,
        observation='displacement and speed continuous; acceleration has finite jumps at stage boundaries',
        inference='actual excitation contains faster modes; midpoint dispersion may shift phase. This does not alone locate all nonlinear velocity error',
        unchanged_loading=True,no_damping=True))
    reports=[];reuse=[]
    for i,(start,end) in enumerate([(1.,1.2),(1.2,1.4)]):
        hist=[]
        for dt,label in [(.0125,'0125'),(.00625,'00625'),(.003125,'003125')]:
            old=APP/'cases'/f'phase-{i}-{label}'
            if selected['selected']=='original144':
                inherited=history(old);a,b=inherited[0]['state'],index[start]['state']
                if any(not np.array_equal(getattr(a,k),getattr(b,k)) for k in ('q','velocity','predictor')):raise ValueError('old window does not share latest sufficient initial state')
                if a.child_states!=b.child_states:raise ValueError('window history changed')
                if a.child_states['identity']!=model.identity:raise ValueError('window model mismatch')
                hist.append(inherited);reuse.append(dict(path=str(old),identity_sha256=sha(old/'identity.json'),initial_sha256=sha(index[start]['folder']/'state.json'),same_q_v_history=True))
            else:
                name=f'phase-{i}-{label}';create_config(run,name,dt=dt,start=start,end=end,initial=index[start]['folder']/'state.json',display_frames=3,field_cache=True)
                run_case(run,name,quiet=True);hist.append(history(run/'cases'/name))
        coarse=compare(hist[0],hist[2],model,V[:,4],start,end);medium=compare(hist[1],hist[2],model,V[:,4],start,end)
        def band_errors(a,b):
            lookup={round(h['state'].time,10):h['state'] for h in b};sums={k:[] for k in groups};balance=[]
            for h in a:
                v=(h['state'].velocity-lookup[round(h['state'].time,10)].velocity)[model.free].ravel();coeff=V.T@M@v
                balance.append(abs(float(coeff@coeff-v@M@v)))
                for k,mask in groups.items():sums[k].append(float(coeff[mask]@coeff[mask]))
            return dict(rms={k:float(np.sqrt(np.mean(x))) for k,x in sums.items()},mass_norm_closure=max(balance))
        ca,me=band_errors(hist[0],hist[2]),band_errors(hist[1],hist[2]);gains={k:1-me['rms'][k]/max(ca['rms'][k],1e-30) for k in groups}
        event=medium['modal_events'];phase=gains['period_0.025_to_0.1s']>=.2 and gains['period_gt_0.1s']>=-.05 and event['same_event_counts'] and event['max_abs_offset_s'] is not None and event['max_abs_offset_s']<=.00625+1e-12
        accepted=phase and medium['all_engineering_fields_passed']
        rec=dict(window=[start,end],coarse_vs_fine=coarse,medium_vs_fine=medium,coarse_bands=ca,medium_bands=me,band_gain=gains,phase_passed=phase,accepted=accepted)
        write(run/f'N2/window{i}.json',rec);reports.append(rec);print('WINDOW',i,gains,accepted,flush=True)
    dt=.00625 if all(r['accepted'] for r in reports) else .0125
    write(run/'N2/reuse-audit.json',dict(verified_inherited_windows=reuse,ancestor_source_chain_verified=True,rewritten_case_identity=False))
    write(run/'N2/time-decision.json',dict(status='phase_improved_scoped' if dt==.00625 else 'retain_dt_joint_gate_not_met',dt_s=dt,times=time_grid(dt),steps=len(time_grid(dt))-1,
        temporal_certified=False,full_cycle='N6 only',windows=[{k:v for k,v in r.items() if k not in ('coarse_vs_fine','medium_vs_fine')} for r in reports]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release')
        study(a.run)
