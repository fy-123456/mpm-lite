"""Observable-weighted phase diagnostics, with at most two short reference probes."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import APP,SPACE_PARENT,read,write,sha,register,serial_lock
from .run import create_config,load_model
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_reference_next.time_study import compare
from engine.aniso_phase1.research_post_release.fields import CachedProbes

def history(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()

def leaf_failures(value,path=''):
    if isinstance(value,dict):
        children=[]
        for k,v in value.items():
            if isinstance(v,(dict,list)):children+=leaf_failures(v,path+'/'+k)
        if children:return children
        if value.get('passed') is False:return [dict(path=path,**{k:v for k,v in value.items() if not isinstance(v,(dict,list))})]
    elif isinstance(value,list):
        return [v for i,x in enumerate(value) for v in leaf_failures(x,path+'/'+str(i))]
    return []

def diagnose(run):
    run=Path(run)
    register(run,'S1/protocol.json',dict(windows=[[1.,1.2],[1.2,1.4]],old_steps=[.0125,.00625,.003125],
        controlled_linear_modes='index4 plus strongest actual rest-linear diagnostic energy and lift forcing',
        reference_probes=[[1.,1.025],[1.2,1.225]],probe_dt=.0015625,probe_steps=16,
        no_damping=True,no_new_full_window_fourth_step=True,source=str(SPACE_PARENT)))
    cfg=create_config(run,'phase-model',end=.025,field_cache=True);m,_=load_model(run,cfg)
    modal=read(SPACE_PARENT/'N2/modal-definition.json')
    with np.load(SPACE_PARENT/'N2/modal-basis.npz') as z:V=z['vectors'].copy();lam=z['values'].copy()
    K=m.rest_K[np.ix_(m.ids,m.ids)];M=m.M3ff
    orth=float(la.norm(V.T@M@V-np.eye(len(lam))));res=float(la.norm(K@V-(M@V)*lam)/la.norm(K@V))
    assert orth<1e-6 and res<1e-7 and modal['space']==m.reduction.signature
    h=history(APP/'cases/final-q7-dt0125');lookup={round(x['state'].time,10):x for x in h}
    times=np.array([x['state'].time for x in h])
    from engine.aniso_phase1.carrier_driven import displacement
    from engine.aniso_phase1.endpoint_boundary import prescribed_speed
    q=np.array([(x['state'].q-m.boundary.unit*displacement(x['state'].time))[m.free].ravel() for x in h])
    v=np.array([(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel() for x in h])
    a=q@M@V;ad=v@M@V;energy=.5*(ad**2+a**2*lam)
    kinetic=.5*np.einsum('ni,ij,nj->n',v,M,v);closure=float(np.max(abs(.5*np.sum(ad**2,axis=1)-kinetic)))
    assert closure<1e-10
    unit=m.boundary.unit
    forcing=[]
    for t in (.1,.499999,.500001,.599999,.600001,.8,1.099999,1.100001):
        accel=.01*np.pi**2*np.cos(2*np.pi*t) if 0<t<.5 else (-.01*np.pi**2*np.cos(2*np.pi*(t-.6)) if .6<t<1.1 else 0.)
        force=-m.M@unit*accel-(m.rest_K@unit.ravel()).reshape(unit.shape)*displacement(t)
        forcing.append(V.T@force[m.free].ravel())
    score=np.max(energy,axis=0);forced=np.max(abs(np.asarray(forcing)),axis=0)
    ids=sorted(set([4,int(np.argmax(score)),int(np.argmax(forced))]))
    cache=CachedProbes(m);sens=[]
    item=lookup[1.1];eps=1e-7;fiber=np.asarray(m.parent.params.fiber_direction)
    for j in ids:
        d=np.zeros_like(item['state'].q);d[m.free]=V[:,j].reshape(-1,3);d/=la.norm(d)
        plus=item['state'].clone();minus=item['state'].clone();plus.q+=eps*d;minus.q-=eps*d
        pa,pb=cache.frame(plus)['PK1'],cache.frame(minus)['PK1'];dp=(pa-pb)/(2*eps)
        va,vb=m.evaluate(plus.q),m.evaluate(minus.q)
        sens.append(dict(mode=j,period_s=float(2*np.pi/np.sqrt(lam[j])),
            rest_linear_energy_peak_J=float(score[j]),lift_force_peak=float(forced[j]),
            PK1_direction_rms=float(np.sqrt(np.mean(dp**2))),
            fiber_direction_rms=float(np.sqrt(np.mean(np.einsum('i,...ij,j->...',fiber,dp,fiber)**2))),
            material_plus_Ks_reaction_derivative=float(np.sum((va['force']-vb['force'])*unit)/(2*eps)),
            scope='unit Euclidean coordinate direction, fixed boundary, display probes diagnostic only; not interval inertial reaction'))
    records=[];fail=[];initial=[]
    for i,(start,end) in enumerate(((1.,1.2),(1.2,1.4))):
        hs=[history(SPACE_PARENT/'cases'/f'phase-{i}-{suffix}') for suffix in ('0125','00625','003125')]
        for hh in hs:
            first=hh[0]['state']
            for key in ('q','velocity','predictor'):
                assert np.array_equal(getattr(first,key),getattr(lookup[start]['state'],key))
            initial.append(dict(case=str(hh[0]['folder'].parents[1]),initial_sha256=sha(hh[0]['folder']/'state.json')))
        old=read(SPACE_PARENT/f'N2/window{i}.json');leaves=leaf_failures(old)
        fail.append(dict(window=[start,end],unique_leaf_failures=leaves,count=len(leaves),old_accepted=old['accepted']))
        create_config(run,f'phase-probe-{i}',dt=.0015625,start=start,end=start+.025,
                      initial=hs[-1][0]['folder']/'state.json',display_frames=2,field_cache=True)
        records.append(dict(window=[start,end],source_result_sha256=sha(SPACE_PARENT/f'N2/window{i}.json')))
    write(run/'S1/reuse-audit.json',dict(status='passed_scoped',records=records,initials=initial,physical_model=m.identity))
    write(run/'S1/failure-map.json',dict(windows=fail,hypotheses=['midpoint phase dispersion of actually observed modes','fine comparator and event sampling underresolution','nonlinear rest-mode approximation not an exact decomposition']))
    write(run/'S1/observable-modal-contribution.json',dict(status='diagnostic',mass_orthogonality=orth,eigen_residual=res,kinetic_closure_J=closure,
        selected_modes=sens,index4_peak_energy_fraction=float(score[4]/score.sum()),top_energy_modes=np.argsort(score)[-10:][::-1].tolist(),
        diagnostic_energy_fractions=(score/score.sum()).tolist(),periods_s=(2*np.pi/np.sqrt(lam)).tolist(),
        scope='homogeneous q/v after prescribed lift; rest linear energy only; sensitivity not a continuum stress test'))
    checks=[]
    for j in ids:
        w=float(np.sqrt(lam[j]))
        for dt in (.0125,.00625,.003125,.0015625):
            A=np.array([[0.,w],[-w,0.]])
            G=la.solve(np.eye(2)-.5*dt*A,np.eye(2)+.5*dt*A)
            theta=2*np.arctan(w*dt/2);N=16
            numerical=np.linalg.matrix_power(G,N)@np.array([1.,0.])
            expected=np.array([np.cos(N*theta),-np.sin(N*theta)])
            exact=la.expm(A*N*dt)@np.array([1.,0.])
            error=float(la.norm(numerical-expected));assert error<1e-10
            checks.append(dict(mode=j,dt_s=dt,period_s=2*np.pi/w,omega_ratio=theta/(w*dt),
                phase_lag_radians=N*(w*dt-theta),formula_error=error,exact_state_error=float(la.norm(numerical-exact)),
                norm_error=float(abs(numerical@numerical-1))))
    write(run/'S1/controlled-phase-check.json',dict(status='passed_scoped',checks=checks,
        preserved_boundary_source_sha256=sha(SPACE_PARENT/'N2/excitation-diagnostic.json'),scope='isolated linear modes, not nonlinear convergence'))
    print('PHASE_DIAGNOSTIC',sens,flush=True)

def finish(run):
    run=Path(run);cfg=read(run/'cases/phase-model/execution-protocol.json');m,_=load_model(run,cfg)
    with np.load(SPACE_PARENT/'N2/modal-basis.npz') as z:mode=z['vectors'][:,4]
    reports=[]
    for i,start in enumerate((1.,1.2)):
        a=history(SPACE_PARENT/'cases'/f'phase-{i}-003125');b=history(run/'cases'/f'phase-probe-{i}')
        rec=compare(a,b,m,mode,start,start+.025)
        reports.append(dict(window=[start,start+.025],**rec));print('PHASE_REFERENCE',i,rec['all_engineering_fields_passed'],flush=True)
    write(run/'S1/reference-resolution.json',dict(status='diagnostic',probes=reports,
        full_window_reference_certified=False,reason='half-step only resolves these short windows; no whole-window convergence certificate'))
    current=read(APP/'P2/time-decision.json')
    write(run/'S1/candidate-times.json',dict(status='not_promoted',times=current['times'],
        proposed='whole-window .00625 repeats old joint failures; .003125 would equal the old comparator, not independent evidence'))
    write(run/'S1/window-comparison.json',dict(status='retained',source_results=[str(SPACE_PARENT/f'N2/window{i}.json') for i in (0,1)],reference_probes_passed=[r['all_engineering_fields_passed'] for r in reports],
        reason='new subwindow evidence does not certify either full-window comparator or a <=256-step independent improvement'))
    write(run/'S1/time-decision.json',dict(status='retain_dt_reference_and_joint_limits',dt_s=.0125,times=current['times'],temporal_accuracy=False,
        observations='observable contribution, controlled dispersion and two actual half-step subwindows added',
        decision='no damping, no changed boundary, no repeated failed schedule or fourth full-window step sweep'))

def schedule(start,end):
    points=[start]
    for lo,hi,h in [(0.,1.,.0125),(1.,1.05,.00625),(1.05,1.225,.003125),(1.225,1.6,.00625)]:
        left=max(start,lo);right=min(end,hi)
        if right>left+1e-12:
            n=int(round((right-left)/h));points.extend(np.linspace(left,right,n+1)[1:].tolist())
    return points

def candidate_prepare(run):
    run=Path(run)
    register(run,'S1/candidate-protocol.json',dict(reason='one new transition-neighborhood schedule after independent half-step probes; no fitted alignment',
        intervals=[[0.,1.,.0125],[1.,1.05,.00625],[1.05,1.225,.003125],[1.225,1.6,.00625]],
        engineering_rtol=.05,main_band_gain=.2,event_budget_s=.00625,independent_subwindow_evidence='S1/reference-resolution.json'))
    for name in ['time-decision.json','candidate-times.json','window-comparison.json']:
        (run/'S1'/('pre-candidate-'+name)).write_bytes((run/'S1'/name).read_bytes())
    for i,(start,end) in enumerate(((1.,1.2),(1.2,1.4))):
        h=history(SPACE_PARENT/'cases'/f'phase-{i}-003125')
        create_config(run,f'phase-candidate-{i}',times=schedule(start,end),start=start,end=end,
            initial=h[0]['folder']/'state.json',display_frames=3,field_cache=True)

def candidate_finish(run):
    run=Path(run);m,_=load_model(run,read(run/'cases/phase-model/execution-protocol.json'))
    with np.load(SPACE_PARENT/'N2/modal-basis.npz') as z:V=z['vectors'];period=z['period_s']
    main=(period>.025)&(period<=.1);records=[];accept=True
    for i,(start,end) in enumerate(((1.,1.2),(1.2,1.4))):
        candidate=history(run/'cases'/f'phase-candidate-{i}');fine=history(SPACE_PARENT/'cases'/f'phase-{i}-003125')
        result=compare(candidate,fine,m,V[:,4],start,end)
        def rms(h):
            lookup={round(x['state'].time,10):x['state'] for x in fine};errors=[]
            for x in h:
                state=x['state'];d=(state.velocity-lookup[round(state.time,10)].velocity)[m.free].ravel()
                errors.append(np.sum((V[:,main].T@m.M3ff@d)**2))
            return float(np.sqrt(np.mean(errors)))
        coarse=history(SPACE_PARENT/'cases'/f'phase-{i}-0125');gain=1-rms(candidate)/max(rms(coarse),1e-30)
        event=result['modal_events'];passed=result['all_engineering_fields_passed'] and gain>=.2 and event['same_event_counts'] and event['max_abs_offset_s'] is not None and event['max_abs_offset_s']<=.00625+1e-12
        records.append(dict(window=[start,end],main_band_gain=gain,accepted=passed,comparison=result));accept &= passed
        print('TIME_CANDIDATE',i,gain,result['all_engineering_fields_passed'],passed,flush=True)
    current=read(APP/'P2/time-decision.json');times=schedule(0.,1.6) if accept else current['times']
    write(run/'S1/candidate-times.json',dict(times=schedule(0.,1.6),steps=len(schedule(0.,1.6))-1,status='accepted_scoped' if accept else 'rejected'))
    write(run/'S1/window-comparison.json',dict(status='passed_scoped' if accept else 'joint_gates_not_met',records=records))
    write(run/'S1/time-decision.json',dict(status='segmented_scoped' if accept else 'retain_dt_joint_gate_not_met',dt_s=None if accept else .0125,
        times=times,steps=len(times)-1,temporal_accuracy=False,scope='two windows, half-step subwindow probes; no full temporal convergence',
        reason='one measured transition-neighborhood schedule; retain baseline if either joint window fails'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['diagnose','finish','candidate-prepare','candidate-finish']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'diagnose':diagnose,'finish':finish,'candidate-prepare':candidate_prepare,'candidate-finish':candidate_finish}[a.phase](a.run)

