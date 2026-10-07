"""One pre-registered observable fixture, full material, bounded short windows."""
from pathlib import Path
import argparse,time,os
import numpy as np
import scipy.linalg as la
from .provenance import *
from .physics import baseline_model
from engine.aniso_phase1.research_candidate_observable_next.fixture import ObservableCoupling
from engine.aniso_phase1.research_pressure_window_next.coupled import OwnedGeometry
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import balanced_solve
from benchmarks.research_phase_reference_next.coupling_study import quadrature
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish


def numerical_sources():
    return dict(source_files(),**{str(Path('engine/aniso_phase1/research_candidate_observable_next/fixture.py')):sha(ROOT/'engine/aniso_phase1/research_candidate_observable_next/fixture.py')})


def algebra(cuts,p):
    top=CartesianTopology(cuts);X,w,ids=quadrature(top);H,_=top.assemble(X,w,ids,np.broadcast_to(np.eye(3),(len(X),3,3)),p['mobility_scale']*MOBILITY)
    C=p['storage']*top.V0;Z=la.solve(H,top.B.T,assume_a='pos');z0=-la.solve(H,top.boundary_term(p['reservoir_Pa']),assume_a='pos');L=top.B@Z;src=top.source(p['source_density_s_inv']);rhs=src-top.B@z0;N=top.cells
    aug=np.zeros((N+2,N+2));aug[:N,:N]=-L/C[:,None];aug[:N,-1]=rhs/C;aug[N,:N]=np.sum(top.B@Z,axis=0);aug[N,-1]=np.sum(top.B@z0)
    return dict(top=top,H=H,C=C,Z=Z,z0=z0,L=L,rhs=rhs,src=src,aug=aug,rate=float(la.eigvalsh(L/np.sqrt(C[:,None]*C[None,:]))[-1]))


def pressure_check(a,p,times):
    N=a['top'].cells;initial=np.r_[np.full(N,p['pressure0_Pa']),0.,1.];exact=np.array([la.expm(a['aug']*t)@initial for t in times]);current=initial[:N].copy();Q=0.;rows=[]
    for i,h in enumerate(np.diff(times),1):
        before=current.copy();C=np.diag(a['C']);current=la.solve(C+.5*h*a['L'],(C-.5*h*a['L'])@current+h*a['rhs'],assume_a='pos');z=a['Z']@(.5*(before+current))+a['z0'];Q+=h*np.sum(a['top'].B@z)
        pe=metric(current,exact[i,:N],.001,.05,a['top'].V0);qe=metric(Q,exact[i,N],1e-10,.05);content=metric(float(a['C']@current),float(a['C']@exact[i,:N]),1e-10,.05)
        fraction=max(x['absolute']/(.25*x['budget']) for x in (pe,qe,content));rows.append(dict(time_s=float(times[i]),pressure=pe,boundary=qe,content=content,time_fraction=fraction,minimum_pressure_Pa=float(current.min()),passed=bool(fraction<=1 and current.min()>=0)))
    return dict(status='passed_scoped' if all(x['passed'] for x in rows) else 'limited',rows=rows,max_time_fraction=max(x['time_fraction'] for x in rows),max_rate_s_inv=a['rate'])


def screen(run):
    run=Path(run);verify(run);m,cfg=baseline_model(run,pressure=True);cuts=read(APP/'S2/coupled-protocol.json')['cuts'];g=OwnedGeometry(m,cuts['coarse']);q=m.rest().q;actual=g.evaluate(q);cache=CachedProbes(m);lam,V=la.eigh(m.rest_K[np.ix_(m.ids,m.ids)],m.M3ff);records=[];chosen=None
    proposals=[dict(pressure0_Pa=.02,window_s=.0002,mobility_scale=1e-7),dict(pressure0_Pa=.2,window_s=.0002,mobility_scale=1e-7)]
    for proposal in proposals:
        p=dict(alpha=.8,storage=.0002,reservoir_Pa=.002,source_density_s_inv=0.,**{k:v for k,v in proposal.items() if k!='window_s'});times=np.linspace(0,proposal['window_s'],17);obs=times[[0,4,8,12,16]];force=p['alpha']*p['pressure0_Pa']*actual['gradient'].sum(axis=0);a=V.T@force[m.free].ravel();outputs=[];max_ratio=0.
        # Constant-pressure linear prediction is screening only, not dynamic evidence.
        for t in obs[1:]:
            velocity=np.zeros_like(q);displacement=np.zeros_like(q);cv=np.sin(np.sqrt(lam)*t)/np.sqrt(lam)*a;cq=(1-np.cos(np.sqrt(lam)*t))/lam*a;velocity[m.free]=(V@cv).reshape(-1,3);displacement[m.free]=(V@cq).reshape(-1,3)
            outputs.append(dict(time_s=float(t),u_rms_m=float(la.norm(cache.maps[0]@displacement)/np.sqrt(cache.maps[0].shape[0])),v_rms_m_s=float(la.norm(cache.maps[0]@velocity)/np.sqrt(cache.maps[0].shape[0]))))
        fixed={label:pressure_check(algebra(c,p),p,times) for label,c in cuts.items()};eligible=sum(x['u_rms_m']>=1.5e-4 or x['v_rms_m_s']>=3e-4 for x in outputs)>=2 and all(x['status']=='passed_scoped' for x in fixed.values())
        # Quantify midpoint phase for the same rest-linear forcing, at all fixed nodes.
        modal_phase=2*np.arctan(.5*np.sqrt(lam)*(times[1]-times[0]));errors=[]
        for i,t in zip((4,8,12,16),obs[1:]):
            exactv=np.sin(np.sqrt(lam)*t)/np.sqrt(lam)*a;discretev=np.sin(i*modal_phase)/np.sqrt(lam)*a;dv=np.zeros_like(q);vv=np.zeros_like(q);dv[m.free]=(V@(discretev-exactv)).reshape(-1,3);vv[m.free]=(V@exactv).reshape(-1,3);error=float(la.norm(cache.maps[0]@dv)/np.sqrt(cache.maps[0].shape[0]));scale=float(la.norm(cache.maps[0]@vv)/np.sqrt(cache.maps[0].shape[0]));errors.append(dict(time_s=float(t),velocity_rms_error=error,budget=1e-4+.05*scale,passed=error<=1e-4+.05*scale))
        eligible &=all(x['passed'] for x in errors)
        rec=dict(parameters=p,window_s=proposal['window_s'],predicted_observations=outputs,pressure_time=fixed,linear_phase_screen=errors,eligible=bool(eligible),pressure_force_max_N=float(np.max(abs(force[m.free]))),static_linear_displacement_coefficient_norm=float(la.norm(la.solve(m.rest_K[np.ix_(m.ids,m.ids)],force[m.free].ravel(),assume_a='pos'))))
        records.append(rec)
        if eligible:chosen=(p,times,obs);break
    write(run/'S3/physical-scale-screening.json',dict(status='passed_scoped' if chosen else 'limited',proposals=records,actual_solid_mass_material_used=True,initial_solid_rest=True,dynamic_steps=0,linear_prediction_not_validation=True))
    if chosen is None:
        write(run/'S3/coupling-scope-decision.json',dict(status='limited',reason='no registered observable and time-resolved fixture within two static proposals',dynamic_steps=0,coupled_q5=False,production_C_E_integration=False));return
    p,times,obs=chosen
    register(run,'S3/new-scene-protocol.json',dict(schema='observable-fixture-protocol-v1',parameters=p,times_s=times.tolist(),observations_s=obs.tolist(),cuts=cuts,coarse_steps=16,fine_steps=32,max_total_attempts=96,mechanical_u_rms_threshold_m=1.5e-4,mechanical_v_rms_threshold_m_s=3e-4,mechanical_min_nonadjacent_observations=2,full_mass=True,full_material_order=7,zero_grips=True,engineering_abs_plus_relative={'u':5e-5,'v':1e-4,'PK1':.02,'reaction':1e-4,'pressure':.001,'flux':1e-10,'content':1e-10,'rtol':.05},energy_scale='initial total energy plus cumulative absolute physical work; minimum 1e-8J; relative budget1%',research_fixture_not_calibrated_material=True))
    write(run/'S3/pressure-schedule-check.json',dict(status='passed_scoped',records=records[-1]['pressure_time'],pressure_error_fraction=.25))
    print('OBSERVABLE_FIXTURE',p,records[-1]['predicted_observations'],flush=True)


def setup(run,grid='coarse',fine=False,state=None):
    p=read(Path(run)/'S3/new-scene-protocol.json');m,cfg=baseline_model(run,pressure=True);return ObservableCoupling(m,cfg,p,grid,fine=fine,state=state),m,cfg


def operators(run):
    run=Path(run);verify(run);c,m,cfg=setup(run);g=c.geometry;q=m.rest().q;v=g.evaluate(q);rng=np.random.default_rng(41);d=rng.normal(size=q.shape);d[m.fixed]=0;d/=la.norm(d);eps=1e-5;v1=g.evaluate(q+eps*d);vm=g.evaluate(q-eps*d);analytic=np.einsum('cij,ij->c',v['gradient'],d);err=float(la.norm((v1['volume']-vm['volume'])/(2*eps)-analytic));dv=v1['volume']-v['volume'];work=np.einsum('cij,ij->c',g.discrete(q,q+eps*d),eps*d);A=c.rest_matrix(c.times[1]);known=rng.normal(size=len(A))*1e-7;answer,diag=balanced_solve(A,A@known,diagnose=True)
    passed=err<1e-9+.001*la.norm(analytic) and np.max(abs(dv-work))<1e-11 and np.linalg.norm(A@answer-A@known)<1e-8
    if not passed:raise ValueError('manufactured coupled operator gate failed')
    write(run/'S3/operator-check.json',dict(status='passed_scoped',gradient_error=float(err),discrete_work_error=float(np.max(abs(dv-work))),H_change_norm=float(la.norm(v1['H']-v['H'])),nonzero_volume_change=float(np.max(abs(dv))),manufactured_nonzero=True,general_LU=diag))
    p=read(run/'S3/new-scene-protocol.json')['parameters'];f=c.state.child_states['fluid']
    if not np.all(np.array(f['pressure_Pa'])==p['pressure0_Pa']) or not np.array_equal(c.geometry.mobility,p['mobility_scale']*MOBILITY):raise ValueError('fixture parameters not consumed')
    write(run/'S3/initial-state-check.json',dict(status='passed_scoped',pressure0_Pa=f['pressure_Pa'],source_m3_s=c.source.tolist(),mobility=c.geometry.mobility.tolist(),initial_state_digest=c.state.digest(),config_consumed=True))
    write(run/'S3/model-parameter-binding.json',dict(status='passed_scoped',coupling=c.identity,numerical_source_sha256=numerical_sources(),driver_sha256=sha(__file__)))
    write(run/'S3/quasi-newton-scope.json',dict(status='limited',matrix='original rest K quasi-Newton with current G/H and original residual verification',exact_nonlinear_jacobian=False,max_iterations=12,max_line_search=10,do_not_relax_residual_on_failure=True))


def cycle(run,grid,fine,stop_after=None):
    run=Path(run);name='observable-'+grid+('-half' if fine else '-h');folder=run/'cases'/name;loaded=None
    if (folder/'identity.json').exists():
        identity=read(folder/'identity.json')
        if identity['numerical_source_sha256']!=numerical_sources() or identity['driver_sha256']!=sha(__file__):raise ValueError('coupling source drift')
        store=GenerationStore(folder,identity);loaded=store.load();store.history()
    begun=time.perf_counter();c,m,cfg=setup(run,grid,fine,loaded['state'] if loaded else None);cache=CachedProbes(m);protocol=read(run/'S3/new-scene-protocol.json');obs=protocol['observations_s']
    if fine:obs=[obs[0],obs[2],obs[4]]
    if loaded is None:
        identity=dict(schema='observable-fixture-case-v1',coupling=c.identity,numerical_source_sha256=numerical_sources(),driver_sha256=sha(__file__),input_lock_sha256=sha(run/'input-lock.json'));write(folder/'identity.json',identity);snapshot(folder/'source',numerical_sources());write(folder/'execution-protocol.json',dict(config=cfg,grid=grid,fine=fine,coupling=c.identity));store=GenerationStore(folder,identity);store.save(c.state,[],frame=frame(c,cache,c.state));rows=[]
    else:
        if identity['coupling']!=c.identity:raise ValueError('fixture identity drift')
        rows=loaded['rows']
    initial=c.state.step;count=0;prior=read(folder/'attempts.json') if (folder/'attempts.json').exists() else {};attempts=prior.get('attempts',0);previous_seconds=prior.get('wall_seconds',0.)
    while c.state.step<len(c.times)-1 and (stop_after is None or count<stop_after):
        other=[read(p) for p in (run/'cases').glob('observable-*/attempts.json') if p.parent!=folder]
        wall_seconds=previous_seconds+time.perf_counter()-begun
        if wall_seconds>=1200 or wall_seconds+sum(x.get('wall_seconds',0.) for x in other)>=2400 or attempts+sum(x['attempts'] for x in other)>=96:raise TimeoutError('registered coupled case/stage/attempt budget exhausted')
        attempts+=1;write(folder/'attempts.json',dict(attempts=attempts,committed=c.state.step,wall_seconds=wall_seconds));t=c.times[c.state.step+1];tick=time.perf_counter()
        try:row=advance_publish(c,store,rows,frame_builder=(lambda s:frame(c,cache,s)) if t in obs[1:] else None)
        except Exception as error:
            write(folder/'attempts.json',dict(attempts=attempts,committed=c.state.step,wall_seconds=previous_seconds+time.perf_counter()-begun))
            write(folder/'failure.json',dict(status='limited',error=repr(error),last_committed_step=c.state.step,last_digest=c.state.digest(),seconds=time.perf_counter()-begun,rolled_back=store.load()['state'].digest()==c.state.digest()));raise
        rows.append(row);count+=1;print(name,c.state.step,'iters',row['iterations'],'seconds',round(time.perf_counter()-tick,3),'minJ',row['min_detF'],flush=True)
        if time.perf_counter()-begun>1200:raise TimeoutError('coupled case time budget')
    write(folder/'attempts.json',dict(attempts=attempts,committed=c.state.step,wall_seconds=previous_seconds+time.perf_counter()-begun))
    segments=read(folder/'segments.json') if (folder/'segments.json').exists() else [];segments.append(dict(pid=os.getpid(),start_step=initial,end_step=c.state.step,seconds=time.perf_counter()-begun));write(folder/'segments.json',segments);write(folder/'ledger.json',rows)
    frames=[]
    for item in store.history():
        path=item['folder']/'frame.npz'
        if path.exists():
            with np.load(path) as a:frames.append(dict(time_s=item['state'].time,u_rms_m=float(la.norm((a['x']-a['X']).reshape(-1,3))/np.sqrt(a['X'].size/3)),v_rms_m_s=float(la.norm(a['velocity'].reshape(-1,3))/np.sqrt(a['X'].size/3))))
    f=c.state.child_states['fluid'];origin=store.history()[0]['state'].child_states['fluid'];mass=float(np.sum(np.array(f['content_m3'])-origin['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']));passed=abs(mass)<=1e-10 and all(r['min_detF']>.1 and r['true_scaled_residual']<=1 and r['darcy_dissipation_J']>=0 for r in rows)
    write(folder/'summary.json',dict(status='passed_scoped' if c.state.step==len(c.times)-1 and passed else 'in_progress',steps=c.state.step,attempts=attempts,end_s=c.state.time,frames=frames,min_detF=min((x['min_detF'] for x in rows),default=1),max_mass_defect=abs(mass),actual_new_process=len({x['pid'] for x in segments})>1,max_balance_J=max((abs(x['energy_balance_J']) for x in rows),default=0),seconds=sum(x['seconds'] for x in segments)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['screen','operators','cycle']);p.add_argument('--run',type=Path,required=True);p.add_argument('--grid',choices=['coarse','fine'],default='coarse');p.add_argument('--fine',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='cycle':cycle(a.run,a.grid,a.fine,a.stop_after)
        else:globals()[a.phase](a.run)
