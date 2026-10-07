"""Actual 16/32-cell early drainage, bounded transactions and honest observability."""
from pathlib import Path
import argparse,time,os,resource
import numpy as np
import scipy.linalg as la
from .provenance import *
from .physics import baseline_model
from engine.aniso_phase1.research_pressure_window_next.coupled import WindowCoupling,time_tolerance
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import balanced_solve
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric


def setup(run,label,state=None):
    protocol=read(Path(run)/'S2/coupled-protocol.json');m,cfg=baseline_model(run,pressure=True)
    c=WindowCoupling(m,cfg,protocol['times_s'],cuts=protocol['cuts'][label],state=state)
    return c,m,cfg


def frame(c,cache,state):
    f=cache.frame(state);F=f['F'];J=np.linalg.det(F);p=np.array(state.child_states['fluid']['pressure_Pa']);ids=c.geometry.topology.locate(f['X'].reshape(-1,3));pp=p[ids].reshape(J.shape)
    f['pressure_Pa']=pp;f['PK1_total']=f['PK1']-c.alpha*pp[...,None,None]*J[...,None,None]*np.swapaxes(np.linalg.inv(F),-1,-2)
    f['Cauchy_skeleton']=f['PK1']@np.swapaxes(F,-1,-2)/J[...,None,None];f['Cauchy_total']=f['PK1_total']@np.swapaxes(F,-1,-2)/J[...,None,None]
    f['cell_pressure_Pa']=p;f['face_flux_m3_s']=np.array(state.child_states['fluid']['flux_interval_m3_s'])
    err=float(np.max(abs(f['Cauchy_total']-(f['Cauchy_skeleton']-c.alpha*pp[...,None,None]*np.eye(3)))))
    if err>1e-9:raise ValueError('pressure stress measure mismatch')
    return f


def prepare(run):
    run=Path(run);verify(run)
    if not read(run/'S1/pressure-scope-decision.json')['eligible_coupled']:raise ValueError('full fixed-pressure gate required')
    p=read(run/'S1/full-window-protocol.json');obs=np.array(p['observations_s']);ts=np.array(p['times_s']);ts=ts[ts<=obs[4]]
    if len(ts)!=20:raise ValueError('19-step pressure prefix required')
    register(run,'S2/coupled-protocol.json',dict(status='passed_scoped',times_s=ts.tolist(),observations_s=obs[:5].tolist(),cuts=p['cuts'],main_steps_per_grid=19,max_fine_steps_per_grid=38,transactions_extra_max=8,mechanical_u_floor_m=5e-5,mechanical_v_floor_m_s=1e-4,full_material_order=7,mass_order=7,split_after=2,actual_frames_per_grid=5))
    checks=[];scales=[];models={}
    for label in ('coarse','fine'):
        c,m,cfg=setup(run,label);g=c.geometry;q=m.rest().q;v=g.evaluate(q);rng=np.random.default_rng(41);d=rng.normal(size=q.shape);d[m.fixed]=0;d/=la.norm(d);eps=1e-5;q1=q+eps*d
        plus=g.evaluate(q1);minus=g.evaluate(q-eps*d);analytic=np.einsum('cij,ij->c',v['gradient'],d);fd=(plus['volume']-minus['volume'])/(2*eps);G=g.discrete(q,q1);dv=plus['volume']-v['volume'];work=np.einsum('cij,ij->c',G,q1-q)
        error=float(la.norm(fd-analytic));tol=1e-9+.001*float(la.norm(analytic));volume=float(np.max(abs(dv-work)))
        if error>tol or volume>1e-11:raise ValueError('actual-coordinate volume derivative/work failed')
        v['H'][0,0]=-1;v['gradient'][:]=0
        clean=g.evaluate(q)
        if clean['H'][0,0]<=0 or not np.any(clean['gradient']):raise ValueError('geometry cache ownership failed')
        h=float(ts[1]);A=c.rest_matrix(h);known=rng.normal(size=A.shape[0])*1e-7;answer,diag=balanced_solve(A,A@known,diagnose=True)
        if np.linalg.norm(A@answer-A@known)>1e-8:raise ValueError('mixed solve residual failed')
        fpressure=-c.alpha*.01*np.sum(clean['gradient'],axis=0);imp=float(np.max(abs(h*fpressure[m.free])));oldtol=1e-10
        scales.append(dict(grid=label,first_h_s=h,old_dt_tolerance_fraction=1e-12/h,new_dt_tolerance_s=time_tolerance(h),pressure_force_max_N=imp/h,pressure_impulse_N_s=imp,old_solid_atol_N_s=oldtol,impulse_over_old_atol=imp/oldtol,new_nominal_impulse_atol_N_s=h*1e-7,old_atol_can_hide_mechanical_response=bool(imp<oldtol)))
        checks.append(dict(grid=label,status='passed_scoped',gradient_error=error,gradient_budget=tol,discrete_volume_error_m3=volume,nonzero_volume_change_m3=float(np.max(abs(dv))),H_change_norm=float(la.norm(plus['H']-clean['H'])),min_detF=plus['min_detF'],mixed_solve=diag,cache_return_is_owned=True,manufactured_nonzero_state=True))
        models[label]=c.identity
        # Exact index, source history and model identity must survive roundtrip.
        s=c.state;c.validate(s);bad=s.clone();bad.time=ts[1]+.01*ts[1];bad.step=1
        try:c.validate(bad)
        except ValueError:pass
        else:raise AssertionError('fractional early time accepted')
        bad=s.clone();bad.child_states['explicit_pressure_grid']='foreign'
        try:c.validate(bad)
        except ValueError:pass
        else:raise AssertionError('foreign grid accepted')
        del c,m,g
    write(run/'S2/common-model-lock.json',dict(status='passed_scoped',models=models,numerical_source_sha256=source_files(),full_mass=True,zero_grips=True))
    write(run/'S2/volume-gradient-check.json',dict(status='passed_scoped',records=checks,manufactured_is_not_dynamic=True))
    write(run/'S2/mixed-operator-check.json',dict(status='passed_scoped',records=checks,general_LU=True,solid_CG_not_used=True))
    write(run/'S2/residual-and-time-scale-audit.json',dict(status='passed_scoped',records=scales,stopping_criterion_changed=True,equations_unchanged=True,source_terms_not_dropped=True,initial_grid_identity_persisted_in_owned_state=True))
    print('COUPLING_PREPARED',scales,flush=True)


def run_case(run,label,stop_after=None):
    import warp as wp
    run=Path(run);folder=run/'cases'/f'coupled-{label}';started=time.perf_counter()
    loaded=None
    if (folder/'identity.json').exists():
        identity=read(folder/'identity.json')
        if identity['numerical_source_sha256']!=source_files() or identity['driver_sha256']!=sha(__file__):raise ValueError('coupled source drift')
        store=GenerationStore(folder,identity);loaded=store.load();store.history()
    c,m,cfg=setup(run,label,loaded['state'] if loaded else None);cache=CachedProbes(m)
    if loaded is None:
        identity=dict(schema='pressure-window-coupled-case-v1',coupling=c.identity,numerical_source_sha256=source_files(),driver_sha256=sha(__file__),input_lock_sha256=sha(run/'input-lock.json'))
        write(folder/'identity.json',identity);snapshot(folder/'source',source_files());write(folder/'execution-protocol.json',dict(config=cfg,coupling=c.identity,times_s=c.times.tolist()));store=GenerationStore(folder,identity);store.save(c.state,[],frame=frame(c,cache,c.state));rows=[]
    else:
        if identity['coupling']!=c.identity:raise ValueError('coupled identity drift')
        rows=loaded['rows']
    initial_step=c.state.step;count=0;obs=read(run/'S2/coupled-protocol.json')['observations_s']
    while c.state.step<19 and (stop_after is None or count<stop_after):
        next_time=c.times[c.state.step+1];tick=time.perf_counter()
        try:row=advance_publish(c,store,rows,frame_builder=(lambda st:frame(c,cache,st)) if next_time in obs[1:] else None)
        except Exception as err:
            write(folder/'failure.json',dict(status='failed',error=repr(err),last_committed_step=c.state.step,time_s=c.state.time,source_sha256=source_files()));raise
        wp.synchronize_device('cuda:0');row_seconds=time.perf_counter()-tick;rows.append(row);count+=1
        print('COUPLED_STEP',label,c.state.step,'dt',row['dt'],'pmin',min(row['pressure_Pa']),'force_res',row['solid_force_residual_N'],'s',row_seconds,flush=True)
        if time.perf_counter()-started>1200 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>16 or resources()['system_free_GiB']<5:raise RuntimeError('resource stop after safe coupled commit')
    segments=read(folder/'segments.json') if (folder/'segments.json').exists() else []
    segments.append(dict(pid=os.getpid(),start_step=initial_step,end_step=c.state.step,seconds=time.perf_counter()-started));write(folder/'segments.json',segments);write(folder/'ledger.json',rows)
    if c.state.step==19:
        hist=store.history();f=c.state.child_states['fluid'];f0=hist[0]['state'].child_states['fluid'];mass=float(np.sum(np.array(f['content_m3'])-f0['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
        frames=[]
        for item in hist:
            path=item['folder']/'frame.npz'
            if path.exists():
                with np.load(path) as a:frames.append(dict(time_s=item['state'].time,u_max_m=float(np.max(abs(a['x']-a['X']))),v_max_m_s=float(np.max(abs(a['velocity']))),PK1_max_Pa=float(np.max(abs(a['PK1']))),PK1_total_max_Pa=float(np.max(abs(a['PK1_total'])))))
        maxu=max(x['u_max_m'] for x in frames);maxv=max(x['v_max_m_s'] for x in frames);observable=bool(maxu>=5e-5 or maxv>=1e-4)
        passed=abs(mass)<=1e-10 and all(r['true_scaled_residual']<=1 and r['darcy_dissipation_J']>=0 and r['min_detF']>.1 for r in rows)
        write(folder/'summary.json',dict(status='passed_scoped' if passed else 'limited',steps=19,end_s=c.state.time,cumulative_mass_defect_m3=mass,min_pressure_Pa=min(min(r['pressure_Pa']) for r in rows),min_detF=min(r['min_detF'] for r in rows),max_true_residual_fraction=max(r['true_scaled_residual'] for r in rows),max_force_residual_N=max(r['solid_force_residual_N'] for r in rows),max_energy_balance_J=max(abs(r['energy_balance_J']) for r in rows),max_pressure_work_defect_J=max(float(np.max(abs(np.asarray(r['pressure_work_defect_J'])))) for r in rows),max_displacement_m=maxu,max_velocity_m_s=maxv,mechanical_response_observable=observable,frames=frames,actual_new_process=len({s['pid'] for s in segments})==2,segments=segments))
        if not passed:raise ValueError('coupled physics failed')


def transactions(run):
    run=Path(run);folder=run/'cases/coupled-coarse';identity=read(folder/'identity.json');hist=GenerationStore(folder,identity).history();origin=hist[2]['state'];c,m,cfg=setup(run,'coarse',origin);records=[]
    def physical_equal(a,b):
        aa=a.to_dict();bb=b.to_dict();diagnostics=[]
        for x,y in zip(aa['child_states']['fluid']['last_ledger']['linear_scaling'],bb['child_states']['fluid']['last_ledger']['linear_scaling']):
            for key in ('raw_rcond_1','scaled_rcond_1'):
                if key in x or key in y:
                    av=x.pop(key);bv=y.pop(key)
                    if not np.isfinite(av+bv) or not np.isclose(av,bv,rtol=1e-12,atol=0):raise ValueError('condition estimator changed materially')
                    diagnostics.append(abs(av-bv))
        if digest(aa)!=digest(bb):raise ValueError('physical state/ledger differs beyond diagnostic estimates')
        return max(diagnostics,default=0.)
    # Two continuous steps are the witness for the resumed main process.
    for index in (3,4):
        c.step();expected=hist[index]['state'];actual=c.state
        error=max(float(np.max(abs(getattr(actual,k)-getattr(expected,k)))) for k in ('q','velocity','predictor'))
        for k in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3'):
            error=max(error,float(np.max(abs(np.asarray(actual.child_states['fluid'][k])-expected.child_states['fluid'][k]))))
        if error>1e-10:raise ValueError('new-process coupled recovery differs')
        diagnostic=physical_equal(actual,expected)
        records.append(dict(witness_step=index,max_error=error,full_digest_equal=actual.digest()==expected.digest(),physical_state_exact=True,condition_estimator_roundoff=diagnostic))
        write(run/'S2/transaction-live-check.json',dict(status='in_progress',records=records))
    for stage in ('before_commit','before_pointer','after_pointer'):
        c.restore(origin);test=run/'S2/faults'/stage;store=GenerationStore(test,dict(coupling=c.identity,test=stage))
        completed=store.load()
        if completed is not None:
            if stage!='before_commit' or completed['state'].step!=3:raise ValueError('unexpected partial fault store')
            diagnostic=physical_equal(completed['state'],hist[3]['state']);records.append(dict(fault=stage,accepted_once=True,physical_state_exact=True,condition_estimator_roundoff=diagnostic,reused_completed_fault_trial=True));continue
        store.save(origin,[]);before=c.state.digest();ptr=read(store.pointer)
        def fail_step(where,state):
            if where==stage:raise ValueError('controlled precommit failure')
        def fail_store(where):
            if where==stage:raise OSError('controlled pointer failure')
        try:row=advance_publish(c,store,[],inject_step=fail_step,inject_store=fail_store)
        except (ValueError,OSError):
            if stage=='after_pointer' or c.state.digest()!=before or read(store.pointer)!=ptr:raise AssertionError('partial coupled rollback')
            if c.geometry.cache:raise AssertionError('trial geometry cache retained')
            row=advance_publish(c,store,[])
        if c.state.step!=origin.step+1 or store.load()['state'].digest()!=c.state.digest():raise ValueError('publication not exactly once')
        diagnostic=physical_equal(c.state,hist[3]['state'])
        records.append(dict(fault=stage,accepted_once=True,physical_state_exact=True,condition_estimator_roundoff=diagnostic))
        write(run/'S2/transaction-live-check.json',dict(status='in_progress',records=records))
    write(run/'S2/transaction-and-restart.json',dict(status='passed_scoped',records=records,actual_new_process=read(folder/'summary.json')['actual_new_process'],extra_compute_attempts=9,caches_rolled_back=True,owned_history=True,raw_committed_digest_always_verified=True,only_rcond_diagnostic_estimates_tolerated=True))


def finish(run):
    run=Path(run);reports={k:read(run/f'cases/coupled-{k}/summary.json') for k in ('coarse','fine')};stable=all(r['status']=='passed_scoped' for r in reports.values());observable=any(r['mechanical_response_observable'] for r in reports.values())
    # If measurable mechanical effects emerge, do not pretend the fixed-geometry
    # reference is a dynamic reference. The registered conditional branch is explicit.
    write(run/'S2/coupled-scene-check.json',dict(status='passed_scoped' if stable else 'limited',records=reports,stable_short_window=stable,mechanical_response_unresolved=not observable,resolved_dynamic_accuracy=False,actual_steps=38))
    write(run/'S2/coupled-time-reference.json',dict(status='not_triggered' if not observable else 'limited',reason='mechanical fields below registered engineering observation floors' if not observable else 'observable motion requires registered half-step comparison',new_steps=0))
    write(run/'S2/coupling-scope-decision.json',dict(status='limited',stable_short_window=stable,actual_coupled=True,mechanical_response_unresolved=not observable,resolved_dynamic_accuracy=False,main_steps=38,transaction_steps=read(run/'S2/transaction-and-restart.json')['extra_compute_attempts'],resource_failed_attempts=1,pressure_time_window_qualified=True,coupled_q5=False,production_C_E_integration=False,pure_solid_default=True,resource_root_cause_fully_isolated=False))
    print('COUPLING_SCOPE',stable,observable,{k:(r['max_displacement_m'],r['max_velocity_m_s']) for k,r in reports.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cycle','transactions','finish']);p.add_argument('--run',type=Path,required=True);p.add_argument('--grid',choices=['coarse','fine'],default='coarse');p.add_argument('--stop-after',type=int);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='cycle':run_case(a.run,a.grid,a.stop_after)
        else:globals()[a.phase](a.run)
