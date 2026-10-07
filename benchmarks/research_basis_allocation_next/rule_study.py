"""Material qualification bound to the actual selected space and trajectory."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import APP,SPACE_PARENT,read,write,sha,digest,register,verify,serial_lock,utc,source_files
from .spaces import load_selected
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.diagnostics import weak_moments
from benchmarks.research_sequential_next.material_study import material_metrics
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.scenarios import install_peak


def qualification(run):
    run=Path(run);verify(run)
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    choice=read(run/'selected-space.json');order=choice['full_order'];mass=choice['mass_order']
    source=run/'cases/final-q7-dt0125'
    register(run,'S6/qualification-protocol.json',dict(space=choice,peak_m=.005,selected_times=[0.,.5,1.1],independent_time=.8,
        full_order=order,higher_order=order+1,compressed_order=5,mass_order=mass,max_dt_s=.0125,source=str(source),directions=['seeded mixed','observed stress-sensitive mode 501 from S1']))
    r,_=load_selected(choice['package']);models={o:install_peak(SegmentedModel(r,order=o,device='cuda:0'),.005) for o in (5,order,order+1)}
    if True:
        from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
        models={o:install_warm(m) for o,m in models.items()}
    hist=GenerationStore(source,read(source/'identity.json')).history();lookup={round(x['state'].time,10):x for x in hist};m=models[order]
    rng=np.random.default_rng(20261001);mixed=rng.normal(size=m.rest().q.shape);mixed[m.fixed]=0;mixed/=np.linalg.norm(mixed)
    sensitive=np.zeros_like(mixed)
    with np.load(SPACE_PARENT/'N2/modal-basis.npz') as z:sensitive[m.free]=z['vectors'][:,501].reshape(-1,3)
    sensitive/=np.linalg.norm(sensitive)
    results=[]
    for t in [0.,.5,1.1,.8]:
        q=lookup[t]['state'].q;weak={o:weak_moments(z,q) for o,z in models.items()}
        for label,d in [('mixed',mixed),('sensitive',sensitive)]:
            values={o:z.evaluate(q,d) for o,z in models.items()}
            results.append(dict(time_s=t,independent_check=t==.8,direction=label,state_sha256=sha(lookup[t]['folder']/'state.json'),
                sufficient=material_metrics(values[order],values[order+1],weak[order],weak[order+1]),compressed=material_metrics(values[5],values[order],weak[5],weak[order])))
        print('QUALIFY',t,'full',all(x['sufficient']['passed'] for x in results),'q5',all(x['compressed']['passed'] for x in results),flush=True)
    assert all(np.array_equal(m.M,z.M) for z in models.values())
    sufficient=all(x['sufficient']['passed'] for x in results);qualified=sufficient and all(x['compressed']['passed'] for x in results)
    q=dict(schema='basis-allocation-q5-qualification-v1',utc=utc(),qualified=qualified,sufficient_qualified=sufficient,results=results,
        reduction_sha256=r.signature,mass_sha256=digest(m.M.tolist()),compressed_model=models[5].identity,full_model=m.identity,source_case=str(source),
        scope=dict(max_dt_s=.0125,peak_m=.005,fiber_angle_degrees=45.,geometry=choice['selected'],mass_order=mass,cycle_end_s=1.6,sentinels=[.5,.6,1.1]),
        scope_limit='registered selected-space trajectory; independent .8 state; no coupled compression claim')
    q['scope']['time_grid_s']=read(source/'execution-protocol.json')['times']
    q['numerical_source_sha256']=source_files()
    q['scope'].update(physical_space_sha256=choice['package']['sha256'],full_order=order,rest_start=True,initial_states=[dict(time_s=x['state'].time,sha256=sha(x['folder']/'state.json')) for x in hist])
    write(run/'S6/qualification-final.json',q)
    write(run/'S6/full-rule-qualification.json',dict(status='passed_scoped' if sufficient else 'failed',order=order,compared_order=order+1,mass_identical=True,qualification_sha256=sha(run/'S6/qualification-final.json')))
    write(run/'S6/compressed-rule-qualification.json',dict(status='passed_scoped' if qualified else 'not_qualified',policy='q5_with_full_retry' if qualified else 'full_only',qualification_sha256=sha(run/'S6/qualification-final.json')))
    write(run/'S6/qualification-scope.json',dict(status='qualified_main' if qualified else 'full_only',space=choice,physical_source=str(source),equivalent_implementation=True,scope=q['scope'],coupled_q5=False,sensitive_full_cycle=False))
    if not sufficient:raise ValueError('full rule not qualified; bounded higher-order analysis required')


def windows_prepare(run):
    from . import config
    from .run import create_config
    from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    source=APP/'cases/sensitive-full-extension';identity=read(source/'identity.json')
    hist=GenerationStore(source,identity).history();lookup={round(x['state'].time,10):x for x in hist}
    research=read(APP/'S5/sensitive-scope.json');window=next(x for x in research['new_windows'] if x['window_s']==[.6,.65])
    if not window['qualified'] or window['start_state_sha256']!=sha(lookup[.6]['folder']/'state.json'):raise ValueError('window inheritance invalid')
    register(run,'S3/window-protocol.json',dict(peak=.0075,window=[.6,.65],steps=4,actual_CLI=True,restart_after=2,
        reused_research_sha256=sha(APP/'S5/sensitive-scope.json'),new_full_prefix_steps=0,full_cycle_qualified=False))
    choice=read(run/'selected-space.json');r,_=load_selected(choice['package'])
    full=install_warm(install_peak(SegmentedModel(r,order=7,device='cuda:0'),.0075))
    compact=install_warm(install_peak(SegmentedModel(r,order=5,device='cuda:0'),.0075))
    if identity['model']!=full.identity:raise ValueError('inherited window physical identity differs')
    if not np.array_equal(compact.M,full.M):raise ValueError('material compression changed mass')
    times=[round(.6+k*.0125,10) for k in range(5)]
    q=dict(schema='basis-allocation-q5-qualification-v1',utc=utc(),qualified=True,sufficient_qualified=True,
        numerical_source_sha256=source_files(),reduction_sha256=r.signature,mass_sha256=digest(full.M.tolist()),
        compressed_model=compact.identity,full_model=full.identity,source_case=str(source),
        evidence=dict(path=str(APP/'S5/sensitive-scope.json'),sha256=sha(APP/'S5/sensitive-scope.json'),window=[.6,.65]),
        scope=dict(time_grid_s=times,max_dt_s=.0125,peak_m=.0075,fiber_angle_degrees=45.,physical_space_sha256=choice['package']['sha256'],
            mass_order=7,full_order=7,rest_start=False,initial_states=[dict(time_s=t,sha256=sha(lookup[t]['folder']/'state.json')) for t in times[:-1]]),
        scope_limit='only exact contiguous window with authenticated sufficient-rule start; full sensitive cycle remains full_only')
    path=run/'S3/window-0060-0065.json';write(path,q)
    write(run/'S3/evidence-vs-permission.json',dict(parent_main_CLI='full_only for all sensitive requests',
        existing_research_windows=[[.5,.55],[.6,.65],[.7,.75]],new_CLI_permission=[.6,.65],other_windows='research only; no automatic permission',
        main_full_cycle_permission='pending final actual sufficient cycle',sensitive_full_cycle_q5=False))
    write(run/'S3/local-material-check.json',dict(status='inherited_same_model_verified',window=window['window_s'],
        material_full=window['metrics']['full']['passed'],material_compressed=window['metrics']['compressed']['passed'],
        source_sha256=sha(APP/'S5/sensitive-scope.json'),model_identical=True,mass_identical=True,main_final_pending=True))
    write(run/'S3/scope-contract.json',dict(schema=q['schema'],contiguous_steps=True,source_identity=True,source_hash=True,
        own_histories_preserved=True,energy_switch_ledger=True,source_mutation=False,damaged_hash='reject',valid_outside_scope='full_only'))
    cfg=create_config(run,'sensitive-window',start=.6,end=.65,peak_m=.0075,initial=lookup[.6]['folder']/'state.json',
        rule_policy='q5_with_full_retry',qualification_path=path,display_frames=3,field_cache=True)
    if cfg['post_release']['rule_policy']!='q5_with_full_retry':raise ValueError('explicit sensitive permission not selected')
    outside=create_config(run,'sensitive-outside-scope',start=.6,end=.6625,peak_m=.0075,initial=lookup[.6]['folder']/'state.json',
        rule_policy='q5_with_full_retry',qualification_path=path,display_frames=2)
    if outside['post_release']['rule_policy']!='full_only':raise ValueError('out of scope compressed config accepted')
    write(run/'S3/entry-check.json',dict(status='passed_scoped',accepted_window=cfg['post_release'],outside=outside['post_release']))
    print('WINDOW_READY',path,flush=True)


def windows_check(run):
    from .run import load_model,load_initial,make_stepper
    from benchmarks.research_sequential_next.material_study import compare_fields
    from benchmarks.research_sequential_next.compare import metric
    run=Path(run);folder=run/('cases/sensitive-window-final' if (run/'cases/sensitive-window-final').exists() else 'cases/sensitive-window');cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg)
    c=make_stepper(run,cfg,m,m.rest());h=GenerationStore(folder,read(folder/'identity.json')).history()
    source=APP/'cases/sensitive-full-extension';ref={round(x['state'].time,10):x for x in GenerationStore(source,read(source/'identity.json')).history()}
    checks=[]
    for x in h[1:]:
        other=ref[round(x['state'].time,10)];fields=compare_fields(c.full,x['state'],other['state'],cfg)
        reaction=metric(x['rows'][-1]['reaction_N'],other['rows'][-1]['reaction_N'],1e-4,.05)
        checks.append(dict(time=x['state'].time,fields=fields,reaction=reaction,passed=bool(reaction['passed'] and all(v['passed'] for reg in fields.values() for v in reg.values()))))
    if len(h)!=5 or not all(x['passed'] for x in checks):raise ValueError('window CLI field mismatch')
    branch=h[0]['state'].child_states['material_scope_bridge'];source0=ref[.6]['state']
    unchanged=all(np.array_equal(getattr(h[0]['state'],k),getattr(source0,k)) for k in ('q','velocity','predictor'))
    if not unchanged or branch['source_state_digest']!=source0.digest():raise ValueError('branch changed physical initial')
    write(run/'S3/window-runtime-check.json',dict(status='passed_scoped',steps=4,checks=checks,actual_new_process_restart=True,
        source_unchanged=True,initial_q_v_predictor_unchanged=True,branch_energy_J=branch['material_energy_difference_J'],
        numeric_sources=source_files(),sensitive_full_cycle_q5=False))
    print('WINDOW_PASSED',branch['material_energy_difference_J'],flush=True)

def windows_refresh(run):
    from .run import create_config
    run=Path(run);old=run/'S3/window-0060-0065.json';q=read(old);q['numerical_source_sha256']=source_files()
    q['source_equivalence']=dict(previous_certificate_sha256=sha(old),reason='q5_fixed nonzero branch now constructs the sufficient model; q5_with_full_retry and q7 equations unchanged')
    path=run/'S3/window-final-0060-0065.json';write(path,q)
    cfg=read(run/'cases/sensitive-window/execution-protocol.json');initial=Path(cfg['initial_state']['path'])
    for name,policy,end in [('sensitive-window-final','q5_with_full_retry',.65),('fixed-window-branch','q5_fixed',.625)]:
        create_config(run,name,start=.6,end=end,peak_m=.0075,initial=initial,rule_policy=policy,qualification_path=path,display_frames=3,field_cache=True)
    write(run/'S3/ordinary-fix-protocol.json',dict(reason='code review found q5_fixed nonzero branch had no full model to authenticate its q7 initial state',
        fix='construct full model for either compressed policy; q5_fixed still never retries',new_steps=6,window='same .6-.65 evidence; no extra material scope',
        previous_runtime_evidence=read(run/'S3/window-runtime-check.json'),numerical_source_sha256=source_files()))

def faults(run):
    from .run import load_model,make_stepper
    from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError,advance_publish
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
    run=Path(run);cfg=read(run/'cases/daily-q5-retry-dt0125/execution-protocol.json');m,_=load_model(run,cfg);base=make_stepper(run,cfg,m,m.rest());full=base.full
    register(run,'S3/fault-protocol.json',dict(final_source=source_files(),faults=['before_material','after_prepare','before_commit','double_failure','KeyError','before_pointer','after_pointer'],
        restart='S6/cache-retry-check.json',qualified_main=True,steps_per_trial=1))
    records=[]
    def children(state,values):values['owned_child']={'pressure':[.1,.2],'time':state.time};return values
    reference=ValidatedAVF(full,cfg);reference.step(.0125,prepare_children=children)
    for stage in ('before_material','after_prepare','before_commit'):
        c=RuntimeRetry(m,full,cfg)
        def fail(attempt,where,state):
            if attempt==0 and where==stage:raise RetryableRuleError('controlled '+stage)
        row=c.step(.0125,inject=fail,prepare_children=children)
        err=max(float(np.max(abs(getattr(c.state,k)-getattr(reference.state,k)))) for k in ('q','velocity','predictor'))
        if err>1e-8 or row['material_attempts']!=2 or c.state.child_states['owned_child']['time']!=.0125:raise ValueError('whole-step retry differs')
        records.append(dict(fault=stage,error=err,attempts=2,owned_history=True,energy_switch_J=row['rule_switch_energy_J']))
    for label,error in [('double_failure',RetryableRuleError),('programming_error',KeyError)]:
        c=RuntimeRetry(m,full,cfg);before=c.state.digest();calls=[]
        def fail(attempt,where,state):
            if where=='before_material':calls.append(attempt);raise error(label)
        try:c.step(.0125,inject=fail)
        except StepRejected:pass
        else:raise AssertionError('failure was swallowed')
        if c.state.digest()!=before or len(calls)!=(2 if label=='double_failure' else 1):raise ValueError('bad error classification or partial rollback')
        records.append(dict(fault=label,attempts=len(calls),unchanged_digest=True))
    c=RuntimeRetry(m,full,cfg);folder=run/'S3/publication-fixture';store=GenerationStore(folder,dict(model=m.identity,source=source_files()));store.save(c.state,[])
    before=c.state.digest();pointer=read(store.pointer)
    def fail_before(where):
        if where=='before_pointer':raise OSError('controlled disk error')
    try:advance_publish(c,store,[],.0125,inject_store=fail_before)
    except OSError:pass
    else:raise AssertionError('disk error swallowed')
    if c.state.digest()!=before or read(store.pointer)!=pointer:raise ValueError('publication rollback failed')
    def fail_after(where):
        if where=='after_pointer':raise OSError('controlled observer error')
    row=advance_publish(c,store,[],.0125,inject_store=fail_after)
    if c.state.step!=1 or len(store.history())!=2:raise ValueError('publication not accepted once')
    records.extend([dict(fault='before_pointer',unchanged=True),dict(fault='after_pointer',accepted_once=True)])
    write(run/'S3/fault-and-restart.json',dict(status='passed_scoped',records=records,numeric_sources=source_files(),actual_restart=read(run/'S6/cache-retry-check.json')))
    write(run/'S3/runtime-policy.json',dict(status='passed_scoped',requested_actual_recorded=True,classification='only numerical/retryable errors',
        whole_step_rollback=True,programming_errors_propagate=True,publication_authoritative=True,sensitive_full_cycle_q5=False))
    print('FAULTS_PASSED',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['qualification','window-prepare','window-check','window-refresh','faults']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release')
        {'qualification':qualification,'window-prepare':windows_prepare,'window-check':windows_check,'window-refresh':windows_refresh,'faults':faults}[a.phase](a.run)
