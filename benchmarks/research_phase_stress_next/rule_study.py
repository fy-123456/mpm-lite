"""Material qualification bound to the actual selected space and trajectory."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import APP,SPACE_PARENT,read,write,sha,digest,register,verify,serial_lock,utc
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
    register(run,'S5/qualification-protocol.json',dict(space=choice,peak_m=.005,selected_times=[0.,.5,1.1],independent_time=.8,
        full_order=order,higher_order=order+1,compressed_order=5,mass_order=mass,max_dt_s=.0125,source=str(source),directions=['seeded mixed','observed stress-sensitive mode 501 from S1']))
    r,_=load_selected(choice['package']);models={o:install_peak(SegmentedModel(r,order=o,device='cuda:0'),.005) for o in (5,order,order+1)}
    if read(run/'S3/performance-decision.json')['warm_adopted']:
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
    q=dict(schema='spatial-phase-q5-qualification-v1',utc=utc(),qualified=qualified,sufficient_qualified=sufficient,results=results,
        reduction_sha256=r.signature,mass_sha256=digest(m.M.tolist()),compressed_model=models[5].identity,full_model=m.identity,source_case=str(source),
        scope=dict(max_dt_s=.0125,peak_m=.005,fiber_angle_degrees=45.,geometry=choice['selected'],mass_order=mass,cycle_end_s=1.6,sentinels=[.5,.6,1.1]),
        scope_limit='registered selected-space trajectory; independent .8 state; no coupled compression claim')
    q['scope']['time_grid_s']=read(source/'execution-protocol.json')['times']
    write(run/'S5/qualification-final.json',q)
    write(run/'S5/full-rule-qualification.json',dict(status='passed_scoped' if sufficient else 'failed',order=order,compared_order=order+1,mass_identical=True,qualification_sha256=sha(run/'S5/qualification-final.json')))
    write(run/'S5/compressed-rule-qualification.json',dict(status='passed_scoped' if qualified else 'not_qualified',policy='q5_with_full_retry' if qualified else 'full_only',qualification_sha256=sha(run/'S5/qualification-final.json')))
    write(run/'S5/qualification-scope.json',dict(status='qualified_main' if qualified else 'full_only',space=choice,physical_source=str(source),equivalent_implementation=True,scope=q['scope'],coupled_q5=False,sensitive_full_cycle=False))
    if not sufficient:raise ValueError('full rule not qualified; bounded higher-order analysis required')


def runtime_models(run):
    from .performance_study import settings
    from .run import load_model,make_stepper
    cfg=settings(run,True)
    cert=Path(run)/'S5/qualification-final.json'
    if cert.exists():cfg['post_release']['qualification']=dict(path=str(cert.resolve()),sha256=sha(cert))
    compact,_=load_model(run,cfg);controller=make_stepper(run,cfg,compact,compact.rest())
    return cfg,compact,controller.full


def faults(run):
    from .provenance import source_files,snapshot
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
    from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError,advance_publish
    run=Path(run)
    register(run,'S5/runtime-protocol.json',dict(steps=4,
        purpose='transaction fixture; an unqualified q5 attempt is deliberately rejected and never committed',
        faults=['before_material','after_prepare','before_commit','double failure','KeyError','before_pointer','after_pointer'],
        actual_restart=True,does_not_grant_q5_qualification=True))
    cfg,m,full=runtime_models(run);results=[]
    for where in ['before_material','after_prepare','before_commit']:
        c=RuntimeRetry(m,full,cfg)
        def fail(attempt,stage,state):
            if attempt==0 and stage==where:raise RetryableRuleError('controlled '+where)
        def children(state,values):values['owned_child']={'pressure':[.1,.2],'time':state.time};return values
        row=c.step(.0125,inject=fail,prepare_children=children)
        ref=ValidatedAVF(full,cfg);ref.step(.0125,prepare_children=children)
        diff=max(float(np.max(abs(getattr(c.state,k)-getattr(ref.state,k)))) for k in ('q','velocity','predictor'))
        assert row['material_attempts']==2 and diff<1e-8 and c.state.child_states['owned_child']['time']==.0125
        assert c.step(.0125)['material_attempts']==1
        results.append(dict(fault=where,q_v_predictor_max=diff,sticky_full=True,owned_children_preserved=True))
    for label,kind in [('double_failure',RetryableRuleError),('programmer_error',KeyError)]:
        c=RuntimeRetry(m,full,cfg);base=c.state.digest()
        def reject(attempt,where,state):
            if where=='before_material':raise kind(label)
        try:c.step(.0125,inject=reject)
        except StepRejected:pass
        else:raise AssertionError('expected rejection')
        assert c.state.digest()==base and len(c.failures)==(2 if kind==RetryableRuleError else 1)
        results.append(dict(fault=label,unchanged=True,attempts=len(c.failures)))
    c=RuntimeRetry(m,full,cfg);folder=run/'cases/retry-fixture'
    identity=dict(schema='spatial-phase-rule-fixture-v1',input_lock_sha256=sha(run/'input-lock.json'),controller=c.identity,
        numerical_source_sha256=source_files(),fixture_source_sha256=sha(Path(__file__)),never_accept_unqualified_q5=True)
    write(folder/'identity.json',identity);write(folder/'execution-protocol.json',cfg);snapshot(folder/'source',source_files())
    store=GenerationStore(folder,identity);store.save(c.state,[]);base=c.state.digest();pointer=read(store.pointer)
    def first_fail(attempt,where,state):
        if attempt==0 and where=='before_material':raise RetryableRuleError('fixture compulsory full retry')
    def disk_fail(where):
        if where=='before_pointer':raise OSError('controlled before pointer')
    try:advance_publish(c,store,[],.0125,inject_step=first_fail,inject_store=disk_fail)
    except OSError:pass
    else:raise AssertionError('expected disk failure')
    assert c.state.digest()==base and read(store.pointer)==pointer and len(store.history())==1
    c=RuntimeRetry(m,full,cfg,store.load()['state'])
    def after(where):
        if where=='after_pointer':raise OSError('controlled postpublication observer')
    row=advance_publish(c,store,[],.0125,inject_step=first_fail,inject_store=after)
    assert len(store.history())==2 and c.state.step==1 and row['material_attempts']==2
    ref=ValidatedAVF(full,cfg)
    for _ in range(4):ref.step(.0125)
    np.savez_compressed(run/'S5/uninterrupted-full.npz',q=ref.state.q,v=ref.state.velocity,predictor=ref.state.predictor)
    write(run/'S5/fault-results.json',dict(status='passed_scoped',records=results,before_pointer_rollback=True,after_pointer_accepted_once=True,
        numeric_sources=source_files(),fixture_does_not_qualify_material=True))
    print('RULE_FAULTS passed',flush=True)


def cycle(run):
    from .provenance import source_files
    from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,advance_publish
    run=Path(run);folder=run/'cases/retry-fixture';identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or identity['fixture_source_sha256']!=sha(Path(__file__)):raise ValueError('fixture source changed')
    cfg,m,full=runtime_models(run);store=GenerationStore(folder,identity);data=store.load();c=RuntimeRetry(m,full,cfg,data['state']);rows=data['rows']
    if c.state.child_states['identity']!=full.identity:raise ValueError('fixture may only resume committed sufficient rule')
    while c.state.step<4:rows.append(advance_publish(c,store,rows,.0125))
    with np.load(run/'S5/uninterrupted-full.npz') as z:
        errors={k:float(np.max(abs(getattr(c.state,k)-z[zk]))) for k,zk in [('q','q'),('velocity','v'),('predictor','predictor')]}
    assert max(errors.values())<1e-8 and len(store.history())==5 and len(c.state.child_states['material_failure_history'])==1
    for row in rows:assert abs(row['delta_total_including_rule_J']-row['delta_total_J']-row['rule_switch_energy_J'])<1e-12
    write(run/'S5/retry-restart.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,generations=5,
        numeric_sources=source_files(),active_rule=full.rule.signature,qualification_separate=True))
    print('RULE_RESTART',errors,flush=True)


def sensitive(run):
    from .run import create_config,run_case,load_model
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
    from benchmarks.research_reference_next.performance_study import branch
    from benchmarks.research_sequential_next.material_study import compare_fields
    from benchmarks.research_sequential_next.compare import metric
    run=Path(run);verify(run)
    source=APP/'cases/sensitive-full-prefix';parent=GenerationStore(source,read(source/'identity.json')).load()
    register(run,'S5/sensitive-protocol.json',dict(peak_m=.0075,main_scope_unchanged=True,
        inherited_prefix=str(source),prefix_state_sha256=sha(parent['folder']/'state.json'),
        bridge_s=[.5,.75],new_full_steps=20,windows=[[.6,.65],[.7,.75]],independent_window=1,
        window_steps=4,dt_s=.0125,full_cycle_qualification=False,full_material_orders=[7,8],
        inheritance='same physical space, mass, full material and actual step; new case identity, old generations untouched'))
    cfg=create_config(run,'sensitive-full-extension',peak_m=.0075,start=.5,end=.75,initial=parent['folder']/'state.json',
        rule_policy='full_only',display_frames=3,field_cache=True)
    run_case(run,'sensitive-full-extension')
    folder=run/'cases/sensitive-full-extension';history=GenerationStore(folder,read(folder/'identity.json')).history()
    lookup={round(x['state'].time,10):x for x in history}
    full,_=load_model(run,cfg);compact=install_peak(SegmentedModel(full.reduction,order=5,device='cuda:0'),.0075)
    higher=install_peak(SegmentedModel(full.reduction,order=8,device='cuda:0'),.0075)
    if read(run/'S3/performance-decision.json')['warm_adopted']:
        from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
        install_warm(compact);install_warm(higher)
    rng=np.random.default_rng(4242);d=rng.normal(size=full.rest().q.shape);d[full.fixed]=0;d/=np.linalg.norm(d)
    windows=[]
    for i,start in enumerate([.6,.7]):
        item=lookup[start];state=item['state'];weak={o:weak_moments(m,state.q) for o,m in [(5,compact),(7,full),(8,higher)]}
        val={o:m.evaluate(state.q,d) for o,m in [(5,compact),(7,full),(8,higher)]}
        metrics=dict(full=material_metrics(val[7],val[8],weak[7],weak[8]),compressed=material_metrics(val[5],val[7],weak[5],weak[7]))
        if not metrics['full']['passed']:raise ValueError('sensitive full rule insufficient')
        initial,origin=branch(item,compact,full);a=ValidatedAVF(compact,cfg,initial);checks=[]
        for k in range(1,5):
            ra=a.step(.0125);target=lookup[round(start+.0125*k,10)];rb=target['rows'][-1]
            fields=compare_fields(full,a.state,target['state'],cfg);reaction=metric(ra['reaction_N'],rb['reaction_N'],1e-4,.05)
            okay=reaction['passed'] and all(v['passed'] for region in fields.values() for v in region.values())
            checks.append(dict(time_s=a.state.time,fields=fields,reaction=reaction,passed=okay))
        qualified=metrics['compressed']['passed'] and all(x['passed'] for x in checks)
        windows.append(dict(window_s=[start,start+.05],independent_check=i==1,qualified=qualified,metrics=metrics,
            start_state_sha256=sha(item['folder']/'state.json'),branch=origin,checks=checks))
    write(run/'S5/sensitive-scope.json',dict(status='qualified_windows' if all(w['qualified'] for w in windows) else 'limited_windows',
        peak_m=.0075,inherited_window_s=[.5,.55],inherited_evidence_sha256=sha(APP/'P5/sensitive-scope.json'),
        inherited_prefix_sha256=sha(source/'identity.json'),new_full_steps=20,new_windows=windows,
        full_cycle_q5=False,main_runner_policy='full_only outside authenticated existing window',
        untested='full loading/hold/unloading/free-vibration coverage; angle/geometry/coupled variants'))
    print('SENSITIVE_NEW_WINDOWS',[(x['window_s'],x['qualified']) for x in windows],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['qualification','faults','cycle','sensitive']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release')
        {'qualification':qualification,'faults':faults,'cycle':cycle,'sensitive':sensitive}[a.phase](a.run)
