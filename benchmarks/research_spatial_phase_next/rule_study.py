"""Material qualification bound to the actual selected space and trajectory."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import APP,read,write,sha,digest,register,verify,serial_lock,utc
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
    source=APP/'cases/final-q7-dt0125' if choice['selected']=='original144' else run/'cases/common-full-prefix'
    register(run,'N3/qualification-protocol.json',dict(space=choice,peak_m=.005,selected_times=[0.,.5,1.1],independent_time=.8,
        full_order=order,higher_order=order+1,compressed_order=5,mass_order=mass,max_dt_s=.0125,source=str(source),directions=['seeded mixed','largest rest diagonal']))
    for name in ['qualification.json','qualification-peak0075.json']:
        (run/'N3'/('inherited-'+name)).write_bytes((run/'N3'/name).read_bytes())
    r,_=load_selected(choice['package']);models={o:install_peak(SegmentedModel(r,order=o,device='cuda:0'),.005) for o in (5,order,order+1)}
    hist=GenerationStore(source,read(source/'identity.json')).history();lookup={round(x['state'].time,10):x for x in hist};m=models[order]
    rng=np.random.default_rng(20261001);mixed=rng.normal(size=m.rest().q.shape);mixed[m.fixed]=0;mixed/=np.linalg.norm(mixed)
    sensitive=np.zeros_like(mixed);sensitive.ravel()[m.ids[int(np.argmax(np.diag(m.rest_K)[m.ids]))]]=1.
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
    write(run/'N3/qualification.json',q)
    write(run/'N3/full-rule-qualification.json',dict(status='passed_scoped' if sufficient else 'failed',order=order,compared_order=order+1,mass_identical=True,qualification_sha256=sha(run/'N3/qualification.json')))
    write(run/'N3/compressed-rule-qualification.json',dict(status='passed_scoped' if qualified else 'not_qualified',policy='q5_with_full_retry' if qualified else 'full_only',qualification_sha256=sha(run/'N3/qualification.json')))
    if choice['selected']!='original144':write(run/'N3/qualification-peak0075.json',dict(qualified=False,scope=dict(max_dt_s=.0125,peak_m=.0075),reason='old-space certificate invalid for new space; no sensitive trajectory sampled'))
    if not sufficient:raise ValueError('full rule not qualified; bounded higher-order analysis required')


def runtime_models(run):
    from . import config
    from .run import load_model
    choice=read(Path(run)/'selected-space.json')
    cfg=config.make(read(Path(run)/'input-lock.json')['energy_scale_J'],end=.05,field_cache=True,
        space=choice['package'],mass_order=choice['mass_order'],full_order=choice['full_order'])
    full,_=load_model(run,cfg)
    compact=install_peak(SegmentedModel(full.reduction,order=5,device='cuda:0'),.005)
    return cfg,compact,full


def faults(run):
    from .provenance import source_files,snapshot
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
    from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError,advance_publish
    run=Path(run)
    register(run,'N3/runtime-protocol.json',dict(steps=4,
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
    np.savez_compressed(run/'N3/uninterrupted-full.npz',q=ref.state.q,v=ref.state.velocity,predictor=ref.state.predictor)
    write(run/'N3/fault-results.json',dict(status='passed_scoped',records=results,before_pointer_rollback=True,after_pointer_accepted_once=True,
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
    with np.load(run/'N3/uninterrupted-full.npz') as z:
        errors={k:float(np.max(abs(getattr(c.state,k)-z[zk]))) for k,zk in [('q','q'),('velocity','v'),('predictor','predictor')]}
    assert max(errors.values())<1e-8 and len(store.history())==5 and len(c.state.child_states['material_failure_history'])==1
    for row in rows:assert abs(row['delta_total_including_rule_J']-row['delta_total_J']-row['rule_switch_energy_J'])<1e-12
    write(run/'N3/retry-restart.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,generations=5,
        numeric_sources=source_files(),active_rule=full.rule.signature,qualification_separate=True))
    print('RULE_RESTART',errors,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['qualification','faults','cycle']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release')
        {'qualification':qualification,'faults':faults,'cycle':cycle}[a.phase](a.run)
