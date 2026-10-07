"""q5 qualification on the selected space and real runtime transaction checks."""
from pathlib import Path
import argparse
import copy
import numpy as np
from .provenance import PARENT,read,write,sha,digest,register,serial_lock,utc,verify
from .run import create_config,load_model,make_stepper,run_case,case_identity
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.model_package import load_reduction
from benchmarks.research_sequential_next.diagnostics import weak_moments
from benchmarks.research_sequential_next.material_study import material_metrics
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError,advance_publish


def qualification(run):
    run=Path(run);verify(run)
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    register(run,'S4/qualification-protocol.json',dict(space='original144',source_case='full-q7-dt0125',
        actual_times=[0.,.5,1.1],directions=['seeded mixed','largest rest diagonal free direction'],
        full_order=7,comparison_order=8,compressed_order=5,mass_order=5,max_dt_s=.0125,
        scope='frozen F45 .005m cycle, fixed M5 and Ks, stage sentinels; not arbitrary material/geometry'))
    r=load_reduction(PARENT);models={o:SegmentedModel(r,order=o,device='cuda:0') for o in (5,7,8)}
    source=run/'cases/full-q7-dt0125';hist=GenerationStore(source,read(source/'identity.json')).history()
    lookup={round(x['state'].time,10):x for x in hist};m=models[7]
    rng=np.random.default_rng(20261001);mixed=rng.normal(size=m.rest().q.shape);mixed[m.fixed]=0;mixed/=np.linalg.norm(mixed)
    sensitive=np.zeros_like(mixed);sensitive.ravel()[m.ids[int(np.argmax(np.diag(m.rest_K)[m.ids]))]]=1.
    results=[]
    for t in [0.,.5,1.1]:
        q=lookup[t]['state'].q;weak={o:weak_moments(z,q) for o,z in models.items()}
        for label,d in [('mixed',mixed),('sensitive',sensitive)]:
            values={o:z.evaluate(q,d) for o,z in models.items()}
            result=dict(time_s=t,direction=label,state_sha256=sha(lookup[t]['folder']/'state.json'),
                        sufficient=material_metrics(values[7],values[8],weak[7],weak[8]),
                        compressed=material_metrics(values[5],values[7],weak[5],weak[7]))
            results.append(result)
        print('QUALIFY',t,all(x['sufficient']['passed'] and x['compressed']['passed'] for x in results),flush=True)
    assert all(np.array_equal(models[7].M,z.M) for z in models.values())
    q=dict(schema='post-release-q5-qualification-v1',utc=utc(),qualified=all(x['sufficient']['passed'] and x['compressed']['passed'] for x in results),
           sufficient_qualified=all(x['sufficient']['passed'] for x in results),results=results,
           reduction_sha256=r.signature,mass_sha256=digest(m.M.tolist()),compressed_model=models[5].identity,
           full_model=models[7].identity,scope=dict(max_dt_s=.0125,peak_m=.005,fiber_angle_degrees=45.,material='unchanged Hencky+fiber',
               geometry='parent original144',mass_order=5,cycle_end_s=1.6,sentinels=[.5,.6,1.1]),
           scope_limit='sampled states and registered cycle; q5 convergence alone does not establish accuracy')
    write(run/'S4/qualification.json',q)
    if not q['sufficient_qualified']:raise ValueError('full q7 is insufficient; do not enable q5')


def prepare(run):
    run=Path(run)
    register(run,'S4/runtime-protocol.json',dict(max_steps=4,
        faults=['q5 before_material','q5 after_prepare','q5 before_commit','both rules fail','programming KeyError','disk before_pointer'],
        expected='owned values rollback; classified retries only; full sticky; one generation; actual CLI restart'))
    for name,policy in [('retry-restart','q5_with_full_retry'),('fixed-q5-short','q5_fixed')]:
        create_config(run,name,dt=.0125,end=.05,rule_policy=policy)
    outside=create_config(run,'outside-dt-scope',dt=.025,end=.05,rule_policy='q5_with_full_retry')
    assert outside['post_release']['rule_policy']=='full_only'
    assert outside['material_order']==7


def faults(run):
    from .provenance import snapshot
    run=Path(run);folder=run/'cases/retry-restart';cfg=read(folder/'execution-protocol.json');model,_=load_model(run,cfg)
    controller=make_stepper(run,cfg,model,model.rest());full=controller.full
    results=[]
    for where in ['before_material','after_prepare','before_commit']:
        c=RuntimeRetry(model,full,cfg);base=c.state
        def fail(attempt,stage,state):
            if attempt==0 and stage==where:raise RetryableRuleError('controlled q5 rejection at '+where)
        def children(state,values):values['owned_child']={'pressure':[.1,.2],'time':state.time};return values
        row=c.step(.0125,inject=fail,prepare_children=children)
        assert row['material_attempts']==2 and c.state.child_states['identity']==full.identity
        assert c.state.child_states['owned_child']['time']==.0125 and len(c.failures)==1
        ref=ValidatedAVF(full,cfg);ref.step(.0125,prepare_children=children)
        difference=max(float(np.max(abs(c.state.q-ref.state.q))),float(np.max(abs(c.state.velocity-ref.state.velocity))))
        assert difference<1e-8
        nextrow=c.step(.0125);assert nextrow['material_attempts']==1
        assert len(c.state.child_states['material_failure_history'])==1
        results.append(dict(fault=where,accepted_attempts=2,sticky_full=True,owned_children=True,q_v_difference=difference))
    for label,exc_type in [('both_rules_fail',RetryableRuleError),('programming_error',KeyError)]:
        c=RuntimeRetry(model,full,cfg);before=c.state.digest()
        def reject(attempt,stage,state):
            if stage=='after_prepare':raise exc_type(label)
        try:c.step(.0125,inject=reject)
        except StepRejected:pass
        else:raise AssertionError('failure should reject')
        assert c.state.digest()==before
        assert len(c.failures)==(2 if label=='both_rules_fail' else 1)
        results.append(dict(fault=label,base_unchanged=True,attempts=len(c.failures)))
    # Actual main-case checkpoint seeded by the same publication helper that
    # the daily CLI uses, then a different process will resume the full rule.
    c=RuntimeRetry(model,full,cfg);identity=case_identity(run,cfg,model,c.state);identity['controller']=c.identity
    snapshot(folder/'source',identity['numerical_source_sha256']);write(folder/'identity.json',identity)
    store=GenerationStore(folder,identity);store.save(c.state,[])
    before=c.state.digest();pointer=read(store.pointer)
    def fail_write(stage):
        if stage=='before_pointer':raise OSError('controlled disk prepublication failure')
    try:advance_publish(c,store,[],.0125,inject_store=fail_write)
    except OSError:pass
    else:raise AssertionError('disk failure not raised')
    assert c.state.digest()==before and read(store.pointer)==pointer
    assert len(store.history())==1
    def first_fail(attempt,stage,state):
        if attempt==0 and stage=='after_prepare':raise RetryableRuleError('controlled checkpoint restart on full rule')
    row=advance_publish(c,store,[],.0125,inject_step=first_fail)
    assert row['material_attempts']==2 and len(store.history())==2
    write(run/'S4/restart-witness.json',dict(state=c.state.to_dict(),digest=c.state.digest(),pointer=read(store.pointer),
        controller=identity['controller'],actual_rule='q7',source=identity['numerical_source_sha256']))
    reference=ValidatedAVF(full,cfg)
    for _ in range(4):reference.step(.0125)
    np.savez_compressed(run/'S4/uninterrupted-full.npz',q=reference.state.q,v=reference.state.velocity,predictor=reference.state.predictor)
    write(run/'S4/fault-results.json',dict(status='passed_scoped',faults=results,
        disk_prepublication_rollback=True,orphan_excluded=True,first_step_full_retry_published=True,
        programming_errors_do_not_retry=True,scope_fallback_to_full=True))


def restart_check(run):
    run=Path(run);folder=run/'cases/retry-restart';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg)
    c=make_stepper(run,cfg,m,m.rest());store=GenerationStore(folder,read(folder/'identity.json'))
    loaded=store.load(validator=c.validate);state=loaded['state'];hist=store.history()
    assert len(hist)==5 and state.step==4 and state.child_states['identity']==c.full.identity
    with np.load(run/'S4/uninterrupted-full.npz',allow_pickle=False) as z:
        errors={key:float(np.max(abs(getattr(state,key)-z[zkey]))) for key,zkey in [('q','q'),('velocity','v'),('predictor','predictor')]}
    assert max(errors.values())<1e-8
    assert len(state.child_states['material_failure_history'])==1
    # The first rejected compressed energy baseline is explicit in the ledger.
    for row in loaded['rows']:
        assert abs(row['delta_total_including_rule_J']-row['delta_total_J']-row['rule_switch_energy_J'])<1e-12
    write(run/'S4/restart-check.json',dict(status='passed_scoped',actual_new_process=True,active_rule='q7',
        history_preserved=True,errors=errors,generations=len(hist),unique_accepted_steps=4))
    print('RULE_RESTART',errors,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['qualification','prepare','faults','restart-check']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'qualification':qualification,'prepare':prepare,'faults':faults,'restart-check':restart_check}[a.phase](a.run)
