"""One final sufficient cycle, conditional daily compression, immutable 29-step seal."""
from pathlib import Path
import argparse,ast,re,unittest
import numpy as np
from .provenance import ROOT,APP,APP_SHA,PARENT,PLAN,PROGRESS,read,write,sha,source_files,all_sources,snapshot,register,verify,audit_parent,resources,serial_lock,utc,check
from .run import create_config,load_model,make_stepper,case_identity
from .publication import audit_release
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_post_release.runtime_rules import RetryableRuleError,advance_publish
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
DEFAULT='daily-q5-retry'
FULL='final-full'

def hist(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()

def prepare(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    for name in all_sources():ast.parse((ROOT/name).read_text(),filename=name)
    for name in ['S2/research-space-decision.json','S1/time-decision.json','S5/performance-decision.json','S3/pressure-scope-decision.json']:
        if not (run/name).exists():raise ValueError('missing decision '+name)
    times=read(run/'S1/time-decision.json')['times']
    modules=['tests.research_local_span_next.test_runtime','tests.research_phase_reference_next.test_events','tests.research_phase_reference_next.test_scaling','tests.research_cross_direction_next.test_runtime','tests.research_phase_boundary_next.test_runtime']
    log=run/'S6/final-tests.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(x) for x in modules]))
    write(run/'S6/final-tests-result.json',dict(status='passed' if result.wasSuccessful() else 'failed',count=result.testsRun,log='S6/final-tests.log',log_sha256=sha(log),numerical_source_sha256=source_files(),modules=modules))
    if not result.wasSuccessful():raise ValueError('targeted regression failed')
    register(run,'S6/numeric-lock.json',dict(numerical_source_sha256=source_files(),selected_space=read(run/'selected-space.json'),times=times,performance=read(run/'S5/performance-decision.json'),max_full_cycles=1,max_daily_cycles=1,end_s=1.6,max_frames=12,steps=len(times)-1))
    write(run/'S6/final-model-lock.json',read(run/'S6/numeric-lock.json'))
    create_config(run,FULL,times=times,end=1.6,rule_policy='full_only',field_cache=True,display_frames=12)
    print('FINAL_PREPARED',result.testsRun,len(times)-1,flush=True)


def daily_prepare(run):
    run=Path(run);q=read(run/'S6/qualification-final.json');times=read(run/'S1/time-decision.json')['times'];qualified=q['qualified']
    register(run,'S6/final-protocol.json',dict(default_case=DEFAULT if qualified else FULL,full_case=FULL,steps=len(times)-1,frames=12,times=times,final_numeric_sha256=source_files(),q5_qualified=qualified))
    if qualified:
        create_config(run,DEFAULT,times=times,end=1.6,rule_policy='q5_with_full_retry',field_cache=True,display_frames=12)
        create_config(run,'final-retry-restart',end=times[2],times=times[:3],rule_policy='q5_with_full_retry',field_cache=True,display_frames=3)
    else:
        write(run/'S5/fault-and-restart.json',dict(status='not_triggered',reason='q5 not qualified; full-only runtime retained'))
    print('DAILY_PREPARED',qualified,flush=True)

def cache_seed(run):
    run=Path(run);folder=run/'cases/final-retry-restart';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);c=make_stepper(run,cfg,m,m.rest())
    identity=case_identity(run,cfg,m,c.state);identity['controller']=c.identity;snapshot(folder/'source',identity['numerical_source_sha256']);write(folder/'identity.json',identity)
    cache=CachedProbes(m,tuple(cfg['probe_shape']));store=GenerationStore(folder,identity);store.save(c.state,[],frame=cache.frame(c.state))
    def fail(attempt,where,state):
        if attempt==0 and where=='after_prepare':raise RetryableRuleError('controlled final cached retry')
    row=advance_publish(c,store,[],.0125,frame_builder=cache.frame,inject_step=fail)
    from benchmarks.research_sequential_next.run import probe_frame
    a,b=cache.frame(c.state),probe_frame(c.full,c.state,cfg['probe_shape']);error=max(float(np.max(abs(a[k]-b[k]))) for k in a)
    if error>1e-7 or row['material_attempts']!=2:raise ValueError('cached sufficient-rule frame differs')
    write(run/'S6/cache-retry-witness.json',dict(status='passed_first_step',field_max=error,source_sha256=source_files()))


def cache_check(run):
    run=Path(run);folder=run/'cases/final-retry-restart';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);c=make_stepper(run,cfg,m,m.rest());h=hist(folder)
    c.validate(h[-1]['state']);full=ValidatedAVF(c.full,cfg);full.step(.0125);full.step(.0125)
    errors={k:float(np.max(abs(getattr(full.state,k)-getattr(h[-1]['state'],k)))) for k in ('q','velocity','predictor')}
    if max(errors.values())>1e-8 or len(h)!=3 or len(h[-1]['state'].child_states['material_failure_history'])!=1:raise ValueError('final retry restart differs')
    write(run/'S6/cache-retry-check.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,cached_fields_after_full_retry=True,source_sha256=source_files()))

def analyze(run):
    run=Path(run);protocol=read(run/'S6/final-protocol.json');default=protocol['default_case'];cfg=read(run/'cases'/FULL/'execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m,tuple(cfg['probe_shape']))
    for case in (default,FULL):
        if read(run/'cases'/case/'identity.json')['numerical_source_sha256']!=source_files():raise ValueError('final source drift')
    a,b=hist(run/'cases'/default),hist(run/'cases'/FULL);times=protocol['times']
    if len(a)!=len(times) or len(b)!=len(times) or abs(a[-1]['state'].time-1.6)>1e-12:raise ValueError('incomplete cycle')
    weights=regions(cache.X);direction=m.parent.params.fiber_direction;records=[];maxima={k:0. for k in ['displacement_m','velocity_m_s','PK1_Pa','fiber_Pa','reaction_N']};passed=True
    for aa,bb,ts in zip(a,b,times):
        if max(abs(aa['state'].time-ts),abs(bb['state'].time-ts))>1e-12:raise ValueError('comparison node mismatch')
        fa,fb=cache.frame(aa['state']),cache.frame(bb['state']);data={}
        for region,w in weights.items():
            data[region]={}
            for key,field,at in [('displacement_m','x',5e-5),('velocity_m_s','velocity',1e-4),('PK1_Pa','PK1',.02)]:
                av,bv=fa[field],fb[field]
                if field=='x':av=av-fa['X'];bv=bv-fb['X']
                err=metric(av,bv,at,.05,w);data[region][key]=err;maxima[key]=max(maxima[key],err['absolute']);passed &=err['passed']
            av=np.einsum('i,...ij,j->...',direction,fa['PK1'],direction);bv=np.einsum('i,...ij,j->...',direction,fb['PK1'],direction);err=metric(av,bv,.02,.05,w);data[region]['fiber_Pa']=err;maxima['fiber_Pa']=max(maxima['fiber_Pa'],err['absolute']);passed &=err['passed']
        records.append(dict(time_s=ts,regions=data))
    reaction=[]
    for aa,bb in zip(a[-1]['rows'],b[-1]['rows']):
        if abs(aa['dt']-bb['dt'])>1e-12:raise ValueError('reaction interval differs')
        err=metric(aa['reaction_N'],bb['reaction_N'],1e-4,.05);passed &=err['passed'];maxima['reaction_N']=max(maxima['reaction_N'],err['absolute']);reaction.append(dict(time_s=aa['time'],**err))
    if not passed:raise ValueError('final material path fields outside practical budgets')
    summary=read(run/'cases'/default/'summary.json');frames=sum((x['folder']/'frame.npz').exists() for x in a)
    if frames>12 or summary['min_detF']<=.1:raise ValueError('scene frame/geometry gate failed')
    sentinels=[r['material_sentinel'] for r in a[-1]['rows'] if r.get('material_sentinel') is not None]
    if protocol['q5_qualified'] and (len(sentinels)!=3 or not all(x['passed'] for x in sentinels)):raise ValueError('missing sentinel evidence')
    write(run/'S6/final-field-comparison.json',dict(status='passed_scoped',maxima=maxima,records=records,reaction_intervals=reaction,scope='same selected space/time; material rule only'))
    parent=hist(APP/'cases/final-full');same=read(run/'selected-space.json')['package']['sha256']==read(run/'baseline-space.json')['package']['sha256']
    errors={}
    if same:
        for key in ('q','velocity','predictor'):
            errors[key]=max(0. if getattr(x['state'],key) is None and getattr(y['state'],key) is None else float(np.max(abs(getattr(x['state'],key)-getattr(y['state'],key)))) for x,y in zip(parent,b))
        if len(parent)!=len(b) or max(errors.values())>1e-8:raise ValueError('unchanged formal trajectory differs')
    write(run/'S6/parent-trajectory-comparison.json',dict(status='passed_scoped' if same else 'new_space_separate_validation',parent_release_sha256=APP_SHA,coordinate_differences=errors,unchanged_space=same,time_unchanged=True,continuum_accuracy=False,local_phase_scope='BASELINE only'))
    write(run/'S6/final-scene.json',dict(status='passed_scoped',default_case=default,steps=len(a)-1,frames=frames,summary=summary,comparison_maxima=maxima,
        fallback_count=sum(r.get('material_attempts',1)-1 for r in a[-1]['rows']),sentinels=sentinels,spatial_accuracy=False,temporal_accuracy=False,resources=resources(),source_sha256=source_files()))
    print('FINAL_SCENE',maxima,summary['this_segment'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','daily-prepare','cache-seed','cache-check','analyze']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; use audit')
        {'prepare':prepare,'daily-prepare':daily_prepare,'cache-seed':cache_seed,'cache-check':cache_check,'analyze':analyze}[a.phase](a.run)
