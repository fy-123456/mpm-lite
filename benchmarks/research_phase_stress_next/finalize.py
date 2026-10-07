"""Final numerical freeze, matched cycles, and immutable 30-step release closure."""
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
DEFAULT='daily-q5-retry-dt0125'
FULL='final-q7-dt0125'

def hist(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()


def prepare(run):
    run=Path(run);verify(run)
    for name in all_sources():ast.parse((ROOT/name).read_text(),filename=name)
    for name in ['S1/time-decision.json','S2/space-decision.json','S3/performance-decision.json','S4/coupling-decision.json']:
        if not (run/name).exists():raise ValueError('missing decision '+name)
    if read(run/'S1/time-decision.json')['dt_s']!=.0125:raise ValueError('adapt final explicit grid before proceeding')
    register(run,'S5/final-model-lock.json',dict(numerical_source_sha256=source_files(),selected_space=read(run/'selected-space.json'),
        time=read(run/'S1/time-decision.json'),performance=read(run/'S3/performance-decision.json')['warm_adopted'],
        max_full_cycles=1,max_daily_cycles=1,end_s=1.6,max_frames=12,steps=128))
    write(run/'S5/evidence-reuse.json',dict(inherited=['all ancestor physical invariants','R4 scoped static','unchanged swap6 full mass','sensitive real q7 prefix to .5'],
        new=['final-source full and daily cycles','q7/q8 and q5/q7 actual final states','two sensitive unloading windows','explicit-grid small-storage transactions'],
        no_old_checkpoint_identity_rewrite=True))
    log=run/'S6/final-tests.log';log.parent.mkdir(exist_ok=True)
    modules=['tests.research_spatial_phase_next.test_runtime','tests.research_cost_phase_next.test_runtime.EntryTests','tests.research_phase_stress_next.test_runtime']
    with log.open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(x) for x in modules]))
    if not result.wasSuccessful():raise ValueError('targeted regression failed; see final-tests.log')
    write(run/'S6/final-tests-result.json',dict(status='passed',count=result.testsRun,log='S6/final-tests.log',log_sha256=sha(log),numerical_source_sha256=source_files(),modules=modules))
    create_config(run,FULL,dt=.0125,end=1.6,rule_policy='full_only',field_cache=True)


def daily_prepare(run):
    run=Path(run)
    if not read(run/'S5/qualification-final.json')['qualified']:raise ValueError('use full-only final case when q5 is unqualified')
    register(run,'S6/final-protocol.json',dict(default_case=DEFAULT,full_case=FULL,steps=128,frames=12,dt=.0125,
        final_numeric_sha256=source_files(),solid_retry_steps=2,pressure_restart='already completed S4 under same numeric sources'))
    create_config(run,DEFAULT,dt=.0125,end=1.6,rule_policy='q5_with_full_retry',field_cache=True)
    create_config(run,'final-retry-restart',dt=.0125,end=.025,rule_policy='q5_with_full_retry',field_cache=True)


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
    run=Path(run);cfg=read(run/'cases'/FULL/'execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m,tuple(cfg['probe_shape']))
    for case in (DEFAULT,FULL,'final-retry-restart','coupled-small2'):
        if read(run/'cases'/case/'identity.json')['numerical_source_sha256']!=source_files():raise ValueError('final numerical identity changed: '+case)
    if read(run/'S6/final-tests-result.json')['numerical_source_sha256']!=source_files():raise ValueError('test numeric source differs')
    a,b=hist(run/'cases'/DEFAULT),hist(run/'cases'/FULL)
    if len(a)!=129 or len(b)!=129 or a[-1]['state'].time!=1.6:raise ValueError('incomplete final cycle')
    weights=regions(cache.X);direction=m.parent.params.fiber_direction;records=[];maxima={k:0. for k in ('displacement_m','velocity_m_s','PK1_Pa','fiber_Pa','reaction_N')};passed=True
    for aa,bb in zip(a,b):
        if abs(aa['state'].time-bb['state'].time)>1e-12:raise ValueError('comparison times differ')
        fa,fb=cache.frame(aa['state']),cache.frame(bb['state']);data={}
        for region,w in weights.items():
            data[region]={}
            for key,field,at in [('displacement_m','x',5e-5),('velocity_m_s','velocity',1e-4),('PK1_Pa','PK1',.02)]:
                av,bv=fa[field],fb[field]
                if field=='x':av=av-fa['X'];bv=bv-fb['X']
                err=metric(av,bv,at,.05,w);data[region][key]=err;maxima[key]=max(maxima[key],err['absolute']);passed&=err['passed']
            av=np.einsum('i,...ij,j->...',direction,fa['PK1'],direction);bv=np.einsum('i,...ij,j->...',direction,fb['PK1'],direction)
            err=metric(av,bv,.02,.05,w);data[region]['fiber_Pa']=err;maxima['fiber_Pa']=max(maxima['fiber_Pa'],err['absolute']);passed&=err['passed']
        records.append(dict(time_s=aa['state'].time,regions=data))
    reaction=[]
    for aa,bb in zip(a[-1]['rows'],b[-1]['rows']):
        if abs(aa['dt']-bb['dt'])>1e-12:raise ValueError('reaction interval differs')
        err=metric(aa['reaction_N'],bb['reaction_N'],1e-4,.05);passed&=err['passed'];maxima['reaction_N']=max(maxima['reaction_N'],err['absolute']);reaction.append(dict(time=aa['time'],**err))
    if not passed:raise ValueError('final q5 vs sufficient fields outside practical budgets')
    sentinels=[r['material_sentinel'] for r in a[-1]['rows'] if r.get('material_sentinel') is not None]
    if len(sentinels)!=3 or not all(x['passed'] for x in sentinels):raise ValueError('registered sentinels missing/failed')
    write(run/'S6/final-field-comparison.json',dict(status='passed_scoped',maxima=maxima,records=records,reaction_intervals=reaction,
        scope='matched selected space/time q5 vs q7; no time or continuum spatial certification'))
    previous=hist(APP/'cases'/DEFAULT)
    if read(APP/'cases'/DEFAULT/'identity.json')['model']!=read(run/'cases'/DEFAULT/'identity.json')['model']:
        raise ValueError('same-space performance comparison requires identical physical model')
    state_errors={k:max(float(np.max(abs(getattr(x['state'],k)-getattr(y['state'],k)))) for x,y in zip(a[1:],previous[1:])) for k in ('q','velocity','predictor')}
    write(run/'S6/parent-trajectory-comparison.json',dict(status='passed_scoped' if max(state_errors.values())<1e-8 else 'different',
        state_errors=state_errors,physical_model_unchanged=True,parent_case_identity_sha256=sha(APP/'cases'/DEFAULT/'identity.json')))
    if max(state_errors.values())>1e-8:raise ValueError('equivalent implementation changed parent trajectory')
    frames=sum((x['folder']/'frame.npz').exists() for x in a)
    if frames!=12:raise ValueError('display frame budget differs')
    summary=read(run/'cases'/DEFAULT/'summary.json');counts=audit_parent(True)[2]
    write(run/'S6/final-scene.json',dict(status='passed_scoped',default_case=DEFAULT,steps=128,frames=frames,generations=len(a),summary=summary,
        comparison_maxima=maxima,sentinels=sentinels,fallback_count=sum(r['material_attempts']-1 for r in a[-1]['rows']),
        source_verified=True,parents_unchanged=counts,temporal_accuracy=False,spatial_accuracy=False,resources=resources()))
    print('FINAL_SCENE',maxima,summary['this_segment'],flush=True)


def seal(run):
    run=Path(run);lock=verify(run)
    for path in ['S6/final-scene.json','S6/cache-retry-check.json','S4/coupled-checks.json','S4/publication-check.json','S5/retry-restart.json']:
        if read(run/path)['status']!='passed_scoped':raise ValueError('missing final gate '+path)
    tests=read(run/'S6/final-tests-result.json')
    if tests['numerical_source_sha256']!=source_files() or sha(run/tests['log'])!=tests['log_sha256']:raise ValueError('test source changed')
    if read(run/'S5/final-model-lock.json')['numerical_source_sha256']!=source_files():raise ValueError('final numeric freeze changed')
    for case in (DEFAULT,FULL,'final-retry-restart','coupled-small2'):
        if read(run/'cases'/case/'identity.json')['numerical_source_sha256']!=source_files():raise ValueError('case numeric source changed')
    if not (run/'S6/physical-review.json').exists():raise ValueError('physical image review required')
    definitions={
      'S0':('passed_scoped',['S0/version-audit.json','S0/compatibility.json','S0/resources.json']),
      'S1':('retained_dt_joint_gate_not_met',['S1/time-decision.json','S1/observable-modal-contribution.json','S1/reference-resolution.json','S1/window-comparison.json']),
      'S2':('retained_swap6_no_resolved_gain',['S2/reference-reload.json','S2/span-diagnostic.json','S2/main-decision.json','S2/space-decision.json','S2/dynamic-qualification.json']),
      'S3':('adopted_coarse_instrumentation',['S3/warm-profile.json','S3/performance-decision.json','S3/operator-equivalence.json','S3/transaction-check.json']),
      'S4':('passed_scoped_explicit_small_storage_grid',['S4/small-storage-diagnostic.json','S4/explicit-grid-check.json','S4/coupled-checks.json','S4/publication-check.json','S4/coupling-decision.json']),
      'S5':('main_qualified_sensitive_windows_only',['S5/final-model-lock.json','S5/qualification-final.json','S5/retry-restart.json','S5/sensitive-scope.json']),
      'S6':('passed_final_source',['S6/final-tests-result.json','S6/final-scene.json','S6/physical-review.json','S6/cache-retry-check.json'])}
    steps=[]
    for code,title in re.findall(r'^\*\*(S\d+\.\d+)｜(.+?)\*\*',(run/'plan-frozen.md').read_text(),re.M):
        status,evidence=definitions[code.split('.')[0]]
        if code=='S2.5':status='condition_not_triggered_heldout_unaccessed'
        if code=='S2.6':status='inherited_same_space_no_new_dynamic_prefix'
        for path in evidence:
            if not (run/path).is_file():raise ValueError('missing evidence '+path)
        steps.append(dict(step=code,title=title,status=status,evidence=evidence))
    if len(steps)!=30:raise ValueError('30-step audit incomplete')
    write(run/'requirement-audit.json',dict(count=30,steps=steps,all_steps_accounted_for=True,original_plan_sha256=lock['plan_sha256'],
        limits=['time and continuum space accuracy not certified','only one stress candidate; second and heldout conditions not triggered',
                'midpoint pressure positivity only fixed solid no source; general 3D pressure not certified','sensitive .0075 full cycle remains full_only']))
    small=read(run/'S4/coupling-decision.json')['small_storage_dt_s'];perf=read(run/'S3/performance-decision.json')
    caps=dict(scene_stability='passed_scoped_128_steps',selected_space=read(run/'selected-space.json')['selected'],functions=144,mass_order=7,dt_s=.0125,
        spatial_accuracy=False,temporal_accuracy=False,reference='reloaded empirical R4; F45 .005 static only',
        material_scope='main .005 exact grid q5; .0075 inherited .5-.55 and new .6-.65/.7-.75 short windows only',
        sensitive_full_cycle_q5=False,shared_reduction=True,uncompressed_raw_archive=True,field_cache=True,
        coarse_instrumentation=perf['warm_adopted'],tangent_cache=False,default_scene='pure solid',
        rollback_restart='final-source solid and explicit-grid fluid passed',physical_3D_solid_coupling=True,
        pressure_monotonicity_scope=dict(fixed_solid=True,source=0.,normal_storage=.2,normal_dt_s=.01,small_storage=.0002,
            small_dt_s=small,cells2_and4=True,small4_dt_s=small/4,steps=4),
        pressure_general_monotonicity=False,pressure_spatial_accuracy=False,production_C_E_integration=False)
    write(run/'capability-matrix.json',caps)
    write(run/'S6/final-resources.json',dict(**resources(),new_results_on_data_disk=True,migration='not triggered; system stayed above 5 GiB'))
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),
        current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    sources=all_sources();snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')}
    write(run/'artifact-sha256.json',artifacts)
    def entry(path):return dict(path=path,sha256=sha(run/path))
    pub=dict(schema='phase-stress-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,
        mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),
        default_case=DEFAULT,sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),
        qualification=entry('S5/qualification-final.json'),sensitive_scope=entry('S5/sensitive-scope.json'),scene=entry('S6/final-scene.json'),
        space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),
        case_identity_sha256=sha(run/'cases'/DEFAULT/'identity.json'),case_protocol_sha256=sha(run/'cases'/DEFAULT/'execution-protocol.json'),
        dt_s=.0125,steps=128,display_frames=12,mass_order=7,full_material_order=7,material_policy='q5_with_full_retry',
        coarse_instrumentation=perf['warm_adopted'],shared_reduction=True,space_array_cache=True,
        spatial_accuracy=False,temporal_accuracy=False,coupling_scope='x-only half-cell pressure, normal and small storage four-step fixtures')
    check(ROOT,sources);check(run/'final-source',sources);check(run,artifacts);audit_parent(True);write(run/'release.json',pub)
    _,counts=audit_release(run);print('SEALED',run,sha(run/'release.json'),counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','daily-prepare','cache-seed','cache-check','analyze','seal']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; use audit')
        {'prepare':prepare,'daily-prepare':daily_prepare,'cache-seed':cache_seed,'cache-check':cache_check,'analyze':analyze,'seal':seal}[a.phase](a.run)
