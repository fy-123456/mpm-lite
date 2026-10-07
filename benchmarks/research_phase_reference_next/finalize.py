"""One final sufficient cycle, conditional daily compression, immutable 32-step seal."""
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
    for name in ['S2/space-decision.json','S1/time-decision.json','S3/performance-decision.json','S4/coupling-decision.json','S5/window-runtime-final.json']:
        if not (run/name).exists():raise ValueError('missing decision '+name)
    if read(run/'S5/window-runtime-final.json')['numeric_sources']!=source_files():raise ValueError('window evidence needs final-source refresh')
    times=read(run/'S1/time-decision.json')['times']
    modules=['tests.research_spatial_phase_next.test_runtime','tests.research_cost_phase_next.test_runtime.EntryTests','tests.research_phase_stress_next.test_runtime','tests.research_basis_allocation_next.test_runtime',
        'tests.research_local_span_next.test_runtime','tests.research_local_span_next.test_rt0','tests.research_local_span_next.test_reuse','tests.research_phase_reference_next.test_events','tests.research_phase_reference_next.test_scaling']
    log=run/'S6/final-tests.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(x) for x in modules]))
    write(run/'S6/final-tests-result.json',dict(status='passed' if result.wasSuccessful() else 'failed',count=result.testsRun,log='S6/final-tests.log',log_sha256=sha(log),numerical_source_sha256=source_files(),modules=modules))
    if not result.wasSuccessful():raise ValueError('targeted regression failed; inspect final-tests.log')
    register(run,'S6/final-model-lock.json',dict(numerical_source_sha256=source_files(),selected_space=read(run/'selected-space.json'),times=times,
        performance=read(run/'S3/performance-decision.json'),max_full_cycles=1,max_daily_cycles=1,end_s=1.6,max_frames=12,steps=len(times)-1))
    write(run/'S6/evidence-reuse.json',dict(new_space=False,new_time_references=True,time_unchanged=True,ancestral_operators_unchanged=True,
        current_model_qualifications=['S2/reference-convergence.json','S2/generalization-check.json','S5/window-runtime-final.json','S3/operator-equivalence.json','S4/coupled-eight-step.json'],
        old_case_identity_rewrite=False,continuum_space_time_certification=False))
    create_config(run,FULL,times=times,end=1.6,rule_policy='full_only',field_cache=True,display_frames=12)
    print('FINAL_PREPARED',result.testsRun,len(times)-1,flush=True)

def daily_prepare(run):
    run=Path(run);q=read(run/'S6/qualification-final.json');times=read(run/'S1/time-decision.json')['times'];qualified=q['qualified']
    register(run,'S6/final-protocol.json',dict(default_case=DEFAULT if qualified else FULL,full_case=FULL,steps=len(times)-1,frames=12,times=times,final_numeric_sha256=source_files(),q5_qualified=qualified))
    if qualified:
        create_config(run,DEFAULT,times=times,end=1.6,rule_policy='q5_with_full_retry',field_cache=True,display_frames=12)
        create_config(run,'final-retry-restart',end=times[2],times=times[:3],rule_policy='q5_with_full_retry',field_cache=True,display_frames=3)
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
    parent=hist(APP/'cases/final-full');coordinate_errors={k:max(0. if getattr(x['state'],k) is None and getattr(y['state'],k) is None else float(np.max(abs(getattr(x['state'],k)-getattr(y['state'],k)))) for x,y in zip(b,parent)) for k in ('q','velocity','predictor')}
    if len(parent)!=len(b) or max(coordinate_errors.values())>1e-8:raise ValueError('unchanged solid model/time differs from direct parent')
    write(run/'S6/parent-trajectory-comparison.json',dict(status='passed_scoped_same_solid_space_time',parent_release_sha256=APP_SHA,errors=coordinate_errors,
        unchanged_space=True,time_unchanged=True,continuum_accuracy=False))
    write(run/'S6/final-scene.json',dict(status='passed_scoped',default_case=default,steps=len(a)-1,frames=frames,summary=summary,comparison_maxima=maxima,
        fallback_count=sum(r.get('material_attempts',1)-1 for r in a[-1]['rows']),sentinels=sentinels,spatial_accuracy=False,temporal_accuracy=False,resources=resources(),source_sha256=source_files()))
    print('FINAL_SCENE',maxima,summary['this_segment'],flush=True)

def seal(run):
    run=Path(run);lock=verify(run);protocol=read(run/'S6/final-protocol.json');default=protocol['default_case']
    required=['S4/config-correction-check.json','S2/reference-displacement-reaction.json','S4/original-equation-equivalence.json','S6/raw-physical-check.json','S4/coupled-eight-step.json','S4/scaling-validation.json','S6/final-scene.json','S5/window-runtime-final.json','S4/coupled-closure.json','S4/rollback-restart.json','S4/fixed-common-time.json','S6/physical-review.json']
    if protocol['q5_qualified']:required+=['S6/cache-retry-check.json','S5/fault-and-restart.json']
    for name in required:
        if read(run/name)['status']!='passed_scoped':raise ValueError('missing final gate '+name)
    if read(run/'S6/final-model-lock.json')['numerical_source_sha256']!=source_files() or read(run/'S6/final-tests-result.json')['numerical_source_sha256']!=source_files():raise ValueError('final numerical source changed')
    stage={
      'S0':('passed_scoped',['S0/version-audit.json','S0/compatibility.json','S0/protocol.json','S0/storage-migration.json']),
      'S1':('finer_reference_dt_retained',['S1/event-roundoff-audit.json','S1/reference-resolution.json','S1/event-resolution.json','S1/impulse-check.json','S1/time-decision.json']),
      'S2':('reference_R5_extended_formal_unchanged',['S2/reference-audit.json','S2/refinement-protocol.json','S2/reference-convergence.json','S2/generalization-check.json','S2/current-vs-R5.json','S2/space-decision.json']),
      'S3':('measured_retained_current_implementation',['S3/current-startup-profile.json','S3/current-warm-profile.json','S3/candidate-protocol.json','S3/operator-equivalence.json','S3/performance-decision.json']),
      'S4':('scaled_actual_RT0_eight_steps',['S4/scaling-protocol.json','S4/scaling-validation.json','S4/config-correction-check.json','S4/original-equation-equivalence.json','S4/fixed-common-time.json','S4/coupled-closure.json','S4/rollback-restart.json','S4/coupled-eight-step.json']),
      'S5':('continuous_eight_step_window_qualified',['S5/evidence-vs-permission.json','S5/continuous-window-qualification.json','S5/window-runtime-final.json','S5/junction-check.json','S5/scope-boundaries.json','S5/fault-and-restart.json']),
      'S6':('passed_final_source',['S6/final-tests-result.json','S6/qualification-final.json','S6/final-scene.json','S6/physical-review.json'])}
    overrides={'S1.5':'not_adopted_unloading_high_frequency_event_count_unresolved','S2.2':'one_R5_level_sufficient_no_second_level_triggered',
        'S3.2':'no_candidate_with_justified_10percent_gain','S3.3':'no_new_performance_candidate_no_speedup_claim'}
    steps=[]
    for code,title in re.findall(r'^\*\*(S\d+\.\d+)｜(.+?)\*\*',(run/'plan-frozen.md').read_text(),re.M):
        status,evidence=stage[code.split('.')[0]]
        for name in evidence:
            if not (run/name).is_file():raise ValueError('missing evidence '+name)
        steps.append(dict(step=code,title=title,status=overrides.get(code,status),evidence=evidence))
    if len(steps)!=32:raise ValueError('32-step closure incomplete')
    write(run/'requirement-audit.json',dict(count=32,steps=steps,all_steps_accounted_for=True,original_plan_sha256=lock['plan_sha256'],
        deviations=['formal solid time grid retained: mode523 event counts differ in unloading window despite small field differences',
        'one R5 reference level suffices under registered conditional budget; no R6 solve',
        'material k_f=220 is a pre-registered variation, not claimed historically unseen',
        'material static solves use original stiffness only as quasi-Newton search matrix, with new physical energy/force and unchanged original Ks',
        'pressure fixture inactive constructor flag corrected to match its actual unchanged SegmentedModel; old source snapshot preserved, final declaration checked by zero-step reload/field equivalence',
        'final main-certificate rollback faults executed after its S6 qualification to avoid using stale permission']))
    perf=read(run/'S3/performance-decision.json');space=read(run/'selected-space.json');scene=read(run/'S6/final-scene.json')
    caps=dict(scene_stability='passed_scoped',selected_space=space['selected'],functions=144,mass_order=7,steps=scene['steps'],dt_s=.0125,
        spatial_accuracy=False,temporal_accuracy=False,scoped_static_accuracy='empirical R5 main .005 and registered k_f=220 material variation',
        material_scope='main .005 final grid; continuous .0075 .6-.7 eight-step window with authenticated sufficient start',sensitive_full_cycle_q5=False,
        shared_reduction=True,uncompressed_raw_archive=True,field_cache=True,coarse_instrumentation=True,reuse_transpose_buffers=perf['reuse_transpose_buffers'],tangent_cache=False,
        pressure_scaling_in_actual_solver=True,two_cell_RT0_coupled_eight_steps=True,pressure_spatial_accuracy=False,pressure_general_monotonicity=False,production_C_E_integration=False,coupled_q5=False,pure_solid_default=True)
    write(run/'capability-matrix.json',caps);write(run/'S6/final-resources.json',dict(**resources(),new_results_on_data_disk=True,migration='see S0/resources and progress'))
    write(run/'S3/transaction-check.json',dict(status='passed_final_source',evidence=['S5/window-runtime-final.json','S6/cache-retry-check.json'],reuse=perf['reuse_transpose_buffers']))
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes());write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    sources=all_sources();snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(path):return dict(path=path,sha256=sha(run/path))
    pub=dict(schema='phase-reference-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),
        default_case=default,sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),qualification=entry('S6/qualification-final.json'),window_qualification=entry('S5/continuous-window-qualification.json'),
        scene=entry('S6/final-scene.json'),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),case_identity_sha256=sha(run/'cases'/default/'identity.json'),case_protocol_sha256=sha(run/'cases'/default/'execution-protocol.json'),
        dt_s=.0125,steps=scene['steps'],display_frames=scene['frames'],mass_order=7,full_material_order=7,material_policy='q5_with_full_retry' if protocol['q5_qualified'] else 'full_only',spatial_accuracy=False,temporal_accuracy=False,coupling_scope='two-cell scaled full-tensor RT0 eight-step fixture; not general production coupling')
    check(ROOT,sources);check(run/'final-source',sources);check(run,artifacts);audit_parent(True);write(run/'release.json',pub);_,counts=audit_release(run);print('SEALED',run,sha(run/'release.json'),counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','daily-prepare','cache-seed','cache-check','analyze','seal']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; use audit')
        {'prepare':prepare,'daily-prepare':daily_prepare,'cache-seed':cache_seed,'cache-check':cache_check,'analyze':analyze,'seal':seal}[a.phase](a.run)
