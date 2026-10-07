"""Final source-bound short restarts, matched solid cycles and release closure."""
from pathlib import Path
import argparse,ast,re,unittest
import numpy as np
from .provenance import ROOT,APP,APP_SHA,PARENT,PLAN,read,write,sha,source_files,all_sources,snapshot,register,verify,audit_parent,resources,serial_lock,utc,check
from .run import create_config,load_model,make_stepper,case_identity
from .publication import audit_release
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_post_release.runtime_rules import RetryableRuleError,advance_publish
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF

DEFAULT='daily-q5-retry-dt0125'
FULL='final-q7-dt0125'
PROGRESS=ROOT/'docs/MPM_LITE_COST_PHASE_PRESSURE_PROGRESS_20261001_ZH.md'
COUPLING=None


def hist(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()


def prepare(run):
    run=Path(run)
    for name in all_sources():ast.parse((ROOT/name).read_text(),filename=name)
    choice=read(run/'selected-space.json');time=read(run/'P2/time-decision.json');rule=read(run/'P5/qualification-final.json')
    if time['dt_s']!=.0125 or not rule['qualified']:raise ValueError('final fixture must be explicitly adapted to changed time/rule selection')
    for name in ['P3/space-decision.json','P4/coupling-decision.json','P1/performance-decision.json']:
        if not (run/name).exists():raise ValueError('missing decision '+name)
    register(run,'P6/final-protocol.json',dict(default_case=DEFAULT,full_case=FULL,steps=128,frames=12,dt=.0125,space=choice,
        material_policy='q5_with_full_retry',mass_order=choice['mass_order'],full_material_order=choice['full_order'],
        field_cache=True,tangent_cache=False,force_only=False,no_damping=True,
        final_coupled_check='four-step drained fixture, real restart after step2, current cached loader',
        final_solid_retry='two steps, real restart after forced full-rule fallback'))
    log=run/'P6/final-tests.log'
    with log.open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName('tests.research_spatial_phase_next.test_runtime'),unittest.defaultTestLoader.loadTestsFromName('tests.research_cost_phase_next.test_runtime.EntryTests')]))
    if not result.wasSuccessful():raise ValueError('targeted regression tests failed; inspect final-tests.log')
    write(run/'P6/final-tests-result.json',dict(status='passed',count=result.testsRun,log='P6/final-tests.log',log_sha256=sha(log),numerical_source_sha256=source_files(),test_source_sha256={name:sha(ROOT/name) for name in ('tests/research_spatial_phase_next/test_runtime.py','tests/research_cost_phase_next/test_runtime.py')}))
    for case,policy,end in [(FULL,'full_only',1.6),(DEFAULT,'q5_with_full_retry',1.6),('final-retry-restart','q5_with_full_retry',.025)]:
        create_config(run,case,dt=.0125,end=end,rule_policy=policy,field_cache=True)


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
    write(run/'P6/cache-retry-witness.json',dict(status='passed_first_step',field_max=error,source_sha256=source_files()))


def cache_check(run):
    run=Path(run);folder=run/'cases/final-retry-restart';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);c=make_stepper(run,cfg,m,m.rest());h=hist(folder)
    c.validate(h[-1]['state']);full=ValidatedAVF(c.full,cfg);full.step(.0125);full.step(.0125)
    errors={k:float(np.max(abs(getattr(full.state,k)-getattr(h[-1]['state'],k)))) for k in ('q','velocity','predictor')}
    if max(errors.values())>1e-8 or len(h)!=3 or len(h[-1]['state'].child_states['material_failure_history'])!=1:raise ValueError('final retry restart differs')
    write(run/'P6/cache-retry-check.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,cached_fields_after_full_retry=True,source_sha256=source_files()))


def coupled(run,resume=False):
    from .coupling_study import model_config,source,advance
    from engine.aniso_phase1.research_cost_phase_next.pressure import MultiCellAVF
    run=Path(run);folder=run/'cases/final-coupled-restart';m,cfg=model_config(run);cache=CachedProbes(m)
    if not resume:
        c=MultiCellAVF(m,cfg,drained=True,reservoir=.002)
        identity=dict(schema='cost-phase-final-coupled-v1',coupling=c.identity,numerical_source_sha256=source_files(),fixture_source_sha256=sha(Path(__file__)),input_lock_sha256=sha(run/'input-lock.json'))
        write(folder/'identity.json',identity);write(folder/'execution-protocol.json',dict(config=cfg,steps=4,dt=.01,coupling=c.identity));snapshot(folder/'source',source_files())
        store=GenerationStore(folder,identity);store.save(c.state,[],frame=c.frame(cache));advance(c,store,[],cache,2)
    else:
        identity=read(folder/'identity.json')
        if identity['numerical_source_sha256']!=source_files() or identity['fixture_source_sha256']!=sha(Path(__file__)):raise ValueError('final coupling source changed')
        store=GenerationStore(folder,identity);data=store.load();c=MultiCellAVF(m,cfg,drained=True,reservoir=.002,state=data['state'])
        if c.state.step!=2:raise ValueError('expected step2 restart')
        rows=advance(c,store,data['rows'],cache,2);old=hist(run/'cases/coupled-drained2')[-1]['state'];errors={}
        for k in ('q','velocity','predictor'):errors[k]=float(np.max(abs(getattr(c.state,k)-getattr(old,k))))
        for k in ('pressure_Pa','content_m3','flux_interval_m3_s','cumulative_source_m3','cumulative_boundary_m3'):
            errors[k]=float(np.max(abs(np.asarray(c.state.child_states['fluid'][k])-old.child_states['fluid'][k])))
        if max(errors.values())>1e-8 or len(store.history())!=5:raise ValueError('cached coupling differs from qualified original loading')
        write(run/'P6/coupled-restart-check.json',dict(status='passed_scoped',actual_new_process=True,source_sha256=source_files(),errors=errors,rows=rows,
            reference='P4 qualified half-cell flux four-step trajectory',cached_loader_qualified=True))
    print('FINAL_COUPLED',c.state.step,flush=True)


def analyze(run):
    run=Path(run);cfg=read(run/'cases'/FULL/'execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m,tuple(cfg['probe_shape']))
    for case in (DEFAULT,FULL,'final-retry-restart','final-coupled-restart'):
        if read(run/'cases'/case/'identity.json')['numerical_source_sha256']!=source_files():raise ValueError('final numerical identity changed: '+case)
    if read(run/'P6/final-tests-result.json')['numerical_source_sha256']!=source_files():raise ValueError('test numeric source differs')
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
    write(run/'P6/final-field-comparison.json',dict(status='passed_scoped',maxima=maxima,records=records,reaction_intervals=reaction,
        scope='matched selected space/time q5 vs q7; no time or continuum spatial certification'))
    previous=hist(APP/'cases'/DEFAULT)
    if read(APP/'cases'/DEFAULT/'identity.json')['model']!=read(run/'cases'/DEFAULT/'identity.json')['model']:
        raise ValueError('same-space performance comparison requires identical physical model')
    state_errors={k:max(float(np.max(abs(getattr(x['state'],k)-getattr(y['state'],k)))) for x,y in zip(a[1:],previous[1:])) for k in ('q','velocity','predictor')}
    write(run/'P6/parent-trajectory-comparison.json',dict(status='passed_scoped' if max(state_errors.values())<1e-8 else 'different',
        state_errors=state_errors,physical_model_unchanged=True,parent_case_identity_sha256=sha(APP/'cases'/DEFAULT/'identity.json')))
    if max(state_errors.values())>1e-8:raise ValueError('equivalent implementation changed parent trajectory')
    frames=sum((x['folder']/'frame.npz').exists() for x in a)
    if frames!=12:raise ValueError('display frame budget differs')
    summary=read(run/'cases'/DEFAULT/'summary.json');counts=audit_parent(True)[2]
    write(run/'P6/final-scene.json',dict(status='passed_scoped',default_case=DEFAULT,steps=128,frames=frames,generations=len(a),summary=summary,
        comparison_maxima=maxima,sentinels=sentinels,fallback_count=sum(r['material_attempts']-1 for r in a[-1]['rows']),
        source_verified=True,parents_unchanged=counts,temporal_accuracy=False,spatial_accuracy=False,resources=resources()))
    print('FINAL_SCENE',maxima,summary['this_segment'],flush=True)


def seal(run):
    run=Path(run);lock=verify(run)
    for path in ['P6/final-scene.json','P6/cache-retry-check.json','P6/coupled-restart-check.json','P4/coupling-decision.json','P4/publication-check.json','P5/retry-restart.json']:
        if read(run/path)['status']!='passed_scoped':raise ValueError('missing final gate '+path)
    tests=read(run/'P6/final-tests-result.json')
    if tests['numerical_source_sha256']!=source_files() or sha(run/tests['log'])!=tests['log_sha256']:raise ValueError('test identity differs')
    for case in (DEFAULT,FULL,'final-retry-restart','final-coupled-restart'):
        if read(run/'cases'/case/'identity.json')['numerical_source_sha256']!=source_files():raise ValueError('final numerical case differs')
    if not (run/'visualization'/DEFAULT/'scene-summary.png').exists() or not (run/'P6/physical-review.json').exists():raise ValueError('visual review missing')
    definitions={
        'P0':('passed_scoped',['P0/version-audit.json','P0/compatibility.json','P0/resources.json']),
        'P1':('adopted_equivalent_cost_changes',['P1/performance-decision.json','P1/compact-representation.json','P1/operator-equivalence.json','P1/transaction-check.json']),
        'P2':('retained_dt_accuracy_uncertified',['P2/reuse-audit.json','P2/phase-cause-analysis.json','P2/time-decision.json']),
        'P3':('reference_improved_candidates_not_promoted',['P3/R4/result.json','P3/reference-decision.json','P3/space-decision.json','P3/dynamic-qualification.json']),
        'P4':('passed_limited_pressure_fixture',['P4/monotonicity-and-time.json','P4/coupling-decision.json','P4/publication-check.json','P4/rollback-check.json']),
        'P5':('main_qualified_sensitive_window_only',['P5/qualification-final.json','P5/qualification-scope.json','P5/retry-restart.json','P5/sensitive-scope.json']),
        'P6':('passed_final_source',['P6/final-tests-result.json','P6/final-scene.json','P6/physical-review.json','P6/cache-retry-check.json','P6/coupled-restart-check.json'])}
    steps=[]
    for code,title in re.findall(r'^\*\*(P\d+\.\d+)｜(.+?)\*\*',(run/'plan-frozen.md').read_text(),re.M):
        status,evidence=definitions[code.split('.')[0]]
        for path in evidence:
            if not (run/path).is_file():raise ValueError('missing step evidence '+path)
        if code=='P3.5':status='inherited_no_physical_space_change'
        steps.append(dict(step=code,title=title,status=status,evidence=evidence))
    if len(steps)!=29:raise ValueError('29-step plan audit incomplete')
    write(run/'requirement-audit.json',dict(count=29,steps=steps,all_steps_accounted_for=True,original_plan_sha256=lock['plan_sha256'],
        decisions=['two measured performance changes, exact arrays and same physical operators',
                   'existing time windows authenticated; proposed segmentation does not pass their existing joint gates',
                   'one R4 reference improves differences; both fixed-budget candidates rejected; new heldout not opened',
                   'new pressure discretization only for zero-transverse x fixture; no general three-dimensional pressure accuracy',
                   '.0075 q5 checked for one actual loaded short window, not a full cycle certificate']))
    caps=dict(scene_stability='passed_scoped_128_steps',selected_space=read(run/'selected-space.json')['selected'],functions=144,mass_order=7,dt_s=.0125,
        spatial_accuracy=False,temporal_accuracy=False,reference='one additional p6 local-h R4; empirical static F45 .005 only',
        material='main F45 .005 actual-grid q5 qualified; .0075 four-step loaded window only',sensitive_full_cycle_q5=False,
        shared_reduction=True,space_archive='exact uncompressed raw CSR',field_cache=True,tangent_cache=False,force_only=False,
        default_scene='pure solid',rollback_restart='passed including final-source numeric archive',
        physical_3D_solid_coupling=True,pressure_dofs=2,four_cell_check=True,pressure_scheme='half-cell conservative face flow, zero transverse flux',
        pressure_monotonicity_scoped=True,pressure_monotonicity_scope='fixed solid, no source, .01 initial/.002 reservoir, normal storage .2, 2/4 cells, dt .01',
        pressure_general_monotonicity=False,pressure_spatial_accuracy=False,production_C_E_integration=False)
    write(run/'capability-matrix.json',caps)
    write(run/'P6/final-resources.json',dict(**resources(),new_results_on_data_disk=True,migration='not triggered; system stayed above 5 GiB'))
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),
        current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    sources=all_sources();snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')}
    write(run/'artifact-sha256.json',artifacts)
    def entry(path):return dict(path=path,sha256=sha(run/path))
    pub=dict(schema='cost-phase-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,
        mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),
        default_case=DEFAULT,sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),
        qualification=entry('P5/qualification-final.json'),sensitive_scope=entry('P5/sensitive-scope.json'),scene=entry('P6/final-scene.json'),
        space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),
        case_identity_sha256=sha(run/'cases'/DEFAULT/'identity.json'),case_protocol_sha256=sha(run/'cases'/DEFAULT/'execution-protocol.json'),
        dt_s=.0125,steps=128,display_frames=12,mass_order=7,full_material_order=7,material_policy='q5_with_full_retry',
        shared_reduction=True,space_array_cache=True,uncompressed_raw_archive=True,spatial_accuracy=False,temporal_accuracy=False,
        coupling_scope='independent normal-storage two/four-cell conservative half-cell flux fixture; no general pressure continuum accuracy')
    check(ROOT,sources);check(run/'final-source',sources);check(run,artifacts);audit_parent(True);write(run/'release.json',pub)
    _,counts=audit_release(run);print('SEALED',run,sha(run/'release.json'),counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cache-seed','cache-check','coupled-seed','coupled-resume','analyze','seal']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; use audit')
        {'prepare':prepare,'cache-seed':cache_seed,'cache-check':cache_check,'coupled-seed':lambda r:coupled(r,False),
         'coupled-resume':lambda r:coupled(r,True),'analyze':analyze,'seal':seal}[a.phase](a.run)
