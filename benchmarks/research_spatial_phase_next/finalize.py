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
PROGRESS=ROOT/'docs/MPM_LITE_SPATIAL_PHASE_PROGRESS_20261001_ZH.md'
COUPLING=ROOT/'docs/MPM_LITE_SPATIAL_PHASE_COUPLING_20261001_ZH.md'


def hist(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()


def prepare(run):
    run=Path(run)
    for name in all_sources():ast.parse((ROOT/name).read_text(),filename=name)
    choice=read(run/'selected-space.json');time=read(run/'N2/time-decision.json');rule=read(run/'N3/qualification.json')
    if time['dt_s']!=.0125 or not rule['qualified']:raise ValueError('final fixture must be explicitly adapted to changed time/rule selection')
    for name in ['N1/space-decision.json','N4/coupling-decision.json','N5/performance-decision.json']:
        if not (run/name).exists():raise ValueError('missing decision '+name)
    register(run,'N6/final-protocol.json',dict(default_case=DEFAULT,full_case=FULL,steps=128,frames=12,dt=.0125,space=choice,
        material_policy='q5_with_full_retry',mass_order=choice['mass_order'],full_material_order=choice['full_order'],
        field_cache=True,tangent_cache=False,force_only=False,no_damping=True,
        final_coupled_check='four-step drained fixture, real restart after step2, current cached loader',
        final_solid_retry='two steps, real restart after forced full-rule fallback'))
    log=run/'N6/final-tests.log'
    with log.open('w') as f:result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.research_spatial_phase_next.test_runtime'))
    if not result.wasSuccessful():raise ValueError('targeted regression tests failed; inspect final-tests.log')
    write(run/'N6/final-tests-result.json',dict(status='passed',count=result.testsRun,log='N6/final-tests.log',log_sha256=sha(log),numerical_source_sha256=source_files(),test_source_sha256=sha(ROOT/'tests/research_spatial_phase_next/test_runtime.py')))
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
    write(run/'N6/cache-retry-witness.json',dict(status='passed_first_step',field_max=error,source_sha256=source_files()))


def cache_check(run):
    run=Path(run);folder=run/'cases/final-retry-restart';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);c=make_stepper(run,cfg,m,m.rest());h=hist(folder)
    c.validate(h[-1]['state']);full=ValidatedAVF(c.full,cfg);full.step(.0125);full.step(.0125)
    errors={k:float(np.max(abs(getattr(full.state,k)-getattr(h[-1]['state'],k)))) for k in ('q','velocity','predictor')}
    if max(errors.values())>1e-8 or len(h)!=3 or len(h[-1]['state'].child_states['material_failure_history'])!=1:raise ValueError('final retry restart differs')
    write(run/'N6/cache-retry-check.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,cached_fields_after_full_retry=True,source_sha256=source_files()))


def coupled(run,resume=False):
    from .coupling_study import model_config,source,advance
    from engine.aniso_phase1.research_spatial_phase_next.multicell import MultiCellAVF
    run=Path(run);folder=run/'cases/final-coupled-restart';m,cfg=model_config(run);cache=CachedProbes(m)
    if not resume:
        c=MultiCellAVF(m,cfg,drained=True,reservoir=.002)
        identity=dict(schema='spatial-phase-final-coupled-v1',coupling=c.identity,numerical_source_sha256=source_files(),fixture_source_sha256=sha(Path(__file__)),input_lock_sha256=sha(run/'input-lock.json'))
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
        write(run/'N6/coupled-restart-check.json',dict(status='passed_scoped',actual_new_process=True,source_sha256=source_files(),errors=errors,rows=rows,
            reference='N4 qualified slow-loader four-step trajectory',cached_loader_qualified=True))
    print('FINAL_COUPLED',c.state.step,flush=True)


def analyze(run):
    run=Path(run);cfg=read(run/'cases'/FULL/'execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m,tuple(cfg['probe_shape']))
    for case in (DEFAULT,FULL,'final-retry-restart','final-coupled-restart'):
        if read(run/'cases'/case/'identity.json')['numerical_source_sha256']!=source_files():raise ValueError('final numerical identity changed: '+case)
    if read(run/'N6/final-tests-result.json')['numerical_source_sha256']!=source_files():raise ValueError('test numeric source differs')
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
    write(run/'N6/final-field-comparison.json',dict(status='passed_scoped',maxima=maxima,records=records,reaction_intervals=reaction,
        scope='matched selected space/time q5 vs q7; no time or continuum spatial certification'))
    # Physical old/new differences are reported separately from implementation error.
    from benchmarks.research_sequential_next.model_package import load_reduction
    from engine.aniso_phase1.research_sequential_next.model import PracticalModel
    oldm=PracticalModel(load_reduction(PARENT),order=7,device='cpu');oldcache=CachedProbes(oldm);old={round(x['state'].time,10):x['state'] for x in hist(APP/'cases/final-q7-dt0125')};new={round(x['state'].time,10):x['state'] for x in b};changes=[]
    for t in [.5,.6,1.1,1.6]:
        aa,bb=cache.frame(new[t]),oldcache.frame(old[t]);delta={}
        for k in ('x','velocity','PK1'):delta[k]=float(np.sqrt(np.mean((aa[k]-bb[k])**2)))
        changes.append(dict(time=t,physical_RMS_changes=delta))
    write(run/'N6/space-change.json',dict(status='diagnostic',records=changes,scope='different approximation spaces and mass coordinates; never compare q/v coefficient arrays across spaces',static_gain='N1 independent common-volume metrics'))
    frames=sum((x['folder']/'frame.npz').exists() for x in a)
    if frames!=12:raise ValueError('display frame budget differs')
    summary=read(run/'cases'/DEFAULT/'summary.json');counts=audit_parent(True)[2]
    write(run/'N6/final-scene.json',dict(status='passed_scoped',default_case=DEFAULT,steps=128,frames=frames,generations=len(a),summary=summary,
        comparison_maxima=maxima,sentinels=sentinels,fallback_count=sum(r['material_attempts']-1 for r in a[-1]['rows']),
        source_verified=True,parents_unchanged=counts,temporal_accuracy=False,spatial_accuracy=False,resources=resources()))
    print('FINAL_SCENE',maxima,summary['this_segment'],flush=True)


def seal(run):
    run=Path(run);lock=verify(run)
    if sha(run/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('original frozen plan changed')
    for name in ['N6/final-scene.json','N6/cache-retry-check.json','N6/coupled-restart-check.json','N4/coupling-decision.json']:
        if read(run/name)['status']!='passed_scoped':raise ValueError('acceptance missing: '+name)
    tests=read(run/'N6/final-tests-result.json')
    if tests['numerical_source_sha256']!=source_files() or sha(run/tests['log'])!=tests['log_sha256']:raise ValueError('test evidence changed')
    for case in (DEFAULT,FULL,'final-retry-restart','final-coupled-restart'):
        if read(run/'cases'/case/'identity.json')['numerical_source_sha256']!=source_files():raise ValueError('final numeric case differs')
    if not (run/'visualization'/DEFAULT/'scene-summary.png').is_file() or not (run/'N6/physical-review.json').exists():raise ValueError('missing visual review')
    definitions={
      'N0':('passed_scoped',['N0/version-audit.json','N0/compatibility.json','N0/resources.json']),
      'N1':('promoted_scoped',['N1/reference-use.json','N1/nonlinear-block-scores.json','N1/heldout-result.json','N1/space-decision.json','N1/dynamic-qualification.json','N1/heldout-displacement.json','N1/normalization-check.json']),
      'N2':('retain_dt_temporal_uncertified',['N2/modal-definition.json','N2/excitation-diagnostic.json','N2/window0.json','N2/window1.json','N2/time-decision.json']),
      'N3':('passed_scoped',['N3/full-rule-qualification.json','N3/compressed-rule-qualification.json','N3/fault-results.json','N3/retry-restart.json','N6/cache-retry-check.json']),
      'N4':('passed_small_multicell_scope',['N4/physical-model.json','N4/coupling-kernel-check.json','N4/flux-operator-check.json','N4/mixed-rank-solver.json','N4/limits-and-closure.json','N4/rollback-check.json','N4/coupling-decision.json','N4/flux-transient-diagnostic.json','N6/coupled-restart-check.json','N6/coupled-publication-check.json']),
      'N5':('exact_array_cache_adopted',['N5/baseline-cost.json','N5/performance-decision.json']),
      'N6':('passed_source_bound',['N6/final-protocol.json','N6/final-tests-result.json','N6/final-scene.json','N6/physical-review.json'])}
    steps=re.findall(r'^\*\*(N\d+\.\d+)｜(.+?)\*\*',(run/'plan-frozen.md').read_text(),re.M);audit=[]
    for code,title in steps:
        status,evidence=definitions[code.split('.')[0]]
        for name in evidence:
            if not (run/name).is_file():raise ValueError('missing step evidence '+name)
        audit.append(dict(step=code,title=title,status=status,evidence=evidence))
    if len(audit)!=28:raise ValueError('unexpected plan step count: '+str(len(audit)))
    write(run/'requirement-audit.json',dict(utc=utc(),count=28,steps=audit,all_steps_accounted_for=True,original_plan_sha256=lock['plan_sha256'],
        deviations=['N1 swap12 not promoted; only swap6 opened preregistered heldout',
          'N2 smaller dt improved modal bands but joint field/event gates failed, current dt retained',
          'N4 report-only abs(list) error fixed; accepted numerical states preserved',
          'N4 field audit corrected positions to displacements and L2 arrays to common volume-weighted RMS; initial unweighted report preserved',
          'N5 target changed to measured repeated R3 construction; exact numeric archive, two warm pairs per window and one build timing per variant'],
        unresolved=['continuum stress accuracy target','full-cycle temporal accuracy','general 3D pressure space and production C-E integration']))
    performance=read(run/'N5/performance-decision.json');choice=read(run/'selected-space.json')
    capabilities=dict(scene_stability='passed_scoped_128_steps',selected_space=choice['selected'],functions=144,mass_order=choice['mass_order'],dt_s=.0125,
        spatial_reference='R3 empirical F45 .005 static; heldout .00375 adjacent R2/R3 checked',
        static_interior_PK1_relative=read(run/'N1/candidates/nonlinear-modes-swap6/result.json')['errors']['interior']['PK1']['relative'],
        spatial_accuracy=False,temporal_accuracy=False,material='new-space F45 .005 q5 qualified against q7; q7 checked against q8; max dt .0125',
        sensitive_peak0075_q5=False,field_cache=True,space_array_cache=performance['selected_space_cache'],tangent_cache=False,force_only=False,
        restart_and_rollback='passed_scoped including final source and cached package',physical_3D_coupling=True,
        pressure_dofs=2,four_cell_check=True,pressure_spatial_accuracy=False,pressure_monotonicity=False,production_C_E_integration=False,default_scene='pure solid',
        pressure_transient_limitation='early reverse internal flux and overshoot reproduced by exact fixed-solid semidiscrete solution; limited mixed-operator fixture only')
    write(run/'capability-matrix.json',capabilities);write(run/'N6/final-resources.json',dict(**resources(),new_results_on_data_disk=True,migration='not triggered: system remained above 5 GiB'))
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes());(run/'coupling-interface.md').write_bytes(COUPLING.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),coupling_path=str(COUPLING.relative_to(ROOT)),coupling_sha256=sha(COUPLING),
        current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    sources=all_sources();snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')}
    write(run/'artifact-sha256.json',artifacts)
    def entry(path):return dict(path=path,sha256=sha(run/path))
    pub=dict(schema='spatial-phase-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,
        mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=DEFAULT,
        sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),
        qualification=entry('N3/qualification.json'),sensitive_qualification=entry('N3/qualification-peak0075.json'),scene=entry('N6/final-scene.json'),
        space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),
        case_identity_sha256=sha(run/'cases'/DEFAULT/'identity.json'),case_protocol_sha256=sha(run/'cases'/DEFAULT/'execution-protocol.json'),
        dt_s=.0125,steps=128,display_frames=12,material_policy='q5_with_full_retry',mass_order=7,full_material_order=7,
        field_cache=True,space_array_cache=performance['selected_space_cache'],force_only=False,tangent_cache=False,
        temporal_accuracy=False,spatial_accuracy=False,physical_3D_coupling=True,coupling_scope='independent conservative two-pressure-cell four-step fixture; nonmonotone early drainage transient; no continuum pore-pressure prediction; pure-solid default')
    check(ROOT,sources);check(run/'final-source',sources);check(run,artifacts);audit_parent(True);write(run/'release.json',pub)
    _,counts=audit_release(run);print('SEALED',run,'release_sha256',sha(run/'release.json'),'counts',counts,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cache-seed','cache-check','coupled-seed','coupled-resume','analyze','seal']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; audit through publication CLI')
        actions={'prepare':prepare,'cache-seed':cache_seed,'cache-check':cache_check,'coupled-seed':lambda r:coupled(r,False),
                 'coupled-resume':lambda r:coupled(r,True),'analyze':analyze,'seal':seal}
        actions[a.phase](a.run)
