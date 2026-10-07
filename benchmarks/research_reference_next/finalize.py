"""Final practical acceptance, explicit decisions and immutable source closure."""
from pathlib import Path
import argparse, re, unittest, ast
import numpy as np
from .provenance import ROOT,PARENT,APP,APP_SHA,PLAN,read,write,sha,digest,source_files,all_sources,snapshot,register,verify,audit_parent,resources,serial_lock,utc,check
from .run import create_config,load_model,make_stepper,case_identity
from .publication import audit_release
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.model_package import load_reduction
from engine.aniso_phase1.research_sequential_next.model import PracticalModel
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError,advance_publish

DEFAULT='daily-q5-retry-dt0125'
PROGRESS=ROOT/'docs/MPM_LITE_REFERENCE_NEXT_PROGRESS_20261001_ZH.md'
COUPLING=ROOT/'docs/MPM_LITE_REFERENCE_NEXT_COUPLING_20261001_ZH.md'


def prepare(run):
    run=Path(run)
    for name in all_sources():ast.parse((ROOT/name).read_text(),filename=name)
    for file in ['Q1/reference-decision.json','Q2/space-decision.json','Q3/time-decision.json','Q4/performance-decision.json','Q5/coupling-decision.json']:
        if not (run/file).is_file():raise ValueError('missing decision '+file)
    if read(run/'Q2/space-decision.json')['selected']!='original144':raise ValueError('runner supports selected original144 only')
    register(run,'Q6/final-protocol.json',dict(default_case=DEFAULT,full_case='final-q7-dt0125',steps=128,frames=12,dt=.0125,
        selected_space='original144',rule_policy='q5_with_full_retry',mass_order=5,full_material_order=7,
        field_cache=True,force_only=False,tangent_cache=False,preconditioner='original',
        energy='original separately budgeted solve/path/rule terms',no_damping=True,
        references='Q1',space_decision='Q2/space-decision.json',time_decision='Q3/time-decision.json',coupling='separate four-step single-pressure-cell fixture'))
    log=run/'Q6/final-tests.log'
    with log.open('w') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.research_reference_next.test_runtime'))
    if not result.wasSuccessful():raise ValueError('targeted regression tests failed')
    write(run/'Q6/final-tests-result.json',dict(status='passed',count=result.testsRun,log='Q6/final-tests.log',log_sha256=sha(log),numerical_source_sha256=source_files(),test_source_sha256=sha(ROOT/'tests/research_reference_next/test_runtime.py')))
    create_config(run,'final-q7-dt0125',dt=.0125,rule_policy='full_only',field_cache=True)
    create_config(run,DEFAULT,dt=.0125,rule_policy='q5_with_full_retry',field_cache=True)
    create_config(run,'final-retry-restart',dt=.0125,end=.025,rule_policy='q5_with_full_retry',field_cache=True)


def hist(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()

def cache_seed(run):
    run=Path(run);folder=run/'cases/final-retry-restart';cfg=read(folder/'execution-protocol.json');model,_=load_model(run,cfg)
    c=make_stepper(run,cfg,model,model.rest());identity=case_identity(run,cfg,model,c.state);identity['controller']=c.identity
    snapshot(folder/'source',identity['numerical_source_sha256']);write(folder/'identity.json',identity)
    cache=CachedProbes(model,tuple(cfg['probe_shape']));store=GenerationStore(folder,identity);store.save(c.state,[],frame=cache.frame(c.state))
    def fail(attempt,where,state):
        if attempt==0 and where=='after_prepare':raise RetryableRuleError('controlled cached retry')
    row=advance_publish(c,store,[],.0125,frame_builder=cache.frame,inject_step=fail)
    from benchmarks.research_sequential_next.run import probe_frame
    a,b=cache.frame(c.state),probe_frame(c.full,c.state,cfg['probe_shape'])
    error=max(float(np.max(abs(a[k]-b[k]))) for k in a)
    assert error<1e-7 and row['material_attempts']==2
    write(run/'Q6/cache-retry-witness.json',dict(status='passed_first_step',q7_frame_error=error,source=source_files(),
         pointer=read(store.pointer),digest=c.state.digest(),failures=c.failures))

def cache_check(run):
    run=Path(run);folder=run/'cases/final-retry-restart';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg)
    c=make_stepper(run,cfg,m,m.rest());history=hist(folder);c.validate(history[-1]['state'])
    assert len(history)==3 and history[-1]['state'].step==2
    assert history[-1]['state'].child_states['identity']==c.full.identity
    assert len(history[-1]['state'].child_states['material_failure_history'])==1
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
    full=ValidatedAVF(c.full,cfg)
    full.step(.0125);full.step(.0125)
    errors={k:float(np.max(abs(getattr(full.state,k)-getattr(history[-1]['state'],k)))) for k in ('q','velocity','predictor')}
    assert max(errors.values())<1e-8
    write(run/'Q6/cache-retry-check.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,
         cached_fields_after_q7_fallback=True,history_preserved=True,source_sha256=source_files()))

def analyze(run):
    run=Path(run);identity=read(run/'cases'/DEFAULT/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or read(run/'cases/final-q7-dt0125/identity.json')['numerical_source_sha256']!=source_files():raise ValueError('final numerical source changed')
    tests=read(run/'Q6/final-tests-result.json')
    if tests['numerical_source_sha256']!=source_files():raise ValueError('tests used other final numeric source')
    a=hist(run/'cases'/DEFAULT);b=hist(run/'cases/final-q7-dt0125')
    if len(a)!=129 or len(b)!=129 or a[-1]['state'].time!=1.6:raise ValueError('incomplete final cycle')
    r=load_reduction(PARENT);m=PracticalModel(r,order=7,device='cpu');cache=CachedProbes(m)
    weights=regions(cache.X);direction=m.parent.params.fiber_direction
    records=[];maxima={k:0. for k in ('displacement_m','velocity_m_s','PK1_Pa','fiber_Pa','reaction_N')}
    all_passed=True
    for aa,bb in zip(a,b):
        sa,sb=aa['state'],bb['state']
        if abs(sa.time-sb.time)>1e-12:raise ValueError('comparison times differ')
        fa,fb=cache.frame(sa),cache.frame(sb);data={}
        for region,w in weights.items():
            data[region]={}
            for key,field,absolute in [('displacement_m','x',5e-5),('velocity_m_s','velocity',1e-4),('PK1_Pa','PK1',.02)]:
                av,bv=fa[field],fb[field]
                if field=='x':av=av-fa['X'];bv=bv-fb['X']
                error=metric(av,bv,absolute,.05,w);data[region][key]=error;maxima[key]=max(maxima[key],error['absolute']);all_passed&=error['passed']
            av=np.einsum('i,...ij,j->...',direction,fa['PK1'],direction);bv=np.einsum('i,...ij,j->...',direction,fb['PK1'],direction)
            error=metric(av,bv,.02,.05,w);data[region]['fiber_Pa']=error;maxima['fiber_Pa']=max(maxima['fiber_Pa'],error['absolute']);all_passed&=error['passed']
        records.append(dict(time_s=sa.time,regions=data))
    reactions=[]
    for aa,bb in zip(a[-1]['rows'],b[-1]['rows']):
        if abs(aa['dt']-bb['dt'])>1e-12:raise ValueError('reaction interval mismatch')
        error=metric(aa['reaction_N'],bb['reaction_N'],1e-4,.05);reactions.append(dict(time=aa['time'],**error));all_passed&=error['passed'];maxima['reaction_N']=max(maxima['reaction_N'],error['absolute'])
    if not all_passed:raise ValueError('final q5 scene differs beyond practical field budget')
    rows=a[-1]['rows'];sentinels=[x['material_sentinel'] for x in rows if x.get('material_sentinel') is not None]
    if len(sentinels)!=3 or not all(x['passed'] for x in sentinels):raise ValueError('missing or failed registered material sentinels')
    write(run/'Q6/final-field-comparison.json',dict(status='passed_scoped',reference_case='final-q7-dt0125',records=records,reactions=reactions,maxima=maxima,
           scope='same selected dt/space, q5 vs q7; not time/space accuracy certification'))
    summary=read(run/'cases'/DEFAULT/'summary.json');frames=sum((x['folder']/'frame.npz').exists() for x in a)
    assert frames==12
    counts=audit_parent(full=True)[2]
    result=dict(status='passed_scoped',default_case=DEFAULT,steps=128,generations=len(a),frames=frames,
                source_verified=True,parent_unchanged=counts,scene=summary,comparison_maxima=maxima,sentinels=sentinels,
                fallback_count=sum(x['material_attempts']-1 for x in rows),actual_rules=sorted({x['material_rule'] for x in rows}),
                time_accuracy=False,spatial_accuracy=False,physical_3D_coupling=read(run/'Q5/coupling-decision.json')['physical_3D_coupling'],resources=resources())
    write(run/'Q6/final-scene-result.json',result)
    print('FINAL_SCENE',maxima,summary['this_segment'],flush=True)

def seal(run):
    run=Path(run);lock=verify(run)
    if sha(run/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('original plan snapshot changed')
    for case in (DEFAULT,'final-q7-dt0125','final-retry-restart','coupled-closed','coupled-drained'):
        identity=read(run/'cases'/case/'identity.json')
        if identity['numerical_source_sha256']!=source_files():raise ValueError('final numeric identity differs: '+case)
    tests=read(run/'Q6/final-tests-result.json')
    if tests['numerical_source_sha256']!=source_files() or tests['log_sha256']!=sha(run/tests['log']):raise ValueError('test evidence changed')
    for name in ['Q6/final-scene-result.json','Q6/cache-retry-check.json','Q5/coupling-decision.json']:
        if read(run/name)['status']!='passed_scoped':raise ValueError('final acceptance incomplete: '+name)
    if not (run/'visualization'/DEFAULT/'scene-summary.png').is_file():raise ValueError('missing final visualization')
    definitions={
      'Q0':('passed_scoped',['Q0/baseline-audit.json','Q0/compatibility.json']),
      'Q1':('reference_usable_registered_static_scope',['Q1/reference-decision.json','Q1/reload-check.json','Q1/common-cell-comparisons.json']),
      'Q2':('candidates_not_promoted',['Q2/space-decision.json']),
      'Q3':('retain_dt_temporal_uncertified',['Q3/time-decision.json','Q3/modal-diagnostic.json']),
      'Q4':('qualification_passed_optimization_not_selected',['Q4/qualification.json','Q4/qualification-peak0075.json','Q4/fault-results.json','Q4/restart-check.json','Q4/performance-decision.json']),
      'Q5':('passed_minimal_3D_single_pressure_cell',['Q5/physical-model.json','Q5/coupling-kernel-check.json','Q5/mixed-rank-solver.json','Q5/limits-check.json','Q5/rollback-check.json','Q5/coupling-decision.json']),
      'Q6':('passed_practical_source_bound',['Q6/final-protocol.json','Q6/final-tests-result.json','Q6/cache-retry-check.json','Q6/final-scene-result.json','Q6/physical-review.json'])}
    steps=re.findall(r'^### (Q\d+\.\d+) (.+)$',(run/'plan-frozen.md').read_text(),re.M);audit=[]
    for code,title in steps:
        status,evidence=definitions[code.split('.')[0]]
        if code=='Q2.4':status='conditional_not_required_original144_retained'
        if code=='Q2.3':status='no_resolved_gain_heldout_not_opened'
        if code=='Q4.4':status='optional_force_only_validated_but_disabled_3_percent_gain'
        for name in evidence:
            if not (run/name).is_file():raise ValueError('missing plan evidence '+name)
        audit.append(dict(step=code,title=title,status=status,evidence=evidence))
    if len(audit)!=27:raise ValueError('unexpected plan step count')
    write(run/'requirement-audit.json',dict(utc=utc(),count=len(audit),steps=audit,all_steps_accounted_for=True,
        original_plan_sha256=sha(run/'plan-frozen.md'),scope='bounded plan with nonpromotion/conditional branches',
        deviations=['Q1 display probes supplemented by common-cell volume quadrature to avoid missing narrow supports',
                    'Q1 R3 first attempt exceeded RSS; root cause explicit zeros on unchanged tensor axes fixed; retry below 16 GiB',
                    'Q5 one constant pressure cell and one drain flux; dense general factorization of 650x650 small reference system, no CG/SPD assumption'],
        unresolved=['original144 spatial field accuracy','full-cycle temporal accuracy','resolved pressure space and production C-E integration']))
    capabilities=dict(scene_stability='passed_scoped_128_steps',selected_space='original144',dt_s=.0125,
        local_modal_phase='smaller step improves registered windows; joint field/reaction criteria fail so dt retained',temporal_accuracy=False,
        spatial_reference='R2/R3 reliable for registered F45 .005m static candidate comparison only',spatial_accuracy=False,
        material='q5 qualified fixed original144 F45 .005/.0075m, max dt .0125; .0075 new run limited to four steps',
        full_cycle_peak_m=.005,field_cache=True,tangent_cache=False,preconditioner='original',force_only_default=False,
        optional_force_only='verified identical response/tangent; 3.02% warm microtrial gain not sufficient for default',
        restart_and_rollback='passed_scoped',physical_3D_coupling=True,pressure_dofs=1,flux_dofs=1,
        pressure_spatial_accuracy=False,production_C_E_integration=False,default_scene='pure solid')
    write(run/'capability-matrix.json',capabilities)
    write(run/'Q6/final-resources.json',dict(**resources(),migration='not_triggered: system above 5 GiB',new_results_on_data_disk=True))
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes());(run/'coupling-interface.md').write_bytes(COUPLING.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),
        coupling_path=str(COUPLING.relative_to(ROOT)),coupling_sha256=sha(COUPLING),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),
        original_plan_sha256=lock['plan_sha256'],snapshot_link_base=str(ROOT/'docs')))
    sources=all_sources();snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')}
    write(run/'artifact-sha256.json',artifacts)
    def entry(path):return dict(path=path,sha256=sha(run/path))
    pub=dict(schema='reference-next-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,
        mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),
        default_case=DEFAULT,sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),
        qualification=entry('Q4/qualification.json'),sensitive_qualification=entry('Q4/qualification-peak0075.json'),scene=entry('Q6/final-scene-result.json'),
        case_identity_sha256=sha(run/'cases'/DEFAULT/'identity.json'),case_protocol_sha256=sha(run/'cases'/DEFAULT/'execution-protocol.json'),
        dt_s=.0125,steps=128,display_frames=12,material_policy='q5_with_full_retry',mass_order=5,full_material_order=7,
        field_cache=True,force_only=False,tangent_cache=False,temporal_accuracy=False,spatial_accuracy=False,
        physical_3D_coupling=True,coupling_scope='independent one-pressure-cell four-step tests, pure-solid default retained')
    check(ROOT,sources);check(run/'final-source',sources);check(run,artifacts);audit_parent(True)
    write(run/'release.json',pub)
    _,counts=audit_release(run)
    print('SEALED',run,'release_sha256',sha(run/'release.json'),'counts',counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cache-seed','cache-check','analyze','seal']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; audit through publication CLI')
        {'prepare':prepare,'cache-seed':cache_seed,'cache-check':cache_check,'analyze':analyze,'seal':seal}[a.phase](a.run)
