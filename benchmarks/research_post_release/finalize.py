"""Final source-bound scene, integration audit and immutable publication."""
from pathlib import Path
import argparse
import copy
import re
import numpy as np
from .provenance import ROOT,PARENT,PARENT_RELEASE_SHA,PLAN,read,write,sha,digest,source_files,snapshot,register,verify,audit_parent,resources,serial_lock,utc,check
from .run import create_config,load_model,make_stepper,case_identity
from .reference_study import reopen
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.model_package import load_reduction
from engine.aniso_phase1.research_sequential_next.model import PracticalModel
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError,advance_publish
from engine.aniso_phase1.consistent_transfer import material_response

DEFAULT='daily-q5-retry-dt0125'
PROGRESS=ROOT/'docs/MPM_LITE_POST_RELEASE_PROGRESS_20261001_ZH.md'
COUPLING=ROOT/'docs/MPM_LITE_POST_RELEASE_COUPLING_INTERFACE_20261001_ZH.md'


def hist(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()


def prepare(run):
    run=Path(run)
    register(run,'S5/publication-guard-revision.json',dict(default_case=DEFAULT,
        previous_full_case='final-q5-retry-dt0125',reason='add sealed-run write protection and application release fork; numerical update mathematics unchanged',
        source_policy='new complete case on final code; retain previous verification',steps=128,frames=12,
        field_cache=True,rule_policy='q5_with_full_retry'))
    create_config(run,DEFAULT,dt=.0125,rule_policy='q5_with_full_retry',field_cache=True)
    create_config(run,'cache-retry-restart',dt=.0125,end=.025,rule_policy='q5_with_full_retry',field_cache=True)
    log=run/'S5/final-tests.log'
    if 'Ran 11 tests' not in log.read_text() or '\nOK\n' not in log.read_text():raise ValueError('final tests not passed')
    write(run/'S5/final-tests-result.json',dict(status='passed',count=11,log='S5/final-tests.log',log_sha256=sha(log),
          numerical_source_sha256=source_files(),test_source_sha256=sha(ROOT/'tests/research_post_release/test_runtime.py')))


def cache_seed(run):
    run=Path(run);folder=run/'cases/cache-retry-restart';cfg=read(folder/'execution-protocol.json');model,_=load_model(run,cfg)
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
    write(run/'S5/cache-retry-witness.json',dict(status='passed_first_step',q7_frame_error=error,source=source_files(),
         pointer=read(store.pointer),digest=c.state.digest(),failures=c.failures))


def cache_check(run):
    run=Path(run);folder=run/'cases/cache-retry-restart';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg)
    c=make_stepper(run,cfg,m,m.rest());history=hist(folder);c.validate(history[-1]['state'])
    assert len(history)==3 and history[-1]['state'].step==2
    assert history[-1]['state'].child_states['identity']==c.full.identity
    assert len(history[-1]['state'].child_states['material_failure_history'])==1
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
    full=ValidatedAVF(c.full,cfg)
    full.step(.0125);full.step(.0125)
    errors={k:float(np.max(abs(getattr(full.state,k)-getattr(history[-1]['state'],k)))) for k in ('q','velocity','predictor')}
    assert max(errors.values())<1e-8
    write(run/'S5/cache-retry-check.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,
         cached_fields_after_q7_fallback=True,history_preserved=True,source_sha256=source_files()))


def reference_invariants(run):
    records=[]
    for name in ('h1','h2','p1'):
        r,m,state,_=reopen(Path(run)/'S2'/name);s=r.parent
        points=[np.linspace(e[0],e[-1],n) for e,n in zip(s.edges,[9,3,3])]
        X=np.stack(np.meshgrid(*points,indexing='ij'),axis=-1)
        A=np.array([[.01,.002,0],[0,-.003,.001],[.001,0,.002]]);b=np.array([.003,-.002,.001])
        q=np.zeros((s.ndof,3));q[:s.n]=s.carrier_X@A.T+b
        u,grad=s._sample(s.nodes(q),points)
        affine=max(float(np.max(abs(u-(X@A.T+b)))),float(np.max(abs(grad-A))))
        theta=.13;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        q[:s.n]=s.carrier_X@(R-np.eye(3)).T
        u,grad=s._sample(s.nodes(q),points);F=np.eye(3)+grad
        _,P=material_response(F.reshape(-1,3,3),np.broadcast_to(s.A,F.reshape(-1,3,3).shape),s.params)
        rigid=float(np.max(abs(P)));carrier=s.reference[:s.n]+q[:s.n]
        if affine>1e-8 or rigid>1e-8:raise ValueError('refined space affine/rigid material check failed')
        records.append(dict(package=name,affine_max=affine,sampled_rigid_PK1_max_Pa=rigid,
                original_stabilization_energy_J=float(.5*np.sum(carrier*(s.Ks@carrier)))))
    write(Path(run)/'S2/affine-rigid-check.json',dict(status='passed_scoped',records=records,scope='affine fields and sampled rigid material response; original Ks reported separately'))


def analyze(run):
    run=Path(run);identity=read(run/'cases'/DEFAULT/'identity.json')
    if identity['numerical_source_sha256']!=source_files():raise ValueError('final numerical source changed')
    tests=read(run/'S5/final-tests-result.json')
    if tests['numerical_source_sha256']!=source_files():raise ValueError('tests used other final numeric source')
    a=hist(run/'cases'/DEFAULT);b=hist(run/'cases/full-q7-dt0125')
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
    write(run/'S5/final-field-comparison.json',dict(status='passed_scoped',reference_case='full-q7-dt0125',records=records,reactions=reactions,maxima=maxima,
           scope='same selected dt/space, q5 vs q7; not time/space accuracy certification'))
    summary=read(run/'cases'/DEFAULT/'summary.json');frames=sum((x['folder']/'frame.npz').exists() for x in a)
    assert frames==12
    counts=audit_parent(full=True)[2]
    result=dict(status='passed_scoped',default_case=DEFAULT,steps=128,generations=len(a),frames=frames,
                source_verified=True,parent_unchanged=counts,scene=summary,comparison_maxima=maxima,sentinels=sentinels,
                fallback_count=sum(x['material_attempts']-1 for x in rows),actual_rules=sorted({x['material_rule'] for x in rows}),
                time_accuracy=False,spatial_accuracy=False,physical_3D_coupling=False,resources=resources())
    write(run/'S5/final-scene-result.json',result)
    print('FINAL_SCENE',maxima,summary['this_segment'],flush=True)


def seal(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('already sealed')
    result=read(run/'S5/final-scene-result.json')
    if read(run/'cases'/DEFAULT/'identity.json')['numerical_source_sha256']!=source_files():raise ValueError('changed final source')
    for p in ['S5/cache-retry-check.json','S5/coupling-check.json','S2/affine-rigid-check.json']:
        if read(run/p)['status']!='passed_scoped':raise ValueError('missing required check: '+p)
    if sha(PLAN)!=read(run/'parent-release-lock.json')['plan_sha256']:
        if sha(run/'plan-frozen.md')!=read(run/'parent-release-lock.json')['plan_sha256']:raise ValueError('lost original registered plan')
    else:(run/'plan-frozen.md').write_bytes(PLAN.read_bytes())
    definitions={
      'S0':('passed_scoped',['parent-release-lock.json','S0/result.json']),
      'S1':('phase_improved_time_uncertified',['S1/time-decision.json','S1/restart-check.json']),
      'S2':('completed_reference_limited',['S2/reference-decision.json','S2/reload-check.json','S2/affine-rigid-check.json']),
      'S3':('not_promoted_reference_limited',['S3/space-decision.json','S3/block-scores.json']),
      'S4':('passed_scoped',['S4/qualification.json','S4/fault-results.json','S4/restart-check.json']),
      'S5':('passed_scoped',['S5/performance-decision.json','S5/cache-retry-check.json','S5/coupling-check.json','S5/final-scene-result.json'])}
    steps=re.findall(r'^### (S\d+\.\d+) (.+)$',(run/'plan-frozen.md').read_text(),re.M);audit=[]
    for code,title in steps:
        status,evidence=definitions[code.split('.')[0]]
        if code=='S3.5':status='conditional_not_required_original_space_retained'
        for name in evidence:
            if not (run/name).is_file():raise ValueError('missing plan evidence')
        audit.append(dict(step=code,title=title,status=status,evidence=evidence))
    assert len(audit)==35
    write(run/'requirement-audit.json',dict(utc=utc(),steps=audit,count=35,all_steps_accounted_for=True,
          original_plan_sha256=sha(run/'plan-frozen.md'),scientific_limits=['time_accuracy_uncertified','reference_limited','no_3D_physical_coupling'],
          scope='bounded original plan, including its explicitly allowed nonpromotion/conditional branches'))
    capabilities=dict(scene_stability='passed_scoped',phase_improvement='passed_scoped_local_windows',temporal_accuracy=False,
       spatial_reference='limited_despite_actual_local_h_and_p',spatial_accuracy=False,selected_space='original144',
       material='q5 qualified on fixed F45/.005m cycle; full-rule retry and three stage sentinels',
       field_cache='passed_scoped_and_enabled',tangent_cache=False,preconditioner='original',
       restart_and_rollback='passed_scoped',independent_2D_coupling_closure=True,physical_3D_coupling=False)
    write(run/'capability-matrix.json',capabilities)
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    (run/'coupling-interface.md').write_bytes(COUPLING.read_bytes())
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),
            coupling_path=str(COUPLING.relative_to(ROOT)),coupling_sha256=sha(COUPLING),
            current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),snapshot_link_base=str(ROOT/'docs')))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('engine/aniso_phase1/research_post_release','benchmarks/research_post_release','tests/research_post_release') for p in sorted((ROOT/folder).glob('*.py'))}
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    files={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')}
    write(run/'artifact-sha256.json',files)
    publish=dict(schema='post-release-practical-v1',utc=utc(),parent_release_sha256=PARENT_RELEASE_SHA,
        parent_lock_sha256=sha(run/'parent-release-lock.json'),default_case=DEFAULT,
        sources=dict(path='final-source-sha256.json',sha256=sha(run/'final-source-sha256.json'),count=len(sources)),
        artifacts=dict(path='artifact-sha256.json',sha256=sha(run/'artifact-sha256.json'),count=len(files)),
        qualification=dict(path='S4/qualification.json',sha256=sha(run/'S4/qualification.json')),
        scene=dict(path='S5/final-scene-result.json',sha256=sha(run/'S5/final-scene-result.json')),
        case_identity_sha256=sha(run/'cases'/DEFAULT/'identity.json'),case_protocol_sha256=sha(run/'cases'/DEFAULT/'execution-protocol.json'),
        dt_s=.0125,steps=128,display_frames=12,material_policy='q5_with_full_retry',full_material_order=7,mass_order=5,
        field_cache=True,tangent_cache=False,preconditioner='original',temporal_accuracy=False,spatial_accuracy=False,physical_3D_coupling=False)
    write(run/'release.json',publish)
    check(ROOT,sources);check(run/'final-source',sources);check(run,files);audit_parent(full=True)
    print('SEALED',run,'sources',len(sources),'artifacts',len(files),'release_sha256',sha(run/'release.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cache-seed','cache-check','reference-invariants','analyze','seal']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'prepare':prepare,'cache-seed':cache_seed,'cache-check':cache_check,'reference-invariants':reference_invariants,'analyze':analyze,'seal':seal}[a.phase](a.run)
