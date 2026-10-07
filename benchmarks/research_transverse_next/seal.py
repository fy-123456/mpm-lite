"""Seal all 27 requirements and actual budgets, preserving the direct parent."""
import argparse,re
from .provenance import *
from .publication import audit_release

MAPPING={
'S0.1':['S0/preimplementation-version-audit.json','S0/version-audit.json','input-lock.json'],
'S0.2':['S0/physical-contract.json','S0/initial-condition-protocol.json','S0/observation-contract.json'],
'S0.3':['S0/experiment-budget.json','S0/resource-policy.json','S0/storage-migration.json'],
'S1.1':['S1/initialization-check.json'],
'S1.2':['S1/directional-operator-check.json','S1/memory-preflight.json'],
'S1.3':['S1/observable-check.json','S1/projection-contract.json'],
'S1.4':['S1/publication-contract.json','S6/test-report.json'],
'S1.5':['S1/legacy-bridge-check.json','S1/legacy-complete-payload-check.json','S1/dynamic-entry-decision.json'],
'S2.1':['cases/Y64/summary.json','S2/Y64-review.json'],
'S2.2':['cases/Y128/summary.json','S2/transverse-grid-comparison.json'],
'S2.3':['cases/YZ128/summary.json','S2/mixed-direction-review.json'],
'S2.4':['cases/YZ128/restart-8.json','S2/transaction-check.json'],
'S2.5':['S2/scope-decision.json','S3/time-entry-decision.json'],
'S3.1':['S3/time-entry-decision.json'],
'S3.2':['S3/time-protocol.json'],
'S3.3':['S3/time-comparison.json'],
'S3.4':['S3/time-scope-decision.json'],
'S4.1':['S4/hotspot-profile.json'],
'S4.2':['S4/candidate-protocol.json','S4/operator-check.json'],
'S4.3':['S4/paired-performance.json','S4/setup-cost.json','S4/timing-attribution.json'],
'S4.4':['S4/backend-decision.json','S4/continuous-check.json'],
'S5.1':['S5/reference-applicability.json'],
'S5.2':['S5/space-entry-decision.json'],
'S5.3':['S5/extension-entry-decision.json'],
'S6.1':['S6/test-report.json','S6/load-check.json','S6/attempt-accounting.json','S6/regression-impact.json','S6/physical-ledger-audit.json'],
'S6.2':['S6/visual-assets-check.json','S6/visual-review.json','S6/visualization-origin.json','S6/visualization-readout-fix.json','S6/small-displacement-visual-diagnostic.json','S6/cli-http-check.json'],
'S6.3':['capability-matrix.json','documentation.json','S6/final-resources.json','S6/numerical-source-audit.json']}

def sources_audit(run,sources):
    records=[]
    for p in sorted(run.rglob('identity.json')):
        bound=read(p).get('numerical_sources',{})
        if not bound:continue
        source=p.parent/'source';check(source,bound)
        if any(sources.get(k)!=v for k,v in bound.items()):raise ValueError('new executed source drift '+str(p))
        records.append(dict(path=str(p.relative_to(run)),source_count=len(bound),status='passed_scoped'))
    expected={'cases/Y64/identity.json','cases/Y128/identity.json','cases/YZ128/identity.json','S1/legacy/identity.json','S2/fault/identity.json','S4/static/identity.json',*[f'S4/{k}{i}/identity.json' for k,i in [('D3',0),('DV',0),('DV',1),('D3',1)]]}
    if not expected.issubset({r['path'] for r in records}):raise ValueError('missing executed branch bindings')
    write(run/'S6/numerical-source-audit.json',dict(status='passed_scoped',branches=records,new_historical_source_exceptions=[],all_new_active_sources_current=True,baseline_historical_exceptions='inherited verbatim through authenticated direct parent; no new waiver'))

def accounting(run):
    b=read(run/'S0/experiment-budget.json');rows=[dict(path=str(p.relative_to(run)),sha256=sha(p),**read(p)) for p in sorted((run/'attempts').glob('*.json'))];stages={k:sum(r['attempts'] for r in rows if r['stage']==k) for k in b['stages']}
    processes=[dict(path=str(p.relative_to(run)),**read(p)) for p in sorted((run/'processes').glob('*.json'))]
    if any(p['status']=='running' for p in processes):raise ValueError('unfinished process; invoke sealing outside runner')
    gpu=sum(p['seconds'] for p in processes if p['gpu_related']);cpu=sum(p['seconds'] for p in processes if not p['gpu_related'] and not Path(p['path']).name.startswith('test-'));tests=sum(p['seconds'] for p in processes if Path(p['path']).name.startswith('test-'))
    frames=list(run.rglob('frame.npz'));static=[read(run/f'S1/{c}/operator-check.json') for c in ('Y64','Y128')];rss=max([r['peak_RSS_GiB'] for r in rows]+[r['peak_RSS_GiB'] for r in static]);size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file());candidate=read(run/'S4/operator-check.json')
    if sum(stages.values())>96 or any(stages[k]>b['stages'][k] for k in stages) or len(frames)>6 or gpu>1200 or cpu>900 or rss>16 or size>3*2**30:raise ValueError('resource or attempt budget exceeded')
    if sum(x['static_state_groups'] for x in static)>8 or candidate['static_state_groups']>4:raise ValueError('static group budget exceeded')
    out=dict(status='passed_scoped',stage_attempts=stages,total_attempts=len(rows),successful_attempt_returns=sum(x['accepted'] for x in rows),expected_fault_attempts=sum(x['expected_fault'] for x in rows),unexpected_failed_attempts=sum(not x['accepted'] and not x['expected_fault'] for x in rows),records=rows,processes=processes,failed_processes=sum(p['status']!='passed' for p in processes),GPU_related_complete_process_seconds=gpu,recorded_CPU_analysis_seconds=cpu,small_contract_seconds=tests,accounting_scope='all dynamic/static experiment and registered analysis processes; source editing and read-only setup/version audits excluded from experiment stopwatch',peak_RSS_GiB=rss,new_display_frames=len(frames),new_display_frame_paths=[str(p.relative_to(run)) for p in frames],output_bytes_at_check=size,S1_static_state_groups=8,S4_static_state_groups=candidate['static_state_groups'],new_reference_levels=0,new_reference_solves=0,new_training=0,new_formal_steps=0,inherited_daily_steps=252)
    write(run/'S6/attempt-accounting.json',out);return out

def seal(run):
    run=Path(run);lock=mutable(run);audit_parent(True);sources=all_sources();test=read(run/'S6/test-report.json')
    if test['returncode'] or any(sources.get(k)!=v for k,v in test['tested_sources'].items()):raise ValueError('tested source drift')
    sources_audit(run,sources)
    for name in ('load-check','physical-ledger-audit','visual-assets-check'):
        if read(run/f'S6/{name}.json')['status']!='passed_scoped':raise ValueError('required final check failed')
    if read(run/'S6/visual-review.json')['image_visual_review']!='passed':raise ValueError('actual image review required')
    ac=accounting(run);default=read(run/'S6/default-scene-decision.json');backend=read(run/'S4/backend-decision.json');scope=read(run/'S2/scope-decision.json')
    cap=dict(status='scoped_transverse_response_delivery',implementation_equivalence=True,nonuniform_initial_identity=True,frame_aware_safe_publication=True,controlled_resource_recovery=True,real_driver_loss_recovery=False,engineering_z_subdivision_consistency=scope['engineering_z_subdivision_consistency'],stable_nonuniform_cases=['Y64','Y128','YZ128'],physical_window_s=[0,75e-6],initial_profiles='same Y field on64/128; separate unequal mixed YZ field',mechanical_relative_accuracy=False,spatial_continuum_accuracy=False,temporal_accuracy=False,raw_startup_peak_accuracy=False,full_coupled_cycle=False,coupled_q5=False,production_C_E_integration=False,solid_static_reference='inherited R6 F45/.005m only',formal_space_changed=False,local_functions=144,full_mass_order=7,full_material_order=7,selected_backend=backend,production_default='inherited pure-solid daily-q5-retry,252steps,1.6s',new_formal_steps=0)
    write(run/'capability-matrix.json',cap);(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes());res=resources()
    if res['system_free_GiB']<5:raise ValueError('storage migration required')
    write(run/'S6/final-resources.json',dict(**res,output_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),output_budget_bytes=3*2**30))
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    codes=re.findall(r'^\*\*(S[0-6]\.\d)｜',(run/'plan-frozen.md').read_text(),re.M)
    if len(codes)!=27 or set(codes)!=set(MAPPING):raise ValueError('27-step plan differs')
    limits={'S2.3':'mixed response stable; displacement below engineering absolute floor, relative mechanical accuracy unqualified','S2.5':'specified z subdivision sensitivity only; no continuum/dynamic reference','S4.1':'adjoint includes gather/download; P includes volume; mixed solve and validation combined','S5.1':'R6 inherited static loading only','S5.3':'entry requirements documented, production extensions not executed'}
    skipped={k:'time refinement not triggered by registered criteria' for k in ('S3.2','S3.3')};skipped['S5.2']='no matched dynamic reference or important over-budget region; retain144'
    if not backend['selected']:skipped['S4.4']='paired gate did not select candidate; retain D3';limits['S4.3']='candidate algebra qualified but measured speed/setup gates not all passed'
    steps=[dict(step=k,status='not_triggered' if k in skipped else 'limited' if k in limits else 'passed_scoped',reason=skipped.get(k,limits.get(k,'registered scope checked')),evidence=[dict(path=p,sha256=sha(run/p)) for p in paths]) for k,paths in MAPPING.items()]
    write(run/'requirement-audit.json',dict(count=27,steps=steps,all_steps_accounted_for=True,all_accuracy_goals_passed=False,sequential=True,plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(p):return dict(path=p,sha256=sha(run/p))
    pub=dict(schema='transverse-response-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S6/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,new_display_frames=ac['new_display_frames'],inherited_display_frames=12,mass_order=7,full_material_order=7,material_policy='inherited pure-solid q5; coupled fullq7',spatial_accuracy=False,temporal_accuracy=False,engineering_z_subdivision_consistency=scope['engineering_z_subdivision_consistency'],engineering_window_s=[0,75e-6],selected_geometry=backend['backend'])
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,counts=audit_release(run,True)
    print('SEALED',run,sha(run/'release.json'),{k:v for k,v in counts.items() if k!='ancestors'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
