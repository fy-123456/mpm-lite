"""Seal all 24 planned steps, including conditional decisions and failures."""
import argparse,re
from .provenance import *
from .publication import audit_release

MAPPING={
'S0.1':['S0/version-audit.json','input-lock.json'],
'S0.2':['S0/physical-contract.json','S0/state-contract.json'],
'S0.3':['S0/experiment-budget.json','S0/resource-policy.json','S0/storage-migration.json'],
'S1.1':['S1/continuous-protocol.json','cases/continuous-B/bridge.json'],
'S1.2':['S1/continuous-comparison.json'],
'S1.3':['S1/transaction-check.json','S1/recovery-comparison.json','cases/continuous-B/restart-8.json'],
'S1.4':['S1/backend-scope-decision.json','S1/resource-interruption.json'],
'S2.1':['S2/optimization-protocol.json'],
'S2.2':['S2/backend-decision.json','S2/operator-check.json'],
'S2.3':['S2/backend-decision.json','S2/paired-performance.json'],
'S2.4':['S2/backend-decision.json'],
'S3.1':['S3/grid-protocol.json','S3/frozen-preflight.json'],
'S3.2':['S3/topology-transfer-check.json','S6/test-report.json'],
'S3.3':['S3/fixed-skeleton-comparison.json','S3/reference-status.json'],
'S3.4':['S3/frozen-geometry-reference.json'],
'S4.1':['S4/extension-protocol.json'],
'S4.2':['cases/extension-h/bridge.json','cases/extension-half/bridge.json','S6/test-report.json'],
'S4.3':['S4/extension-comparison.json'],
'S4.4':['S4/transaction-check.json','cases/extension-h/restart-36.json','S4/extension-decision.json'],
'S5.1':['S5/space-entry-decision.json'],
'S5.2':['S5/bottleneck-review.md','implementation-report.md'],
'S6.1':['S6/test-report.json','S6/load-check.json','S6/attempt-accounting.json','S6/regression-impact.json','S6/physical-ledger-audit.json'],
'S6.2':['S6/visual-assets-check.json','S6/visual-review.json','S6/visualization-origin.json'],
'S6.3':['capability-matrix.json','documentation.json','S6/final-resources.json']}

def accounting(run):
    run=Path(run);budget=read(run/'S0/experiment-budget.json');records=[dict(path=str(p.relative_to(run)),sha256=sha(p),**read(p)) for p in sorted((run/'attempts').glob('*.json'))];stage={s:sum(x['attempts'] for x in records if x['stage']==s) for s in budget['stages']};frames=list(run.rglob('frame.npz'));processes=[dict(path=str(p.relative_to(run)),**read(p)) for p in sorted((run/'processes').glob('*.json'))];gpu=sum(p['seconds'] for p in processes if p['gpu_related']);peak=max([x['peak_RSS_GiB'] for x in records]+[read(run/'S3/reference-status.json')['peak_RSS_GiB']]);size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    if sum(stage.values())>70 or any(stage[k]>budget['stages'][k] for k in stage) or len(frames)>6 or gpu>2400 or peak>16 or size>3*2**30:raise ValueError('experiment budget exceeded')
    if any(p['status']=='running' for p in processes):raise ValueError('unfinished process')
    cpu=read(run/'S3/reference-status.json')
    if cpu['seconds']>900 or cpu['assemblies']>12 or cpu['spectral_evaluations']>4 or any(p['seconds']>1200 for p in processes):raise ValueError('static or per-process budget exceeded')
    out=dict(status='passed_scoped',stage_attempts=stage,total_attempts=sum(stage.values()),new_committed_steps=sum(x['accepted'] for x in records),injected_failure_attempts=sum(x['expected_fault'] for x in records),other_failed_attempts=sum(not x['accepted'] and not x['expected_fault'] for x in records),records=records,processes=processes,GPU_related_complete_process_seconds=gpu,failed_processes=sum(p['status']=='failed' for p in processes),construction_only_failed_processes=sum(p['status']=='failed' and p['path'].endswith('s1-resource-retry.json') for p in processes),peak_RSS_GiB=peak,new_display_frames=len(frames),new_display_frame_paths=[str(p.relative_to(run)) for p in frames],output_bytes_at_check=size,CPU_reference=read(run/'S3/reference-status.json'),inherited_daily_steps=252,inherited_daily_frames=12,new_formal_steps=0)
    write(run/'S6/attempt-accounting.json',out);return out

def capabilities(run):
    run=Path(run);s1=read(run/'S1/backend-scope-decision.json');s2=read(run/'S2/backend-decision.json');s4=read(run/'S4/extension-decision.json')
    return dict(status='scoped_continuous_and_extension_delivery',continuous_B=s1['status'],continuous_B_window_s=[1.25e-5,7.5e-5],G1=s2,extension=s4,frozen_geometry=read(run/'S3/frozen-geometry-reference.json')['status'],actual_coupled_spatial_accuracy=False,raw_substep_accuracy=False,exclusive_new_performance=s2['selected'],production_default='inherited daily-q5-retry; pure solid,252steps,1.6s',formal_space_changed=False,local_functions=144,mass_order=7,full_material_order=7,new_formal_steps=0,full_coupled_cycle=False,coupled_q5=False,production_C_E_integration=False)

def seal(run):
    run=Path(run);lock=mutable(run);audit_parent(True);sources=all_sources();test=read(run/'S6/test-report.json')
    if test['returncode']!=0 or any(sources.get(k)!=v for k,v in test['tested_sources'].items()):raise ValueError('tested source drift')
    bindings=[]
    for p in sorted(run.rglob('identity.json')):
        identity=read(p);bound=identity.get('numerical_sources',{})
        for key,value in bound.items():
            if sha(ROOT/key)!=value:raise ValueError('executed branch source drift: '+str(p)+' '+key)
        if bound:bindings.append(dict(path=str(p.relative_to(run)),sources=len(bound),passed=True))
    if len(bindings)!=9:raise ValueError('unexpected executed branch source inventory')
    write(run/'S6/numerical-source-audit.json',dict(status='passed_scoped',branches=bindings,ancestors_checked=True))
    if read(run/'S6/visual-review.json')['image_visual_review']!='passed' or read(run/'S6/load-check.json')['status']!='passed_scoped':raise ValueError('visual/default load required')
    ac=accounting(run);default=read(run/'S6/default-scene-decision.json');s1=read(run/'S1/backend-scope-decision.json');s2=read(run/'S2/backend-decision.json');s4=read(run/'S4/extension-decision.json')
    cap=capabilities(run)
    write(run/'capability-matrix.json',cap);(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes());write(run/'S6/final-resources.json',dict(**resources(),output_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),output_budget_bytes=3*2**30));write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    codes=re.findall(r'^### (S[0-6]\.[1-4]) ',(run/'plan-frozen.md').read_text(),re.M)
    if len(codes)!=24 or set(codes)!=set(MAPPING):raise ValueError('24-step plan differs')
    steps=[]
    for code,paths in MAPPING.items():
        status='passed_scoped';reason='scoped evidence and acceptance complete'
        if code=='S2.2' and (run/'S2/operator-check.json').exists():status=read(run/'S2/operator-check.json')['status'];reason='one bounded G1 candidate implemented and operator checked'
        if code=='S2.3' and (run/'S2/paired-performance.json').exists():status=read(run/'S2/paired-performance.json')['status'];reason=s2['reason']
        if code=='S2.4' and not s2.get('continuous_G1_qualified'):status='not_triggered';reason=s2['reason']
        if code=='S3.3':status='limited';reason='two transverse resolutions diagnose change; they do not certify spatial convergence'
        if code=='S3.4':status=cap['frozen_geometry'];reason='two frozen displacements and RT0 subspace only; no new fine-grid dynamics'
        if code=='S4.3' and s4['status']!='extended_window_qualified':status='limited';reason='extension engineering gate not passed'
        steps.append(dict(step=code,status=status,reason=reason,evidence=[dict(path=p,sha256=sha(run/p)) for p in paths]))
    write(run/'requirement-audit.json',dict(count=24,steps=steps,all_steps_accounted_for=True,all_accuracy_goals_passed=False,sequential=True,resource_interruption_reordering='S3 CPU work while GPU unavailable; S1 then resumed',plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources);artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def e(n):return dict(path=n,sha256=sha(run/n))
    pub=dict(schema='continuous-geometry-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**e('final-source-sha256.json'),count=len(sources)),artifacts=dict(**e('artifact-sha256.json'),count=len(artifacts)),space=e('selected-space.json'),capabilities=e('capability-matrix.json'),step_audit=e('requirement-audit.json'),attempts=e('S6/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,new_display_frames=ac['new_display_frames'],inherited_display_frames=12,mass_order=7,full_material_order=7,material_policy='inherited pure-solid q5; coupled fullq7',spatial_accuracy=False,temporal_accuracy=False,engineering_window_accuracy=s4['status']=='extended_window_qualified',selected_geometry=s4['backend']+' in explicitly qualified research windows')
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,counts=audit_release(run,True);print('SEALED',run,sha(run/'release.json'),{k:v for k,v in counts.items() if k!='ancestors'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
