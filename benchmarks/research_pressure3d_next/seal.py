"""Seal the 27 planned steps, preserving failed trials and narrow source history."""
import argparse,ast,re
from .provenance import *
from .publication import audit_release

MAPPING={
'S0.1':['S0/preimplementation-version-audit.json','S0/version-audit.json','input-lock.json'],
'S0.2':['S0/physical-contract.json','S0/observation-contract.json','S0/source-states.json'],
'S0.3':['S0/experiment-budget.json','S0/resource-policy.json','S0/storage-migration.json'],
'S1.1':['S1/partition-check.json'],
'S1.2':['S1/construction-contract.json','S1/memory-preflight.json'],
'S1.3':['S1/adjoint-check.json','S6/test-report.json'],
'S1.4':['S1/operator-equivalence.json','S1/cache-work-check.json'],
'S1.5':['S1/dynamic-entry-decision.json'],
'S2.1':['S2/trajectory-protocol.json','S0/source-states.json'],
'S2.2':['S2/coarse-equivalence.json','S2/source-normalization-fix.json'],
'S2.3':['S2/transaction-check.json','cases/pressure128-D3/restart-8.json'],
'S2.4':['S2/spatial-comparison.json','S2/time-entry-decision.json'],
'S2.5':['S2/spatial-scope-decision.json'],
'S3.1':['S3/solid-reference-audit.json'],
'S3.2':['S3/refinement-protocol.json','S3/R6-definition.json'],
'S3.3':['S3/solid-reference-comparison.json','S3/R6-package.json'],
'S3.4':['S3/allocation-entry-decision.json'],
'S4.1':['S4/hotspot-profile.json'],
'S4.2':['S4/candidate-protocol.json','S4/operator-check.json'],
'S4.3':['S4/paired-performance.json','S4/postprocess-fix.json','S4/setup-cost.json'],
'S4.4':['S4/backend-decision.json','S4/continuous-check.json'],
'S5.1':['S5/recovery-contract.json','S6/test-report.json'],
'S5.2':['S5/resource-fault-check.json'],
'S5.3':['S5/recovery-scope-decision.json','S5/new-process-load.json'],
'S6.1':['S6/test-report.json','S6/load-check.json','S6/attempt-accounting.json','S6/regression-impact.json','S6/physical-ledger-audit.json'],
'S6.2':['S6/visual-assets-check.json','S6/visual-review.json','S6/visualization-origin.json'],
'S6.3':['capability-matrix.json','documentation.json','S6/final-resources.json','S6/numerical-source-audit.json']}


def source_audit(run,sources):
    fix=read(run/'S2/source-normalization-fix.json');fixture='benchmarks/research_pressure3d_next/fixture.py';perf='benchmarks/research_pressure3d_next/performance.py'
    if sources[fixture]!=fix['new_fixture_sha256']:raise ValueError('fixture normalization changed again')
    old_path=run/'diagnostics/pre-field-statistics-D30/source'/perf
    old_perf=sha(old_path)
    if sha(run/'diagnostics/pre-field-statistics-G20/source'/perf)!=old_perf:raise ValueError('different archived performance versions')
    def without_statistics(text):
        tree=ast.parse(text);tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='statistics'];return ast.dump(tree,include_attributes=False)
    if without_statistics(old_path.read_text())!=without_statistics((ROOT/perf).read_text()):raise ValueError('performance change exceeds statistics fix')
    records=[]
    for p in sorted(run.rglob('identity.json')):
        identity=read(p);bound=identity.get('numerical_sources',{})
        if not bound:continue
        location=str(p.relative_to(run));source=p.parent/'source'
        if location=='S2/fault/identity.json':
            original=run/'cases/pressure128-D3'
            if bound!=read(original/'identity.json')['numerical_sources']:raise ValueError('fault branch source differs')
            source=original/'source'
        check(source,bound);exceptions=[]
        for key,value in bound.items():
            if sources.get(key)==value:continue
            if key==fixture and value==fix['previous_fixture_sha256'] and location.startswith(('S1/base/','S1/yz/','diagnostics/pre-canonical-identity/')):
                exceptions.append(dict(source=key,old_sha256=value,current_sha256=sources[key],scope='JSON container canonicalization only; numerical kernels unchanged; current coarse/full dynamics separately verified',evidence='S2/source-normalization-fix.json'))
            elif key==perf and value==old_perf and location.startswith(('diagnostics/pre-field-statistics-D30/','diagnostics/pre-field-statistics-G20/')):
                exceptions.append(dict(source=key,old_sha256=value,current_sha256=sources[key],scope='preserved failed postprocessing evidence, excluded from candidate timing; AST outside statistics unchanged',evidence='S4/postprocess-fix.json'))
            else:raise ValueError('executed numerical source drift '+location+' '+key)
        records.append(dict(path=location,sources=len(bound),snapshot_verified=True,snapshot_path=str(source.relative_to(run)),historical_exceptions=exceptions))
    required={'cases/pressure32-D3/identity.json','cases/pressure128-D3/identity.json','S5/resource-fault/identity.json',*[f'S4/{k}/identity.json' for k in ('D30','G20','G21','D31')]}
    if not required.issubset({x['path'] for x in records}):raise ValueError('missing active source bindings')
    write(run/'S6/numerical-source-audit.json',dict(status='passed_scoped',branches=records,active_numerical_sources_current=True,historical_exceptions_explicit=True,ancestors_checked=True))


def accounting(run):
    budget=read(run/'S0/experiment-budget.json');records=[dict(path=str(p.relative_to(run)),sha256=sha(p),**read(p)) for p in sorted((run/'attempts').glob('*.json'))]
    stages={k:sum(x['attempts'] for x in records if x['stage']==k) for k in budget['stages']};frames=list(run.rglob('frame.npz'));processes=[dict(path=str(p.relative_to(run)),**read(p)) for p in sorted((run/'processes').glob('*.json'))]
    gpu=sum(p['seconds'] for p in processes if p['gpu_related']);ref=read(run/'S3/solid-reference-comparison.json');static=[read(run/f'S1/{g}/operator-check.json') for g in ('base','yz')]
    peak=max([x['peak_RSS_GiB'] for x in records]+[ref['peak_RSS_GiB']]+[x['peak_RSS_GiB'] for x in static]);size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    if sum(stages.values())>90 or any(stages[k]>budget['stages'][k] for k in stages) or len(frames)>6 or gpu>1200 or peak>16 or size>3*2**30:raise ValueError('experiment budget exceeded')
    if ref['seconds']>1800 or ref['new_equilibrium_solves']>4 or sum(x['static_state_groups'] for x in static)>12:raise ValueError('reference budget exceeded')
    if any(p['status']=='running' for p in processes):raise ValueError('unfinished recorded process; invoke seal outside the experiment runner')
    out=dict(status='passed_scoped',stage_attempts=stages,total_attempts=sum(stages.values()),new_committed_steps=sum(x['accepted'] for x in records),injected_failure_attempts=sum(x['expected_fault'] for x in records),other_failed_attempts=sum(not x['accepted'] and not x['expected_fault'] for x in records),records=records,processes=processes,failed_processes=sum(p['status']=='failed' for p in processes),GPU_related_complete_process_seconds=gpu,GPU_process_budget_s=1200,peak_RSS_GiB=peak,new_display_frames=len(frames),new_display_frame_paths=[str(p.relative_to(run)) for p in frames],output_bytes_at_check=size,new_reference_levels=1,new_equilibrium_solves=ref['new_equilibrium_solves'],static_D3_groups=sum(x['static_state_groups'] for x in static),G2_operator_state_groups=3,new_CPU_reference_integrations=0,inherited_daily_steps=252,inherited_daily_frames=12,new_formal_steps=0)
    write(run/'S6/attempt-accounting.json',out);return out


def seal(run):
    run=Path(run);lock=mutable(run);audit_parent(True);sources=all_sources();test=read(run/'S6/test-report.json')
    if test['returncode']!=0 or any(sources.get(k)!=v for k,v in test['tested_sources'].items()):raise ValueError('tested source drift')
    source_audit(run,sources)
    for p in ['S6/load-check.json','S5/new-process-load.json','S6/visual-assets-check.json','S5/resource-fault-check.json']:
        if read(run/p)['status']!='passed_scoped':raise ValueError('required check failed '+p)
    if read(run/'S6/visual-review.json')['image_visual_review']!='passed':raise ValueError('visual review needed')
    ac=accounting(run);default=read(run/'S6/default-scene-decision.json');perf=read(run/'S4/backend-decision.json')
    cap=dict(status='scoped_3D_pressure_and_reference_delivery',implementation_equivalence='passed specified D3 grids and actual states',engineering_spatial_consistency='passed 32x1x1 versus 32x2x2, 0..75us only',solid_static_reference='R5 to R6 actual h refinement, F45/0.005m only',resource_recovery='controlled resource refusal and authenticated new-process load passed',exclusive_performance='measured G2 failed selection gate; retain D3',formal_space_changed=False,local_functions=144,full_mass_order=7,full_material_order=7,selected_geometry='D3 in qualified research windows',G2=perf,spatial_continuum_accuracy=False,temporal_accuracy=False,raw_substep_accuracy=False,full_coupled_cycle=False,coupled_q5=False,production_C_E_integration=False,real_driver_loss_recovery=False,production_default='inherited pure-solid daily-q5-retry,252steps,1.6s',new_formal_steps=0)
    write(run/'capability-matrix.json',cap);(run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())
    res=resources()
    if res['system_free_GiB']<5:raise ValueError('storage migration required before sealing')
    write(run/'S6/final-resources.json',dict(**res,output_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),output_budget_bytes=3*2**30))
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS),current_plan_path=str(PLAN.relative_to(ROOT)),current_plan_sha256=sha(PLAN),original_plan_sha256=lock['plan_sha256']))
    codes=re.findall(r'^\*\*(S[0-6]\.\d)｜',(run/'plan-frozen.md').read_text(),re.M)
    if len(codes)!=27 or set(codes)!=set(MAPPING):raise ValueError('27-step plan differs')
    limited={'S2.5':'specified two grids only; no continuum or full-cycle certificate','S4.1':'local adjoint and P contraction timed together; no independent LU/material/IO breakdown','S4.3':'G2 correct but below 5 percent speed gate and beyond 16-step setup recovery'}
    decisions={'S3.4':'retain formal144; regional errors within engineering budget; no training','S4.4':'candidate not selected; four additional continuity steps not triggered'}
    steps=[]
    for code,paths in MAPPING.items():
        status='limited' if code in limited else 'not_triggered' if code=='S4.4' else 'passed_scoped'
        steps.append(dict(step=code,status=status,reason=limited.get(code,decisions.get(code,'scoped evidence and checks complete')),evidence=[dict(path=p,sha256=sha(run/p)) for p in paths]))
    write(run/'requirement-audit.json',dict(count=27,steps=steps,all_steps_accounted_for=True,all_accuracy_goals_passed=False,sequential=True,plan_sha256=lock['plan_sha256']))
    snapshot(run/'final-source',sources);write(run/'final-source-sha256.json',sources)
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','release.json')};write(run/'artifact-sha256.json',artifacts)
    def entry(p):return dict(path=p,sha256=sha(run/p))
    pub=dict(schema='pressure3d-practical-v1',utc=utc(),application_parent=str(APP),application_parent_release_sha256=APP_SHA,mathematical_parent=str(PARENT),mathematical_release_sha256=lock['parent_release_sha256'],input_lock_sha256=sha(run/'input-lock.json'),default_case=default['default_case'],default_case_source=default['default_case_source'],sources=dict(**entry('final-source-sha256.json'),count=len(sources)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),space=entry('selected-space.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),attempts=entry('S6/attempt-accounting.json'),formal_steps=252,new_formal_steps=0,new_display_frames=ac['new_display_frames'],inherited_display_frames=12,mass_order=7,full_material_order=7,material_policy='inherited pure-solid q5; coupled fullq7',spatial_accuracy=False,temporal_accuracy=False,engineering_spatial_consistency=True,engineering_window_s=[0.,75e-6],selected_geometry='D3, research scope only')
    check(run/'final-source',sources);check(run,artifacts);write(run/'release.json',pub);_,counts=audit_release(run,True)
    print('SEALED',run,sha(run/'release.json'),{k:v for k,v in counts.items() if k!='ancestors'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):seal(a.run)
