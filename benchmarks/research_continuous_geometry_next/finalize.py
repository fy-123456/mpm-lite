"""Honest branch decisions, inherited default and practical limitations."""
import argparse,re
import numpy as np
from .provenance import *
from .runtime import update
from .lineage import default_source
from benchmarks.research_phase_stress_next.time_study import history

def continuous_decision(run):
    run=Path(run);mutable(run);review=read(run/'S1/continuous-comparison.json')
    if review['status']!='passed_scoped':raise ValueError('continuous branch is not qualified')
    write(run/'S1/backend-scope-decision.json',dict(status='continuous_B_qualified',backend='B',window_s=[1.25e-5,7.5e-5],same_new32_grid=True,old_original16_single_step_qualification='inherited only',new_performance_qualification=False,explicit_A_source=str(APP/'cases/boundary32-h')))
    jobs=[dict(path=str(p.relative_to(run)),**read(p)) for p in (run/'processes').glob('s1-*.json') if read(p)['status']=='failed'];write(run/'S1/resource-interruption.json',dict(status='diagnosed_and_resumed',records=jobs,cause='external GPU processes started after initial memory sample; unchanged 70% free-memory guard rejected transient reserve',numerical_failures_detected=False,committed_state_preserved=True,in_process_rollback_validation_resource_limited=True,rollback_note='The memory exception also blocked geometry validation during in-process restore. Persisted GenerationStore stayed at step9; a fresh process authenticated and resumed it. No in-process recovery under external memory starvation is claimed.',guard_not_relaxed=True,other_processes_not_stopped=True,CPU_reference_run_during_GPU_unavailability=True))

def finish(run):
    run=Path(run);mutable(run);opt=read(run/'S2/optimization-protocol.json')
    if opt['status']=='not_triggered':write(run/'S2/backend-decision.json',dict(status='not_triggered',selected=False,backend='B',reason=opt['reason'],predicted_optimistic_gain_upper=opt['optimistic_whole_step_gain_upper'],new_dynamic_attempts=0,continuous_G1_qualified=False,new_exclusive_performance=False))
    decision=read(run/'S2/backend-decision.json')
    if decision['selected'] and not decision.get('continuous_G1_qualified'):raise ValueError('G1 continuous evidence required')
    s1=read(run/'S1/continuous-comparison.json');write(run/'S1/raw-comparison.json',dict(status=s1['status'],states=s1['states'],balances=s1['balances']));write(run/'S1/engineering-comparison.json',dict(status=s1['status'],fields=s1['fields'],source='continuous-comparison.json',new_window_only=True));write(run/'S1/restart.json',read(run/'cases/continuous-B/restart-8.json'))
    perf=read(run/'S2/paired-performance.json');write(run/'S2/setup-cost.json',dict(status='limited' if not decision['selected'] else 'passed_scoped',records=[dict(input=x['input'],B_build_s=x['B']['build_s'],G1_build_s=x['G1']['build_s'],additional_setup_s=x['G1']['additional_setup_s'],setup_recovery_steps=x['setup_recovery_steps']) for x in perf['records']],max_allowed_recovery_steps=16))
    write(run/'S2/continuous-check.json',dict(status='not_triggered',reason=decision['reason'],new_continuous_G1_attempts=0))
    write(run/'S4/time-extension-bridge.json',dict(status='passed_scoped',h=read(run/'cases/extension-h/bridge.json'),half=read(run/'cases/extension-half/bridge.json'),residual_normalization_window_s=2e-4,new_execution_end_s=3e-4,one_common_physical_source=True))
    from .review import balances
    physical=[]
    for name in ('B0','G10','G11','B1'):
        value=balances(history(run/'S2'/name));physical.append(dict(case=name,**value))
    if not all(x['passed'] for x in physical):raise ValueError('performance case physical ledger failed')
    write(run/'S6/physical-ledger-audit.json',dict(status='passed_scoped',performance_cases=physical,continuous=read(run/'S1/continuous-comparison.json')['balances'],extension=read(run/'S4/extension-comparison.json')['balances']))
    same={n:sha(ROOT/'benchmarks/research_startup_substeps_next'/n)==sha(ROOT/'benchmarks/research_continuous_geometry_next'/n) for n in ('base_config.py','config.py','physics.py','spaces.py','run.py')}
    if not all(same.values()):raise ValueError('inherited solid modules differ')
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=same,ancestor_sources_immutable=True,full_mass_and_material_unchanged=True,production_q5_permission_not_extended=True))
    default,chain=default_source(APP,APP_SHA);owner=default.parent.parent;h=history(default)
    if len(h[-1]['rows'])!=252 or h[-1]['state'].time!=1.6:raise ValueError('default scene differs')
    write(run/'S6/default-scene-decision.json',dict(status='inherited',default_case=default.name,default_case_source=dict(release_root=str(owner),release_sha256=sha(owner/'release.json'),case=default.name,identity_sha256=sha(default/'identity.json'),execution_protocol_sha256=sha(default/'execution-protocol.json')),formal_steps=252,new_formal_steps=0))
    log=(run/'processes/test-contracts.log').read_text();match=re.search(r'Ran (\d+) tests',log)
    if not match or not re.search(r'^OK\s*$',log,re.M):raise ValueError('tests not passed')
    extra=(run/'processes/test-batch.log').read_text() if (run/'processes/test-batch.log').exists() else '';extra_count=int(re.search(r'Ran (\d+) test',extra)[1]) if extra else 0
    if extra and not re.search(r'^OK\s*$',extra,re.M):raise ValueError('batch tests not passed')
    tested={k:v for k,v in all_sources().items() if k.startswith(('engine/','tests/'))}
    write(run/'S6/test-report.json',dict(status='passed_scoped',commands=['.venv/bin/python -B -m unittest discover -s tests/research_continuous_geometry_next -v','.venv/bin/python -B -m unittest discover -s tests/research_continuous_geometry_next -p test_batch.py -v'],returncode=0,distinct_tests=int(match[1])+extra_count,suite_executions=1+bool(extra),total_test_case_executions=int(match[1])+extra_count,tested_sources=tested,logs=['processes/test-contracts.log','processes/test-batch.log'],inherited_tests_not_rerun=True,actual_checks=['S1/continuous-comparison.json','S1/transaction-check.json','S2/operator-check.json','S3/frozen-geometry-reference.json','S4/extension-comparison.json','S4/transaction-check.json']))
    directions=[dict(direction='general 3D pressure geometry',status='can_prepare',needs='prove per-cell ownership and bounded noncontiguous tensor adjoints without deleting current x-only guard; static reference already provided'),dict(direction='actual deforming spatial trajectory',status='needs_evidence',needs='same time protocol on two actual 3D coupled grids, preserve current F, full mobility, q7/M7, then compare pressure and solid regional stresses'),dict(direction='144-function allocation',status='needs_evidence',needs='trustworthy deforming spatial reference and regional error localization; no training this round'),dict(direction='startup raw microstep peaks',status='needs_evidence',needs='application requires microstep peak accuracy; inherited first BE interval relative error remains about12.45%'),dict(direction='coupled q5 / production C/E',status='not_this_round',needs='independent work/tangent/transaction qualification')]
    write(run/'S5/space-entry-decision.json',dict(directions=directions,new_spaces=0,training_solves=0,production_C_E_changes=0,coupled_q5=False,frozen_geometry=read(run/'S3/frozen-geometry-reference.json')['status'],actual_coupled_spatial_accuracy=False))
    frozen=read(run/'S3/frozen-geometry-reference.json');ref=read(run/'S3/fixed-skeleton-comparison.json');comparisons=[dict(coarse=x['coarse'],fine=x['fine'],max_budget_ratios=x['comparison']['max_budget_ratios']) for x in ref['records']]
    lines=['# 当前热点与后续进入条件','','连续B和200→300微秒续算分别验收；不把历史单步43.17%降幅当成本轮连续加速率。',f'G1：{decision["reason"]}。预评估乐观收益上界{opt["optimistic_whole_step_gain_upper"]:.2%}，最终采用{decision["backend"]}；算子及实测结果单独保存。',f'冻结真实位移参考：{frozen["status"]}，{len(frozen["completed"])}组；这不是实际动态空间收敛。','','横向固定骨架差异：']
    for x in comparisons:lines.append(f'- {x["coarse"]}→{x["fine"]}：{x["max_budget_ratios"]}')
    lines+=['','优先实现有界三维单元几何，再用两档真实动态轨迹建立空间参考。参考可信后才重分配144个局部函数。原始首微步约12.45%流量误差仍保留；是否另改时间格式由应用峰值需求决定。','发生1次非预设动态显存失败，另一次构造受限；诊断为外部GPU竞争。未放宽guard，原存储提交未污染，空闲后续算。','本轮正式纯固体场景不变，日常q5证书不推广到耦合。']
    (run/'S5').mkdir(exist_ok=True);(run/'S5/bottleneck-review.md').write_text('\n'.join(lines)+'\n');update(run,'S5/S6：参考与范围已分别记录；正式144局部函数、q7材料/M7质量保持原模型。默认252步日常场景只读加载，未重跑长周期。原始微步精度与实际空间收敛继续标为未认证。')
    (run/'implementation-report.md').write_bytes(PROGRESS.read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['continuous','finish']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):continuous_decision(a.run) if a.phase=='continuous' else finish(a.run)
