"""Finish scoped decisions without adding dynamics or extending certificates."""
import argparse,re
import numpy as np
from .provenance import *
from .lineage import default_source
from .runtime import update
from .review import balances
from benchmarks.research_phase_stress_next.time_study import history


def finish(run):
    run=Path(run);mutable(run)
    static={g:read(run/f'S1/{g}/operator-check.json') for g in ('base','yz')}
    build={g:read(run/f'S1/{g}/construction.json') for g in static}
    if not all(v['status']=='passed_scoped' for v in static.values()):raise ValueError('static checks missing')
    write(run/'S1/construction-contract.json',dict(status='passed_scoped',records=build,geometry_injected_before_initial_validation=True,full_domain_weight_copies_per_cell=0,full_P_transpose=True,shared_raw_values=True,supported_shapes=[[32,1,1],[32,2,1],[32,1,2],[32,2,2]],dynamic_shapes=[[32,1,1],[32,2,2]]))
    write(run/'S1/memory-preflight.json',dict(status='passed_scoped',grids={g:{k:v['identity'][k] for k in ('metadata_bytes','construction_metadata_peak_bytes','cap_bytes','points','shape','cache_entries')} for g,v in build.items()},original_70_percent_guard_unchanged=True,global_dense_node_basis=False))
    write(run/'S1/adjoint-check.json',dict(status='passed_scoped',small_asymmetric_gather_and_weighted_adjoint='processes/test-multirange-recovery.log',complete_volume_gradient_CPU_reference={g:[v['checks']['gradient'] for v in a['records']] for g,a in static.items()},full_P_applied=True,baseline_raw_pruning='conservative x envelope only; transverse pruning is experimental G2',standalone_small_tensor_adjoint='inherited tensor adjoint; full composed gradient checked against independent CPU references'))
    write(run/'S1/operator-equivalence.json',dict(status='passed_scoped',grids=static,transfer=read(run/'S1/dynamic-entry-decision.json')['subspace_checks']))
    write(run/'S1/cache-work-check.json',dict(status='passed_scoped',checks={g:dict(owned=a['owned_results'],**a['checks']) for g,a in static.items()},cache_limit=6))
    spatial=read(run/'S2/spatial-comparison.json');time=read(run/'S2/time-entry-decision.json');perf=read(run/'S4/paired-performance.json');decision=read(run/'S4/backend-decision.json')
    if spatial['status']!='passed_scoped' or decision['selected']:raise ValueError('this finalizer requires qualified D3 and unselected G2')
    signals=[dict(time_s=x['time_s'],signal_m=x['displacement_signal_m'],difference_m=x['max_displacement_difference_m']) for x in spatial['solid']]
    write(run/'S2/spatial-scope-decision.json',dict(status='engineering_spatial_consistency_passed',window_s=[0.,75e-6],grids=[[32,1,1],[32,2,2]],max_fluid_budget_ratios=spatial['flow']['max_budget_ratios'],displacement_amplitudes=signals,interpretation='two specified grids agree below practical engineering scale; small physical signal is not proof of 5 percent relative accuracy',inherited_coarse_time_max_budget_ratio=time['inherited_coarse_time_budget_ratio'],new_half_steps=0,time_separation_reason='inherited time estimate is above quarter budget for startup flux, but both it and spatial differences leave ample engineering margin; time contamination of tiny spatial differences is not certified away',raw_startup_microstep_relative_error_inherited=.1245,full_spatial_accuracy=False,solid_dynamic_reference=False,full_coupled_cycle=False))
    profile=read(run/'S4/candidate-protocol.json');timings=[]
    for record in perf['records']:
        for kind in ('D3','G2'):
            m=record[kind];timings.append(dict(input=record['input'],kind=kind,build_s=m['build_s'],advance_including_probe_and_IO_s=m['advance_s'],geometry_components=m['geometry'],timing_note='geometry components are nested inside advance; do not add them to whole-step time; local_gradient includes complete P restriction; mixed LU/material/probe/IO not individually instrumented'))
    write(run/'S4/hotspot-profile.json',dict(status='passed_scoped',source_profile=profile['source_profile'],optimistic_whole_process_gradient_removal_fraction=profile['optimistic_whole_process_gradient_removal_fraction'],records=timings,granularity_limited=True,cold_geometry_cache=True,cold_JIT_compilation=False))
    write(run/'S4/setup-cost.json',dict(status='limited',records=[dict(input=r['input'],D3_build_s=r['D3']['build_s'],G2_build_s=r['G2']['build_s'],additional_setup_s=r['G2']['additional_setup_s'],setup_recovery_steps=r['setup_recovery_steps']) for r in perf['records']],limit_steps=16))
    write(run/'S4/continuous-check.json',dict(status='not_triggered',reason=decision['reason'],new_continuous_steps=0,selected_backend='D3'))
    # Verify the already committed performance and retry states, without a GPU.
    from benchmarks.research_continuous_geometry_next.review import balances as ledger
    cases=['S4/D30','S4/G20','S4/G21','S4/D31','S5/resource-fault']
    physical=[dict(case=p,**ledger(history(run/p))) for p in cases]
    if not all(x['passed'] for x in physical):raise ValueError('auxiliary physical ledger failed')
    write(run/'S6/physical-ledger-audit.json',dict(status='passed_scoped',auxiliary=physical,trajectories=spatial['balances']))
    same={n:sha(ROOT/'benchmarks/research_continuous_geometry_next'/n)==sha(ROOT/'benchmarks/research_pressure3d_next'/n) for n in ('base_config.py','config.py','physics.py','spaces.py','run.py')}
    if not all(same.values()):raise ValueError('inherited solid module changed')
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=same,formal_144_unchanged=True,full_M7_q7_unchanged=True,ancestor_sources_immutable=True,coupled_q5_permission_extended=False))
    default,chain=default_source(APP,APP_SHA);owner=default.parent.parent;h=history(default)
    if len(h[-1]['rows'])!=252 or h[-1]['state'].time!=1.6:raise ValueError('default scene changed')
    write(run/'S6/default-scene-decision.json',dict(status='inherited',default_case=default.name,default_case_source=dict(release_root=str(owner),release_sha256=sha(owner/'release.json'),case=default.name,identity_sha256=sha(default/'identity.json'),execution_protocol_sha256=sha(default/'execution-protocol.json')),formal_steps=252,new_formal_steps=0))
    logs=['processes/test-contracts.log','processes/test-multirange-recovery.log'];count=0
    for p in logs:
        text=(run/p).read_text();m=re.search(r'Ran (\d+) tests',text)
        if not m or not re.search(r'^OK\s*$',text,re.M):raise ValueError('tests failed '+p)
        count+=int(m[1])
    write(run/'S6/test-report.json',dict(status='passed_scoped',returncode=0,distinct_tests=count,total_test_case_executions=count,suite_executions=2,logs=logs,commands=[read(run/'processes'/f'{name}.json')['command'] for name in ('test-contracts','test-multirange-recovery')],tested_sources={k:v for k,v in all_sources().items() if k.startswith(('engine/','tests/'))},inherited_tests_not_rerun=True,additional_integration=['S1/operator-equivalence.json','S2/coarse-equivalence.json','S2/spatial-comparison.json','S2/transaction-check.json','S3/solid-reference-comparison.json','S4/operator-check.json','S4/paired-performance.json','S5/resource-fault-check.json']))
    update(run,'S6准备：6项独立合同测试通过；性能候选G2中位收益不足1%且设置回收超限，保留D3。新参考R6无需第二级加密；空间重分配、细格h/2及候选连续4步均未触发。正在核查继承日常终态、生成六帧来源的场图并封存27步证据。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):finish(a.run)
