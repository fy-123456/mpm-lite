"""Conditional time/performance entry and read-only reference applicability."""
import argparse
import numpy as np
from .provenance import *
from .runtime import update

def time_scope(run):
    run=Path(run);mutable(run);d=read(run/'S3/time-entry-decision.json')
    if d['need_half']:
        times=np.array(read(run/'S0/coupled-protocol.json')['times']['h']);fine=np.sort(np.r_[times,.5*(times[:12]+times[1:13])])
        write(run/'S3/time-protocol.json',dict(status='registered',case=d['case'],full_times_s=fine.tolist(),stop_step=24,stop_s=37.5e-6,residual_window_s=200e-6,new_frames=0))
        raise RuntimeError('registered one half-step trajectory; execute before final scope decision')
    for name in ('time-protocol','time-comparison','time-scope-decision'):
        write(run/f'S3/{name}.json',dict(status='not_triggered',reason=d['reason'],new_steps=0,new_display_frames=0,selected_schedule='original h and original theta',scope='0..75us engineering response; raw startup peak accuracy remains limited',no_h4=True,inherited_uniform_raw_microstep_difference=.1245,new_nonuniform_time_error_estimate=None))
    update(run,'S3：未触发h/2。新工况工程比较和硬条件均通过，尚无同工况时间误差或启动峰值应用需求；旧均匀工况12.45%微步差仅作背景，未冒充新工况误差。')

def hotspot(run):
    run=Path(run);mutable(run);rows=[]
    for case in ('Y64','Y128','YZ128'):
        for path in sorted((run/'cases'/case).glob('profile-*.json')):
            a=read(path);geo={k:v['seconds'] for k,v in a['geometry'].items()};physical=a['publication']['physical_step']['seconds'];material=a['material']['material_evaluate']['seconds']
            # AVF path contains model.evaluate; do NOT add both a second time.
            other=max(0.,physical-sum(geo.values())-material)
            rows.append(dict(case=case,source=str(path.relative_to(run)),sha256=sha(path),build_s=a['build_s'],advance_s=a['advance_s'],geometry=geo,material_evaluate_including_path_s=material,AVF_path_nested_s=a['material']['AVF_material_path_inclusive']['seconds'],mixed_LU_validation_and_unseparated_s=other,publication=a['publication'],probes_s=a['probes_s'],cache_hits=a['cache_hits'],cache_misses=a['cache_misses']))
    denominator=sum(r['advance_s'] for r in rows);totals={k:sum(r['geometry'][k] for r in rows) for k in rows[0]['geometry']}
    f=(totals['cofactor_download']+totals['P_restriction_and_volume'])/denominator
    write(run/'S4/hotspot-profile.json',dict(status='passed_scoped',records=rows,total_advance_s=denominator,geometry_total_s=totals,whole_advance_fractions={k:v/denominator for k,v in totals.items()},optimistic_target_removal_bound=f,optimistic_Tnew_over_Told_lower_bound=1-f,granularity_limit='P restriction shares CPU slice/sum volume work; adjoint includes gather/download; mixed LU/validation not independently split',synchronization='component and material hooks synchronize; qualification runs omit new diagnostic hooks',cache_scope='owned six-entry q-keyed current V/G/H; no cross-state reuse',extra_dynamic_steps=0))
    if f<.05:
        for name in ('candidate-protocol','operator-check','paired-performance','continuous-check'):
            write(run/f'S4/{name}.json',dict(status='not_triggered',reason='complete target elimination gives less than 5 percent',new_steps=0,selected=False))
        write(run/'S4/backend-decision.json',dict(status='retained',backend='D3',selected=False,reason='measured target below whole-step gate'))
    else:
        write(run/'S4/candidate-protocol.json',dict(status='registered',candidate='device-cell-volume',mechanism='deterministic GPU weighted cell-volume reduction reuses existing RT0 blocks; download only one scalar per cell instead of full J array; retain full P and all current-state G/H',reason='local buffer allocation and isolated P metadata are too small; full J transfer and CPU volume work jointly measurable',optimistic_fraction=f,expected_fraction=.6*f,expected_estimate_not_measurement=True,profile_partition_is_upper_bound=True,pairs=[dict(input_step=8,order=['D3','DV']),dict(input_step=16,order=['DV','D3'])],scope='YZ128 certified states, unchanged full equations and q7/M7',max_static_state_groups=4,max_dynamic_attempts=10,selection=dict(median_whole_advance_gain=.05,both_inputs_nonnegative=True,gain_spread_max=.05,setup_recovery_steps_max=16),cache_key='same authenticated per-model RT0 ownership metadata: space, P, cuts, quadrature, device and source; only q-independent metadata reused',extra_peak_bytes_estimate='8*(RT0 block count+pressure cell count), below 1MiB; current J already allocated in D3',current_F_G_H_recomputed=True,owned_return_arrays=True,global_dense_node_basis=False))
    update(run,f'S4实测：cofactor/整域下载与P/体积CPU段合计占推进{f:.1%}；该占比只是可删除上界。'+('登记唯一GPU单元体积汇总候选，完整P不变。' if f>=.05 else '上界不足5%，不触发新候选。'))
    print('HOTSPOT',f,flush=True)

def extensions(run):
    run=Path(run);mutable(run);ref=read(APP/'S3/solid-reference-comparison.json');entries={}
    for name in ('R6-package.json','R6-definition.json','R6.npz','solid-reference-comparison.json'):
        p=APP/'S3'/name;expected=read(APP/'artifact-sha256.json')['S3/'+name]
        if sha(p)!=expected:raise ValueError('R6 evidence changed')
        entries[name]=dict(path=str(p),sha256=expected)
    current=read(run/'S2/scope-decision.json')
    write(run/'S5/reference-applicability.json',dict(status='inherited_static_only',authenticated=entries,reference_scope='pure-solid F45 / .005m static loading only',formal_vs_R6=ref['formal_vs_R6'],R5_vs_R6=ref['adjacent'],new_reference_levels=0,new_equilibrium_solves=0,dynamic_reference=False,error_sources=dict(initial_projection='same continuous Y field restricted exactly, separately checked',pressure_grid='specified z subdivision sensitivity only',time='new nonuniform time error unmeasured; engineering response passed',implementation='directional and legacy checks passed',solid_space='no matched dynamic reference; tiny displacement does not prove relative accuracy')))
    write(run/'S5/space-entry-decision.json',dict(status='not_triggered',formal_functions=144,formal_space_changed=False,new_training=0,reason='regional engineering comparisons pass; matched dynamic solid reference absent; no basis replacement justified',future_gate=['important region exceeds practical budget','pressure and time effects separated','independent solid reference at same physical problem'],group_gain='0.5*rS.T@solve(KSS,rS) valid only for SPD local tangent and shared quadratic approximation, including off-diagonal coupling'))
    write(run/'S5/extension-entry-decision.json',dict(status='future_work_only',preferred_next='one selected nonuniform short window with a semidiscrete fixed-skeleton reference for transverse pressure modes; retain current production default',entries=[
        dict(direction='longer window',input='one YZ128 checkpoint at75us with identical200us protocol',missing=['new interval stability and accumulated mass/energy','mode and stress signal resolution'],cost='first extend one case by 4..8 steps, separately registered',stop='negative pressure, detF/residual/ledger failure or fixed resource cap'),
        dict(direction='more pressure cells',input='same continuous initial condition and full tensor',missing=['new bounded geometry construction and rank checks','memory cap accounting','controlled refinement reference'],cost='static preflight first; no automatic >128 allocation',stop='metadata exceeds bound or reference unidentifiable'),
        dict(direction='production C/E integration',input='explicit inertia, initial condition, solid/pressure spaces, pressure-work and exchange interfaces',missing=['mass and energy exchange contracts','mixed solver suitability','joint commit/rollback and real IO'],cost='one minimal separate interface fixture before any application scene',stop='transaction or physical exchange ambiguity'),
        dict(direction='coupled q5',input='new same-space coupled sufficient q7 qualification',missing=['energy/internal force/tangent and hidden states','whole-step full-rule retry transaction','pressure geometry/material separation'],cost='small static qualification first; full M7 unchanged',stop='compression certificate fails; retain q7')],coupled_q5=False,production_C_E=False,full_cycle=False))
    update(run,'S5：R6包/定义/数据/比较哈希认证通过，仅继承F45静态适用范围。未触发144函数重分配；新增参考平衡和训练均为0。后续优先补一个同工况横向压力参考，再讨论扩窗或生产耦合。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['time','hotspot','extensions']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'time':time_scope,'hotspot':hotspot,'extensions':extensions}[a.phase](a.run)
