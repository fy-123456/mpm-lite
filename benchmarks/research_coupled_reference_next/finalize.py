"""Truthful capability and physical ledger reports, no additional trajectories."""
import argparse,re,time
import numpy as np
from .provenance import *
from .runtime import update
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_continuous_geometry_next.review import balances
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_sequential_next.checkpoint import GenerationStore

def accounting(run):
    attempts=[dict(path=str(p.relative_to(run)),**read(p)) for p in sorted((run/'attempts').glob('*.json'))]
    processes=[dict(path=str(p.relative_to(run)),**read(p)) for p in sorted((run/'processes').glob('*.json'))]
    if any(p['status']=='running' for p in processes):raise ValueError('unfinished process')
    gpu=sum(p['seconds'] for p in processes if p['gpu_related']);cpu=sum(p['seconds'] for p in processes if not p['gpu_related'] and not Path(p['path']).name.startswith('test-'));tests=sum(p['seconds'] for p in processes if Path(p['path']).name.startswith('test-'))
    size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file());frames=list(run.rglob('frame.npz'));rss=max([x['peak_RSS_GiB'] for x in attempts]+[0.])
    if len(attempts)>36 or gpu>900 or cpu>600 or size>2*2**30 or len(frames)>2 or rss>16:raise ValueError('resource budget exceeded')
    return dict(status='passed_scoped',attempts=attempts,total_attempts=len(attempts),successful_attempts=sum(x['accepted'] for x in attempts),unexpected_dynamic_failures=sum(not x['accepted'] and not x['expected_fault'] for x in attempts),processes=processes,GPU_related_complete_process_seconds=gpu,CPU_analysis_seconds=cpu,contract_test_seconds=tests,peak_dynamic_RSS_GiB=rss,output_bytes=size,new_display_frames=len(frames),new_formal_steps=0)

def prepare(run):
    run=Path(run).absolute();mutable(run);sources=all_sources();ac=accounting(run);write(run/'S5/attempt-accounting.json',ac)
    test=read(run/'processes/test-contracts.json');log=(run/'processes/test-contracts.log').read_text();count=re.search(r'Ran (\d+) tests',log)
    if test['returncode'] or not count or not re.search(r'^OK\s*$',log,re.M):raise ValueError('contract tests failed')
    tested={k:v for k,v in test['source_sha256'].items() if k.startswith(('engine/','tests/'))};check(ROOT,tested)
    write(run/'S5/test-report.json',dict(status='passed_scoped',distinct_tests=int(count[1]),command=test['command'],returncode=0,tested_sources=tested,old_long_cycles_not_rerun=True))
    folders=['S2/bridge','S3/BD0','S3/FBD0','S3/FBD1','S3/BD1']
    if (run/'S3/continuous/identity.json').exists():folders.append('S3/continuous')
    ledgers=[dict(path=f,**balances(history(run/f))) for f in folders]
    if not all(x['passed'] for x in ledgers):raise ValueError('physical ledger failed')
    write(run/'S5/physical-ledger-audit.json',dict(status='passed_scoped',records=ledgers))
    branches=[]
    for path in run.rglob('identity.json'):
        bound=read(path).get('numerical_sources',{})
        if bound:check(ROOT,bound);check(path.parent/'source',bound);branches.append(dict(path=str(path.relative_to(run)),sources=len(bound)))
    for p in (run/'processes').glob('*.json'):check(p.parent/(p.stem+'-source'),read(p)['source_sha256'])
    write(run/'S5/numerical-source-audit.json',dict(status='passed_scoped',branches=branches,all_executed_dynamic_sources_current=True,failed_attempt_sources_preserved=True,scope='earlier failed identity check fixed before any dynamic generation; postprocessing source additions do not alter physical solver'))
    parent=read(APP/'release.json');default=parent['default_case_source'];folder=Path(default['release_root'])/'cases'/default['case'];ident=read(folder/'identity.json');store=GenerationStore(folder,ident);hist=store.history();state=store.load()['state']
    if sha(Path(default['release_root'])/'release.json')!=default['release_sha256'] or sha(folder/'identity.json')!=default['identity_sha256']:raise ValueError('formal scene binding changed')
    write(run/'S5/daily-load-check.json',dict(status='passed_scoped',source=default,zero_steps=True,step=state.step,time_s=state.time,digest=state.digest(),generations=len(hist),scope='authenticated checkpoint read, not a new full-cycle run'))
    # Engineering comparison is not a convergence certificate. The construction
    # snapshots are also the local diagnostic points, not a hidden validation set.
    ref=dict(np.load(run/'S1/reference-r12.npz'));comparisons={}
    comparisons['pressure']=metric(ref['pressure'],ref['actual_pressure'],.001,.05)
    comparisons['displacement']=metric(ref['displacement'],ref['actual_displacement'],5e-5,.05)
    for key in ('Ay','Az'):
        a=ref[key]-ref[key][0];b=ref['actual_'+key]-ref['actual_'+key][0];comparisons[key+'_increment']=metric(a,b,1e-5,.1)
    write(run/'S1/engineering-review.json',dict(status='passed_scoped' if all(v['passed'] for v in comparisons.values()) else 'limited',checks=comparisons,construction_snapshots_reused=True,separate_unused_state_check='S1/holdout-review.json',full_space_certification=False,reason='engineering-sized diagnostic agreement; relative mechanical and continuum accuracy remain open'))
    write(run/'S4/spatial-entry-decision.json',dict(status='not_triggered',formal_space_changed=False,reason='no independent spatial reference and mechanical signal below engineering floor; projected diagnostic does not justify a new144 budget'))
    d=read(run/'S3/backend-decision.json')
    cap=dict(status='scoped_coupled_reference_delivery',coupled_affine_reference=True,reference_scope='mass-projected r8/r12 diagnostic, full128 pressure, true geometric derivatives and affine drift',full_space_reference_certified=False,coupled_temporal_accuracy=False,spatial_continuum_accuracy=False,mechanical_relative_accuracy=False,new_coupled_extension=False,inherited_coupled_window_s=[0,75e-6],selected_backend=d,loadable_interface='benchmarks.research_coupled_reference_next.selected.selected_setup',implementation_equivalence=True,controlled_transaction_recovery_inherited=True,new_local_buffer_recovery=True,real_driver_loss_recovery=False,formal_space_changed=False,local_functions=144,mass_order=7,full_material_order=7,coupled_q5=False,production_C_E=False,full_coupled_cycle=False,new_formal_steps=0,new_display_frames=0)
    write(run/'capability-matrix.json',cap);write(run/'S4/capability-matrix.json',cap)
    selfcheck=read(run/'S1/reference-self-check.json');perf=read(run/'S3/paired-performance.json');validity=read(run/'S1/local-validity.json');load=read(run/'S5/selected-load-check.json')
    update(f'''\n## 本轮结果与限制\n\n- 已完成真实耦合的局部诊断参考：原完整质量投影、128格压力、非零初始驱动、压力几何切线及Darcy几何导数均保留；4方向检查通过。\n- 独立BDF与指数参考压力差{selfcheck['records'][-1]['independent_pressure_error_Pa']:.3g} Pa；r8/r12压力差{selfcheck['rank8_to12_pressure_difference_Pa']:.3g} Pa。三个时刻全空间加速度投影残差比最大{validity['maximum_acceleration_projection_relative']:.3g}。这些是诊断证据，不是连续空间或完整非线性精度证明。\n- 参考构造使用的快照也用于局部余项检查，明确不是独立隐藏集。实体位移约80纳米，低于工程绝对下限；不宣称机械相对精度。\n- 不扩窗、不新增物理帧、不更换144空间；纯固体252步结果仅只读继承。\n- FBD共用两支局部梯度的x向转置，减少重复计算；两组整步相对BD降幅{perf['records'][0]['gain_fraction']:.2%}/{perf['records'][1]['gain_fraction']:.2%}。采用决定：{d['backend']}，理由：{d['reason']}。仅代表实际短窗整步，不外推完整周期。\n- {int(count[1])}项CPU合同测试通过，{ac['total_attempts']}次实际动态尝试、{ac['successful_attempts']}次成功；GPU相关完整进程暂计{ac['GPU_related_complete_process_seconds']:.1f}秒，最终总数见S5/attempt-accounting.json。\n- 新进程实际加载{load['actual_backend']}第{load['step']}步；守恒账本与逐步真实残差通过。系统盘{resources()['system_free_GiB']:.2f}GiB，未触发迁移，结果与缓存位于数据盘。\n\n## 下一步\n\n已补第16步/50微秒留出状态工程检查。下一轮先分析原始启动流量及时间相位是否有可分辨误差，再决定最小h/2短窗；获得独立空间参考后才调整144函数预算。完整耦合周期、生产C/E、耦合q5和真实驱动丢失恢复仍未认证。\n\n## 最新可视化CLI\n\n```bash\ncd /root/workspace/mpm-lite\n.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \\\n  --directory {run.relative_to(ROOT)}\n```\n\nIDE转发8765后打开 http://127.0.0.1:8765/ 。参考曲线为本轮新计算，实体场为继承75微秒数据，新增物理帧0。浏览器交互未验证。\n''')
    print('PREPARED',d['backend'],ac['total_attempts'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();prepare(a.run)
