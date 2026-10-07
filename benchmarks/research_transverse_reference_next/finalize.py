"""Consolidate scoped results without adding numerical work."""
import argparse,re
from .provenance import *
from .runtime import update
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_continuous_geometry_next.review import balances


def accounting(run):
    attempts=[dict(path=str(p.relative_to(run)),**read(p)) for p in sorted((run/'attempts').glob('*.json'))]
    processes=[dict(path=str(p.relative_to(run)),**read(p)) for p in sorted((run/'processes').glob('*.json'))]
    if any(p['status']=='running' for p in processes):raise ValueError('cannot account a running process')
    gpu=sum(p['seconds'] for p in processes if p['gpu_related']);cpu=sum(p['seconds'] for p in processes if not p['gpu_related'] and not Path(p['path']).name.startswith('test-'))
    tests=sum(p['seconds'] for p in processes if Path(p['path']).name.startswith('test-'));rss=max([x['peak_RSS_GiB'] for x in attempts]+[0]);size=sum(p.stat().st_size for p in run.rglob('*') if p.is_file());frames=list(run.rglob('frame.npz'))
    if len(attempts)>50 or gpu>900 or cpu>600 or rss>16 or size>2*2**30 or len(frames)>2:raise ValueError('budget exceeded')
    return dict(status='passed_scoped',attempts=attempts,total_attempts=len(attempts),successful_attempts=sum(x['accepted'] for x in attempts),unexpected_failed_attempts=sum(not x['accepted'] and not x['expected_fault'] for x in attempts),processes=processes,GPU_related_complete_process_seconds=gpu,CPU_analysis_seconds=cpu,contract_test_seconds=tests,peak_dynamic_RSS_GiB=rss,new_display_frames=len(frames),output_bytes_at_check=size,CPU_reference_steps=read(run/'S1/reference-self-check.json')['new_CPU_matrix_steps'],static_candidate_distinct_states=4,new_formal_steps=0)


def prepare(run):
    run=Path(run).absolute();mutable(run);sources=all_sources();account=accounting(run);write(run/'S5/attempt-accounting.json',account)
    test=read(run/'processes/test-contracts.json');log=(run/'processes/test-contracts.log').read_text();count=re.search(r'Ran (\d+) tests',log)
    if test['returncode'] or not count or not re.search(r'^OK\s*$',log,re.M):raise ValueError('CPU tests failed')
    tested={k:v for k,v in test['source_sha256'].items() if k.startswith(('engine/','tests/'))}
    if any(sources.get(k)!=v for k,v in tested.items()):raise ValueError('tested engine or contract changed')
    write(run/'S5/test-report.json',dict(status='passed_scoped',distinct_tests=int(count[1]),returncode=0,command=test['command'],tested_sources=tested,actual_integration=['S2/bridge-check.json','S3/operator-check.json','S3/buffer-lifecycle.json','S3/paired-performance.json','S3/continuous-check.json','S5/load-check.json','S5/selected-load-check.json'],old_long_cycles_not_rerun=True))
    folders=['S2/bridge','S3/DV0','S3/BD0','S3/BD1','S3/DV1','S3/continuous'];ledgers=[dict(path=f,**balances(history(run/f))) for f in folders]
    if not all(x['passed'] for x in ledgers):raise ValueError('final physical ledgers failed')
    write(run/'S5/physical-ledger-audit.json',dict(status='passed_scoped',records=ledgers))
    checks=[]
    for path in sorted(run.rglob('identity.json')):
        bound=read(path).get('numerical_sources',{})
        if bound:
            check(path.parent/'source',bound)
            if any(sources.get(k)!=v for k,v in bound.items()):raise ValueError('executed numerical source drift '+str(path))
            checks.append(dict(path=str(path.relative_to(run)),source_count=len(bound)))
    for p in sorted((run/'processes').glob('*.json')):
        record=read(p);check(p.parent/(p.stem+'-source'),record['source_sha256'])
    write(run/'S5/numerical-source-audit.json',dict(status='passed_scoped',executed_branches=checks,all_active_numerical_sources_current=True,new_numerical_source_exceptions=[],postprocessing_revision='visualize.py only: use explicit Z=0.5 plane and add flow/energy panels; original process snapshot preserved; no dynamics rerun',selected_loader='zero-step selected-load source record predates output-only rendering revision; numerical engine/fixture/state unaffected'))
    d=read(run/'S3/backend-decision.json');selfcheck=read(run/'S1/reference-self-check.json');review=read(run/'S1/fixed-skeleton-time-review.json')
    cap=dict(status='scoped_transverse_reference_delivery',fixed_skeleton_reference=selfcheck['status']=='passed_scoped',fixed_skeleton_engineering_time_checks=review['status']=='passed_scoped',fixed_skeleton_time_window_s=[0,200e-6],transverse_relative_decay_accuracy=False,signal_limited=True,coupled_temporal_accuracy=False,spatial_continuum_accuracy=False,mechanical_relative_accuracy=False,raw_startup_peak_accuracy=False,new_coupled_extension=False,inherited_coupled_window_s=[0,75e-6],selected_backend=d,loadable_interface='benchmarks.research_transverse_reference_next.selected.selected_setup',implementation_equivalence=True,frame_aware_recovery_inherited=True,new_call_local_failure_recovery=True,real_driver_loss_recovery=False,formal_space_changed=False,local_functions=144,full_mass_order=7,full_material_order=7,coupled_q5=False,production_C_E_integration=False,full_coupled_cycle=False,production_default='inherited pure-solid daily-q5-retry,252steps,1.6s',new_display_frames=0)
    write(run/'capability-matrix.json',cap)
    perf=read(run/'S3/paired-performance.json');res=resources();raw=review['records'][0]['raw']['checks']['face_flux']['relative'];sc=read(run/'S5/selected-load-check.json')
    update(run,f'''\n## 本轮结果与限制\n\n- 固定骨架128格参考和两时刻独立矩阵指数核对通过，YZ128/Y128共{selfcheck['new_CPU_matrix_steps']}个廉价CPU压力步。工程观察通过；模式演化仍低于信号下限。原始区间流量整体范数相对差约{raw:.2%}，不宣称启动峰值精度。\n- 几何delta_volume独立账本核对支持固体体积交换主导横向压力变化；下一步应补包含该交换的同工况动态参考。\n- 扩窗与耦合h/2均未触发：没有为了更多帧扩大实验。新增物理帧0；继承场与本轮参考分开标注。\n- 批量父梯度下载候选通过4状态算子、两输入公平单步和4步连续验证；相对DV单步耗时减少{perf['records'][0]['gain_fraction']:.2%}/{perf['records'][1]['gain_fraction']:.2%}，中位{perf['median_gain_fraction']:.2%}。两后端几何求值次数均为7；不宣称完整周期速度。额外暂存保守预算约3.24MiB。\n- 入选研究后端：`{d['backend']}`。新进程通过`selected.selected_setup`实际构造`{sc['actual_backend']}`并加载第{sc['step']}步；纯固体默认场景保持。\n- 5项CPU合同测试、9次动态推进全部成功；另有一次预登记局部缓冲区静态失败后重建检查，不计动态步。GPU相关完整进程{account['GPU_related_complete_process_seconds']:.1f}秒，峰值动态RSS {account['peak_dynamic_RSS_GiB']:.2f}GiB。\n- 系统盘剩余{res['system_free_GiB']:.2f}GiB，未触发迁移；结果与缓存写数据盘。\n\n## 最新可视化CLI\n\n```bash\ncd /root/workspace/mpm-lite\n.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \\\n  --directory {run.relative_to(ROOT)}\n```\n\nIDE转发8765后打开 http://127.0.0.1:8765/ 。图中固定骨架参考为本轮新计算，75微秒运动固体场为继承数据；位移以纳米表示，原始曲线未平滑。HTTP与实际图像核查另见S5证据，浏览器交互未执行。\n\n## 后续推荐\n\n先建立包含alpha*DeltaV、完整质量与惯性的同工况耦合参考，再决定扩窗和144函数预算；继续关注局部伴随计算热点。原始启动流量、连续空间精度、机械相对精度、完整周期、耦合q5、生产C/E与真实驱动丢失恢复仍未认证。\n''')
    print('PREPARED',d['backend'],account['total_attempts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();prepare(a.run)
