"""Consolidate scoped evidence and inherited daily scene without new dynamics."""
import argparse,re
from .provenance import *
from .runtime import update
from .lineage import default_source
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_continuous_geometry_next.review import balances

def prepare(run):
    run=Path(run);mutable(run)
    paths=['S1/legacy-bridge-check.json','S1/directional-operator-check.json','S2/transaction-check.json','S2/transverse-grid-comparison.json']
    if not all(read(run/p)['status']=='passed_scoped' for p in paths):raise ValueError('required physical entry failed')
    default,chain=default_source(APP,APP_SHA);owner=default.parent.parent;h=history(default)
    if len(h[-1]['rows'])!=252 or h[-1]['state'].time!=1.6:raise ValueError('default scene changed')
    write(run/'S6/default-scene-decision.json',dict(status='inherited',default_case=default.name,default_case_source=dict(release_root=str(owner),release_sha256=sha(owner/'release.json'),case=default.name,identity_sha256=sha(default/'identity.json'),execution_protocol_sha256=sha(default/'execution-protocol.json')),formal_steps=252,new_formal_steps=0))
    names=['S1/legacy','S2/fault',*[f'S4/{k}{i}' for k,i in [('D3',0),('DV',0),('DV',1),('D3',1)]]]
    if (run/'S4/continuous/CURRENT.json').exists():names+=['S4/continuous']
    records=[dict(case=n,**balances(history(run/n))) for n in names]
    if not all(x['passed'] for x in records):raise ValueError('auxiliary physical ledger failed')
    write(run/'S6/physical-ledger-audit.json',dict(status='passed_scoped',auxiliary=records,trajectories={c:read(run/f'S2/{c}-review.json')['balances'] for c in ('Y64','Y128','YZ128')}))
    same={n:sha(ROOT/'benchmarks/research_pressure3d_next'/n)==sha(ROOT/'benchmarks/research_transverse_next'/n) for n in ('base_config.py','config.py','physics.py','spaces.py','run.py')}
    if not all(same.values()):raise ValueError('inherited solid module changed')
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=same,formal_144_unchanged=True,full_M7_q7_unchanged=True,ancestor_sources_immutable=True,coupled_q5_permission_extended=False))
    path=run/'processes/test-contracts.log';text=path.read_text();match=re.search(r'Ran (\d+) tests',text)
    if not match or not re.search(r'^OK\s*$',text,re.M):raise ValueError('contract suite failed')
    test=read(run/'processes/test-contracts.json')
    write(run/'S6/test-report.json',dict(status='passed_scoped',returncode=test['returncode'],distinct_tests=int(match[1]),suite_executions=1,command=test['command'],logs=['processes/test-contracts.log'],tested_sources={k:v for k,v in all_sources().items() if k.startswith(('engine/','tests/'))},inherited_tests_not_rerun=True,new_GPU_integration=['S1/directional-operator-check.json','S1/legacy-bridge-check.json','S2/transverse-grid-comparison.json','S2/transaction-check.json','S4/operator-check.json','S4/paired-performance.json'],scope='five CPU contracts plus actual GPU static/dynamic/transaction integrations, no old long-cycle rerun'))
    update(run,'S6准备：新入口的5项CPU合同及真实静态/短窗/故障/候选集成均已归档；正式144空间、完整M7/q7和继承纯固体模块保持。开始新进程只读加载与六帧可视化检查。')

def finish(run):
    run=Path(run).absolute();mutable(run);daily=read(run/'S6/daily-load-check.json');new=read(run/'S6/new-checkpoint-load.json')
    if daily['status']!='passed_scoped' or new['status']!='passed_scoped':raise ValueError('load checks failed')
    write(run/'S6/load-check.json',dict(status='passed_scoped',inherited_daily=daily,new_nonuniform=new,new_dynamic_steps=0))
    scope=read(run/'S2/scope-decision.json');perf=read(run/'S4/backend-decision.json');stats=read(run/'S4/paired-performance.json');res=resources()
    balances0=[read(run/f'S2/{c}-review.json')['balances'] for c in ('Y64','Y128','YZ128')]
    attempts=[read(p) for p in sorted((run/'attempts').glob('*.json'))]
    update(run,f'''\n## 本轮交付结论\n\n三条非均匀初压轨迹各18步到75微秒，6个新物理帧；Y64/Y128仅认证增加z分区后的工程一致性，YZ128混合方向稳定。最大质量缺口{max(abs(x['mass_defect_m3']) for x in balances0):.3g}m³，最大能量缺口{max(abs(x['energy_balance_J']) for x in balances0):.3g}J，最小detF={min(x['min_detF'] for x in balances0):.9f}。位移约百纳米，低于绝对工程误差下限，不能据此宣称相对空间精度。\n\nS3未触发新时间细分，启动峰值精度仍受限。R6仅为继承静态参考，本轮0新参考平衡、0训练、144函数未换包。候选单步收益{stats['records'][0]['gain_fraction']:.2%}/{stats['records'][1]['gain_fraction']:.2%}，中位{stats['median_gain_fraction']:.2%}；最终研究后端：{perf['backend']}。不将4个单步或4步连续验证冒充整18步测速。\n\n当前累计动态尝试{len(attempts)}次，其中{sum(x['accepted'] for x in attempts)}次返回成功，{sum(x['expected_fault'] for x in attempts)}次预登记故障。系统盘剩余{res['system_free_GiB']:.2f}GiB，未触发迁移。\n\n## 可视化 CLI\n\n```bash\ncd /root/workspace/mpm-lite\n.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \\\n  --directory {run.relative_to(ROOT)}\n```\n\nIDE转发8765端口后打开 http://127.0.0.1:8765/ 。页面展示本轮6个真实帧、横向模式、分量位移、区域应力和资源拒绝恢复。继承纯固体动画单独标为252步/1.6秒/12帧、位移10倍显示；本轮未重跑。HTTP、图像/GIF解码及脚本检查见S6证据；浏览器交互是否执行单独记录。\n\n发布后只读全审计命令：\n\n```bash\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \\\n  -m benchmarks.research_transverse_next.publication \\\n  --run {run.relative_to(ROOT)}\n```\n\n## 下一步建议\n\n优先为一个非均匀工况补同问题的横向压力参考与可分辨时间尺度，再选择一个短窗扩展；不同时增加网格、加载和时间长度。真实驱动丢失恢复、完整耦合周期、生产C/E及耦合q5仍未认证。一般混合LU和原残差保护保留。\n''')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','finish']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):prepare(a.run) if a.phase=='prepare' else finish(a.run)
