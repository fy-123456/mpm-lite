"""python -m benchmarks.research_e.stage2.run: freeze, restore, run, seal.

All outputs and the independent checkout live in E's data-disk prefix. The
repository result path is a symlink. Historical files remain byte-identical.
"""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
from time import perf_counter
import traceback
import zipfile
from engine.aniso_phase1.research_e.stage2.handoff import sha,save

ROOT=Path(__file__).resolve().parents[3]
PARENT=Path('docs/results/parallel-v22/integration/20260930T054100Z-common-inputs')
OLD_E=Path('docs/results/parallel-v22/E/20260930T040636Z_E_full')
PROTOCOL=Path(__file__).with_name('protocol.json')
DATA=Path('/root/autodl-tmp/mpm-lite-research-e/stage2')
TRUST=Path('/root/autodl-tmp/mpm-lite-research-d/common-inputs/20260930T054100Z-common-inputs')
EXTENSION_DIRS=('engine/aniso_phase1/research_e/stage2','benchmarks/research_e/stage2','tests/research_e/stage2')


def extension_manifest(root):
    return {str(p.relative_to(root)):sha(p) for d in EXTENSION_DIRS for p in sorted((root/d).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def check_manifest(root,manifest):
    bad=[rel for rel,digest in manifest.items() if not (root/rel).is_file() or sha(root/rel)!=digest]
    if bad:raise ValueError('identity mismatch: '+str(bad))
    return dict(passed=True,count=len(manifest),mismatches=[])


def baseline(root,protocol,*,trusted=None):
    from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
    space,inertia,state,verified=load_frozen_inputs(root/PARENT,root,trusted_data_root=trusted,expected_sha256=protocol['parent_bundle_sha256'])
    manifest=json.loads((root/OLD_E/'source-sha256.json').read_text())
    old=check_manifest(root,manifest)
    artifacts=json.loads((root/OLD_E/'artifact-sha256.json').read_text())
    oldart=check_manifest(root/OLD_E,artifacts)
    common=json.loads((root/PARENT/'bundle.json').read_text())['code_sha256']
    extras={p:h for p,h in manifest.items() if p not in common}
    negative={}
    try:load_frozen_inputs(root/PARENT,root,trusted_data_root=trusted,expected_sha256='0'*64);negative['wrong_parent']=False
    except ValueError:negative['wrong_parent']=True
    from engine.aniso_phase1.research_d.common_state import CommonState,StateTransaction
    import numpy as np
    try:StateTransaction(CommonState(np.zeros((10,2)),np.zeros((10,2))));negative['E_2d_as_A_3d']=False
    except ValueError:negative['E_2d_as_A_3d']=True
    return dict(passed=bool(verified['passed'] and old['passed'] and oldart['passed'] and all(negative.values())),
        common=verified,E_source=old,E_artifacts=oldart,readonly_supplements=extras,negative_controls=negative,
        source_checkout=str(root),A_space_is_not_E_space=True)


def launch(args):
    protocol=json.loads(PROTOCOL.read_text());run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_E_stage2')
    if Path(run_id).name!=run_id or run_id in ('.','..'):raise ValueError('single component run id required')
    workspace=DATA/run_id;workspace.mkdir(parents=True,exist_ok=False)
    out=workspace/'results';out.mkdir()
    link=ROOT/'docs/results/parallel-v22-stage2/E'/run_id;link.parent.mkdir(parents=True,exist_ok=True);link.symlink_to(out,target_is_directory=True)
    save(out/'protocol.json',protocol)
    from .storage import guard
    guard(out)
    d_source=ROOT/'engine/aniso_phase1/research_d/stage2/contracts.py'
    if d_source.is_file():shutil.copy2(d_source,out/'D-contract-source.py')
    before=baseline(ROOT,protocol,trusted=TRUST);save(out/'baseline-live-before.json',before)
    # Validate original v22 archive as well; this reads but never rewrites assets.
    from benchmarks.research_e.run import baseline_audit
    archive=baseline_audit();save(out/'v22-baseline-before.json',archive)
    if any(v['mismatches'] for v in archive.values() if isinstance(v,dict)):raise ValueError('v22 changed')
    ext=extension_manifest(ROOT);save(out/'extension-source-sha256.json',ext)
    with zipfile.ZipFile(out/'source-e-stage2.zip','w',zipfile.ZIP_DEFLATED) as z:
        for rel in ext:z.write(ROOT/rel,rel)
    isolated=workspace/'checkout';isolated.mkdir()
    with zipfile.ZipFile(ROOT/PARENT/'code-snapshot.zip') as z:
        if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise ValueError('unsafe snapshot member')
        z.extractall(isolated)
    oldmanifest=json.loads((ROOT/OLD_E/'source-sha256.json').read_text())
    for rel in set(oldmanifest)|set(ext):
        target=isolated/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    shutil.copytree(ROOT/PARENT,isolated/PARENT,symlinks=False)
    shutil.copytree(ROOT/OLD_E,isolated/OLD_E,symlinks=False)
    save(out/'independent-copy.json',dict(checkout=str(isolated),source_snapshot_sha256=sha(ROOT/PARENT/'code-snapshot.zip'),
        supplemented_E_sources=check_manifest(isolated,oldmanifest),extension=check_manifest(isolated,ext),
        arrays='dereferenced byte copies; no external symlinks or disabled path checks'))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
             MPLCONFIGDIR=str(workspace/'mpl-cache'),TMPDIR=str(workspace/'tmp'))
    (workspace/'tmp').mkdir();(workspace/'mpl-cache').mkdir()
    command=[sys.executable,'-m','benchmarks.research_e.stage2.run','--worker','--output',str(out)]
    with (out/'execution.log').open('w') as log:
        process=subprocess.Popen(command,cwd=isolated,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        code=process.wait()
    guard(out)
    save(out/'baseline-live-after.json',baseline(ROOT,protocol,trusted=TRUST))
    save(out/'v22-baseline-after.json',baseline_audit())
    check_manifest(ROOT,ext)
    consumer=subprocess.run([sys.executable,'-m','benchmarks.research_e.stage2.verify','--output',str(out)],cwd=isolated,env=env,capture_output=True,text=True)
    (out/'consumer-smoke.log').write_text(consumer.stdout+consumer.stderr)
    if consumer.returncode:raise RuntimeError('independent consumer failed: '+consumer.stderr)
    save(out/'consumer-smoke.json',json.loads(consumer.stdout))
    # Worker output is already closed, so archive includes the complete execution log.
    save(out/'artifact-sha256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact-sha256.json'})
    print('RESULT',link,flush=True)
    return code


def timeout_handler(signum,frame):raise TimeoutError('pre-registered experiment time budget exceeded')


def worker(out):
    import numpy as np
    import scipy
    started=perf_counter();protocol=json.loads((out/'protocol.json').read_text())
    ext=json.loads((out/'extension-source-sha256.json').read_text());check_manifest(ROOT,ext)
    save(out/'baseline-readonly-check.json',baseline(ROOT,protocol))
    save(out/'environment.json',dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
         utc=datetime.now(timezone.utc).isoformat(),loadavg=os.getloadavg(),cpu_count=os.cpu_count(),device='cpu',
         thread_env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
         system_free_bytes=shutil.disk_usage('/').free,data_free_bytes=shutil.disk_usage(DATA).free,
         output_root=str(out),checkout=str(ROOT),budget=protocol['resources'],timing='shared host; no performance claim'))
    from . import experiments as ex
    from .adapters import transaction_audit,handoff_audit
    checks={}
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/research_e','-v'],capture_output=True,text=True)
    (out/'tests.log').write_text(tests.stdout+tests.stderr);checks['E2_E4_evaluation_tests']=dict(passed=tests.returncode==0)
    signal.signal(signal.SIGALRM,timeout_handler)
    for name,fn in [('E5_consolidation',ex.consolidation_audit),('E5_manufactured',ex.manufactured_audit),
                    ('E6_pressure',ex.pressure_audit),('E7_cycle',ex.cycle_audit),('E8_robustness',ex.robustness_audit),
                    ('E9_transaction',lambda o,p:transaction_audit(o,p,ROOT/PARENT))]:
        from .storage import guard
        guard(out)
        print(name,'START',flush=True);t=perf_counter();signal.alarm(protocol['resources']['per_experiment_timeout_s'])
        try:result=fn(out,protocol)
        except Exception as err:result=dict(passed=False,error=str(err),traceback=traceback.format_exc())
        finally:signal.alarm(0)
        result['wall_seconds']=perf_counter()-t;save(out/(name+'.json'),result)
        checks[name]=dict(passed=result['passed'],wall_seconds=result['wall_seconds'])
        print(name,'PASS' if result['passed'] else 'FAIL',round(result['wall_seconds'],2),flush=True)
        if 'error' in result:print(result['traceback'],flush=True)
        if perf_counter()-started>protocol['resources']['total_budget_s']:raise TimeoutError('total budget')
    evidence={k:dict(v,artifact_sha256=sha(out/(k+'.json'))) for k,v in checks.items() if (out/(k+'.json')).exists()}
    result=handoff_audit(out,protocol,sha(out/'extension-source-sha256.json'),sha(out/'protocol.json'),evidence)
    save(out/'E10_handoff.json',result);checks['E10_private_handoff']=dict(passed=result['passed'])
    from engine.aniso_phase1.research_e.stage2.handoff import load
    from engine.aniso_phase1.research_e.stage2.d_adapter import consume
    m,M,v,_=load(out/'handoff',expected_manifest_sha256=result['manifest_sha256'],expected=result['expected'])
    d_result=consume(out/'handoff'/'d-mixed',m,M,v,out/'D-contract-source.py')
    save(out/'D-contract-consumer.json',d_result)
    if d_result.get('D_contract_consumed'):checks['E10_D_contract']=dict(passed=d_result['passed'])
    baseline_after=baseline(ROOT,protocol);save(out/'baseline-independent-after.json',baseline_after)
    checks['E1_E11_baseline_unchanged']=dict(passed=baseline_after['passed'] and check_manifest(ROOT,ext)['passed'])
    total=perf_counter()-started
    acceptance=dict(checks=checks,independent_physics_passed=all(v['passed'] for v in checks.values()),
        capabilities={'static_operator':checks['E10_private_handoff']['passed'],
            'bounded_material_reference':False,'dynamic_cycle':False,'cuda':False,
            'coupled_physics':all(checks[k]['passed'] for k in ('E5_consolidation','E5_manufactured','E6_pressure','E7_cycle','E8_robustness'))},
        scope='fixed 2D quasistatic E; dynamic_cycle flag reserved for actual shared solid dynamics',
        C_parent_value_transaction=checks['E9_transaction']['passed'],C_integrator_coupling=False,D_public_consumer=d_result.get('passed',False),D_solver_integration=False,
        total_seconds=total,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        parent_bundle_sha256=protocol['parent_bundle_sha256'],extension_source_sha256=sha(out/'extension-source-sha256.json'),protocol_sha256=sha(out/'protocol.json'))
    save(out/'acceptance.json',acceptance)
    try:
        from .visualize import render
        render(out)
    except Exception:(out/'visualization-error.log').write_text(traceback.format_exc())
    report(out,acceptance)
    print('INDEPENDENT_PHYSICS_PASSED',acceptance['independent_physics_passed'],flush=True)
    return 0 if acceptance['independent_physics_passed'] else 1


def report(out,a):
    lines=['# E 第二阶段固定二维耦合验证','',f"独立物理验收：**{'通过' if a['independent_physics_passed'] else '未全部通过'}**。",'',
        '按指定最新共同冻结包恢复独立源码副本，补齐 E 的四项额外只读资产；运行前后核对父包、原 E 交付及扩展源码。旧实现、协议和报告未改写。',
        '采用 4% 场误差与 5% 循环差异预算；守恒、有限值、真实混合残差和事务失败保护仍是硬约束。','',
        '| 项目 | 状态 | 证据 |','| --- | --- | --- |']
    for k,v in a['checks'].items():
        evidence=k+'.json' if (out/(k+'.json')).exists() else ('tests.log' if 'tests' in k else 'acceptance.json')
        lines.append(f"| {k} | {'通过' if v['passed'] else '未通过'} | [{evidence}]({evidence}) |")
    manufactured=json.loads((out/'E5_manufactured.json').read_text())
    cycle=json.loads((out/'E7_cycle.json').read_text())
    if 'rows' in manufactured and 'rows' in cycle:
        row=manufactured['rows'][-1];cr=cycle['rows'][-2]
        worst=max(v['error'] for region in row['stress'].values() for v in region.values())
        cycle_worst=max(v['error'] for part in [cr['whole']]+cr['stages']+cr['ends'] for v in part.values())
        lines += ['',f"制造解 {row['n']}×{row['n']}：压力误差 {row['pressure_error']:.3%}，位移 {row['displacement_error']:.3%}，最差必检分区/口径应力 {worst:.3%}；循环 dt=0.01 对 0.005 s 的全程/阶段/阶段末最大差 {cycle_worst:.3%}。"]
    lines+=['','## 修正的应力口径','',
        'sigma_eff=C:epsilon，sigma=sigma_eff-alpha*p*I。二维平面应变输出完整 3×3 张量，包括 zz；P0 压力在单元内恒定。中心、积分均值、同积分点全场分别输出。制造解全域、边界带、内部的有效及总应力均值与全场全部进入验收，缺数据、空分区与错误符号不会自动通过。',
        '旧 stress_cell_mean_error 混用了数值中心值和解析平均；新报告单列该旧口径诊断与正确平均口径，保留历史数字。制造解线性时间依赖使解析后向 Euler 导数精确；时间加密差异还包含离散空间投影响应，不能据此宣称任意时间依赖精确。','',
        '## 场景与原始数据','',
        '[场景与循环账本](scene-summary.png)、[空间收敛](convergence.png)、[场景动画](scene-preview.gif)。所有 NPZ 保存未平滑字段，循环保存每步质量与功/耗散账本以及每 0.04 s 的完整场；Darcy 物理耗散与后向 Euler 数值损失分开，准静态动能为零。',
        '36 个稳健性工况按原参数组合执行，固定 0/17/35 子集做网格敏感性。粗网格稳定不等于精度认证，尤其渗透率比 1000 不获得解析精度声明。','',
        '## 接口与尚未认证的能力','',
        'E 子状态以拥有副本的纯值放入真正父 StateTransaction，由父事务一次发布；E 没有提前提交外部对象。通过失败、异常、父拒绝、过期 token、双提交、错误重启参数和连续/重启一致性检查。使用真实 A 父状态只验证事务容器，未把二维 E 位移伪装成三维 A 系数。',
        '[混合算子包](handoff/operator-package.json) 保存 A/G/C/B/H、精确混合矩阵、参考解和作用。原混合矩阵非对称、非 SPD，交付使用一般求解器及分块量纲残差。',
        'D 在本轮并行工作中新加 MixedBlockContract，允许独立二维 (u,p) 块。E 精确消元 RT0 通量后调用真实 D 合同及 validate_handoff，并重建通量核查原三块残差和作用；[消费证据](D-contract-consumer.json)绑定当时 D 源码摘要，随后在独立进程再次复载。D 求解后端与 GPU 没有参与。C 新动力学积分器与 E 的真实耦合未认证。E 不修改 C/D 公共接口。三维、有限变形、移动粒子、GPU 和生产集成未开展。','',
        '## 复现与资源','',
        '`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m benchmarks.research_e.stage2.run`',
        f"共享主机观测耗时 {a['total_seconds']:.2f} s，峰值 RSS {a['max_rss_kib']/1024:.1f} MiB，不构成性能加速声明。输出及独立副本均在 E 私有数据盘目录，仓库结果路径为符号链接。",
        '[预登记协议](protocol.json)、[验收](acceptance.json)、[基线核验](baseline-readonly-check.json)、[扩展源码](extension-source-sha256.json)、[完整成果摘要](artifact-sha256.json)。','']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id');ap.add_argument('--worker',action='store_true');ap.add_argument('--output',type=Path)
    args=ap.parse_args();return worker(args.output) if args.worker else launch(args)


if __name__=='__main__':raise SystemExit(main())
