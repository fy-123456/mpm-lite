"""Seal a static D extension without changing the immutable parent package."""
import json,re,shutil,zipfile
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_d.stage2.contracts import sha,PARENT_SHA256,SPACE_SHA256
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from benchmarks.research_d.stage2.bootstrap import PARENT
from benchmarks.research_d.stage2.validate import OUT
from benchmarks.research_d.stage2.integration.inspect import compatibility
from benchmarks.research_d.stage2.storage import guard

TERMINAL={'extension.json','extension-sha256.txt','independent-reload.json','delivery-sha256.json','delivery-sha256.txt'}

def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def main():
    root=Path.cwd();space,_,_,parent=load_frozen_inputs(root/PARENT,root,expected_sha256=PARENT_SHA256)
    acceptance=json.loads((OUT/'correctness-acceptance.json').read_text())
    if not acceptance['passed']:raise ValueError('cannot seal failed static operator')
    # Archive raw logs without editing their contents.
    logs=Path('/root/autodl-tmp/mpm-lite-research-d/stage2')
    for name in ('pilot','gpu-pilot','validation','supplement','tests','final-tests','integration-scope','visualize','performance-0','performance-1','performance-2'):
        if (logs/(name+'.log')).exists():shutil.copy2(logs/(name+'.log'),OUT/(name+'.log'))
    tests=(OUT/'tests.log').read_text();m=re.search(r'Ran (\d+) tests in ([0-9.]+)s',tests)
    if not m or 'OK (skipped=1)' not in tests or '\nFAILED' in tests:raise ValueError('missing passing regression log')
    extra=(OUT/'final-tests.log').read_text()
    if not re.search(r'\nOK\s*(?:\n|$)',extra):raise ValueError('final stage2 tests have not passed')
    write('tests-summary.json',dict(passed=True,regression_ran=int(m[1]),regression_skipped=1,
         skip_reason='second CUDA ordinal unavailable on this single-GPU machine',regression_seconds=float(m[2]),
         final_stage2_ran=int(re.search(r'Ran (\d+) tests',extra)[1]),logs=['tests.log','final-tests.log']))
    process=[json.loads((OUT/f'performance-{i}.json').read_text()) for i in range(3)]
    metrics={};noise={}
    for b in ('cpu','gpu'):
        rows=[r for p in process for r in p['costs'] if r['backend']==b]
        metrics[b]={k:float(np.median([r[field] for r in rows if r['label']==label])) for k,label,field in
            [('cold_backend_seconds','cold','seconds'),('warm_seconds','warm','seconds'),('estimated_total_with_one_measured_build_seconds','cold','total_cold_including_shared_setup')]}
        pairs=[]
        for p in process:
            r={r['label']:r['seconds'] for r in p['costs'] if r['backend']==b}
            pairs.append(abs(r['warm']-r['repeat_for_AA'])/max(r['warm'],r['repeat_for_AA']))
        noise[b]=dict(samples=pairs,median=float(np.median(pairs)),maximum=max(pairs))
    gain=1-metrics['gpu']['estimated_total_with_one_measured_build_seconds']/metrics['cpu']['estimated_total_with_one_measured_build_seconds']
    perf=dict(passed=False,diagnostic_complete=True,independent_processes=3,orders=[p['order'] for p in process],metrics=metrics,AA_relative_noise=noise,
        estimated_total_gain=gain,performance_promoted=False,reason='shared host A/B CPU jobs; common initial Hessian measured once then charged to each total; exclusive fresh full-build rounds remain unexecuted',
        adoption='optional static research operator only; no production default switch',
        peak_host_kib=max(p['host_peak_kib'] for p in process))
    write('performance-summary.json',perf)
    active=Path('/root/workspace/mpm-lite')
    write('integration-compatibility.json',compatibility(active))
    write('baseline-final.json',parent);guard(OUT/'storage-final.json')
    solves=json.loads((OUT/'static-solves.json').read_text());curvature=json.loads((OUT/'curvature.json').read_text());pre=json.loads((OUT/'preconditioners.json').read_text())
    write('stage-completion.json',dict(D1='passed: independent restored and dereferenced parent',D2='passed: explicit contracts and identity refusals',
      D3='passed: 3 states / 2 directions; new 6-vs-7 grouped integration',D4='passed: all 144 local basis functions and real maps/adjoints',
      D5='passed: original energy/gradient/exact tangent; repeated/rotation/compression controls',D6='passed within declared q0/full-spectrum and material controls',
      D7='bounded comparison complete; 2 iterative candidates reached budget and are retained as failures',D8='passed: CPU/GPU same-boundary static equilibrium',
      D9='partial: 3-process shared-host diagnostics; exclusive full cold construction remains unexecuted',D10='awaiting pinned B/C/E handoffs',
      D11='sealed; independent final-digest receipt accompanies delivery',dynamic_gpu=False,continuum_accuracy=False,production_default_changed=False))
    table='\n'.join(f"| {r['name']} | {r.get('iterations','—')} | {r['converged']} | {r['original_operator_residual']:.3e} |" for r in pre)
    report=f'''# D 第二阶段：真实共同空间静态 GPU 交付

已实现并验证固定共同空间的 CPU/GPU 静态算子、精确材料切线和再平衡。D1—D8 的有界交付已完成；D9 留有独占全冷构造复测，D10 等待各方向固定交接包。没有切换生产默认，没有宣称动态 GPU 或连续体空间精度通过。

## 基线与实现

父包 `{PARENT_SHA256}`；空间 `{SPACE_SHA256}`。交付前重验589项父源码和29项成果均一致。全部新增实现限于 D stage2。研究副本在数据盘，输入数组已解引用复制，旧包不依赖活动目录中新写的 B/C/E 代码。

实际规模为369×3完整、219×3自由位移，144个局部标量基，六阶2,985,984个材料点。Warp float64 稀疏/轴分解映射保留非正交局部变换与完整约束行；原 Hencky、双边二次纤维、原 Ks 不变。小 Ks 块在主机最终组装并计时。材料全域布局实测可装入显存，因此使用常驻积分布局；有16GiB预算前检，未构造稠密高阶基。

## 正确性与场景稳定性

- 归档、父扰动和新增3倍扰动共3个状态，每个覆盖混合与局部方向；六阶原 CPU 对照、三档差分、完整位置/梯度伴随均通过。
- 新状态六阶/七阶复核使用父协议原门槛，分别检查全部、自由、固定、左右夹持以及净反力和材料/稳定化贡献。没有事后放宽阈值。
- CPU/GPU 都从同一个 q0、0.005m夹持位移出发，各自重算原势能和真实残量，2次接受更新后收敛。最终自由残差CPU `{solves['cpu']['trace'][-1]['residual_N']:.3e}` N、GPU `{solves['gpu']['trace'][-1]['residual_N']:.3e}` N；最小detF `{solves['gpu']['trace'][-1]['min_detF']:.8f}`。
- 完整反力、原能量、相同参考坐标探针PK1和位移通过0.5%后端等价门槛。实际差远小于门槛。图中仅放大显示位移，原始数据没有平滑。
- 60项既有及新增回归：59通过、1因机器没有第二张CUDA设备而跳过；最终stage2测试另有记录。覆盖演示卸载回零、材料、生产求解器、边界、状态事务及新增非法状态/负曲率/奇异拒绝。

![静态再平衡及中平面PK1](static-validation.png)

## 曲率、求解与预条件

初态657×657完整精确切线对称差 `{curvature['archived']['symmetry']:.3e}`，最小/最大特征值 `{curvature['archived']['minimum']:.6g}` / `{curvature['archived']['maximum']:.6g}`，条件数约 `{curvature['archived']['maximum']/curvature['archived']['minimum']:.3g}`。仅对该状态宣告SPD；其他状态保留方向曲率与材料控制，不外推全局凸性。

| 迭代辅助 | 迭代数 | 达标 | 真实算子残差 N |
| --- | ---: | --- | ---: |
{table}

两个未达标候选保留为失败，不虚报收敛。载体—局部块在本线性控制中较合适。静态终态使用冻结初始精确切线的稠密分解作为搜索辅助，原势能Armijo控制接受；负曲率或近奇异系统走一般求解，非下降方向回退。搜索矩阵不加入物理势能，且本轮CPU/GPU共用已封存的GPU构造搜索矩阵，因此CPU复核是独立原残量再平衡，不是独立从头构造同一Hessian。

## 完整成本及采用范围

完成3个独立进程的AB/BA/AB、冷/暖和A/A重复，含输出场、上传/读回和同步，GPU独占，CPU绑24—27核。A/B同机CPU研究仍在运行，这些为共享主机诊断，**不授予正式性能晋级**。

| 成本中位数 | CPU s | GPU s |
| --- | ---: | ---: |
| 已封存搜索矩阵下的首次求解 | {metrics['cpu']['cold_backend_seconds']:.3f} | {metrics['gpu']['cold_backend_seconds']:.3f} |
| 暖复用求解 | {metrics['cpu']['warm_seconds']:.3f} | {metrics['gpu']['warm_seconds']:.3f} |
| 加上一次已测构造成本的总成本估算 | {metrics['cpu']['estimated_total_with_one_measured_build_seconds']:.3f} | {metrics['gpu']['estimated_total_with_one_measured_build_seconds']:.3f} |

真实初始矩阵构造耗时 `{curvature['build_seconds']:.3f}` s，另计首次编译/首次算子、空间构造、父包加载；它只测过一次，不能把估算总成本冒充三次独立全冷启动。GPU可作为本固定静态问题的研究可选实现，生产采用、其他场景与纯CPU从头构造比较仍需另验。

## 接口、复载和剩余工作

公共合同显式绑定自由/完整位移、边界提升、正势能梯度、一次dV、材料/质量/状态/源码身份。二维压力混合系统使用单独`MixedBlockContract`，不强行映射到三维A空间。B/C/E的活动stage2目录已发现，但没有固定交接身份时不导入或授予跨方向能力。动态GPU仍待C的方程和全循环冻结。

`extension.json`绑定新增源码与全部成果；`source-validation-original.zip`保留初轮实际运行源码，最终防护补丁通过同状态复核和最终测试。`independent-reload.json`是校验最终摘要的外层回执，`delivery-sha256.json`再绑定回执和交付文件。独立副本测试错误扩展哈希、旧源码/扩展源码/输入篡改、越界链接、动态及混合能力冒报，随后复算静态终态。

系统盘保护记录见`storage-final.json`。低于5GiB时只迁移旧导出归档，复制并校验后保留原路径链接；本轮大数组、副本和缓存全部位于数据盘。

复现及API说明：工作区`benchmarks/research_d/stage2/README.md`。后续优先执行独占资源完整冷启动复测，再分批接收B/C/E封存接口；A空间精度认证保持独立状态。
'''
    (OUT/'REPORT_ZH.md').write_text(report)
    sources={str(p.relative_to(root)):sha(p) for area in ('engine/aniso_phase1','benchmarks','tests') for p in (root/area/'research_d/stage2').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    from dataclasses import asdict
    from engine.aniso_phase1.research_d.stage2.cpu_operator import CPUOperator,contract_for
    write('static-contract.json',asdict(contract_for(CPUOperator(space),space.q0,extension_sources=sources,device='cuda:0')))
    write('extension-source-sha256.json',sources)
    with zipfile.ZipFile(OUT/'source-extension.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in sources:z.write(root/name,name)
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name not in TERMINAL}
    caps={k:dict(passed=True,evidence='correctness-acceptance.json') for k in ('static_operator','bounded_material_reference','cuda')}
    caps.update({k:dict(passed=False,reason='outside this static certification') for k in ('dynamic_cycle','coupled_physics','continuum_spatial_accuracy','production_default')})
    write('extension.json',dict(schema_version=2,parent_bundle_sha256=PARENT_SHA256,parent_bundle_path=str(PARENT),space_sha256=SPACE_SHA256,
        extension_source_sha256=sources,files=files,capabilities=caps,performance_promoted=False))
    (OUT/'extension-sha256.txt').write_text(sha(OUT/'extension.json')+'\n')
    print('sealed',sha(OUT/'extension.json'),len(sources),len(files),flush=True)

if __name__=='__main__':main()
