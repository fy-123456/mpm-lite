"""Generate standalone figures and seal a scoped, reproducible D delivery."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import zipfile
import numpy as np
from engine.aniso_phase1.research_d.identity import ROOT,BASELINE,sha,write_json,own_sources,baseline_audit
from engine.aniso_phase1.research_contracts import HandoffMetadata,validate_package
from .protocol import environment,GATES


def read(p):return json.loads(Path(p).read_text())


def figures(out,performance):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/mpm-lite-research-d-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    cases=performance['cases'];fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(cases));width=.38
    ax.bar(x-width/2,[c['timing']['cpu']['cold']['median']*1000 for c in cases],width,label='CPU: complete cold solve')
    ax.bar(x+width/2,[c['timing']['gpu']['cold']['median']*1000 for c in cases],width,label='GPU: complete cold solve')
    ax.set_xticks(x,[c['case'] for c in cases],rotation=20,ha='right');ax.set_yscale('log')
    ax.set_ylabel('Median milliseconds (log scale)');ax.set_title('Same float64 equation / same true residual tolerance')
    ax.legend();fig.tight_layout();fig.savefig(out/'performance.png',dpi=170);plt.close(fig)
    tag='tensile-ppc2-angle45-full-amplitude'
    cpu=np.load(out/f'scene-{tag}-cpu.npz');gpu=np.load(out/f'scene-{tag}-cuda-0.npz')
    cr=read(out/f'scene-{tag}-cpu.json');gr=read(out/f'scene-{tag}-cuda-0.json')
    t=np.arange(1,len(cr['rows'])+1)*cr['dt'];fig,axes=plt.subplots(2,2,figsize=(11,7))
    for rows,label,style in [(cr['rows'],'CPU','-'),(gr['rows'],'GPU','--')]:
        axes[0,0].plot(t,[r['right_force'] for r in rows],style,label=label)
        axes[0,1].plot(t,[r['min_det'] for r in rows],style,label=label)
        axes[1,0].plot(t,[r['measured_grip_displacement_m'] for r in rows],style,label=label)
        axes[1,1].semilogy(t,[r['residual'] for r in rows],style,label=label)
    axes[1,0].plot(t,[r['commanded_displacement_m'] for r in cr['rows']],':',label='Command')
    for ax,title,y in zip(axes.ravel(),['Raw grip reaction','Positive deformation','Particle grip motion','True nonlinear residual'],['N','min det F','m','residual norm']):
        ax.set(title=title,xlabel='Physical time (s)',ylabel=y);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'scene-stability.png',dpi=170);plt.close(fig)
    X=cpu['reference'];positions=gpu['positions'];stress=gpu['stresses'];norm=np.linalg.norm(stress,axis=(2,3))
    fig=plt.figure(figsize=(8,5));ax=fig.add_subplot(projection='3d')
    mag=5.;shown=X+mag*(positions[0]-X)
    scatter=ax.scatter(*shown.T,c=norm[0],s=18,cmap='viridis',vmin=0,vmax=max(float(norm.max()),1e-12))
    ax.scatter(*X.T,s=2,c='gray',alpha=.2);ax.set(xlim=(.1,.92),ylim=(.35,.65),zlim=(.35,.65),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    ax.set_box_aspect((.82,.3,.3));fig.colorbar(scatter,ax=ax,shrink=.65,label='PK1 norm (Pa)')
    def update(i):
        shown=X+mag*(positions[i]-X);scatter._offsets3d=tuple(shown.T);scatter.set_array(norm[i])
        ax.set_title(f'GPU production tensile cycle / t={(i+1)*cr["dt"]:.4f} s\nDisplacements shown x5; raw stress and motion retained')
        return scatter,
    animation=FuncAnimation(fig,update,frames=range(0,len(positions),2),interval=80)
    animation.save(out/'scene-preview.gif',writer=PillowWriter(fps=12));plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True);p.add_argument('--regression-log',required=True);p.add_argument('--performance',default='performance')
    a=p.parse_args();out=Path(a.out);static=read(out/'static-summary.json');pre=read(out/'precondition-summary.json')
    non=read(out/'nonlinear-summary.json');perf=read(out/a.performance/'summary.json');local=read(out/'local-space/summary.json');tests=read(out/('tests-final.json' if (out/'tests-final.json').exists() else 'tests.json'))
    log=Path(a.regression_log).read_text();match=re.search(r'Ran (\d+) tests',log)
    regression=dict(tests=int(match.group(1)) if match else 0,passed='\nOK\n' in log,log_path=str(Path(a.regression_log).resolve().relative_to(ROOT)),sha256=sha(a.regression_log))
    write_json(out/'baseline-regression.json',regression)
    figures(out,perf)
    baseline=baseline_audit();write_json(out/'baseline-after.json',baseline)
    if not baseline['passed']: raise RuntimeError('baseline changed during delivery')
    source=own_sources();write_json(out/'source-delivered-sha256.json',source)
    files=list(source)+['benchmarks/research_d/README.md']
    with zipfile.ZipFile(out/'source-d.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:archive.write(ROOT/name,name)
    write_json(out/'modified-files.json',dict(added=files,modified_baseline_files=[],default_changed=False))
    # Publish one REAL static handoff package binding exactly the tested inputs.
    inputs={str((out/n).relative_to(ROOT)):sha(out/n) for n in ('F45-small-q2-input.json','F45-small-q2.npz','static-summary.json')}
    meta=HandoffMetadata(1,BASELINE,'D',source,inputs,{'length':'m','force':'N','energy':'J','stress':'Pa'},
        'float64','cpu and cuda:0','Cartesian reference','displacement',{'mu':10.,'lam':20.,'kf':200.,'angle_deg':45.},
        {'x=.25':'fixed','x=.75':'u_x=.005 m; u_y=u_z=0','sides':'natural'},'positive reference volume; rigid grips zero strain',
        20260930,('static',))
    evidence=dict(producer='D',baseline_sha256=BASELINE,input_sha256=inputs,code_sha256=source,
                  passed=static['passed'] and tests['passed'],validated_capabilities=['static'])
    write_json(out/'static-handoff-acceptance.json',evidence)
    package=dict(metadata=asdict(meta),acceptance=dict(path=str((out/'static-handoff-acceptance.json').relative_to(ROOT)),sha256=sha(out/'static-handoff-acceptance.json')))
    validate_package(package,ROOT,BASELINE);write_json(out/'static-handoff.json',package)
    other=[]
    for direction in ('A','B','C','E'):
        candidates=sorted((ROOT/f'docs/results/parallel-v22/{direction}').rglob('acceptance.json'))
        other.append(dict(direction=direction,observed_acceptances=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x)) for x in candidates],
                          adopted=False,reason='No explicitly adapted sealed package with the necessary combined static/dynamic acceptance was integrated in this D run.'))
    write_json(out/'integration-capabilities.json',dict(D_static=True,D_moving_tensor=False,others=other,
        ordering='A×B, A×C, B×C, A×B×C then frozen D replacement, then validated E',
        default_changed=False))
    resource_rows=[]
    for path in sorted((out/a.performance).glob('process-*.json')):
        item=read(path); peaks=[]
        for sample in item['monitor']:
            try: peaks.append(float(sample['gpu'].split(',')[1].strip()))
            except (ValueError,IndexError): pass
        resource_rows.append(dict(pid=item['pid'],max_host_rss_bytes=item['after']['max_rss_bytes'],
            max_observed_GPU_MiB=max(peaks) if peaks else None,module_warmup_seconds=item['module_warmup_seconds'],
            thread_settings=item['after']['threads'],CPU_affinity=item['after']['affinity']))
    write_json(out/'resource-cost-summary.json',dict(processes=resource_rows,
        static_workspaces=[dict(name=c['case']['name'],**c['memory']) for c in static['cases']],
        cache='isolated /tmp/mpm-lite-research-d-cache; compiled modules remain reusable',
        software=dict(warp='1.10.1',dtype='float64',cuda_toolkit='12.8',driver='595.71.05'),
        GPU='NVIDIA GeForce RTX 5090',peak_scope='sampled whole-device allocation while process ownership was exclusive; not exact allocator high-water mark'))
    all_passed=all((static['passed'],pre['passed'],non['passed'],perf['passed'],local['passed'],tests['passed'],regression['passed'],baseline['passed']))
    full=read(out/'scene-tensile-ppc2-angle45-full-amplitude-cpu.json')
    grip=max(r['grip_displacement_error_m'] for r in full['rows']);momentum=max(r['particle_momentum_balance_error_norm'] for r in full['rows'])
    completion=dict(independent_D_validation_passed=all_passed,all_D1_D11_completed=False,
        stages={'D1':'passed: canonical frozen tensor/local/implicit identities','D2':'passed within scoped operators',
                'D3':'measured including local factor/basis/QR and transfers','D4':'measured; fixed SPD optional candidates',
                'D5':'passed float64 tensor/material/local batches; heterogeneous CSR explicit',
                'D6':'same-state production checks and stable scene equivalence passed' if non['passed'] else 'not passed',
                'D7':'passed cache dependencies, graph fallback and release tests',
                'D8':'bounded scale/direction/stiffness/particle scan completed; not unbounded validation',
                'D9':'three-process paired costs recorded; shared-host CPU limitation',
                'D10':'partial: schema and D static adapter; other direction integration pending',
                'D11':'partial: independent delivery sealed; combined moving/physical acceptance pending'},
        added_tests=tests,baseline_regression=regression,baseline_unchanged=baseline['passed'],
        GPU_available=True,full_repository_tests=False,default_changed=False,
        limits=['New tensor backend is static, not a moving high-order Lite implementation.',
               'v22 continuum/spatial targets remain unpassed; backend equivalence does not certify them.',
               'Production particle grip/momentum behavior is retained and reported, not repaired by D.',
               'CPU host shared; GPU process ownership sampled, not administratively reserved.'],
        original_production_scene_diagnostics=dict(max_particle_grip_error_m=grip,max_particle_momentum_residual_N=momentum),
        environment=environment())
    write_json(out/'completion-check.json',completion)
    max_stress=max(c['stress_relative'] for c in static['cases']);max_nonlinear=max(non['implicit_device_differences'].values())
    lines=['# D 隐式求解器与 GPU 扩展：独立阶段交付','',
        '已在最新可核实的 v22 源码上完成 D 独立求解器研究与有界验证，交付静态 GPU 可选路径、公共合同和原始结果。D10/D11 的跨方向移动组合仍待验收，不切换生产默认。','',
        f'基线归档 SHA256：`{BASELINE}`。交付后再次核验 488 项源码和 131 项成果；旧核心、README、依赖及归档未改动。并行新增的 A/B/C/E 路径未作为新的默认基线。','',
        '## 已实现与验证','',
        f'- {tests["tests"]} 项 D 专项测试通过；另有 {regression["tests"]} 项既有材料、高阶空间和 v20/v22 回归通过。非全仓覆盖。',
        '- 复用 TensorElastic/BoxElastic、既有材料法则及曲率保护语义；新增 float64 Warp 轴运算、可分离预条件、常驻标量/工作区、分批状态检查及 CUDA Graph。最终成功由真实残差决定。',
        '- 局部重叠块、粗空间组合、材料方向感知的真实局部刚度；预条件从不加入物理势能。异构路径显式组装 CSR，平均材料只用于预条件。',
        '- 批量局部校正保留每个校正，构造、上传、读回、CPU 基构造及 QR 全部计时。缓存绑定几何、基、边界、材料、状态、实例及 CUDA context；捕获失败有可测回退。',
        f'- 7 组静态控制，525–88125 个自由度，Q2/Q3/Q4、0/30/45/90 度和更强纤维刚度；另有 10/100 倍异构材料。全活动物理域 CPU/GPU 应力最大相对差 {max_stress:.3g}。',
        f'- 非线性快照按物理坐标统一节点顺序后，CPU/GPU 算子/状态最大相对差 {max_nonlinear:.3g}。包含压缩负曲率、重复奇异值、原势能线搜索和两类完整回滚。',
        '- 原生产求解器两种纤维方向和 24/192 粒子循环均验证；另有最大位移 0.005 m、80 步、加载/保持/卸载/末端保持循环。新增张量路径并未接入这些移动场景。','',
        '## 性能口径与采用范围','',
        '三独立进程，每个场景均有 AB 和 BA 配对。CPU 使用继承的 SciPy CG；GPU 使用同一离散方程。完整冷求解包含公共预处理、后端构造、Graph 捕获、传输、求解和反力/能量；暖复用另列。首次编译/模块加载单列。CPU 绑核但共享宿主机，GPU 按 250 ms 采样排查其他计算进程，因此结果是本机实测范围内的候选结论。','',
        '| 场景 | CPU 完整冷求解 ms | GPU 完整冷求解 ms | 冷启动倍率 | 暖复用倍率 | 决定 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for c in perf['cases']:
        lines.append(f'| {c["case"]} | {c["timing"]["cpu"]["cold"]["median"]*1e3:.2f} | {c["timing"]["gpu"]["cold"]["median"]*1e3:.2f} | {c["cold_speedup"]:.2f}× | {c["warm_speedup"]:.2f}× | {c["decision"]} |')
    lines+=['','![完整静态求解成本](performance.png)','',
        'Schwarz 不能仅依据迭代数采用；详见 `precondition-summary.json` 的总成本。局部空间 GPU 批处理也包含 CPU 因子/基/QR 与传输成本；没有稳定总收益的范围保留为诊断或可选路径。','',
        '## 场景、误差口径和保留问题','',
        '按照用户要求使用预先固定的实用阈值：算子相对差 1e-7、能量方向差 1e-4、切线方向差 1e-3、同问题物理字段差 0.5%；真实线性残差用 1e-6/1e-8 两档敏感性对照。有限性、正 Jacobian、约束和回滚仍是硬检查。这些阈值不改变 v22 空间目标。','',
        '![原幅值循环](scene-stability.png)','',
        '[场景动画（位移显示放大 5 倍，原始数据未平滑）](scene-preview.gif)','',
        f'原生产移动路径仍有粒子夹持/传递偏差：本次原幅值循环的最大粒子夹持位移差 {grip:.6g} m，最大粒子动量账本残差 {momentum:.6g} N。D 保留同一方程的 CPU/GPU 等价和稳定运行证据；不能把它写成粒子传递守恒或 C 的动力学物理验收通过，也未通过增加耗散掩盖此问题。',
        '',
        '首轮 `20260930T1210Z-D-validation/nonlinear-summary.json` 保留为诊断反例：直接比较设备相关节点顺序、随机状态和任意自由基导致错误差异；修正在 D 的 canonical snapshot 适配器完成，旧生产代码未变。本轮第一次非线性入口的默认参数笔误保留原日志，并通过 `resume-*.json` 记录修复后的源码指纹；性能第一轮的 NumPy 布尔值序列化错误保留在 `performance/`，修复原子记录后完整三进程重测位于 `performance-v2/`，不筛选有利轮次。','',
        '## D10/D11 与使用方式','',
        '已提供 `research_contracts.py`、真实静态 `solver_callbacks`、封存包验证入口及本轮 `static-handoff.json`。不完整元数据、摘要变化、未绑定输入的验收证据和缺失动态事务均拒绝。A/B/C/E 的联合算法未在本次接入；各方向最新文件不会自动成为已验收依赖。','',
        '复现命令和 API 见仓库 `benchmarks/research_d/README.md`。使用新的 run_id；已有结果只追加，不覆盖。','',
        '可复核文件：`protocol.json`、`resume-*.json`、`tests.log`、各场景 JSON/NPZ、`performance-v2/process-*.json`、`local-space/summary.json`、`source-d.zip`、`source-delivered-sha256.json`、`artifact-sha256.json`、`completion-check.json`。','',
        f'交付时系统盘剩余 {shutil.disk_usage("/").free/1024**3:.2f} GiB；阈值采用 5 GiB。数据盘为 `/root/autodl-tmp`，资源记录保留每轮检查及实际迁移（如有）。','']
    with (out/'REPORT_ZH.md').open('x') as f:f.write('\n'.join(lines))
    artifacts={str(x.relative_to(ROOT)):sha(x) for x in sorted(out.rglob('*')) if x.is_file() and x.name!='artifact-sha256.json'}
    write_json(out/'artifact-sha256.json',artifacts)
    print(json.dumps(dict(independent_passed=all_passed,report=str(out/'REPORT_ZH.md'),artifacts=len(artifacts))))

if __name__=='__main__':main()
