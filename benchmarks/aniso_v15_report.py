"""Append measured v15 tables only after independent analysis completes."""
import json
from pathlib import Path
import numpy as np
from benchmarks.aniso_v15_analysis import OUT,BASE,ROOT,load,rows,arrays,norm,field_stress


def main():
    s=load(OUT/'summary.json');assert s['completed'];p=ROOT/'docs/ANISO_LITE_COMPATIBLE_HISTORY_ZH.md'
    doc=p.read_text();assert '## 最终结果' not in doc
    doc=doc.replace('既有 109 项加新增 5 项回归','既有 109 项加新增 6 项回归')
    lines=['## 最终结果','',
        '共同历史在本轮 F45 支撑上的实现检查通过，但通用支撑扩张的去质量门槛失败，已加拒绝保护；**本轮没有获得整体时间/空间精度认证，默认算法不切换**。下面将同输入短程比较与从头运行的完整循环分开报告。','',
        '同输入续算采用最后相邻时间步 0.00025 / 0.000125 s，应力按相同物理采样点的时间/粒子 RMS 比较。它测量的是时间敏感性，不是真解误差。','',
        '| 同起点区间 / s | 原方案相对差 | 共同历史相对差 | 原方案绝对差 / Pa | 共同历史绝对差 / Pa |',
        '|---|---:|---:|---:|---:|']
    names={'unload':'0.85–0.90，卸载','early_hold':'1.10–1.15，早期保持','late_hold':'1.60–1.65，末端续算'}
    for phase,label in names.items():
        a=s['branch_refinement'][phase]['baseline'][-1];b=s['branch_refinement'][phase]['compatible'][-1]
        lines.append(f"| {label} | {100*a['P_relative']:.4f}% | {100*b['P_relative']:.4f}% | {a['P_absolute_Pa']:.7f} | {b['P_absolute_Pa']:.7f} |")
    lines += ['', '终态差也单独保留，不能仅凭区间 RMS 达到 2% 就判定通过：','', '| 区间 | 原方案终态应力差 | 共同历史终态应力差 |','|---|---:|---:|']
    for phase,label in names.items():
        a=s['branch_refinement'][phase]['baseline'][-1];b=s['branch_refinement'][phase]['compatible'][-1]
        lines.append(f"| {label} | {100*a['P_terminal_relative']:.4f}% | {100*b['P_terminal_relative']:.4f}% |")
    old=max(c['max_history_rms'] for n,c in s['cases'].items() if '-baseline-' in n)
    new=max(c['max_history_rms'] for n,c in s['cases'].items() if '-compatible-' in n)
    lines += ['',f'同输入对照中，原方案新增相容性漂移的最大 RMS 为 {old:.6e}；共同历史全部轨迹的最大 RMS 为 {new:.6e}。这里衡量 F−(G₀Y)R：旧快照的既有差异保存在 R 内，没有被清空。','',
        '从全新参考态开始，两档完整循环采用 0.001 / 0.0005 s。下表的原方案来自 v14 相同参数归档；不能将这些粗/细差直接与上表最细两档差比较。','',
        '| 完整循环阶段 | 原方案相对差 | 共同历史相对差 | 原方案绝对差 / Pa | 共同历史绝对差 / Pa |','|---|---:|---:|---:|---:|']
    for phase,label in [('ramp','加载'),('loaded_hold','满载保持'),('unload','卸载'),('final_hold','末端保持')]:
        a=s['two_level_full_cycle_refinement']['baseline'][phase];b=s['two_level_full_cycle_refinement']['compatible'][phase]
        lines.append(f"| {label} | {100*a['P_relative']:.4f}% | {100*b['P_relative']:.4f}% | {a['P_absolute_Pa']:.7f} | {b['P_absolute_Pa']:.7f} |")
    lines += ['', 't=1.6 s 的绝对恢复状态：','', '| 时间步 / s | 方案 | 应力 RMS / Pa | 动能 / J | 稳定化能 / J |','|---|---|---:|---:|---:|']
    for level,dt in [('coarse',.001),('fine',.0005)]:
        oldrows=rows(BASE/'v14/cases'/('material-'+level)/'steps.jsonl');P=field_stress(arrays(BASE/'v14/cases'/('material-'+level)/'frames.npz'))
        v=oldrows[-1];lines.append(f"| {dt:g} | 原方案 | {norm(P[-1]):.7f} | {v['kinetic']:.6e} | {v['stabilization_energy']:.6e} |")
        v=s['cases']['cycle-compatible-'+level];lines.append(f"| {dt:g} | 共同历史 | {v['terminal_P_rms_Pa']:.7f} | {v['terminal_kinetic_J']:.6e} | {v['terminal_stabilization_J']:.6e} |")
    maxima={k:max(v['checks'][k] for v in s['cases'].values()) for k in next(iter(s['cases'].values()))['checks']}
    errors=max(max(r['errors'].values()) for r in s['independent_snapshots'])
    maxF=max(r['errors']['F'] for r in s['independent_snapshots'])
    maxforce=max(v for r in s['independent_snapshots'] for k,v in r['errors'].items() if k.endswith('_N'))
    maxenergy=max(v for r in s['independent_snapshots'] for k,v in r['errors'].items() if k.endswith('_J'))
    lines += ['', '这些终态仍是有限时间的动态状态，不能当作塑性残余或已经认证的平衡。共同历史消除了指定关系的新增漂移；应力的时间变化必须另外验收。','',
        '![储能分布和历史差异](results/lite-aniso-mainline/v15/energy-location.png)','',
        '![短程与完整循环的时间差](results/lite-aniso-mainline/v15/time-comparison.png)','',
        '| 检查 | 结果 |','|---|---|',
        '| 原有及新增回归 | 正式计算前完整 115 项通过；加保护后候选 7 项重测通过，包含新增拒绝测试（共 116 个不同测试） |',
        '| 同状态诊断 | 24 份 v14 快照，分区能量加和与历史增量关系通过 |',
        '| 同态去质量原始 Hessian | 48 组通过，有限差分步长减半复核通过 |',
        '| 新完整循环中的去质量原始 Hessian | 12 组通过 |',
        '| 正式动态计算 | 26 条，共 9300 步完成 |',
        '| 独立快照与反力复算 | 36 份通过；保存帧与即时快照逐项完全一致 |',
        '| 保护前跨支撑刚体平移 | 180 步运动学通过；补充静态谱检查失败，不能算跨支撑刚度通过 |',
        '| 支撑扩张反例与交付保护 | 64→80 活跃节点，候选额外零模态 48 个；交付代码提前拒绝且不提交物理状态 |',
        '| 公开 CLI | 加保护后 20 步重测通过 |',
        '| 四档完整循环及新空间精度 | 本轮未执行，不宣告通过 |',
        '| 旧版冻结归档 | 35 个归档、3488 份文件校验未变 |','',
        f"全轨迹阶段能量闭合最大 {maxima['stage_budget_error']:.3e} J；参考重建能量跳变最大 {maxima['stabilization_rebuild_delta']:.3e} J；粒子试探/提交差最大 {maxima['particle_trial_commit_max']:.3e}；共同 F/Y 关系闭合最大 {maxima['common_history_max']:.3e}。独立快照 F 误差最大 {maxF:.3e}，力误差最大 {maxforce:.3e} N，能量误差最大 {maxenergy:.3e} J。",'',
        '## 运行记录与复现','',
        '系统盘余量触发旧快照测试的 5 GiB 保护；将可重算的临时测试输出放到独立盘后完整重跑，保护阈值未改。正式计算的磁盘旧文件始终保留，新输出写入私有临时目录，完成后逐文件 SHA256 校验复制回项目。','',
        '首轮新实验脚本把 Warp 的 NumPy 视图直接存为帧，导致后续计算改写历史帧。该轮未用于时间精度结论；原始尝试保存在 `frame-alias-attempt`。修复为独立副本，新增真实步进回归，重跑全部 26 条正式轨迹。最终分析还逐项比对起点及中间即时快照。早期测试夹具与极小能量相减的舍入诊断记录也保留，未将失败记录混入正式通过结果。','',
        '公开选项示例（需要可用的存储/缓存目录）：','',
        '```bash','OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso \\',
        '  --device cpu --scene tensile --grid 9 --dt 0.001 --fiber-angle 45 \\',
        '  --smooth-loading --history-consistency residual_center --stabilization compatible_patch \\',
        '  --boundary-impulse-transfer --apic-transfer incremental --affine-flip-ratio 1 \\',
        '  --velocity-dissipation null --reaction-force-atol 1e-7 --headless --steps 20','```','',
        '专用诊断与驱动使用固定 v14/v15 路径；协议及轨迹目录拒绝覆盖，部分分析脚本会重写汇总，因此请在独立副本运行。精确复现保护前正式轨迹应使用 `source-executed-before-support-guard.zip`；交付源码保存在 `source-delivered.zip`，包含支撑拒绝保护、补充诊断与最终文档。两者均应解包到独立工作副本，并提供对应 v14 输入归档；依次运行 `benchmarks.aniso_compatible_diagnosis`、`benchmarks.aniso_compatible_controls`、`benchmarks.aniso_compatible_history freeze/tests/run`、`benchmarks.aniso_v15_analysis`。`--output` 供诊断输出或驱动内部 worker 临时目录使用，不会重定向驱动的固定协议根目录。','',
        '源码：[共同历史实现](../engine/aniso_phase1/compatible_patch.py)、[新增测试](../tests/test_aniso_compatible_patch.py)。数据：[空间分账](results/lite-aniso-mainline/v15/diagnosis/summary.json)、[固定映射松弛](results/lite-aniso-mainline/v15/frozen-relaxation/summary.json)、[力传递可见性](results/lite-aniso-mainline/v15/force-visibility.json)、[正式动态汇总](results/lite-aniso-mainline/v15/summary.json)、[最终归档检查](results/lite-aniso-mainline/v15/completion-check.json)。']
    p.write_text(doc.rstrip()+'\n\n'+'\n'.join(lines)+'\n')

if __name__=='__main__':main()
