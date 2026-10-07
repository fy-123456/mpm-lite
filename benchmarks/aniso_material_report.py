"""Fill the human-readable v14 report from sealed-format numerical summaries."""
from pathlib import Path
from benchmarks.aniso_material_history import BASE,OUT,ROOT,load

def main():
    s=load(OUT/'summary.json');a=load(OUT/'artifact-check.json');d=load(OUT/'deformed-massless.json');cross=load(BASE/'v14-exploration/material-support-crossing.json');p=ROOT/'docs/ANISO_LITE_MATERIAL_REFERENCE_ENERGY_ZH.md';txt=p.read_text()
    lines=['最细两档相邻差如下，同时给出绝对值，避免小残余放大百分比。反力、F 和能量的完整门槛保存在 JSON 中。','','| 阶段 | 原参考相对差 | 新方案相对差 | 原参考绝对差 / Pa | 新方案绝对差 / Pa |','|---|---:|---:|---:|---:|']
    labels={'ramp':'拉伸 0.05–0.5 s','hold':'满载保持 0.5–0.6 s','unload':'卸载 0.6–1.1 s','early_hold':'卸载后早期 1.1–1.2 s','late_hold':'延长保持 1.2–1.6 s','final_hold':'全部末端保持 1.1–1.6 s','whole':'完整观测 0.05–1.6 s'}
    for phase,label in labels.items():lines.append(f'| {label} | {100*s["refinement"]["baseline"][phase]["pairs"][-1]["P_relative"]:.4f}% | {100*s["refinement"]["material"][phase]["pairs"][-1]["P_relative"]:.4f}% | {s["refinement"]["baseline"][phase]["pairs"][-1]["P_absolute"]:.6f} | {s["refinement"]["material"][phase]["pairs"][-1]["P_absolute"]:.6f} |')
    lines+=['',f'新方案所有阶段的 R/F/P/U 及 F/P 终态 2% 门槛：**{"通过" if s["material_all_phase_time_2pct_passed"] else "未通过"}**；所有阶段规定的下降阶检查：**{"通过" if s["material_all_phase_orders_passed"] else "未通过"}**。相邻差反映时间敏感性，不是真实解误差。','', '第四档卸载后的恢复（应力除以各自 t=0.5 s 的应力 RMS）：','','| 时刻 | 原参考残余比例 | 新方案残余比例 | 新方案应力 RMS / Pa | 新方案动能 / J |','|---|---:|---:|---:|---:|']
    aa=s['phase_metrics']['baseline-fourth']['recovery_samples'];bb=s['phase_metrics']['material-fourth']['recovery_samples']
    for old,new in zip(aa,bb):
        if new['time'] in (1.1,1.2,1.4,1.6):lines.append(f'| {new["time"]:.1f} s | {100*old["P_over_load_peak"]:.3f}% | {100*new["P_over_load_peak"]:.3f}% | {new["P_rms_Pa"]:.6f} | {new["kinetic_J"]:.4e} |')
    end=bb[-1];lines+=['',f'新方案在 1.6 s 的稳定化储能占总弹性能 {100*end["stabilization_J"]/end["elastic_J"]:.2f}%，动能/弹性能为 {end["kinetic_J"]/end["elastic_J"]:.3f}。这仍是有限时间的动态观察，不能当作已达到静力平衡的残余应力。']
    lines+=['','![完整循环与延长保持](results/lite-aniso-mainline/v14/cycle-recovery.png)','', '![分阶段时间差](results/lite-aniso-mainline/v14/time-errors.png)'];txt=txt.replace('<!-- CYCLE_RESULTS -->','\n'.join(lines))
    lines=[f'新方案所有 24,000 步的参考重建能量跳变最大绝对值为 **{s["material_reference_rebuild_max_J"]:.3e} J**。四档所有满载/末端保持区间的机械能净增量检查：**{"均未增加" if s["material_all_holds_nonincrease"] else "仍有增加，见原始分账"}**。以下为第四档，单位均为 J：','','| 保持区间 | 原参考机械能变化 | 新方案机械能变化 | 原参考重建累计 | 新方案重建累计 |','|---|---:|---:|---:|']
    for phase,label in [('hold','满载保持'),('early_hold','末端前 0.1 s'),('late_hold','追加的 0.4 s'),('final_hold','末端完整 0.5 s')]:
        u=s['phase_metrics']['baseline-fourth'][phase];v=s['phase_metrics']['material-fourth'][phase];lines.append(f'| {label} | {u["mechanical_change_J"]:+.6e} | {v["mechanical_change_J"]:+.6e} | {u["stages_J"]["stabilization_rebuild_delta"]:+.6e} | {v["stages_J"]["stabilization_rebuild_delta"]:+.6e} |')
    lines+=['','新方案第四档满载/末端保持的分账：','','| 环节 | 满载保持 / J | 末端保持 / J |','|---|---:|---:|']
    labels2={'transfer_roundtrip_delta':'粒子—网格往返','boundary_projection_delta':'边界投影','solve_delta':'网格求解','final_projection_damping_delta':'末端投影','kinetic_metric_change':'移动后的动能度量','selective_dissipation_delta':'严格零空间耗散','stabilization_rebuild_delta':'参考重建'}
    for k,label in labels2.items():lines.append(f'| {label} | {s["phase_metrics"]["material-fourth"]["hold"]["stages_J"][k]:+.6e} | {s["phase_metrics"]["material-fourth"]["final_hold"]["stages_J"][k]:+.6e} |')
    v=s['phase_metrics']['material-fourth'];lines+=['',f'新方案第四档满载/末端保持中，单步机械能增量大于 $10^{{-12}}$ J 的步数分别为 {v["hold"]["positive_energy_steps"]}/{v["final_hold"]["positive_energy_steps"]}。净能量下降与每步不增是不同条件。'];txt=txt.replace('<!-- ENERGY_RESULTS -->','\n'.join(lines))
    lines=['| 验收项目 | 结果 |','|---|---|','| 原有＋新增回归 | 109 项通过，其中新增 6 项 |','| 正式完整循环 | 8 条、48,000 步完成 |','| 独立快照复算 | 80 份通过 |','| 真实公开 CLI | 20 步通过 |','| 初始去质量静态/空间组 | 12 组秩与求解通过，空间应力精度未通过 |',f'| 变形后原始去质量 Hessian | 12 组，{"全部通过" if d["all_rank_gates_passed"] else "未全部通过"} |',f'| 原对照前 1.2 s 复现 | 四档与 v13 严格零空间对照一致 |',f'| 新方案完整时间精度 | {"通过" if s["material_all_phase_time_2pct_passed"] and s["material_all_phase_orders_passed"] else "未通过"} |','| 整体精度与默认切换 | 不宣告通过；默认不切换 |','',f'独立快照状态误差最大 {a["max_state_error"]:.3e}，反力差最大 {a["max_force_error"]:.3e} N；局部历史闭合最大 {s["history_closure_max"]:.3e}，粒子 F 提交误差最大 {s["particle_commit_max"]:.3e}，标记提交误差最大 {s["material_carrier_commit_max"]:.3e}。全程阶段能量闭合最大 {max(r["max_stage_budget_error"] for r in s["checks"].values()):.3e} J。','',f'补充 180 步实际刚体平移，x 方向累计 {cross["rigid_translation_cells"][0]:.2f} 个网格单元，网格节点支撑成员变化 {cross["support_changes"]} 次；位置/F/v/C/材料标记和能量检查通过。最大位置误差 {cross["errors"]["x_error"]:.3e} m，最大重建能量跳变 {cross["errors"]["energy_rebuild_J"]:.3e} J。','', '失败诊断原样保留：首轮短程耦合脚本错误关闭了反力计算依赖的账本，修复后全 80 组重跑；联合线性诊断初次调用梯度数组索引不正确，修正后全组重跑。这些失败未计入正式通过结果。前两档预分析只作中间诊断，其最后打印语句要求第四档而退出；正式全四档分析另行完整运行。','', '![独立空间应力差](results/lite-aniso-mainline/v14/spatial-stress.png)'];txt=txt.replace('<!-- ACCEPTANCE_RESULTS -->','\n'.join(lines))
    txt=txt.replace('<!-- NEXT_RESULTS -->','## 下一步建议\n\n保留材料携带参考作为能量一致的候选，同时保留原参考＋严格零空间对照。优先在相同输入状态下检查标记更新、粒子局部历史与速度传递的相容性，比较更一致的时间耦合，尤其关注卸载后低应力阶段的绝对误差与振荡衰减。线性对照说明对称分步有价值，但不能直接推断它会修复完整非线性误差。\n\n空间方向继续以固定几何的独立参考为准，优先处理夹持过渡和梯度插值；增加粒子采样已经有小幅改善，单凭它仍不足以通过内部和边界应力精度。每个候选仍先过势能—力—切线、刚体/仿射、去质量刚度及历史闭合检查，再做整段四档循环。')
    final_old=s['phase_metrics']['baseline-fourth']['residual'];final_new=s['phase_metrics']['material-fourth']['residual']
    explanation=f'末端绝对应力：原参考 {final_old["P_rms_Pa"]:.6f} Pa，新方案 {final_new["P_rms_Pa"]:.6f} Pa，几乎相同。各自满载归一化的比例略有下降，不能据此判定卸载恢复更完整。新方案明显降低了末端动能，但仍保留较多稳定化储能；这提示下一步检查材料标记与粒子历史的相容性，尚不能证明该环节解释了全部残余。'
    txt=txt.replace('卸载残余只是该有限观察时段的响应；',explanation+'\n\n卸载残余只是该有限观察时段的响应；')
    txt=txt.replace('本轮优先处理参考重建能量，', '本轮修复了参考重建注能，并通过实现检查；完整时间/空间精度与卸载恢复仍未通过。\n\n本轮优先处理参考重建能量，')
    p.write_text(txt)
    load_old=100*s['refinement']['baseline']['ramp']['pairs'][-1]['P_relative'];load_new=100*s['refinement']['material']['ramp']['pairs'][-1]['P_relative'];uold=100*s['phase_metrics']['baseline-fourth']['residual']['P_over_load_peak'];unew=100*s['phase_metrics']['material-fourth']['residual']['P_over_load_peak']
    entry=f'> **v14 材料参考能量一致性（2026-09-29）：** [实现、公式与完整验收](docs/ANISO_LITE_MATERIAL_REFERENCE_ENERGY_ZH.md)。新增可选 `--stabilization material_patch`。109 项回归、8 条四档循环 48,000 步、80 份独立快照及 180 步实际跨网格检查完成；参考重建能量跳变为零。最细加载应力差 {load_old:.4f}%→{load_new:.4f}%；延长保持至 1.6 s 后绝对应力 {final_old["P_rms_Pa"]:.5f}→{final_new["P_rms_Pa"]:.5f} Pa，几乎相同；卸载恢复与整体时间/空间精度仍未通过，默认未切换。\n\n'
    for file in ('README.md','RUNNING_RESTORED.md'):
        q=ROOT/file;old=q.read_text();first,rest=old.split('\n',1);q.write_text(first+'\n\n'+entry+rest.lstrip('\n'))

if __name__=='__main__':main()
