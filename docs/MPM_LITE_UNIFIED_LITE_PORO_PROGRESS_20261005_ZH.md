# 统一粒子—富集—渗流 Lite 实施记录

更新时间：2026-10-05 21:03:00 +0800。

执行 [计划](MPM_LITE_PARTICLE_DECOUPLED_ANISOTROPIC_POROUS_UNIFIED_PLAN_20261005_ZH.md)。本轮已完成**有界 CPU 工程桥接和诊断交付**，并非整篇论文或原 Lite 生产集成全部完成。

## 基线与来源

- 开工时最新发布：`20261005T105825Z-coupled-reference`；SHA-256：`8428612da43ed943a47e4e3417d2439ba13d615a5fa29c15f29a3d53aaa43413`。
- 26 个带时间发布重新扫描；20 源码、20 快照、478 产物及祖先完整审计通过。
- 旧 1294 项 Python 来源冻结，新代码只在 `research_unified_lite_poro` 命名空间。原 144 局部函数、M7/q7、BD、纯固体默认周期及其许可保持原版。
- 本轮为新的小工况：2 个压力控制体、12 个 Q1 载体标量函数、2 个局部鼓包函数、30 个自由矢量分量。它不等于正式144空间，也不继承其精度或GPU结论。
- 成果目录：[20261005T123342Z-unified-lite-poro](results/unified-lite-poro/20261005T123342Z-unified-lite-poro/index.html)。

## 已实现

1. **实际粒子历史闭环。** Unload 用粒子位置、F、速度和参考速度梯度做加权 Hermite 重建；Solve 只使用有界冻结包；Load 用同一位移/梯度增量更新真实粒子，下一步重新读入。位置样本不足以区分低 PPC 的鼓包模式，联合 F/梯度重建恢复了所测三档 PPC 的完整秩；不以清零或覆盖 F 掩盖误差。
2. **完整惯性和同一形变场。** 质量包括载体—局部交叉项；材料复用 Hencky+二次纤维能；几何复用全张量 RT0 Darcy 积分。中点惯性、三点 AVF 材料力和 Simpson 体积链用于一般 Newton 混合残差；数值 Jacobian 是小模型的实现选择，不宣称解析切线或高性能生产解法。
3. **有界方向分布表示。** 保留旧 `moments` 作诊断对照，新增并在小场景选择 `fixed-positive`：固定高斯位置、正权三线性 A2/A4 插值，保留位置关联，不平均成单一方向。只支持完整笛卡尔参考粒子模板；边界常值延拓、非均匀密度/任意散乱粒子尚不支持。
4. **完整事务和恢复。** 试探期间不修改已提交粒子、系数、压力、含量和累计量；受控失败后原状态不变，重算一致。检查点具有配置与状态摘要，独立进程已恢复第12步。
5. **真正的内层隔离检查。** 冻结包在新进程中不加载粒子数组，完成材料/几何求值和一个完整耦合求解；不是只报告计数器为零。

## 异常归因与处理

首轮32点规则的粒子最大位移随 PPC 变化很大。首先在同一物理状态比较分组积分、逐粒子积分和七阶连续方向场积分，确认低 PPC 采样偏差与二阶位置矩无法充分积分局部鼓包梯度均有贡献。随后发现不同 PPC 的粒子最大值还混入了观察位置差异，因此改用共同物理点，不将原最大值差直接称作空间误差。

共同位置、共同 t=0.02 s 的 PPC2/PPC4 对照：

| 指标 | 原32点规则 | 新128点固定正权规则 |
| --- | --- | --- |
| 位移场相对差 | 7.282% | 1.857% |
| 位移最大分量差 | 2.21919e-05 m | 5.10049e-06 m |
| 压力最大差 | 0.005818 Pa | 0.000787554 Pa |

这证明所测短窗的采样敏感性降低，不等于相对真解误差为1.86%。候选的某个固定状态材料力相对同空间积分参考仍约13.4%（PPC4），因此不签发统一应力精度结论。积分参考五阶/七阶力差约3.16e-5；它仅控制同空间材料积分误差，不控制固体或压力空间误差。

HTTP首次用默认 urllib 开启器请求回环地址返回502，改成直接回环 HTTPConnection 后六项资源均正常；未修改应用或代理配置。现有 Conda Node 因 SQLite 符号缺失无法运行，未修改全局依赖；静态图已经实际查看，HTTP已验收，但未把 JavaScript 语法或浏览器交互写为通过。

## 实际运行结果

- 入选小周期：12 步至0.06 s，最大粒子位移 **0.888 mm**，最小 detF **0.998727**。加载、保持和卸载输入已覆盖，但终态仍有惯性运动，不称完全排水或完全松弛。
- 最大整步能量缺陷 5.23e-18 J，最大实际含量缺陷 2.54e-17 m³。账本小误差不能替代场精度。压力为相对排水边界的有符号表压，短暂小负值未裁剪。
- 最终 **12 项唯一合同测试通过**；此前9项属于早期回归，不叠加为21项。独立进程的完整状态摘要与第12步一致。
- 16/54/128/1024 粒子的每次材料求值均为128点；直接数组字段合计191472字节，不含空间/拓扑嵌套数组和进程峰值。Prepare与粒子存储仍依赖粒子数。
- 单次材料计时约0.59/1.76/0.59/0.58 ms，存在抖动，不能宣称时间曲线严格恒定或给通用加速比。新128点小周期约6.29 s，原32点约3.96 s；这是以较多积分换取更稳健表示的运行记录，不是加速成果。
- 固定骨架两压力单元参考中，独立 BDF 与矩阵指数压力差约1.29e-8 Pa；不把该参考推广为运动骨架或完整非线性真值。
- 全部物理尝试50次，成功提交或独立返回46步，受控故障/非法步拒绝4次；上限64。旧发布正式周期新增步数为0，全部新模拟为CPU小工况。

## 尚未通过的关键门槛

本实现采用固定材料坐标，未完成原 Lite 的 Eulerian 网格重定位和 APIC 传递适配。世界坐标粒子能够移动并反馈历史，但这不是 Eulerian 单元跨越的验证。直接把当前 BD 的固定参考几何和原 Lite 中心梯度拼接，会混用体积、材料参考和质量语义；在缺少明确适配推导的情况下，没有修改原生产路径或宣称问题已解决。

正式144空间的端到端桥接和独立充分空间参考仍缺失。按计划门槛，未开展72/144/288正式空间重分配、完整三贡献消融或隐藏集泛化，也未继承耦合q5许可。PPC静态检查、同空间材料诊断和固定骨架参考属于可独立推进的有界检查，不冒充这些未执行步骤。

## 28步逐项状态

| 步骤 | 状态 | 实际原因／范围 |
| --- | --- | --- |
| S0.1 | passed_scoped | 26 releases rechecked; full source/snapshot/artifact ancestor audit passed |
| S0.2 | passed_scoped | existing physical contract and new model identity separated |
| S0.3 | passed_scoped | exact new namespace; frozen old sources; scoped budgets |
| S1.1 | passed_scoped | explicit material-reference particle and fluid state |
| S1.2 | passed_scoped | owned frozen inner interface and whole-step commit |
| S1.3 | limited | engineering protocol frozen; paper heldout matrix not activated |
| S2.1 | passed_scoped | moment8 diagnosed; fixed positive 128-site candidate selected for structured particles |
| S2.2 | limited | small enriched consistent-mass model passed; formal 144 adapter not integrated |
| S2.3 | passed_scoped | shared finite-deformation volume and inherited full RT0 assembly |
| S2.4 | passed_scoped | independent process coupled solve loads no particle arrays |
| S3.1 | limited | real particle history feedback in material coordinates; Eulerian relocation remains unvalidated |
| S3.2 | passed_scoped | 12-step load/hold/unload engineering cycle, observable displacement |
| S3.3 | passed_scoped | complete rollback/retry and identity-bound new-process checkpoint load |
| S3.4 | limited | CPU bridge gate passed; full Eulerian Lite production gate not passed |
| S4.1 | limited | 0.06 s mechanical signal is visible; no full drainage or temporal accuracy claim |
| S4.2 | limited | independent fixed-skeleton time reference, no full moving-solid spatial reference |
| S4.3 | limited | same-space material order5/7 convergence only |
| S5.1 | not_triggered | full spatial reference and original-Lite adapter gates are unmet |
| S5.2 | not_triggered | do not train or replace formal144 against unqualified reference |
| S5.3 | not_triggered | formal space never replaced; no transfer of q5 licenses |
| S6.1 | limited | bounded CPU operator and short same-site PPC comparison; no general Eulerian claim |
| S6.2 | passed_scoped | per-call work and scoped short-run costs reported separately |
| S6.3 | limited | representation correction compared; full three-contribution ablation deferred with integration gate |
| S6.4 | not_triggered | no paper heldout generalization claim before independent spatial reference |
| S6.5 | limited | raw costs include preparation; higher Nq costs more, no speedup claim |
| S7.1 | passed_scoped | 12 unique final contract tests and fresh-process recovery |
| S7.2 | limited | claim/evidence limitations documented; paper results not complete |
| S7.3 | passed_scoped | independent scoped release, original source unchanged, viewer and CLI |

原始索引：[能力矩阵](results/unified-lite-poro/20261005T123342Z-unified-lite-poro/capability-matrix.json)、[步骤审计](results/unified-lite-poro/20261005T123342Z-unified-lite-poro/requirement-audit.json)、[测试](results/unified-lite-poro/20261005T123342Z-unified-lite-poro/S7/test-report.json)、[共同位置对照](results/unified-lite-poro/20261005T123342Z-unified-lite-poro/S6/common-site-ppc.json)。

## 下一步

先完成材料参考桥接与原 Lite Eulerian 传递之间的数学和状态映射，重点检查当前梯度与参考梯度、粒子 F 增量、质量及压力控制体的一致性；随后连接正式144空间。并行扩大算例或重新训练空间不能代替这一步。进入论文精度阶段前，再建立同物理的独立固体/压力空间参考，不能用当前两单元固定骨架参考代替。

## 最新可视化 CLI

```bash
cd /root/workspace/mpm-lite
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \
  --directory docs/results/unified-lite-poro/20261005T123342Z-unified-lite-poro
```

在 IDE 转发8765后访问 `http://127.0.0.1:8765/`。页面显示实际四个关键帧和原始曲线；形变倍数只影响显示。

独立重跑小场景（新目录/新case，避免覆盖证据）：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -B -m benchmarks.research_unified_lite_poro.run \
  --output /tmp/mpm-unified-reproduce --case fixed-cycle --ppc 3 --steps 12 \
  --rule fixed-positive
```

重跑CLI输出数值与检查点，不自动将其标记为认证发布。现有图页面使用本次封存成果。

## 资源与交付

结束检查系统盘剩余约6.63 GiB，数据盘约8.03 GiB；没有触发小于5 GiB的迁移条件。新结果体积很小，未迁移或删除旧数据。
本轮发布保留原应用默认入口和BD选择，新增CPU研究入口。最终来源/产物/祖先复核通过后封存；版本哈希以新 release.json 和审计CLI为准。
