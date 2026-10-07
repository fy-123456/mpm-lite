# 恢复环境与运行方法

**最新验证进展（v22）：** 完成两级夹持参考加密、Q3/Q4 交叉验证及同预算高阶/重叠局部支撑。40 项相关测试、96 组去质量静态候选和 4 份实际非线性检查通过。最终 Q4 重叠支撑全域应力差 12.96%、内部应力差 13.35%；全区应力验收：未通过，参考全区自检：未通过。生产默认未切换。参见 [v22 中文报告](docs/ANISO_LITE_GRIP_HIGH_ORDER_ZH.md)。


**最新验证进展（v21）：** 完成纤维轴向应变和分区应力参考加密、Q3/Q4 交叉验证及局部自由度选取。32 项相关测试、176 组去质量静态候选和 3 份真实大空间非线性复核通过。144 个收益选取局部标量函数对最终独立参考的全域应力差 43.83%、内部应力差 42.27%；空间精度仍未通过，参考也尚未完全认证。v20 时间算法保留，默认未切换。参见 [v21 中文报告](docs/ANISO_LITE_FIBER_SPACE_ADAPTIVITY_ZH.md)。

**最新验证进展（v20）：** 完成反力模式归因、运动状态分段惯性积分、6 条完整循环 73600 步和 38 组去质量空间检查。38 项相关测试、42 份独立快照复核通过。Gauss3＋材料不可见自由度静态消元的最细相邻原始反力差 0.0167145%、全程应力差 0.00667142%；时间反力通过、时间应力通过。夹持局部自由度减轻过约束，空间应力仍未通过，默认未切换。参见 [v20 中文报告](docs/ANISO_LITE_REACTION_MODES_LOCAL_DOF_ZH.md)。

**最新验证进展（v19）：** 边界速度与实际冲量共同更新完成四档 48000 步循环；最细相邻原始反力差 2.9654%、全程应力差 0.0075%；应力通过 2% 时间门槛，原始反力未通过。30 项测试、28 份独立快照、9 条细步长移动恢复完成。约束损失极小但未证明趋零；充分积分消除初始惯性欠采样，相容梯度候选的 F45 反力仍偏硬约 29.7%，空间精度未通过，默认未切换。参见 [v19 中文报告](docs/ANISO_LITE_ENDPOINT_INERTIA_COMPATIBILITY_ZH.md)。

**最新验证进展（v18）：** 移动粒子 8 条短窗轨迹 / 12000 步通过本轮 2% 门槛；四档完整循环 96000 步已完成，时间验收未通过，最细相邻档全程应力差 0.0021%、终态差 0.1376%，原始反力差 18.2291%。21 项相关测试、56 份独立快照复核完成。同一稳定化主导形状的充分积分惯性约为原采样的 4.57 倍，空间应力精度仍未通过。完整循环使用新增的受限 CPU 加载边界实现，默认未切换。参见 [v18 中文报告](docs/ANISO_LITE_MOVING_CYCLE_SPACE_ZH.md)。

**最新验证进展（v17）：** 保持 v16 算法，完成应力模态归因及两份固定几何快照的六档时间细化，12 条轨迹 / 25200 步。最细相邻档全过程应力差 0.0183% / 0.0170%，终态差 0.0127% / 0.0278%。约 2835 rad/s 的稳定化主导模式解释了原最细档的主要相位误差；最大频率不是主要指标。18 项相关测试通过，完整移动循环和空间精度尚未验收，默认算法不切换。参见 [v17 中文报告](docs/ANISO_LITE_STRESS_MODES_TIME_ZH.md)。


> **v16 支撑刚度与联合时间更新：** [实现、公式与测试结果](docs/ANISO_LITE_CARRIER_JOINT_ZH.md)。保留材料自由度并使用二次场保持的网格映射，24 组四方向支撑检查、两种联合方案各 180 步跨支撑检查通过。完成 124 项完整回归、4 项 AVF 补充测试、64 条正式轨迹 12000 步和 64 份独立终态复算。AVF 将联合后向欧拉的移动粒子最细应力差约 15.6%–16.4% 降至 2.3%–2.4%，但末端保持终态差仍约 5%；原分步方案在部分区间的相邻差仍更小。**仅为受限 CPU 研究原型，完整时间/空间精度未通过，默认未切换。**

> **v15 储能定位与共同历史实验（2026-09-29）：** [实现、公式与测试结果](docs/ANISO_LITE_COMPATIBLE_HISTORY_ZH.md)。完成 24 份卸载相关快照诊断、26 条对照轨迹 9300 步及 36 份独立复算。可选 `compatible_patch` 将新增 F/Y 历史漂移降至约 3.45×10⁻¹⁴，但应力恢复仅略有改善。115 项完整回归通过；加支撑保护后 7 项候选测试和 20 步 CLI 重测通过。**支撑扩张反例出现 48 个额外零刚度模式，现已提前拒绝该情况；候选仅限已验支撑，整体精度未通过，默认未切换。**

> **v14 材料参考能量一致性（2026-09-29）：** [实现、公式与完整验收](docs/ANISO_LITE_MATERIAL_REFERENCE_ENERGY_ZH.md)。新增可选 `--stabilization material_patch`。109 项回归、8 条四档循环 48,000 步、80 份独立快照及 180 步实际跨网格检查完成；参考重建能量跳变为零。最细加载应力差 2.5979%→2.0998%；延长保持至 1.6 s 后绝对应力 0.09771→0.09810 Pa，几乎相同；卸载恢复与整体时间/空间精度仍未通过，默认未切换。

2026-09-29 v13：**v13 停载能量与联合速度耗散（2026-09-29）：** [实现、公式与完整循环验收](docs/ANISO_LITE_UNRESOLVED_VELOCITY_ZH.md)。新增可选 `--velocity-dissipation weak`；保持静态势能与局部历史，103 项回归、三方案四档完整循环 54,000 步及 96 份独立快照复算完成。F45 最细加载应力差：基线 3.1349%，严格零空间 2.5979%，弱传递 5.1157%；两种耗散改善停载能量，但均未达到应力 2% 门槛。参考重建仍存在正能量增量，空间应力精度尚未通过。**整体精度未通过，默认未切换。**

完整循环时序、守恒证明、未通过门槛与复现命令见报告。公共选项仅启用速度耗散，完整循环使用专门验收运行器。

<!-- v13 end -->

2026-09-28 最新实验为 [v12 仿射历史与时间应力改善](docs/ANISO_LITE_AFFINE_HISTORY_TIME_ZH.md)。保持 v11 静态改善，新增独立 `--affine-flip-ratio 1`，用于 `--apic-transfer incremental`。95 项回归和四方向四档完整加载通过实现/物理检查；F45 应力时间敏感性显著减小，但仍未达到 2%，短停载漂移增大，默认未切换。

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso --device cpu --scene tensile --grid 9 --dt 0.001 --fiber-angle 45 --smooth-loading --history-consistency residual_center --stabilization selective_patch --boundary-impulse-transfer --apic-transfer incremental --affine-flip-ratio 1 --reaction-force-atol 1e-7 --headless --steps 20
```

该命令已实际运行。完整四档配置、归因对照和剩余误差见报告。以下为历史记录。

2026-09-28 最新实验为 [v11 二次保持稳定化](docs/ANISO_LITE_SELECTIVE_STABILIZATION_ZH.md)。保持锁定环境，使用 CPU float64。新选项在保留逐粒子历史的同时不惩罚二次多项式变形；去质量门槛通过，F45 静态过度约束明显减轻。90 项回归、30,000 步完整慢加载和 64 份独立快照复算完成；F45 时间应力差仍未达标，默认未切换。

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso --device cpu --scene tensile --grid 9 --dt 0.001 --fiber-angle 45 --history-consistency residual_center --stabilization selective_patch --boundary-impulse-transfer --apic-transfer incremental --headless --steps 20
```

该命令已实际验证。完整验收参数以报告中的冻结协议为准。旧版源码和结果应使用对应归档复现，不用新源码替换旧冻结协议。

以下保留 2026-09-28 的 [v10 局部历史候选](docs/ANISO_LITE_LOCAL_HISTORY_ZH.md)，仍使用本文件所述锁定环境与 CPU float64。新增选项 `--history-consistency residual_center`；与 `--stabilization corotated` 组合时，通过了本轮去质量静态门槛。83 项回归、3 项 Q1 参考检查和 2 项 Q2 参考检查通过。F45 空间偏硬及应力参考不确定性仍存在，未切换默认。

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso --scene tensile --device cpu --grid 17 --fiber-angle 45 --smooth-loading --boundary-impulse-transfer --apic-transfer incremental --history-consistency residual_center --stabilization corotated --headless --steps 20
```

以下 v9 与更早版本的运行记录保留供对照。

当前工作区：`/root/workspace/mpm-lite`。2026-09-28 按原 `uv.lock` 恢复到 Python 3.11.16；当前 v9 验证使用 CPU float64。结果与复现见 [v9 报告](docs/ANISO_LITE_PROJECTED_HISTORY_ZH.md)。本轮相关回归 69/69、静态算子检查 3/3 通过；CUDA 本轮未运行，下面的 GPU 结果属于历史环境。

```bash
cd /root/workspace/mpm-lite
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m demos.aniso --scene tensile --device cpu --grid 9 --fiber-angle 45 --smooth-loading --boundary-impulse-transfer --apic-transfer incremental --history-consistency projected_center --headless --steps 20
```

该命令启用实验候选；不传 `--history-consistency` 时仍使用原默认。整体精度限制见报告，勿将单步或回归通过当作空间/时间收敛证明。

以下保留 2026-09-26 的历史恢复记录，路径不是当前工作区。

恢复日期：2026-09-26。Python 3.11.15，使用原 uv.lock 安装依赖。

```bash
cd /home/yin/mpm-lite
.venv/bin/python -m demos.aniso --device cuda:0
```

浏览器地址：http://127.0.0.1:8080。远程使用 SSH 端口转发。

无界面验证：

```bash
cd /home/yin/mpm-lite
.venv/bin/python -m demos.aniso --device cuda:0 --headless --steps 20
```

验证：183 项 unittest，175 通过、8 跳过、0 失败；20 步 A100 GPU 仿真通过（converged=true），Viser HTTP 200、正常退出。跳过项为可选 CUDA 审计和多卡用例。
完整日志：/home/yin/restore-validation。
