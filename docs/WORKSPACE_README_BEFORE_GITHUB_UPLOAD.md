# MPM Lite: Linear Kernels and Integration without Particles

**最新验证进展（v22）：** 完成两级夹持参考加密、Q3/Q4 交叉验证及同预算高阶/重叠局部支撑。40 项相关测试、96 组去质量静态候选和 4 份实际非线性检查通过。最终 Q4 重叠支撑全域应力差 12.96%、内部应力差 13.35%；全区应力验收：未通过，参考全区自检：未通过。生产默认未切换。参见 [v22 中文报告](docs/ANISO_LITE_GRIP_HIGH_ORDER_ZH.md)。


**最新验证进展（v21）：** 完成纤维轴向应变和分区应力参考加密、Q3/Q4 交叉验证及局部自由度选取。32 项相关测试、176 组去质量静态候选和 3 份真实大空间非线性复核通过。144 个收益选取局部标量函数对最终独立参考的全域应力差 43.83%、内部应力差 42.27%；空间精度仍未通过，参考也尚未完全认证。v20 时间算法保留，默认未切换。参见 [v21 中文报告](docs/ANISO_LITE_FIBER_SPACE_ADAPTIVITY_ZH.md)。

**最新验证进展（v20）：** 完成反力模式归因、运动状态分段惯性积分、6 条完整循环 73600 步和 38 组去质量空间检查。38 项相关测试、42 份独立快照复核通过。Gauss3＋材料不可见自由度静态消元的最细相邻原始反力差 0.0167145%、全程应力差 0.00667142%；时间反力通过、时间应力通过。夹持局部自由度减轻过约束，空间应力仍未通过，默认未切换。参见 [v20 中文报告](docs/ANISO_LITE_REACTION_MODES_LOCAL_DOF_ZH.md)。

**最新验证进展（v19）：** 边界速度与实际冲量共同更新完成四档 48000 步循环；最细相邻原始反力差 2.9654%、全程应力差 0.0075%；应力通过 2% 时间门槛，原始反力未通过。30 项测试、28 份独立快照、9 条细步长移动恢复完成。约束损失极小但未证明趋零；充分积分消除初始惯性欠采样，相容梯度候选的 F45 反力仍偏硬约 29.7%，空间精度未通过，默认未切换。参见 [v19 中文报告](docs/ANISO_LITE_ENDPOINT_INERTIA_COMPATIBILITY_ZH.md)。

**最新验证进展（v18）：** 移动粒子 8 条短窗轨迹 / 12000 步通过本轮 2% 门槛；四档完整循环 96000 步已完成，时间验收未通过，最细相邻档全程应力差 0.0021%、终态差 0.1376%，原始反力差 18.2291%。21 项相关测试、56 份独立快照复核完成。同一稳定化主导形状的充分积分惯性约为原采样的 4.57 倍，空间应力精度仍未通过。完整循环使用新增的受限 CPU 加载边界实现，默认未切换。参见 [v18 中文报告](docs/ANISO_LITE_MOVING_CYCLE_SPACE_ZH.md)。

**最新验证进展（v17）：** 保持 v16 算法，完成应力模态归因及两份固定几何快照的六档时间细化，12 条轨迹 / 25200 步。最细相邻档全过程应力差 0.0183% / 0.0170%，终态差 0.0127% / 0.0278%。约 2835 rad/s 的稳定化主导模式解释了原最细档的主要相位误差；最大频率不是主要指标。18 项相关测试通过，完整移动循环和空间精度尚未验收，默认算法不切换。参见 [v17 中文报告](docs/ANISO_LITE_STRESS_MODES_TIME_ZH.md)。


> **v16 支撑刚度与联合时间更新：** [实现、公式与测试结果](docs/ANISO_LITE_CARRIER_JOINT_ZH.md)。保留材料自由度并使用二次场保持的网格映射，24 组四方向支撑检查、两种联合方案各 180 步跨支撑检查通过。完成 124 项完整回归、4 项 AVF 补充测试、64 条正式轨迹 12000 步和 64 份独立终态复算。AVF 将联合后向欧拉的移动粒子最细应力差约 15.6%–16.4% 降至 2.3%–2.4%，但末端保持终态差仍约 5%；原分步方案在部分区间的相邻差仍更小。**仅为受限 CPU 研究原型，完整时间/空间精度未通过，默认未切换。**

> **v15 储能定位与共同历史实验（2026-09-29）：** [实现、公式与测试结果](docs/ANISO_LITE_COMPATIBLE_HISTORY_ZH.md)。完成 24 份卸载相关快照诊断、26 条对照轨迹 9300 步及 36 份独立复算。可选 `compatible_patch` 将新增 F/Y 历史漂移降至约 3.45×10⁻¹⁴，但应力恢复仅略有改善。115 项完整回归通过；加支撑保护后 7 项候选测试和 20 步 CLI 重测通过。**支撑扩张反例出现 48 个额外零刚度模式，现已提前拒绝该情况；候选仅限已验支撑，整体精度未通过，默认未切换。**

> **v14 材料参考能量一致性（2026-09-29）：** [实现、公式与完整验收](docs/ANISO_LITE_MATERIAL_REFERENCE_ENERGY_ZH.md)。新增可选 `--stabilization material_patch`。109 项回归、8 条四档循环 48,000 步、80 份独立快照及 180 步实际跨网格检查完成；参考重建能量跳变为零。最细加载应力差 2.5979%→2.0998%；延长保持至 1.6 s 后绝对应力 0.09771→0.09810 Pa，几乎相同；卸载恢复与整体时间/空间精度仍未通过，默认未切换。

> **v13 停载能量与联合速度耗散（2026-09-29）：** [实现、公式与完整循环验收](docs/ANISO_LITE_UNRESOLVED_VELOCITY_ZH.md)。新增可选 `--velocity-dissipation weak`；保持静态势能与局部历史，103 项回归、三方案四档完整循环 54,000 步及 96 份独立快照复算完成。F45 最细加载应力差：基线 3.1349%，严格零空间 2.5979%，弱传递 5.1157%；两种耗散改善停载能量，但均未达到应力 2% 门槛。参考重建仍存在正能量增量，空间应力精度尚未通过。**整体精度未通过，默认未切换。**

> **v12 仿射历史与时间应力改善（2026-09-28）：** [实现、公式与完整归因](docs/ANISO_LITE_AFFINE_HISTORY_TIME_ZH.md)。保留 v11 静态能量与刚度，新增可选 `--affine-flip-ratio 1`。95 项回归通过；正式四方向四档 30,000 步及归因对照 22,500 步完成。F45 最细全程应力差由 6.77% 降至 3.135%，末态应力差由 6.56% 降至 2.526%；主要改善来自保留 APIC 仿射速度历史。**应力尚未达到 2%，短停载漂移增大；整体精度未通过，默认未切换。**

> **v11 二次保持稳定化与四档慢加载（2026-09-28）：** [实现、公式与完整结果](docs/ANISO_LITE_SELECTIVE_STABILIZATION_ZH.md)。新增可选 `--history-consistency residual_center --stabilization selective_patch`。90 项回归通过，24 组静态拉伸与 16 组梁检查通过去质量刚度门槛；F45 最细候选反力相对同一旧参考由偏高 10.08% 降为偏低 2.19%。四方向四档完整慢加载共 30,000 步，64 份独立快照复算通过；F45 最细反力差 2.23%、终态应力差 6.56%，时间精度仍未通过。局部高阶参考与分区比较见报告。**默认未切换，未宣布整体精度通过。**

> **v10 局部历史与静态刚度门槛（2026-09-28）：** [实现、公式和测试结果](docs/ANISO_LITE_LOCAL_HISTORY_ZH.md)。新增 `--history-consistency residual_center`，保留逐粒子材料历史；可组合客观稳定化。83 项回归、3 项 Q1 和 2 项 Q2 参考检查通过；完成 20 个参考解、36 组静态拉伸、24 组梁检查及 1200 步短程加载。稳定化候选通过本轮全部去质量刚度门槛，但最细候选网格 F45 反力仍偏硬约 10%，参考应力及动态时间精度仍未全部达标。**默认未切换，未宣布整体精度通过。**

> **v9 一致历史与空间验证（2026-09-28）：** [实现、公式与完整结果](docs/ANISO_LITE_PROJECTED_HISTORY_ZH.md)。v8 已完成归档。新增可选 `--history-consistency projected_center`，69 项回归与 3 项静态算子检查通过；完成 8000 步四档/锚点、60 组静态空间对照、8 组参考、8 组去质量梁检查，以及 24 条动态补充轨迹（5600 步）。同状态历史差异率降低 98.24%–99.87%，但完整 F45 最细反力差为 2.452%、观测阶仅 0.174；新投影偏软，零刚度模式和空间差仍明显，F45 动态应力时间筛查未通过。**整体精度尚未验收通过，默认未切换。**

> **F45 第四档与历史重采样诊断（2026-09-28）：** [结果与公式](docs/ANISO_LITE_FOURTH_STEP_ZH.md)。新增 4000 步，3 项诊断测试与 16 份快照复核完成。最细相邻反力差 2.303%，rho=0.933。第四档的相邻反力差仍仅缓慢收缩，尚未建立可靠时间收敛。 固定粒子状态的重复重建不改变结果；局部历史差异以梯度项为主，生产未改。

> **C/L 分离与 APIC 增量回传（2026-09-28）：** [实现与完整 F45 验收](docs/ANISO_LITE_AFFINE_CONSISTENCY_ZH.md)。61 项回归、420 次有效隔离传递和 4000 步完整加载完成。最细相邻反力差 2.525%，rho=0.995，满足本轮 F45 预定数值门槛，但尚未建立可靠时间收敛；默认未切换。

> **APIC 传递频率检查（2026-09-27）：** [隔离实验与结果](docs/ANISO_LITE_APIC_FREQUENCY_ZH.md)。3 项新测试及 26 组、1840 次传递通过公式与守恒复核。确认梯度回传的空间平滑及速度／梯度反馈效应；12 份真实 F45 快照存在 21%–63% 的中心梯度往返差。该比例不是反力误差，生产默认未改，完整 F45 精度仍待验收。

> **FLIP 物理时间标定验证（2026-09-27）：** [完整结果](docs/ANISO_LITE_FLIP_TIME_ZH.md)。52 项检查通过，新增三档完整加载共 3500 步；相邻反力差由固定比例的 12.53%→15.35% 改善到 9.30%→10.19%，最细两档绝对差减少约 40%，但未达到 5% 且未显示收敛。物理检查通过，生产默认未改。

> **F45 完整三档验证（2026-09-27）：** [实验与定位报告](docs/ANISO_LITE_F45_REFINEMENT_ZH.md)。50 项检查通过，6 条完整加载共 7000 步；原路径反力相邻差 12.53%→15.35%，历史梯度对照为 11.76%→15.06%，时间精度未通过。24 份快照确认局部梯度历史差异；补充实际内核冻结测试确认固定 FLIP 混合的传递频率效应。生产默认未改，下一步优先验证按物理时间校准混合强度。

> **边界冲量修复（2026-09-27）：** [实现与测试报告](docs/ANISO_LITE_BOUNDARY_IMPULSE_ZH.md)。新增可关闭的 FLIP 边界冲量回传，47 项检查通过，最终 8 条短轨迹共 1200 步；粒子合力误差最大 9.54e-10 N。非均匀解析场测试已落地，梯度算法保持原样；F45 短程时间差仍约 9.57%，完整加载精度尚未验收。另修复初始 F 快照持有实时数组的问题，旧记录原样保留并标注。

> **原 Lite 后续优化（2026-09-27）：** [反力停止条件、F45 传递诊断与慢加载结果](docs/ANISO_LITE_FORCE_ACCURACY_ZH.md)。39 项测试通过，16 条轨迹共 18,000 步完成；网格动量误差降至 1.52e-9 N。慢加载改善反力时间差但仍未整体达到 5%；诊断确认两层梯度传递差异与初始边界投影/FLIP 冲量不一致，原传递保持不变。

> **原 Lite 各向异性主线（2026-09-27）：** [入口与本轮验收报告](docs/ANISO_LITE_MAINLINE_RESULTS_ZH.md)。35 项必要检查通过，四组两档共 6000 步完整运行，F0 方向效应可分辨；三组反力时间敏感性未通过 5% 门槛，整体预设尚未通过验收。CPU 已验证，CUDA 本轮未运行。以下保留历史研究索引。

> 新增诊断：[跨单元、平滑插值与速度保留](docs/ANISO_MPM_CROSSING_RESULTS_ZH.md)。完成事件定位、一档空间加密、仿射边界约束和速度残差实验；32 项测试通过。平滑插值＋约束降低质量病态和网格原点敏感性；速度保留虽消除投影损失，综合精度尚未改善。空间/时间收敛未建立，生产默认未替换。

> 新增验证：[X/Z 共同加密与实际空间网格 MPM](docs/ANISO_XZ_MPM_VALIDATION_ZH.md)。XZ 两档短时对照显示未加密方向贡献同量级差异；432 粒子实际跨网格验证通过 F 预测/提交与完整质量动能一致性，26 项测试通过。MPM 时间精度检查未通过，网格相位敏感性仍待解决。回放：`.venv/bin/python -m demos.aniso_spatial_mpm --port 8088`。

> 最新完成：[非均匀 Y 参考与一致质量传递](docs/ANISO_PRACTICAL_VALIDATION_ZH.md)。6 条完整网格短轨迹、576 步及 60 项回归通过，时间与终态积分复核通过；本轮均匀 Y24 优于指定局部 Y24。新增 GPU 参考 `consistent_pic` 完整质量与历史闭环，保留生产默认。查看器：`.venv/bin/python -m demos.aniso_practical --port 8087`。

> 最新静态诊断：[group4x8 的夹持、加载与插值一致性](docs/ANISO_BOUNDARY_CONSISTENCY_ZH.md)。精确材料面约束能消除夹持遗漏并改善梁位移，但不能解释全部偏软；全局 MLS 候选暴露新的积分不足，未替换生产路径。新增两个网格的位移/反力数据及 `demos.aniso_boundary` 查看器。

> 本轮进展：[group4x8 可选材料积分与稳定化组合](docs/ANISO_GROUPED_QUADRATURE_ZH.md)。已接入冻结采样势能、残差和切线，完成去质量静态核对及短程试验；默认未变。试验中不需旧稳定化消除零模态，但相对独立梁参考仍有空间误差，保留实验选项。下面“未接入”的表述指上一阶段。

上一轮离线研究：[位置—变形—方向配对采样](docs/ANISO_JOINT_SAMPLING_ZH.md)。已比较代表点与分组条件矩，21 份冻结快照中最多四组方案的能量/中心平均应力误差均低于约 1.6%；简单代表点存在明显空间方差损失，分组方案成本仍高。该轮未接入生产求解。可用 `.venv/bin/python -m demos.aniso_snapshot --data docs/results/joint-sampling --port 8083` 查看。

最新诊断：[固定快照材料积分对照](docs/ANISO_MATERIAL_SNAPSHOT_ZH.md)。已比较单中心、重建八点与粒子材料能/应力；定位到空间采样分布和方向—变形相关性误差。仅增加离线诊断与 Viser 查看器，未修改生产求解。运行 `.venv/bin/python -m demos.aniso_snapshot --port 8083` 查看。

当前实现：[二次空间重建与材料转动模板](docs/ANISO_QUADRATIC_RECONSTRUCTION_ZH.md)。新增可选 `quadratic` / `material_quadratic`，保持势能、残差与切线一致；测试梁旋转的粗网格能量变化由约 24% 降至约 1.22%，静态零模态检查通过。动态耗散与 CPU 重建成本仍待改善，默认模式未改。以下为前期记录。

最新实现与命令见[生产共旋稳定化](docs/ANISO_COROTATED_STABILIZATION_ZH.md)：新增可选 `corotated`、完整旋转导数、精确/修改切线与验证；固定网格重建的部分朝向误差仍存在。[上一轮旋转与耗散](docs/ANISO_ROTATION_DISSIPATION_ZH.md)、[空间稳定化与四阶矩](docs/ANISO_STABILIZATION_MOMENTS_ZH.md)、[中心历史迁移](docs/ANISO_HISTORY_REFERENCE_RESULTS_ZH.md)与[第二轮定量结果](docs/ANISO_QUANTITATIVE_RESULTS_ZH.md)保留前期数据。本文件中的早期阶段记录不替代当前验收。

This is the open-source reference implementation of the SIGGRAPH 2026 paper [MPM Lite: Linear Kernels and Integration without Particles](https://mpmlite.github.io/).

![teaser](assets/banner.jpg)

算法框架、公式推导、代码流程、隐式求解限制和与近期开源/论文工作的比较见：[中文技术文档](docs/MPM_LITE_ALGORITHM_ZH.md)。

## Quick Start

### Dependencies

First clone the repository via git. We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python packages.

```shell
# install required python packages
uv sync
```

### 3D Demos

The 3D demos use a browser-based Viser viewer. Install the extra viewer
dependency with `uv add viser` (or run `uv sync` after cloning a checkout that
already contains it), then open the URL printed by the demo:

| | | |
| --- | --- | --- |
| [<img src="assets/wheel.gif" width="220">](demos/)<br>Run: <code>uv run -m demos.wheel --device cuda:0</code> | [<img src="assets/noodles.gif" width="220">](demos/)<br>Run: <code>uv run -m demos.noodles --device cuda:0</code> | [<img src="assets/snow.gif" width="220">](demos/)<br>Run: <code>uv run -m demos.snow --device cuda:0</code> |

For remote access, bind the server to all interfaces and choose a port:

```shell
uv run -m demos.wheel --device cuda:0 --viser-host 0.0.0.0 --viser-port 8080
```

Then open `http://127.0.0.1:8080` locally, or forward the port over SSH.
`--viser-max-points` controls browser-side downsampling for large particle
sets. The old `--gui` flag is accepted for compatibility but is no longer
needed.

### Fixed-reference anisotropic elasticity

```shell
uv run -m demos.aniso --device auto
```

Open the printed Viser URL (default `http://127.0.0.1:8080`). The scene menu
includes a fixed-boundary fiber block, a uniform affine field, and a material-point
directional response display. Play, pause, single-step, reset, fiber angle/stiffness,
reference positions, and display-only displacement magnification are available.
See [中文运行与验收说明](docs/ANISO_VISER_DEMO_ZH.md).

### 2D Demo

Run a minimal, self-contained single-file 2D example with GUI for a simulation on pure elasticity. Add `-X utf8` for UTF-8 characters compatibility.

```shell
uv add glfw # for GUI
uv run python -X utf8 mpmlite2d.py
```

## Acknowledgements

We acknowledge support from the National Science Foundation under Grants 2153851 and 2301040, the Toyota Research Institute, Sony Corporation, and NVIDIA Corporation.

If you find this repository useful in your project, please cite the following work:

```bibtex
@article{feng2026mpmlite,
  title={MPM Lite: Linear kernels and integration without particles},
  author={Feng, Xiang and Chen, Yunuo and Yu, Chang and Su, Hao and Terzopoulos, Demetri and Yang, Yin and Masterjohn, Joseph and Castro, Alejandro and Jiang, Chenfanfu},
  journal={ACM Trans. Graph.},
  publisher = {Association for Computing Machinery},
  volume = {45},
  number = {4},
  url = {https://doi.org/10.1145/3811294},
  doi = {10.1145/3811294},
  year={2026}
}
```

能量诊断、拉伸试验、真实稀疏算子与公平基线的后续工作见 [第二轮量化验收进度](docs/ANISO_VALIDATION_PROGRESS_ZH.md)。

当前各向异性稀疏求解器采用统一增量势能残差、带曲率检查及正定近似回退的 PCG、原势能线搜索。推导与新旧离散对照见 [变分离散验证结果](docs/ANISO_VARIATIONAL_RESULTS_ZH.md)。

材料积分最新验证：[3³ 对齐积分与成本](docs/ANISO_QUADRATURE_VALIDATION_RESULTS_ZH.md)、[生产接入门槛](docs/ANISO_QUADRATURE_PRODUCTION_GATE_ZH.md)。只读可视化：`.venv/bin/python -m demos.aniso_quadrature --port 8085`。

模板切换研究：[固定材料样本的历史＋增量原型](docs/ANISO_HISTORY_INCREMENT_ZH.md)。保留旧严格恢复对照，附单次非嵌套切换、解析切线与短程释放结果；刚体转动增量表达仍有已量化限制。

后续进展：[形状余量模式增强](docs/ANISO_RESIDUAL_ENRICHMENT_ZH.md)已通过有限转动、解析导数和静态秩检查，并完成积分及时间步扫描；释放轨迹的部分变形误差仍需研究。

最新研究：[可靠积分下的时间收敛与细参考诊断](docs/ANISO_CONVERGENCE_REFERENCE_ZH.md)。共同初态、固定运动空间下完成四档 dt 和 grid 49/65 参考，40 项测试通过；时间差递减，但参考自身尚未充分收敛。新增模式造成的局部 F 差与整根梁的参考误差已分开定位。

继续加密：[候选时间达标与 grid 65/81/97 参考](docs/ANISO_REFINEMENT_ZH.md)。完成 10,560 个更新步、44 项测试；六条候选的速度/F/P 相邻时间步差均低于 1%。细参考仍未充分收敛，暂不增加局部余量模式。新增精确张量质量分解、计算复用及经 CPU 对照验证的可选双精度 GPU 本构评估。

上一轮：[参考时间达标与独立光滑相容初态对照](docs/ANISO_REFERENCE_CONTROL_ZH.md)。完成 9 条轨迹、19,200 步、48 项测试；原/光滑初态的 grid 81/97 时间相邻差均低于 1%。空间参考仍未充分收敛，局部增强条件未满足。原历史完整保留，光滑位置场及解析梯度另存，材料与初始速度保持一致。

最新结果：[空间参考加密与等能量初态、速度、边界对照](docs/ANISO_SPATIAL_ENERGY_ZH.md)。完成 25 条轨迹、119,808 步、53 项测试；原态及等能量光滑态加密至 grid 129，14 组最细时间检查与积分复核全部通过。R_F 下降至约 2.34–3.80，仍未达到 .1。匹配能量后光滑态的空间差仍较小，但参考仍不充分；速度和边界对照没有单独消除空间分辨率问题。

上一轮完成：[按方案差异尺度加密参考，并约束初态动量和应力差](docs/ANISO_REFERENCE_SCALE_ZH.md)。14 条轨迹、147,456 步、56 项相关测试通过；原态参考加密至 grid 161，共同最细时间步 .1220703125 μs，T_F/T_P/T_v 全部低于预定 .05。145→161 的 R_F=1.967–3.192，仍未达到 .1，继续保留现有整体余量模式。新增连续初态动能/总线动量匹配，以及等弹性能、位置/F 差受限的应力拟合对照；独立积分复核通过，初态改善未带来全部空间指标一致改善。

最新完成：[原态参考的轴向／截面分别加密与误差定位](docs/ANISO_DIRECTIONAL_REFERENCE_ZH.md)。10 条轨迹、307,200 步；62 项相关回归及最终 6 项专项复查通过。共同时间步减至 .06103515625 μs 后，四组 T_F/T_P/T_v 全部 <.05，独立积分复核通过；R_F=1.841–2.988，空间仍未达标。截面加密对 F 更敏感，轴向加密对速度更敏感；切换区约占 29% 的 F 差平方，继续保留整体余量模式，下一轮优先拆分 Y/Z 参考加密。

最新完成：[等单元数的 X／Y／Z 分别加密](docs/ANISO_AXIS_REFERENCE_ZH.md)。6 条轨迹、221,184 步、10 项专项回归通过；四组时间及独立积分复核通过，最大 T=0.029677。三方向空间指标仍未全部达标。Y 加密 F 变化约为 X 的 1.56 倍，X 速度变化约为 Y 的 1.89 倍；12.5% 体积的 Y 条带包含 Y 加密 F 差平方的 55.76%，支持下一轮试验非均匀 Y 参考并保留轴向对照。

旧长程实验已暂停：[等成本非均匀 Y 参考加密](docs/ANISO_NONUNIFORM_REFERENCE_ZH.md)。二分预先指定 Y 条带中的两个基线单元，与均匀 Y24 等单元数、自由度及积分点数；保留已验证的 X/Z 对照，另以 Y48 作诊断比较网格。CUDA 已恢复，17 项包含真实 GPU 的回归全部通过；六组初态预检及新网格初态高阶积分通过，四条新增轨迹的归档状态为暂停，优劣结论未完成终态与时间验收。本轮短程验证单独保存，不与旧 3 ms 轨迹混用。
