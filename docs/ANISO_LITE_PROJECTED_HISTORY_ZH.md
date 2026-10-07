# v9：v8 归档、一致中心历史与独立空间验证

日期：2026-09-28。当前交付为 **v9 实验候选**；`pyproject.toml` 的包版本仍为 `0.1.0`。全部新增数值实验使用 CPU、float64 和锁定依赖，CUDA 未验收。默认路径保持 `history_consistency=standard`。

本轮完成用户要求的三项工作。实现、公式核对和实验流程完成，不等于整体精度通过：一致投影显著降低局部历史差异，但完整反力尚无可靠时间收敛，静态空间测试暴露额外软化和零刚度模式。

## 1. v8 已完成验收归档

已有第四档 `dt=0.000125 s` 的 4000 步轨迹直接复用，没有重跑这一档。补齐第四档真实 P2C/history 内核重建诊断，与前三档统一成 16 份快照；再次执行 3 项诊断测试，完成独立 NumPy 检查、四档汇总、图表、中文报告、README 和 SHA256 清单。

16 份快照分解闭合最大差 `1.077e-15`，独立重建与真实内核最大差 `8.882e-16`；固定粒子状态下再次重建的中心 F 变化精确为 0。最细相邻反力差为 **2.303%**，观测阶约 **0.100**。v8 的数值门槛通过，可靠收敛仍未建立。

完整结果：[v8 报告](ANISO_LITE_FOURTH_STEP_ZH.md)、[16 份快照](results/lite-aniso-mainline/v8/diagnosis/summary.json)、[独立核对](results/lite-aniso-mainline/v8/artifact-check.json)、[归档清单](results/lite-aniso-mainline/v8/artifact-sha256.json)。旧实现已保存为 [source-before-v9.zip](results/lite-aniso-mainline/v8/source-before-v9.zip)，便于在单独目录复现旧冻结协议。

## 2. 实现：让中心提交与粒子更新使用同一变形映射

记 `D` 为节点速度到中心材料梯度的算子，`S` 为中心到粒子的线性插值。粒子材料梯度仍采用原规则，而非位置插值的导数：

\[
L_p=\sum_c S_{pc}L_c,\quad L_c=\sum_i v_i\otimes D_{ci},\quad
F_p^{n+1}=(I+\Delta t L_p)F_p^n.
\]

历史平均采用参考体积权重：

\[
V_c=\sum_p S_{pc}V_p,\qquad W_{cp}=\frac{S_{pc}V_p}{V_c},\qquad \bar F_c=\sum_pW_{cp}F_p^n.
\]

原中心规则为 `(I+dt L_c) bar F_c`。它与粒子更新后再平均的差，包含梯度差、梯度与历史的关联项，以及移动后的权重变化。新增选项 `--history-consistency projected_center` 改用

\[
\boxed{F_c(v)=\sum_p W_{cp}(I+\Delta t L_p(v))F_p^n}
=\bar F_c+\Delta t\sum_i v_i\otimes b_{ci},
\]

\[
g_{pi}=\sum_d S_{pd}D_{di},\qquad
b_{ci}=\sum_p W_{cp}(F_p^n)^Tg_{pi}.
\]

通俗说：先按粒子真正使用的梯度更新历史，再把这些更新合成中心的材料状态。实现每步准备一次稀疏映射；Newton 迭代中权重、旧历史和映射均冻结。粒子速度、位置、APIC 系数 C、材料梯度 L 和 F 的回传规则保持原公式。

材料能仍为原来的 Hencky 基体与纤维项：

\[
\psi(F,A)=\mu\sum_j(\log\sigma_j)^2+\frac\lambda2\Big(\sum_j\log\sigma_j\Big)^2
+\frac{k_f}{2}(I_4-1)^2,\qquad I_4=\operatorname{tr}(AF^TF).
\]

令 `P=∂ψ/∂F`、`H=∂P/∂F`。统一增量势能、残差及切线为

\[
\Phi(v)=\frac12\sum_i m_i\|v_i-\hat v_i\|^2+\sum_cV_c\psi(F_c(v),A_c),
\]

\[
r_i=m_i(v_i-\hat v_i)+\Delta t\sum_cV_cP_cb_{ci},
\]

\[
(K\delta v)_i=m_i\delta v_i+
\Delta t^2\sum_cV_c\left[H_c:\Big(\sum_j\delta v_j\otimes b_{cj}\Big)\right]b_{ci}.
\]

重力等已包含在预测速度中；约束在自由度投影时施加。夹持反力也使用此映射的未投影材料力，并加入惯性贡献。精确切线用于导数核对；压缩时允许原有正定近似作为求解方向，线搜索仍评估原势能。

实现见 [projected_history.py](../engine/aniso_phase1/projected_history.py)、[入口](../demos/aniso.py)、[反力计算](../engine/aniso_phase1/tensile.py)。候选限于中心积分、`particle_resample`、`mean_tensor`、变分力和无稳定化；其余组合明确拒绝。当前稀疏映射在 CPU 上通过 SciPy 准备，尚未做 GPU 性能或复杂方向场验收。

## 3. 同状态检查与完整四档加载

采用用户指定指标及参考体积 RMS：

\[
\|Q\|_V=\sqrt{\frac{\sum_c V_c\|Q_c\|_F^2}{\sum_c V_c}},\qquad
\eta_{hist}=\frac{\|F_{rebuild}-F_{commit}\|_V}{\Delta t}.
\]

在 v7/v8 的 16 份完全相同输入状态上，新映射的固定权重闭合最大差为 `6.661e-16`，总 `η_hist` 比原方案降低 **98.24%–99.87%**。这是同状态局部诊断，不能解释成反力精度提高相同比例。

权重随粒子移动后仍有

\[
F_{rebuild}-F_{commit}=(W_{n+1}-W_n)F_p^{n+1}.
\]

新方案实际加载快照的固定权重闭合最大差 `1.776e-15`，总 `η_hist` 为 `4.373e-6–2.337e-4 /s`。移动重采样尚未消除；非线性材料平均也仍有 `ψ(WF) ≠ Wψ(F)`。

相关回归 **69/69 通过，0 跳过**，包含 5 项新测试：独立映射、非均匀历史、仿射保持、势能方向导数、精确切线与对称性、压缩、实际步的动量/历史闭合、失败回滚和非法组合。另有 **3/3 静态算子检查**，包括实际 Warp 切线去掉质量后与独立静态装配的对应。

F45 同一 `0.5 s` 加载运行四档 `dt=1/0.5/0.25/0.125 ms`，合计 **7500 步**；增加原方案 500 步锚点，总计 **8000 步**。锚点反力与已有 v7 记录精确一致。全部轨迹求解与物理检查通过，最大粒子合力误差 `7.178e-10 N`。原默认未切换。

反力差在相同物理时间 `0.05–0.5 s` 比较，定义

\[
E_k=\operatorname{RMS}_t(R_{\Delta t_k}-R_{\Delta t_{k+1}}),\quad
e_k=\frac{E_k}{\max(\operatorname{RMS}_t(R_{\Delta t_{k+1}}),10^{-3}\,N)},\quad
\rho_k=E_k/E_{k-1},\ p_k=-\log_2\rho_k.
\]

| 相邻 dt（ms） | v8 总反力相对差 | 新投影相对差 | 新投影绝对 RMS（N） | 新投影观测阶 |
|---|---:|---:|---:|---:|
| 1 → 0.5 | 2.602% | 3.018% | 3.61093e-4 | — |
| 0.5 → 0.25 | 2.525% | 2.835% | 3.48974e-4 | 0.049 |
| 0.25 → 0.125 | 2.303% | 2.452% | 3.09354e-4 | 0.174 |

最细相邻绝对差减少约 **9.49%**，但反力幅度变化使相对差从 **2.303% 增至 2.452%**。预设的“低于 5% 且绝对差下降”通过；连续两组观测阶至少 0.5 的筛查未通过，不能宣布可靠时间收敛或整体精度改善。

![新投影四档反力分解](results/lite-aniso-mainline/v9/projected-reactions.png)

20 份新加载快照还以独立 NumPy 公式重放 `x/v/C/L/F`、中心 F、SVD 应力与两侧未投影反力，最大状态/力差 `7.386e-15`。能量账本显示新候选第四档状态重建净变化约 `+1.410e-7 J`，而动能传递净变化约 `−1.039e-5 J`。这些是有符号累积量，不能用来分配全局误差因果占比，也不构成能量守恒证明。

## 4. 独立于时间误差的空间测试

固定几何 `[.125,.875]×[.375,.625]×[.375,.625] m`、夹持位置 `x≤.25 / x≥.75`、右侧位移 `5 mm`、`μ=10 Pa, λ=20 Pa`。方向组为 ISO（`kf=0`）、F0/F45/F90（`kf=200 Pa`）。

空间主测试直接求小应变静力平衡 `K_s u=f`，不含时间积分、速度、质量或惯性。材料取同一非线性势能在 `F=I` 的 Hessian；它提供静态空间验证，不是有限变形真解。

- 网格扫描：`h=1/8,1/16,1/24 m`，每单元轴向 2 个粒子（8 PPC）。
- 采样扫描：固定 `h=1/16`，轴向 `2/3/4` 个粒子，即 `8/27/64 PPC`。
- 积分对照：原中心 `D`、一致投影 `WSD`、粒子材料积分 `SD`；共 **60 组**。
- 独立参考：物理体内全积分 Q1 FEM，`h=1/32,1/48`，四方向共 **8 组**。参考 Gauss2 与 Gauss3 单元矩阵独立核对。
- 场比较：在共同 `1/96 m` 分区的 Gauss2 点计算体积范数，共 331,776 点；报告 `F−I`、PK1 应力和能量。Lite 场按实际粒子使用的中心梯度插值重建。

60 组线性残差检查通过，但全部存在可明确构造的自由零刚度模式。使用兼容奇异系统的 MINRES 代表解，未人为加质量或对角刚度；不能把位移代表解称作唯一静态解。共同梯度场会消去所展示的棋盘零模态，但完整唯一性仍未建立。

在最细 Lite 网格 `h=1/24`，相对较细 Q1 参考的差异如下。**参考自身尚未全通过加密门槛，表中是两种离散的差异，不是真实误差。**

| 方向 | 原中心反力差 | 一致投影反力差 | 粒子积分反力差 | 一致投影 F−I 差 | 一致投影 P 差 |
|---|---:|---:|---:|---:|---:|
| ISO | 0.235% | 8.786% | 4.426% | 52.112% | 45.170% |
| F0 | 0.00458% | 7.736% | 4.018% | 53.824% | 24.516% |
| F45 | 0.275% | 18.024% | 10.877% | 52.974% | 116.834% |
| F90 | 0.214% | 8.552% | 4.322% | 48.409% | 41.450% |

静态加载下 `U=R d/2`，因此总能量相对差与反力相对差相同。局部采样能量另存于逐例 JSON；它与求解器积分能不混用。

参考自身 `h=1/32 → 1/48` 的变化：

| 方向 | 反力 | F−I | P |
|---|---:|---:|---:|
| ISO | 0.258% | 6.720% | 7.850% |
| F0 | 0.00757% | 7.261% | 0.314% |
| F45 | 5.516% | 8.464% | 87.366% |
| F90 | 0.181% | 6.013% | 6.898% |

尤其 F45 应力参考尚不稳定；不能凭原中心反力与它接近就判断原中心最准确。8 个参考的独立积分能量与装配能量最大相对差仅 `1.122e-12`，所以积分实现核对通过与参考空间加密通过是两个不同条件。

F45 在 `h=1/16` 的 8/27/64 PPC 反力分别为：原中心 `0.025344/0.025142/0.025344 N`，一致投影 `0.018729/0.019826/0.019286 N`，粒子积分 `0.021058/0.021909/0.021593 N`。增加采样没有提供单调改善，也没有消除零模态。

![空间反力对照](results/lite-aniso-mainline/v9-space/spatial-reactions.png)

### 为什么闭合更好，却更软？

在 `F=I`、均匀材料、固定支撑和半正定材料 Hessian 下，`W` 与 `S` 都是平均操作。由二次能的凸性可得

\[
K_{projected}\preceq K_{particle}\preceq K_{center}.
\]

这里的次序表示同一节点位移下的弹性能次序，不是有限变形下的普遍定理。新规则引入的两层平均会滤掉部分局部变形方差；让两份平均历史一致，并不自动恢复被平均掉的刚度。

### 弯曲必须去掉质量项

速度未知量的动态切线为 `K_v=M+dt² K_s`，因此静态刚度应检查

\[
K_s=(K_v-M)/\Delta t^2.
\]

实际 Warp 算子去质量检查已通过。在 17/33 两个梁网格、四个材料方向，共 **8 组**静态特征值测试中，中心积分分别出现 **32/64 个零或极软模态**，全积分参考均为 **0**。因此动态方程可解不能证明弯曲刚度正确。粒子采样和一致投影继承 `D` 的零空间，不能单独解决这个问题。

## 5. 有限变形动态补充：先筛时间差，再读空间差

四方向 × 两个网格（9/17）× 三档时间步（0.5/0.25/0.125 ms），共 **24 条完整轨迹、5600 步**。物理几何、材料、夹持、终点位移相同；全部使用新投影。此补充试验加载时长为 **0.05 s**、速度参数为 **0.1 m/s**，与上一节完整四档的 0.5 s 慢加载是不同协议，不能相互替代或混算观测阶。

初始两档有 6/8 个方向/网格组未通过时间门槛；完整保留原结果后，给全部组统一增加第三档，未挑选有利案例。全部 24 条轨迹求解与物理检查通过。

时间门槛预设为：反力曲线以及终态 `F−I`、`P`、弹性能的相邻差均 ≤2%。第三档加入后，最细两档的结果如下：

| 方向/网格 | 反力差 | F−I 差 | P 差 | 能量差 | 时间门槛 |
|---|---:|---:|---:|---:|---|
| ISO/9 | 1.222% | 0.795% | 1.093% | 0.270% | 通过 |
| ISO/17 | 0.965% | 0.748% | 1.324% | 0.148% | 通过 |
| F0/9 | 0.691% | 0.950% | 0.858% | 0.193% | 通过 |
| F0/17 | 1.037% | 0.634% | 0.485% | 0.009% | 通过 |
| F45/9 | 0.853% | 1.269% | 4.821% | 3.639% | 未通过 |
| F45/17 | 0.744% | 1.014% | 4.618% | 1.148% | 未通过 |
| F90/9 | 1.243% | 0.764% | 0.955% | 0.189% | 通过 |
| F90/17 | 1.010% | 0.685% | 1.388% | 0.192% | 通过 |

这一较快加载协议的反力观测阶为 0.913–1.009，但只有一组三档阶数，且 F45 应力时间差仍超标。`time_separation_passed=false` 被保留，分析程序相应返回 2；所有仿真子进程退出码均为 0。

在公共 `dt=0.125 ms` 下，`h=1/8 → 1/16` 的空间差异为：

| 方向 | 反力 | F−I | P | 能量 | 两网格时间筛查 |
|---|---:|---:|---:|---:|---|
| ISO | 37.318% | 46.390% | 75.601% | 16.540% | 通过 |
| F0 | 11.218% | 39.480% | 33.709% | 5.633% | 通过 |
| F45 | 15.064% | 50.732% | 160.864% | 6.785% | 未通过 |
| F90 | 40.242% | 41.907% | 55.266% | 14.344% | 通过 |

场差在 192 个共同材料位置上比较，将细网格粒子 F 插值到粗网格初始采样位置，再计算相同材料的应力。它是有限采样诊断，不等同于上一节的全域积分范数。对于 ISO/F0/F90，空间差显著大于已筛查的时间差；F45 尚不能完全排除时间混入。只有一档空间加密，不能据此计算渐近空间收敛阶或估计真解误差。

24 条轨迹的 96 份原始快照另以独立 NumPy/SciPy 稀疏公式检查粒子 `x/v/C/L/F`、中心提交、未投影夹持反力、初始 F、最终时间和位移，全部通过，最大状态/力差为 `4.430e-14`；结果见下方独立核对文件。

## 6. 验收边界与后续方向

本轮完成的是：v8 归档；一个可开关、势能/残差/切线一致的历史候选；同输入比较和完整四档复验；与时间误差分开的静态空间矩阵，以及有限变形动态补充。没有把候选设为默认，也没有以负结果为由删去案例或放宽判据。

下一轮可按以下顺序推进：

1. **补回局部变形方差，并保持变分一致。** 本轮确认单纯平均闭合会软化。可以在积分中保留 `δF_p=F_p−\bar F_c`，或研究与 `D` 零空间对应的客观稳定化势能；所有补偿均从同一势能导出力和切线，并先验证静态特征值、刚体转动和仿射场。不可直接给反力乘系数。
2. **先使参考可信，再评价空间准确性。** 优先复核 F45 的夹持过渡与应力集中，增加独立参考分辨率，分开报告全域场、远离夹持区的场和总反力。当前参考应力差太大，尚不足以给候选排名。
3. **继续分开测时间、空间与传递耗散。** 保留 `η_hist` 的固定权重/移动权重分解、弹性/惯性反力分解，以及传递能量账本。优先 ISO/F0/F45/F90；达到稳定趋势后再研究复杂方向场与 GPU 性能。

上述是本轮结果支持的后续工作，尚未作为通过验收的修复实现。

## 7. 文件与复现

- [总体交付核对](results/lite-aniso-mainline/v9/completion-check.json)、[69 项回归](results/lite-aniso-mainline/v9/tests.json)、[新测试代码](../tests/test_aniso_projected_history.py)。
- [同状态 16 例](results/lite-aniso-mainline/v9/same-state.json)、[完整四档结果](results/lite-aniso-mainline/v9/summary.json)、[20 快照独立核对](results/lite-aniso-mainline/v9/artifact-check.json)。
- [静态空间汇总](results/lite-aniso-mainline/v9-space/summary.json)、[逐例结果](results/lite-aniso-mainline/v9-space/results.json)、[梁特征值](results/lite-aniso-mainline/v9-space/beam.json)。
- [动态初始两档](results/lite-aniso-mainline/v9-dynamic-space/summary.json)、[统一第三档](results/lite-aniso-mainline/v9-dynamic-refined/summary.json)、[动态原始快照核对](results/lite-aniso-mainline/v9-dynamic-refined/artifact-check.json)。
- 各输出目录均保留 `protocol.json`、配置、逐步记录、快照和日志；最终 SHA256 清单在各目录的 `artifact-sha256.json`，交付源码为 [source-delivered.zip](results/lite-aniso-mainline/v9/source-delivered.zip)。

运行候选的短程示例：

```bash
cd /root/workspace/mpm-lite
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m demos.aniso --scene tensile --device cpu --grid 9 \
  --fiber-angle 45 --dt .001 --smooth-loading \
  --boundary-impulse-transfer --apic-transfer incremental \
  --history-consistency projected_center --reaction-force-atol 1e-8 \
  --headless --steps 20
```

复跑完整 v9 时使用空目录，避免覆盖已封存结果：

```bash
.venv/bin/python -m benchmarks.aniso_projected_history freeze --output /tmp/v9-repeat
.venv/bin/python -m benchmarks.aniso_projected_history tests --output /tmp/v9-repeat
.venv/bin/python -m benchmarks.aniso_projected_history probes --output /tmp/v9-repeat
.venv/bin/python -m benchmarks.aniso_projected_history run --output /tmp/v9-repeat --jobs 2
.venv/bin/python -m benchmarks.aniso_projected_history analyze --output /tmp/v9-repeat
.venv/bin/python -m benchmarks.aniso_projected_check --output /tmp/v9-repeat

.venv/bin/python -m benchmarks.aniso_static_space freeze --output /tmp/v9-space-repeat
.venv/bin/python -m benchmarks.aniso_static_space tests --output /tmp/v9-space-repeat
.venv/bin/python -m benchmarks.aniso_static_space run --output /tmp/v9-space-repeat
.venv/bin/python -m benchmarks.aniso_static_space analyze --output /tmp/v9-space-repeat
```

动态补充运行器是 `benchmarks.aniso_dynamic_space` 和 `benchmarks.aniso_dynamic_refine`，依次执行 `freeze/run/analyze`；`--output` 同样应选空目录。细化脚本会引用仓库内已保存的初始两档并校验其哈希，改变参考路径或协议必须形成新的实验记录。独立动态核对使用 `benchmarks.aniso_dynamic_check`。

分析返回 2 表示对应精度门槛未全部满足；不等于仿真进程失败。静态分析返回 0 表示算子与运行检查通过，空间精度要另读 `reference_resolution_passed` 和 `five_percent_screen_passed`；完整时间实验同样需另读可靠趋势标志。
