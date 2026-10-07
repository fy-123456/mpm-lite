# Lite 各向异性：反力停止条件、F45 传递诊断与慢加载对照

2026-09-27。执行用户指定的三项后续工作；上一轮 [v1 报告](ANISO_LITE_MAINLINE_RESULTS_ZH.md) 和数据保留。本轮结果位于 `docs/results/lite-aniso-mainline/v2/`。

**本轮实现与测试已完成：39 项必要回归通过，16 条轨迹共 18,000 步完整完成；网格合力残差门槛已通过，整体时间敏感性仍未通过。** 最大网格动量误差降到 1.52e-9 N；半速时间差为 ISO 7.71%、F0 0.43%、F45 10.85%、F90 8.02%，仍仅 F0 达到 5%。

## 1. 实际修改

新增可选 `reaction_force_atol`，单位 N。`None` 保持旧停止行为；本轮显式设置 `1e-7`。求解器、demo 的 `Config` 和命令行 `--reaction-force-atol` 已接通。没有改材料模型、网格、APIC/FLIP、历史模式或稳定化。

新增 `TransferAuditLedger`，在现有能量诊断上记录各阶段线动量，并在选定 F45 步保存冻结状态。该诊断只读，不改变粒子、中心或网格状态；测试已验证开启/关闭诊断得到逐元素相同的 x/v/F。

修改前的 `solver.py`、`demos/aniso.py` 保存在 `v2/before/`。工作区无 Git 元数据，采用冻结源码 SHA-256；v1 协议仍指向旧源码，其哈希校验不会假装新代码等同旧代码。

## 2. 停止条件如何与力精度对应

速度方程残差具有动量量纲：

\[
r_i=m_i(v_i^{n+1}-v_i^*)+\Delta t f_i^{\rm int},
\qquad f_i^{\rm res}=r_i/\Delta t.
\]

对自由节点合力残差，使用保守界

\[
\left|\sum_i f_{i,x}^{\rm res}\right|
\le\frac{\sqrt{N_{\rm free}}}{\Delta t}\|r\|_2
\le\frac{\sqrt{N_{\rm active}}}{\Delta t}\|r\|_2.
\]

代码取全部活跃节点数作为上界，不依赖 tensile 的固定夹具形状。当前场景一般有 225 个活跃节点，比实际自由节点数更保守。该界控制的是**投影后的平衡方程合力残差**，不是单个夹具反力相对连续真解的误差，也不消除时间或空间离散误差。

设用户要求 \(\varepsilon_R\)，先保留一半裕量：

\[
\tau_R=\frac{0.5\Delta t\varepsilon_R}{\sqrt{N_{\rm active}}},\qquad
\tau_N=\min(\tau_{\rm old},\tau_R).
\]

原本取多种容差最大值的 \(\tau_{\rm old}\) 仍保留，但不能盖过新的力误差上限。线性求解同时设置

\[
\mathrm{CG}_{\rm atol}\le0.1\tau_N,\qquad
\mathrm{CG}_{\rm rtol}\|r\|_2\le0.1\tau_N.
\]

小更新量分支也必须满足 \(\|r\|\le\tau_N\)，不能绕过上限。该选项只允许 variational 力离散及 damping=1；非法容差和非法时间步在推进状态前拒绝。达到最大迭代数仍未收敛时照常失败回滚，不伪装成功。

每步保存实际 Newton/CG 目标、残差及 `reaction_force_residual_bound`。最终仍用独立的夹具反力/动量诊断核对 `1e-7 N` 门槛。

## 3. F45 的同状态审计

F45 原速与半速、粗细时间步共四条轨迹，分别在加载进度 20%、50%、100% 的步上保存审计，合计 12 个冻结快照。每个快照包含重采样前后中心 F/P/能量、粒子 F/A0、提交后的粒子和中心 F，以及两侧速度梯度。

独立 CPU 三线性权重重新计算

\[
W_{cp}=\frac{w_{cp}V_p}{\sum_qw_{cq}V_q},\quad
\bar F_c=\sum_p W_{cp}F_p.
\]

核对实际中心 F 与这个独立加权平均的一致性，随后测量

\[
P(\bar F_c)-\sum_pW_{cp}P(F_p),\qquad
\psi(\bar F_c)-\sum_pW_{cp}\psi(F_p).
\]

这些是材料平均的非交换差异，不直接称为连续真解误差。旧中心与重采样中心按全局坐标配对，用相同当前中心体积比较，避免把体积变化混在其中；配对数单独记录。

粒子使用自己的插值速度梯度 \(G_p\)，中心使用 \(G_c\)。因此真实冻结权重下的更新差异要分成两项：

\[
\begin{aligned}
\overline{F_p^{n+1}}-F_c^{n+1}
={}&\Delta t\bigl(\overline{G_pF_p^n}-\bar G_p\bar F_p^n\bigr)\\
&+\Delta t(\bar G_p-G_c)\bar F_p^n.
\end{aligned}
\]

第一项是乘积平均的非交换项，第二项是两层梯度传递差异。再独立计算

\[
\bar F_{\rm moving}-\bar F_{\rm frozen}
=(W(x^{n+1})-W(x^n))F_p^{n+1}.
\]

分解恒等式、粒子更新恒等式以及独立重采样核对使用执行前固定的 `1e-12` 绝对门槛。该分解将相关误差分开测量，但有限快照不构成整条反力时间误差的因果证明。

每步还保存粒子起点、P2C、C2G 原始场、边界投影、隐式求解、最终网格、G2C 和粒子终点的线动量差。外力/边界造成的动量变化是物理或离散加载的一部分，不能将所有阶段变化都称为动量误差。

## 4. 冻结对照与资源规模

[protocol.json](results/lite-aniso-mainline/v2/protocol.json) 在正式运行前冻结。

| 项目 | 原速 fast | 半速 slow |
|---|---:|---:|
| 加载时间 T | 0.5 s | 1 s |
| loading_speed | 0.01 m/s | 0.005 m/s |
| 最终命令位移 | 0.005 m | 0.005 m |
| dt | 0.001 / 0.0005 s | 相同 |
| 每组粗/细步数 | 500 / 1000 | 1000 / 2000 |

每种加载率均运行 ISO/F0/F45/F90，共 16 条、18,000 步。四组只有 k_f 和角度不同；除上述加载参数外原速/半速保持一致。相同原速的新结果与 v1 已有数据比较，用来区分“调紧求解”与“减慢加载”的影响。

保持

\[
u(t)=\frac{u_*}{2}(1-\cos(\pi t/T)),\qquad u_*=0.005\ \mathrm m.
\]

时间差使用同一加载率的共同物理时间，比较区间为 `[0.1T,T]`，梯形积分 RMS，反力分母下限 `0.001 N`，门槛仍为 **5%**。方向分辨门槛仍为两条曲线绝对时间扰动之和的 2 倍。不同加载率只在相同加载进度/位移下比较，不把它们当成相同物理时间。

反力动量差门槛保持 `1e-7 N`，网格夹具速度误差 `1e-12 m/s`，夹具区域粒子平均位移误差 `0.001 m`，中心/粒子 detF 必须为正。最终摘要区分完整运行、力精度、审计恒等式和时间敏感性。

沿用 9³ 网格、192 粒子、μ=10、λ=20、k_f=0/200 Pa、密度 1、重力 0、APIC、FLIP=0.9、center/mean_tensor/particle_resample/variational、无稳定化及额外阻尼。CPU float64，最多三个单线程独立进程；唯一 GPU 满负载，本轮不声明 CUDA 验证。沿用已有磁盘保护。

## 5. 测试与复现

[tests.json](results/lite-aniso-mainline/v2/tests.json)：39 项通过，包含原主线 35 项及新增 4 项；没有跳过。另已完成一次实际 CLI 冒烟，记录见 `v2/cli-smoke.log`；该启动步不计入 18,000 个正式时间步。

新增检查覆盖：故意宽松的 Newton/CG/小更新容差无法绕过力上限；dt 减半时目标同步减半；非法参数/耗尽迭代不提交；诊断无副作用及冻结分解正确；半速加载与原速的相位和终点位移一致。

单组示例：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso \
  --scene tensile --device cpu --grid 9 --dt .001 --kf 200 --fiber-angle 45 \
  --loading-speed .005 --loading-time 1 --smooth-loading \
  --reaction-force-atol 1e-7 --linear-solver pcg \
  --headless --steps 1000 --csv /tmp/lite-aniso-slow-F45.csv
```

完整复现到新目录：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.aniso_force_accuracy freeze --output /tmp/lite-aniso-v2
.venv/bin/python -m benchmarks.aniso_force_accuracy tests --output /tmp/lite-aniso-v2
.venv/bin/python -m benchmarks.aniso_force_accuracy run --output /tmp/lite-aniso-v2 --jobs 3
.venv/bin/python -m benchmarks.aniso_force_accuracy analyze --output /tmp/lite-aniso-v2
```

已有结果不覆盖，源码哈希不匹配时拒绝继续；分析退出码 2 表示至少一项验收未通过。现场 Viser 可复用单组命令并移除 headless/steps/csv 参数；本轮没有新增 GUI 控件，也没有声称浏览器交互测试通过。

## 6. 额外定位：网格边界投影与 FLIP 回传

审计中发现并从原传递代码核对了一个确定的离散关系。令 \(M_p^n\) 为粒子总动量，\(M_g^{raw}\) 为未投影网格动量，\(M_g^{bc}\) 为施加初始边界投影后的网格动量，\(M_g^{new}\) 为求解后的网格动量。完整支持、均匀密度及本轮传递下，P2C/C2G 的总线动量近似守恒：\(M_g^{raw}=M_p^n\)。

原 C2P 使用 FLIP/PIC 混合，FLIP 增量以已经投影的 `grid_v` 为起点，因此

\[
M_p^{n+1}=\beta(M_p^n+M_g^{new}-M_g^{bc})+(1-\beta)M_g^{new}
=M_g^{new}-\beta(M_g^{bc}-M_g^{raw}).
\]

也就是

\[
\Delta M_{C2P}=-\beta\Delta M_{initial\ projection},\qquad\beta=0.9.
\]

本轮反力定义包含原始网格到最终网格的边界冲量；它与粒子最终净动量变化之间会相差约 \(0.9\Delta M_{initial\ projection}\)。这与 Newton 未收敛是两件不同的事。新的停止条件控制网格平衡残差，不会自动修复这个回传关系。

正式对照中保持原传递不变；已用所有步核对该关系并报告粒子层面的动量差，不能仅凭网格动量门槛通过就宣称整条粒子动力学守恒。该机制对全部反力时间差的贡献仍需受控修改来建立因果，不能由关系本身直接断言。

## 7. 正式结果

16 条正式轨迹共 **18,000 步**；完整运行：**通过**。必要测试 39 项全部通过。

网格合力/夹具/合法状态检查：**通过**；冻结传递审计恒等式：**通过**。时间敏感性仍按原先 5% 门槛判断，不以新的力残差通过替代。

| 组别 | 原速总反力时间差 | 半速总反力时间差 | 原速弹性分量时间差 | 半速弹性分量时间差 |
|---|---:|---:|---:|---:|
| ISO | 12.066% | 7.707% | 0.933% | 0.517% |
| F0 | 0.794% | 0.430% | 0.046% | 0.023% |
| F45 | 15.137% | 10.850% | 10.308% | 7.352% |
| F90 | 12.877% | 8.016% | 0.834% | 0.348% |

弹性分量列只帮助定位，不替代总反力判据；分量相对差使用各自 RMS 与 0.001 N 下限。

![两种加载率的反力分量](results/lite-aniso-mainline/v2/reaction-rate-comparison.png)

| 组别 | 原速惯性/总反力 RMS | 半速惯性/总反力 RMS | 半速总反力时间检查 |
|---|---:|---:|---|
| ISO | 60.13% | 19.33% | 未通过 |
| F0 | 2.51% | 1.00% | 通过 |
| F45 | 34.86% | 10.90% | 未通过 |
| F90 | 54.41% | 19.05% | 未通过 |

RMS 比值不是可相加的“力占比”，尤其在弹性与惯性反力互相抵消时不能这样解释。

| 原速细步组 | 改停止条件前最大网格动量差 (N) | 修改后 (N) | 反力曲线变化 RMS (N) |
|---|---:|---:|---:|
| ISO | 2.467e-07 | 8.174e-10 | 1.933e-09 |
| F0 | 2.729e-07 | 8.847e-10 | 8.237e-08 |
| F45 | 9.092e-08 | 8.041e-10 | 1.537e-07 |
| F90 | 2.467e-07 | 5.387e-10 | 4.253e-09 |

这一对照直接区分停止条件的作用与加载变化的作用。修改前数据引用完整保留的 v1，相同原速、时间步、材料、传递与边界；审计开启/关闭的逐元素等价已由测试确认。

| 轨迹 | 步数 | 最大网格动量差 (N) | 最大合力残差上界 (N) | 最大 Newton 次数 |
|---|---:|---:|---:|---:|
| fast-ISO-coarse | 500 | 4.124e-10 | 4.938e-09 | 1 |
| fast-ISO-fine | 1000 | 8.174e-10 | 3.939e-09 | 1 |
| fast-F0-coarse | 500 | 6.096e-10 | 5.092e-09 | 1 |
| fast-F0-fine | 1000 | 8.847e-10 | 4.970e-09 | 1 |
| fast-F45-coarse | 500 | 7.164e-10 | 1.651e-08 | 1 |
| fast-F45-fine | 1000 | 8.041e-10 | 5.039e-09 | 1 |
| fast-F90-coarse | 500 | 8.396e-10 | 4.992e-09 | 1 |
| fast-F90-fine | 1000 | 5.387e-10 | 4.614e-09 | 1 |
| slow-ISO-coarse | 1000 | 6.372e-10 | 4.895e-09 | 1 |
| slow-ISO-fine | 2000 | 1.190e-09 | 4.602e-09 | 1 |
| slow-F0-coarse | 1000 | 4.732e-10 | 3.878e-09 | 1 |
| slow-F0-fine | 2000 | 1.517e-09 | 5.015e-09 | 1 |
| slow-F45-coarse | 1000 | 5.807e-10 | 6.304e-09 | 1 |
| slow-F45-fine | 2000 | 3.851e-10 | 4.997e-09 | 1 |
| slow-F90-coarse | 1000 | 4.987e-10 | 5.000e-09 | 1 |
| slow-F90-fine | 2000 | 6.738e-10 | 4.982e-09 | 1 |

## 8. F45 冻结审计的实测分解

| 轨迹 / 加载进度 | 梯度传递项 F RMS | GF 非交换项 F RMS | 移动权重项 F RMS | 重采样 PK1 变化 RMS (Pa) |
|---|---:|---:|---:|---:|
| fast-F45-coarse / 20% | 5.695e-06 | 1.143e-09 | 3.053e-08 | 1.925e-03 |
| fast-F45-coarse / 50% | 1.214e-05 | 7.912e-09 | 2.250e-07 | 3.800e-03 |
| fast-F45-coarse / 100% | 5.198e-06 | 3.958e-09 | 1.027e-08 | 1.499e-03 |
| fast-F45-fine / 20% | 2.477e-06 | 5.317e-10 | 1.431e-08 | 9.067e-04 |
| fast-F45-fine / 50% | 5.337e-06 | 3.938e-09 | 1.078e-07 | 1.833e-03 |
| fast-F45-fine / 100% | 2.757e-06 | 2.241e-09 | 5.452e-09 | 9.334e-04 |
| slow-F45-coarse / 20% | 3.182e-06 | 4.447e-10 | 1.285e-08 | 1.032e-03 |
| slow-F45-coarse / 50% | 6.671e-06 | 4.028e-09 | 1.150e-07 | 1.890e-03 |
| slow-F45-coarse / 100% | 2.759e-06 | 1.753e-09 | 3.031e-09 | 9.267e-04 |
| slow-F45-fine / 20% | 1.393e-06 | 2.226e-10 | 6.115e-09 | 4.897e-04 |
| slow-F45-fine / 50% | 3.047e-06 | 1.992e-09 | 5.529e-08 | 9.546e-04 |
| slow-F45-fine / 100% | 1.281e-06 | 9.806e-10 | 2.084e-09 | 3.727e-04 |

这些快照中梯度传递项与 GF 非交换项的 RMS 比值为 **1230–7154 倍**，完整张量分解的最大绝对闭合误差为 **1.248e-15**。因此，本轮观察到的局部更新差异主要来自 \(\bar G_p-G_c\)，并非 GF 非交换项主导。不能将这个局部结论自动等同“已经解释整条反力误差”。

![F45 局部传递项](results/lite-aniso-mainline/v2/f45-transfer-audit.png)

三个分量的范数不能直接相加：张量可能方向不同甚至抵消。重采样前后的 F/P/能量、材料平均非交换差异及全部配对数量保留在每个 `audit-*.json/.npz`；并未用修正过的 F 替换实际模拟状态。

## 9. 网格反力与粒子动量的区别

| 轨迹 | 粒子合力平衡差 RMS (N) | 由 0.9 倍初始投影预测的 RMS (N) | C2P 冲量关系最大误差 (kg·m/s) |
|---|---:|---:|---:|
| fast-ISO-fine | 0.00630451 | 0.00630451 | 1.366e-18 |
| slow-ISO-fine | 0.00144472 | 0.00144472 | 5.638e-19 |
| fast-F45-fine | 0.00577029 | 0.00577029 | 1.231e-18 |
| slow-F45-fine | 0.00136682 | 0.00136682 | 6.397e-19 |

![网格反力与粒子动量诊断](results/lite-aniso-mainline/v2/particle-grid-momentum.png)

这里的“粒子合力平衡差”为 `(R_left+R_right) - (M_particle_end-M_particle_start)/dt`，与既有 `momentum_balance_error` 的网格定义不同。两种量都保留，不能用网格检查通过掩盖粒子端差异。全量数值见 [particle-grid-momentum.json](results/lite-aniso-mainline/v2/particle-grid-momentum.json)。

## 10. 本轮边界与后续方向

本轮完成用户要求的停止条件修正、同状态诊断与同终点慢加载对照。没有更换材料、采用独立 MPM、修改原传递、增加阻尼或事后放宽门槛；没有额外空间扫描。CUDA 和空间精度均未验证。

下一轮优先将边界投影冲量纳入 FLIP 增量的一致账目，作为独立可关闭修正，用相同状态与粒子动量核对验证；另对 F45 的中心梯度与粒子回传梯度不一致开展局部控制实验。两项都需单独隔离验证，不能把只修诊断定义当作修复动力学，也不能直接把中心 F 覆盖回粒子以求数值相同。
