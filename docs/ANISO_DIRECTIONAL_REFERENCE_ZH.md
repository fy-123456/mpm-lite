# 原态参考的方向加密与误差定位

本轮已完成十条新增参考轨迹、307,200 步。四组参考在共同时间步 **0.06103515625 μs** 下全部通过时间门槛，独立积分复核也通过；空间门槛仍未通过，**R_F=1.84–2.99，目标为 <0.1**。

方向对照给出了更明确的下一步依据：截面加密对 F 的影响约为轴向加密的 1.66 倍，轴向加密对速度的影响约为截面的 2.02 倍。两种分辨率都需要继续检查。切换区没有独占参考变化，因此本轮保留原有整体余量模式，不推进局部增强、历史压缩或反复切换，也未继续拟合初态。

本轮扩展的是诊断参考和分析工具，候选的历史增量算法、本构及 `recover()` 的严格检查保持原实现。前轮记录见[方案差异尺度报告](ANISO_REFERENCE_SCALE_ZH.md)。

## 1. 固定条件与四组参考

原初态、材料方向与参数、3 ms 释放、左夹具、完整一致质量及 L-BFGS/Armijo 采用同一协议。初态指纹为 `8bab2b477a16f84ab2f84b6368655e249757e5a3aaccef01ff5d91d560882295`。

| 名称 | 梁内单元数 (X,Y,Z) | 自由位移分量数 | 材料积分点数 | 用途 |
|---|---|---:|---:|---|
| base | (80,20,20) | 105,840 | 4,000,000 | 原 grid 161 |
| axial | (88,20,20) | 116,424 | 4,400,000 | 仅轴向加密 |
| cross | (80,22,22) | 126,960 | 4,840,000 | 仅截面加密 |
| both | (88,22,22) | 139,656 | 5,324,000 | 共同加密，等价 grid 177 |

各被调整方向的单元数增加 10%。自由度已扣除左端夹具；axial/cross/both 相对 base 的自由度增量分别为 10.00%/19.95%/31.95%。截面组同时增加两个方向的分辨率，不能把原始变化量直接解释为等成本效率，也不能单凭该组把影响全部归给 Y。

四组都包含旧 grid 17 的全部材料界面。材料使用新旧分界公共细分上的 Gauss 5，质量使用 Gauss 2 精确积分。历史场直接保留；全部差值在相同材料参考位置 X 上比较，而非对齐新旧节点编号。

独立重建初态的最大位置差为 5.6e−16，F 差为 7.7e−14，速度及速度梯度差分别为 4.2e−17/5.8e−15。这排除了本轮方向差异由初态重新近似造成的可能。详见[初态可表达性检查](results/directional-reference/v1/initial-representation.json)。

## 2. 时间与空间门槛

统一使用体积加权 RMS；矩阵场使用 Frobenius 范数：

\[
\|f\|_{\rm RMS}=\sqrt{\frac{\sum_q V_q^0|f(X_q)|^2}{\sum_q V_q^0}},\qquad
S_f=\|f_{\rm 增强候选}-f_{\rm 未增强候选}\|_{\rm RMS}.
\]

候选 17、33 分别提供自己的固定方案差异尺度 S_f；这些编号不是本轮新参考的编号。定义

\[
T_f=\frac{\|f_{dt}-f_{dt/2}\|_{\rm RMS}}{S_f},\qquad
R_f=\frac{\|f_{\rm both}-f_{\rm base}\|_{\rm RMS}}{S_f}.
\]

预定门槛为每个网格、两个候选尺度下的 T_F/T_P/T_v 均 <0.05，空间 R_F/R_P/R_v 均 <0.1。它们衡量变化相对于方案差异的大小，不是相对于解本身的百分比。

第一档比较 .244140625/.1220703125 μs 时，共同加密组 T_v=0.050943 超限。因此继续减半，并为四组补齐共同时间步 .06103515625 μs；没有沿用 grid 161 的通过结论或放宽门槛。

| 参考 | 首档 max T | 最终 max T_F | 最终 max T_P | 最终 max T_v | 最终时间验收 |
|---|---:|---:|---:|---:|---|
| base | .044414 | .005865 | .011641 | .022646 | 通过 |
| axial | .049067 | .007250 | .013485 | .025159 | 通过 |
| cross | .045247 | .005985 | .011693 | .023097 | 通过 |
| both | **.050943，未通过** | .007091 | .013255 | .026130 | 通过 |

每列最大值均在两个候选尺度中取最大。最终列比较 .1220703125/.06103515625 μs。时间步减半后，观察到的时间变化约减半。

最终共同时间步下的空间比值为：

| 方案差异尺度 | R_F | R_P | R_v | 空间验收 |
|---|---:|---:|---:|---|
| 候选 17 | 2.9882 | 2.5934 | 1.4574 | 未通过 |
| 候选 33 | 1.8414 | 1.8942 | 2.5448 | 未通过 |

共同加密的位置差 RMS 为 5.7096e−7，F 差为 5.2758e−4，P 差为 .147560，速度差为 .00254920。位置差较小不能替代 F/P/v 的验收。

为与前轮作同时间步比较，在 .1220703125 μs 下，145→161 的 R_F 为 3.1920/1.9670，161→177 为 2.9858/1.8399；变化有所减小，但仍远高于门槛。相邻离散结果的变化不是连续精确解误差的严格上界，也不足以单独证明渐近空间收敛。

![最终时间、空间及方向对照](results/directional-reference/v1/convergence.png)

## 3. 轴向、截面及交互影响

同时检查两个条件下的方向加密：base→axial 与 cross→both 是轴向变化，base→cross 与 axial→both 是截面变化。最终共同时间步的 RMS 如下。

| 对照 | F 差 | P 差 | 速度差 |
|---|---:|---:|---:|
| base→axial | 2.7264e−4 | .096354 | .00220034 |
| base→cross | 4.5125e−4 | .112240 | .00109000 |
| cross→both | 2.7327e−4 | .097569 | .00224933 |
| axial→both | 4.5337e−4 | .116339 | .00113845 |
| base→both | 5.2758e−4 | .147560 | .00254920 |

以 base 为起点，截面 F 变化约为轴向的 1.655 倍；轴向速度变化约为截面的 2.019 倍。提高另一方向分辨率后，这一排序仍保留，但不代表两个方向相互独立。

定义交互场

\[
I_f=f_{\rm both}-f_{\rm axial}-f_{\rm cross}+f_{\rm base}.
\]

其范数与共同加密变化范数之比为 **F 11.00%、P 23.12%、v 21.23%**。因此不把各方向差的范数相加当作总误差的百分比。P 从每个完整 F 场分别计算，再做差；相加的 F 扰动并不保证 P 的交互项为零。

还独立检查方向效应的时间稳定性，例如

\[
D_{x,f}=\frac{\|(f_{\rm axial}-f_{\rm base})_{dt}-(f_{\rm axial}-f_{\rm base})_{dt/2}\|}
{\|(f_{\rm axial}-f_{\rm base})_{dt/2}\|}.
\]

最终全部方向、共同加密及交互场的 F/P/v 指标均 <.05，最大 D=.012939，即 **1.294%**。交互场的 D_F/D_P/D_v 分别为 .893%/.885%/1.126%。第一档最大 D 为 2.543%，补算后方向趋势保持稳定。

## 4. 变化集中在哪里

下表为最终共同加密变化的平方积分占比；区域体积占比同时列出，避免把区域大小误当成集中程度。

| 区域 | 体积占比 | F 差平方占比 | P 差平方占比 | 速度差平方占比 |
|---|---:|---:|---:|---:|
| 夹具邻区，X∈[.25,.3125) | 12.5% | 11.43% | 11.39% | 13.12% |
| 切换区，X∈[.4375,.5625] | 25.0% | 29.10% | 27.83% | 26.58% |
| 其余梁体 | 62.5% | 59.47% | 60.78% | 60.29% |

切换区有一定偏高，但没有独占参考变化，现有证据不支持仅在该处增加候选模式。

Y∈[.453125,.46875] 的条带占体积 12.5%，包含共同加密变化的 **42.43% F 差平方、33.62% P 差平方**。仅截面加密的对应占比为 53.62%/46.77%。轴向贡献则沿梁分布，并对速度较敏感。

共同加密 F 差平方按参考梯度的 X/Y/Z 列分解为 25.28%/69.80%/4.91%。这与 Y 条带的定位相呼应，但梯度列是场的分量，不能代替只加密 Y、只加密 Z 的因果对照。

![最终轴向、截面与共同加密剖面](results/directional-reference/v1/profiles.png)

每个点是材料条带内的体积加权 RMS，不是中心线采样；同一物理量的三个面板使用相同纵轴。灰色为夹具邻区，橙色为切换区。

原态界面只读诊断也保留：ε=1e−8 时，X/Y/Z 法向旧界面的 F 跳变 RMS 约 .002499/.0004073/5.77e−8，牵引差约 .5255/.1622/5.76e−7；另有 ε=1e−7 对照。初态 X 界面跳变较大，而终态 F 参考变化主要在 Y 导数列，说明不能从一个初态分量直接解释全部动态误差。见[界面诊断](results/directional-reference/v1/initial-interfaces.json)与[前轮剖面](results/directional-reference/v1/previous-location.png)。本轮没有用新初态覆盖原历史。

## 5. 数值与积分验收

十条新增轨迹共 307,200 步，全部沿程验收通过：

| 检查 | 本轮极值 |
|---|---:|
| 最大缩放残差 | 3.815e−13 |
| 最小 det F | .99612437 |
| 最大夹具速度 | 0 |
| 历史场与材料／质量点 F 最大差 | 8.89e−16 |
| 动能读回最大绝对差 | 2.55e−21 |
| 能量预算最大绝对残差 | 6.69e−22 |

十条轨迹初始总机械能均约 3.696019937625e−6，耗散检查全部通过。见[数值汇总](results/directional-reference/v1/numerical-summary.json)。

独立积分复核的统一门槛为相对差 <1e−6，实际结果：

- 四组终态 Gauss 5/7：能量最大差 6.32e−15，力 7.97e−13，切线作用 6.33e−15。
- 共同空间差：主积分与独立 Gauss 5 的最大差 1.08e−10。
- 四组分别复核时间差范数：最大差 9.15e−9；没有只检查最大的网格。
- 两个方案差异分母与独立 Gauss 7：最大差 3.41e−10。

这些结果支持当前比较不是由积分精度主导。它们不改变空间门槛未通过的结论。完整数据见[独立复核](results/directional-reference/v1/audit.json)。

相关回归 62 项通过，包含真实 CUDA、0 跳过；最终版本另复查 6 项方向专项测试。覆盖矩形仿射位置与梯度、一致质量、原历史保留、序列化、CPU/CUDA 力与短程轨迹，以及方向统计、非线性应力交互和时间效应指标。日志分别为 [完整回归](results/directional-reference/v1/tests.txt)与[最终专项回归](results/directional-reference/v1/tests-final-directional.txt)。

## 6. 实现、资源与复现

实现入口：

- [`directional_reference.py`](../engine/aniso_phase1/directional_reference.py)：显式矩形轴坐标，插值与梯度使用实际轴间距。
- [`aniso_spatial_energy.py`](../benchmarks/aniso_spatial_energy.py)：为原轨迹入口增加可选参考几何，原默认行为保留。
- [`aniso_directional_reference.py`](../benchmarks/aniso_directional_reference.py)：初态检查、方向对照、T/R/D、剖面、独立积分与进度记录。
- [`test_aniso_directional_reference.py`](../tests/test_aniso_directional_reference.py)：六项专项回归。

本轮运行期间定期更新文档，[进度与容量记录](results/directional-reference/v1/PROGRESS.md)已完成。系统盘低于 2 GiB 时迁移可再生缓存，两盘剩余总量低于 5 GiB 时暂停的保护持续启用。结束复查时系统／数据盘约剩 **5.34/48.9 GiB**，未需迁移或暂停。

结果目录约 217 MiB。十条轨迹进程累计耗时约 17.35 小时，采用四张 GPU 并行；这包括初始化、读回及共享负载影响，不是纯内核耗时。GPU 3 期间出现其他共享任务，运行时长不能用于等成本效率排名。第一轮补算调度的等待条件曾修正并重启，原轨迹保留；完成记录见 `schedule.json`、`finer-schedule.json`，重启前日志保留为 `attempt1-*`。

[最终结果](results/directional-reference/v1/results.json)记录共同时间步及所有门槛；[第一档归档](results/directional-reference/v1/first-time-level/results.json)保留时间门槛未全部通过的原结果。原始轨迹位于结果目录 `original/`，比较缓存包含场指纹。

以下为完整重跑命令，需保留前轮 `reference-scale/v1`、`refinement/v1` 和原始快照。读取已有结果只需执行 time/analyze/audit/plot；轨迹缓存配置包含设备编号，设备不同会重算。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ANISO_TEST_CUDA=1
export MPLCONFIGDIR=/tmp/mpm-lite-mpl-cache
for stage in initial interfaces locate; do
  .venv/bin/python -m benchmarks.aniso_directional_reference --stage "$stage"
done
for case in axial cross both; do
  for dt in 2.44140625e-7 1.220703125e-7; do
    .venv/bin/python -m benchmarks.aniso_directional_reference --stage run --case "$case" --dt "$dt" --device cuda:0
  done
done
# 首档 both 的 T_v 未通过，因此四组补齐共同半时间步。
for case in base axial cross both; do
  .venv/bin/python -m benchmarks.aniso_directional_reference --stage run --case "$case" --dt 6.103515625e-8 --device cuda:0
  .venv/bin/python -m benchmarks.aniso_directional_reference --stage time --case "$case"
done
for stage in analyze audit plot; do
  .venv/bin/python -m benchmarks.aniso_directional_reference --stage "$stage"
done
.venv/bin/python -m unittest -v tests.test_aniso_directional_reference tests.test_aniso_consistent_transfer tests.test_aniso_constrained_initial tests.test_aniso_convergence_reference tests.test_aniso_history_increment tests.test_aniso_initial_controls tests.test_aniso_reference_control tests.test_aniso_refinement tests.test_aniso_resident_reference tests.test_aniso_residual_enrichment tests.test_aniso_template_remap tests.test_resource_guard
```

## 7. 下一步依据

优先继续改进空间参考。可以从 (88,22,22) 出发，分别比较 (96,22,22)、(88,24,22)、(88,22,24)：三个方向的单元增幅均为 9.09%，材料积分点数均为 5,808,000，仍包含旧历史界面。这比同时加密 Y/Z 更容易区分原因，也使方向比较的积分成本更接近。上述三组是下一轮建议，本轮未运行；每个新网格仍需重查 T_f。

结合 Y 条带剖面判断是否值得采用非均匀参考加密，同时保留轴向速度敏感性的对照。现在定位的是参考变化，还不能把它等同于候选相对可信参考的误差。

只有参考尺度足以分辨方案差异后，才评估是否需要局部余量模式。届时若采用 χ_k e，梯度必须包含 χ_k∇_X e+e⊗∇_Xχ_k，并重新验收质量、总增量夹具、有限转动和静态模式。初态拟合、历史压缩和反复切换继续后置。
