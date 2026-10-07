# 参考时间加密与光滑相容初态对照（2026-09-23）

本轮完成 **9 条轨迹、19,200 个更新步、48 项相关测试**。原初态及独立光滑初态的 grid 81/97 均通过速度/F/P 相邻时间步差小于 1% 的阶段门槛。光滑初态较早达到门槛，但没有显示出更高的时间收敛阶；两组空间参考变化仍明显。**本轮保留现有余量模式，没有进入局部增强。**

完整数据：[results.json](results/reference-control/v1/results.json)。上一阶段：[候选时间达标、参考尚未达标](ANISO_REFINEMENT_ZH.md)。本轮脚本：[aniso_reference_control.py](../benchmarks/aniso_reference_control.py)。

## 协议与比较尺度

原历史沿用同一 t=1 ms 保存快照，再释放 3 ms；固定 grid 97 和新旧分界公共细分 Gauss 5，从已有 dt=3.90625 μs 继续计算 1.953125、.9765625 μs。另为 grid 81 计算相同两档，检查其时间敏感性后，在相同最细 dt 下比较 81→97。

光滑对照单独保存，只替换初始位置场及其解析梯度。初始速度、材料方向、参数、质量、夹具、材料积分、释放时间和 Q1 运动空间保持一致。原快照及上一轮结果保留。光滑组不是原实验的新参考，不能用它直接评判原初态候选。

时间差定义为

\[
E_v(h)=\frac{\|v_h-v_{h/2}\|_M}{\|v_{h/2}\|_M},\quad
E_F(h)=\frac{\|F_h-F_{h/2}\|_V}{\|F_{h/2}-I\|_V},\quad
E_P(h)=\frac{\|P_h-P_{h/2}\|_V}{\|P_{h/2}\|_V}.
\]

这里 h 是时间步，\(\|f\|_V^2=\sum_qw_q|f_q|^2/\sum_qw_q\)，张量使用 Frobenius 范数。当前密度为 1，速度体积范数比等于质量范数比。比较均在相同参考位置及材料方向上进行。

这些是固定 3 ms 释放**终态**的相邻差，不是整条轨迹的最大误差，也不是严格真误差上界。候选仍采用上一轮已经达到时间门槛的终态。时间检查达标不自动意味着空间参考可信。

## 光滑历史如何构造

以无量纲坐标

\[
s=\left((X_1-.25)/.5,\ (X_2-.5)/.0625,\ (X_3-.5)/.0625\right)
\]

在旧材料分界 Gauss 5 点做体积加权最小二乘，拟合原初始位移：

\[
g(X)=X+\sum_{i=1}^{5}\sum_{j=0}^{3}\sum_{k=0}^{2}c_{ijk}s_1^i s_2^j s_3^k,
\qquad F_{\rm smooth}(X)=\nabla_Xg(X).
\]

所有位移项含 s₁，左端位置因此严格保持。60 个标量多项式项只保存历史，不成为新的运动未知量。没有单独拟合 F，也没有先将 g 投影回 Q1。后续位置和梯度一起更新：

\[
x=g+\sum_n\Delta u_h^n,\qquad F=\nabla_Xg+\sum_n\nabla_X\Delta u_h^n.
\]

后续 Q1 增量仍可产生梯度跳变，所以“光滑”只描述初始位置历史；原初始速度也没有被光滑化。新增解析历史的保存、载入和场相加支持，并将多项式系数与网格分界明确区分。

初态审计：[initial-audit.json](results/reference-control/v1/initial-audit.json)。相对原初态，位置差/位移尺度约 **.278%**，F 差/‖F−I‖ 约 **4.43%**，P 相对差约 **26.87%**。弹性能从 3.32026e-6 变为 3.11434e-6，下降约 **6.20%**；初始速度与动能不变。因此这不是只改变连续性、同时保持应力和能量不变的实验。

在旧 x 界面两侧 ε=1e-8 处，光滑历史的 F 差约 1.1e-9、牵引差约 7e-9–7e-8，并随 ε 缩小；原历史则有稳定的 F 跳变约 .0025、牵引跳变约 .49–.58。拟合积分改用 Gauss 7 后，x/F/P 相对变化仅约 5.1e-13 / 3.3e-12 / 3.7e-11，见 [拟合积分复查](results/reference-control/v1/fit-integration-check.json)。

## 时间加密结果

原 grid 97 前一轮的最后相邻 v/F/P 差为 2.3473% / .14278% / 2.2328%。本轮继续减半得到：

| 网格与初态 | 较细 dt（μs） | v 相邻差 | F 差/‖F−I‖ | P 相邻差 |
|---|---:|---:|---:|---:|
| 97 原初态 | 1.953125 | 1.41198% | .083665% | 1.37462% |
| 97 光滑初态 | 1.953125 | .87279% | .042627% | .75634% |
| 97 原初态 | .9765625 | **.78100%** | **.045728%** | **.77073%** |
| 97 光滑初态 | .9765625 | **.49532%** | **.023344%** | **.42902%** |
| 81 原初态 | .9765625 | **.60724%** | **.036846%** | **.61849%** |
| 81 光滑初态 | .9765625 | **.36447%** | **.020940%** | **.38522%** |

光滑 grid 97 在 1.953125 μs 已过门槛，原初态需要继续到 .9765625 μs。最后一组中，光滑组 v/F/P 的**绝对相邻差**分别为原组的 .612 / .510 / .543，因此改善不只是归一化分母变化。

但按最后两组绝对差计算的时间观测阶数，原 grid 97 的 v/F/P 约 .852 / .872 / .833，光滑组约 .816 / .869 / .817。光滑组差值更小、较早达标，**不等于收敛阶提高**。

![固定 grid 97 的时间相邻差](results/reference-control/v1/time-control.png)

## 时间干扰压低后，空间参考仍未可信

在相同 dt=.9765625 μs 下比较 81→97，绝对体积加权 RMS 为：

| 初态 | x 差 | F 差 | P 差 | v 差 |
|---|---:|---:|---:|---:|
| 原初态 | 1.838214e-6 | 8.294622e-4 | .227044 | .00366641 |
| 光滑初态 | 1.442004e-6 | 7.158272e-4 | .192015 | .00211890 |

对应的相对差为：

| 初态 | x 差/位移尺度 | F 差/‖F−I‖ | P 相对差 | v 相对差 |
|---|---:|---:|---:|---:|
| 原初态 | .08056% | **4.9443%** | **37.9617%** | **18.6474%** |
| 光滑初态 | .06320% | **4.2692%** | **32.8898%** | **11.1676%** |

光滑组的空间 F/P/v 绝对差约为原组的 .863 / .846 / .578。它有所改善，但远没有消除空间变化。两组的网格差都明显大于各自最后的相邻时间差；这使空间问题更容易辨认，但仍不能把网格差当成严格空间真误差。

只比较两档空间参考，也不足以估计可靠的空间收敛阶。本对照支持初始位置历史影响参考收敛，不能证明界面跳变是唯一或全部原因；初始应力、能量同时改变，原速度与释放边界仍被保留。

![同时间步的两档空间参考差](results/reference-control/v1/space-control.png)

## 局部增强的条件检查

继续使用上一轮预先约定的门槛：候选及参考时间三指标通过；参考空间变化小于方案场差的 1/10；增强相对参考的 F 误差平方超过 50% 集中在占体积 25% 的切换区。

\[
R_f=\frac{\|f_{81}-f_{97}\|_V}{\|f_{\rm enriched}-f_{\rm shifted}\|_V}.
\]

分母为两方案的场差，不是两项误差范数之差。新参考仍不能满足 R_f<.1：

| 候选网格 | R_x | R_F | R_P | R_v |
|---|---:|---:|---:|---:|
| 17 | .866 | 4.698 | 3.990 | 2.096 |
| 33 | .961 | 2.895 | 2.915 | 3.660 |

原初态增强候选的切换区 F 误差平方份额为 **29.93% / 26.99%**，未超过 50%。四项门槛中，候选时间、参考时间通过；参考变化量和局部集中门槛未通过。因此本轮保持已有 Δu=NΔuₐ+Ce，没有增加局部自由度。

下一步有依据优先继续检验空间参考，或另立实验区分初始应力、速度与释放边界的影响；本轮没有证明后两者是主因。若后续可靠参考确实显示局部集中，再试

\[
\Delta u=N\Delta u_a+\sum_k C_k\chi_k e,\qquad
\nabla_X(\chi_ke)=\chi_k\nabla_Xe+e\otimes\nabla_X\chi_k.
\]

此时仍需 Σχₖ=1 保留整体余量转动，重新检查相关模式、质量、总增量夹具、静态模式和有限转动；这些属于后续有条件的工作。

## 数值检查与资源

9 条轨迹共 19,200 步，全部通过求解、历史一致性、正 Jacobian、夹具、能量预算与耗散检查。最大实际求解残差 3.8025e-10，最小 det(F)=.9959703，最大夹具速度 0，最大能量预算闭合差 6.16e-22，最大历史 x/F 重建差 1.63e-14。最后两档的实际残差均低于 7e-11。

终态 Gauss 5/7 能量/力/切线作用的相对差：原组约 8.96e-15 / 6.96e-13 / 1.22e-13；光滑组约 1.18e-14 / 2.92e-13 / 1.22e-13。共同探针 Gauss 3→5 后，原/光滑组应力差指标变化约 4.45e-9 / 3.07e-9，其余指标低于 8e-15。全部低于预设 1e-6 复查门槛，详见 [quadrature-audit.json](results/reference-control/v1/quadrature-audit.json)。

**48 项相关测试通过，0 失败、0 跳过**，包含解析梯度、夹具、历史提交和重启、积分分界识别、原路径回归及真实 CUDA 对照，见 [tests.txt](results/reference-control/v1/tests.txt)。本轮新增 [解析历史实现](../engine/aniso_phase1/smooth_history.py) 和 [专项测试](../tests/test_aniso_reference_control.py)；原 recover() 严格检查、现有余量模式及求解方程保持原协议。

grid 81/97 分别保留 500,000 / 864,000 个材料积分点，未做材料采样压缩。开工系统盘约 9 GiB、数据盘约 41 GiB；汇总结束约 **7.08 / 49.48 GiB**，合计约 56.55 GiB。每步检查容量，未触发系统盘低于 2 GiB 的缓存迁移或两盘合计低于 5 GiB 的暂停。只保存场系数与数值日志，不落盘材料点云。

## 复现命令

以下设备编号与已保存配置一致，可复用通过验收的缓存；换设备仍求解相同方程，但配置检查会重新计算对应轨迹。均从相同初态释放 3 ms。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.aniso_reference_control --stage prepare
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind original --grid 97 --dt .000001953125 --device cuda:0
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind original --grid 97 --dt .0000009765625 --device cuda:2
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind original --grid 81 --dt .000001953125 --device cuda:0
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind original --grid 81 --dt .0000009765625 --device cuda:2
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind smooth --grid 97 --dt .00000390625 --device cuda:1
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind smooth --grid 97 --dt .000001953125 --device cuda:3
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind smooth --grid 97 --dt .0000009765625 --device cuda:0
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind smooth --grid 81 --dt .000001953125 --device cuda:1
.venv/bin/python -m benchmarks.aniso_reference_control --stage run --kind smooth --grid 81 --dt .0000009765625 --device cuda:3
.venv/bin/python -m benchmarks.aniso_reference_control --stage audit
.venv/bin/python -m benchmarks.aniso_reference_control --stage analyze
ANISO_TEST_CUDA=1 .venv/bin/python -m unittest tests.test_aniso_reference_control tests.test_aniso_refinement tests.test_aniso_convergence_reference tests.test_aniso_residual_enrichment tests.test_aniso_history_increment tests.test_aniso_consistent_transfer tests.test_aniso_template_remap tests.test_resource_guard -v
```
