# F45 第四档时间步与中心材料历史重采样诊断

**第四档的相邻反力差仍仅缓慢收缩，尚未建立可靠时间收敛。** 新增第四档完整加载和 16 份重采样快照诊断均已完成。最细相邻总反力差 2.303%，绝对差 0.000341781 N；相较上一对差异，rho=0.9331，观测阶 p=0.100。生产算法未修改。

## 1. 对照配置和完成情况

只新增 dt=0.000125 s 一条完整 F45，beta=0.9^(dt/0.001)=0.9869162813660015；共 4000 步，到达 0.5 s，余弦加载终点位移 0.005 m。保留 APIC incremental、独立 C/L、边界冲量修复、1e-7 N 反力停止条件及全部原物理参数。CPU float64，CUDA 未运行。

前三档 dt=0.001/0.0005/0.00025 s 复用 v7 已验证记录。生产代码未改；运行前及完成后验证 v7 冻结源码以及 v1–v7 成果哈希。新增 3 项诊断测试通过：真实快照分解与实际内核重建、均匀仿射零差异、非法时间步／变化支撑拒绝。此前 61 项回归对应的生产源码保持一致，本轮未重复执行。

四档均在物理时刻 0.05、0.1、0.25、0.5 s 诊断，共 16 份原始快照。先分析前三档 12 份，再追加第四档 4 份；旧诊断直接复用，不重复模拟轨迹。

## 2. 四档反力趋势

在同一粗档物理时刻集合、0.05–0.5 s 区间上进行梯形积分 RMS；相对差分母为较细反力曲线 RMS、底限 0.001 N。定义

\[
D_{i,i+1}=\operatorname{RMS}(R_{dt_i}-R_{dt_{i+1}}),\qquad
\rho_i=D_{i,i+1}/D_{i-1,i},\qquad p_i=-\log_2\rho_i.
\]

这些是时间步间差异和观测趋势，不是连续真解误差。三档旧数值按同一采样方式重新计算，避免比较口径变化。

| 分量 | 时间步对 | 绝对差 RMS (N) | 相对差 | 相邻差比 rho | 观测阶 p |
|---|---|---:|---:|---:|---:|
| 总反力 | coarse-fine | 0.0003680564 | 2.602% | — | — |
| 总反力 | fine-finest | 0.0003662687 | 2.525% | 0.9951 | 0.007 |
| 总反力 | finest-fourth | 0.0003417808 | 2.303% | 0.9331 | 0.100 |
| 弹性 | coarse-fine | 0.0003510342 | 2.431% | — | — |
| 弹性 | fine-finest | 0.0003546734 | 2.399% | 1.0104 | -0.015 |
| 弹性 | finest-fourth | 0.0003348725 | 2.216% | 0.9442 | 0.083 |
| 惯性 | coarse-fine | 2.021834e-05 | 2.022% | — | — |
| 惯性 | fine-finest | 1.412076e-05 | 1.412% | 0.6984 | 0.518 |
| 惯性 | finest-fourth | 8.838775e-06 | 0.884% | 0.6259 | 0.676 |

![四档反力](results/lite-aniso-mainline/v8/four-step-reactions.png)

第四档物理检查 通过。最大粒子合力误差 7.491e-10 N，最小粒子 J=0.99998519，最大夹持位移差 1.419e-04 m；终点反力 0.01983641 N。

预定数值条件（最细相邻总反力差 ≤5% 且绝对差下降）：通过。

## 3. 重采样诊断的精确分解

每份诊断从同一物理时刻的“已提交中心状态”和“已更新粒子状态”出发。W0 是本步开始位置的体积归一化粒子→中心权重，W1 是本步结束位置的权重。重建中心状态为

\[
\widetilde F_c^{n+1}=W_1F_p^{n+1}.
\]

记 $\bar F=W_0F_p^n$、$\bar L=W_0L_p^{n+1}$，则

\[
\widetilde F_c^{n+1}-F_c^{n+1}
=\underbrace{\Delta t[W_0(L_pF_p^n)-\bar L\bar F]}_{\delta F_{cov}}
+\underbrace{\Delta t(\bar L-L_c)\bar F}_{\delta F_{grad}}
+\underbrace{(W_1-W_0)F_p^{n+1}}_{\delta F_{move}}.
\]

三项分别表示粒子梯度与变形的关联、中心／粒子梯度更新不一致、位置变化引起的权重变化。计算使用材料梯度 L，不使用 APIC 系数 C。

对这三项同时报告单步 RMS 和除以 dt 后的变化率。单步差随 dt 变小，并不自动代表单位物理时间的影响消失。

还计算重建前后的 PK1 应力 P 和弹性能。由于材料非线性，应力与能量的分项采用固定顺序：提交状态 → 加关联项 → 加梯度项 → 加移动项 → 更换中心体积权重。有限变化之和与总变化闭合，但这种顺序分配不是唯一的因果占比。

本轮所有快照的占用中心集合不变，参考材料方向均匀；均已检查。诊断遇到改变支撑或非均匀方向时会拒绝套用这组简化分解。

## 4. 诊断结果

16 份快照的三项分解最大差 1.077e-15；独立重建与实际内核的中心 F 最大差 8.882e-16。实际内核重复重建的 F 最大变化 0.000e+00，粒子状态最大变化 0.000e+00。

**固定粒子状态下，重复中心重建没有持续改变结果。** 真实 P2C/history 内核连续执行两次，中心 F 第二次变化为零；粒子的 x/v/C/L/F 均精确不变，未推进时间。这说明固定输入的重采样会得到相同中心表示。实际加载每一步都会生成新的粒子历史，因此不能据此排除随运动出现的中心／粒子差异。

全部 16 个状态中，梯度项 RMS 是移动权重项的 56.8–745.3 倍，是关联项的 1170–14671 倍。这是局部变形差的比较，不是全局反力误差来源的占比。

下面比较同一物理时刻 t=0.25 s，单位均为 1/s。它们是变形差异除以 dt 后的 RMS，不能相加为一个反力误差比例。

| 档位 | 关联项变化率 | 梯度项变化率 | 权重移动项变化率 | 总差变化率 |
|---|---:|---:|---:|---:|
| incremental-coarse | 7.96065e-06 | 0.0133519 | 0.000233703 | 0.0133605 |
| incremental-fine | 7.99157e-06 | 0.013115 | 0.000230411 | 0.0131236 |
| incremental-finest | 8.01374e-06 | 0.0129507 | 0.000227709 | 0.0129592 |
| incremental-fourth | 8.03166e-06 | 0.0128143 | 0.000225541 | 0.0128227 |

![历史差异的单位时间变化率](results/lite-aniso-mainline/v8/history-rates.png)

重采样总弹性能变化（含体积权重变化）在 t=0.25 s 的记录：

| 档位 | 单次变化 (J) | 变化／dt (W) |
|---|---:|---:|
| incremental-coarse | -2.55134e-08 | -2.55134e-05 |
| incremental-fine | -1.32305e-08 | -2.6461e-05 |
| incremental-finest | -6.82142e-09 | -2.72857e-05 |
| incremental-fourth | -3.50846e-09 | -2.80677e-05 |

非线性材料平均另有

\[
P(W_1F_p)\ne W_1P(F_p),\qquad
\sum_c V_c\psi(W_1F_p)\ne\sum_p V_p\psi(F_p).
\]

它与上面的中心／粒子更新差异是不同诊断量；不能将两者直接相加为误差预算。原始应力与能量、绝对值及归一化值见 diagnostic-indicators.json 和 diagnosis/summary.json。

## 5. 解释与下一步

第四档的相邻反力差仍仅缓慢收缩，尚未建立可靠时间收敛。 与上一对相比，绝对相邻差变化为 -6.69%。小于 5% 的差异只是预定一致性门槛，不代表反力离真解小于 5%。

诊断确认：梯度更新不一致是这组局部历史差异的主要项；固定状态反复重建不会自行累积新变化。非零的单位时间差异率说明中心和粒子的空间表示仍不一致，但它本身既不能证明时间步不收敛，也不能证明它解释了全部反力漂移。

下一步应围绕中心／粒子材料更新的一致性做单一受控对照：先在同状态检查梯度场和历史平均的空间加密效应，再选择材料历史更新方案。若调整中心历史，应同步核对势能、残差和切线的一致性，并继续保留粒子动量与边界冲量验收。空间误差和角动量保持仍未得到本轮验证。

## 6. 文件和复现

[冻结协议](results/lite-aniso-mainline/v8/protocol.json)、[四档汇总](results/lite-aniso-mainline/v8/summary.json)、[逐快照分解](results/lite-aniso-mainline/v8/diagnosis/summary.json)、[独立成果核对](results/lite-aniso-mainline/v8/artifact-check.json)、[补充指标](results/lite-aniso-mainline/v8/diagnostic-indicators.json)。

[运行器](../benchmarks/aniso_fourth_step.py)、[重采样诊断](../benchmarks/aniso_resample_diagnostic.py)、[新增测试](../tests/test_aniso_resample_diagnostic.py)。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.aniso_fourth_step freeze --output /tmp/fourth-repeat
.venv/bin/python -m benchmarks.aniso_fourth_step tests --output /tmp/fourth-repeat
.venv/bin/python -m benchmarks.aniso_fourth_step run --output /tmp/fourth-repeat
.venv/bin/python -m benchmarks.aniso_fourth_step diagnostic --output /tmp/fourth-repeat
.venv/bin/python -m benchmarks.aniso_fourth_step analyze --output /tmp/fourth-repeat
```

分析退出码 2 表示冻结数值门槛未全部满足；退出码 0 也不自动证明可靠收敛，须查看 rho 和观测阶。输出目录拒绝覆盖已有案例。独立核对脚本 `v8/check_results.py` 可复制到新输出目录后，从仓库根目录用 `PYTHONPATH=.` 执行。

2026-09-28 后续说明：v8 已封存，工作区已增加 [v9 一致历史候选](ANISO_LITE_PROJECTED_HISTORY_ZH.md)。复现 v8 冻结源码时应在单独目录解压 [source-before-v9.zip](results/lite-aniso-mainline/v8/source-before-v9.zip)，并提供协议所引用的既有结果；不要以新源码覆盖旧协议或放宽哈希检查。此处的“下一步”为 v8 当时结论，当前进展以 v9 报告为准。
