"""Publish Chinese v20 evidence report only after actual final artifacts exist."""
from benchmarks.aniso_v20_common import *

def pct(x):return f'{100*x:.6g}%'
def main():
    c=load(OUT/'cycle-acceptance.json');assert c['completed'];ab=load(OUT/'full-cycle-ablation.json');audits=load(OUT/'independent-audits.json');assert audits['completed'];q=load(OUT/'runtime-quadrature.json');assert q['completed'];s=load(OUT/'local-space.json');r=load(OUT/'local-relaxation.json');d=load(OUT/'condensation-diagnosis.json');tests=load(OUT/'tests.json');last=c['pairs'][-1]
    abtable='\n'.join(f"| {name} | {pct(v['final_hold_spectrum']['above_80pct_Nyquist_fraction'])} | {v['phase_reaction_rms_N']['final_hold']:.6g} | {v['terminal_stress_rms_Pa']:.6g} |" for name,v in ab['cases'].items())
    table='\n'.join(f"| {name} | {pct(v['raw_reaction_relative'])} | {pct(v['relative'])} | {pct(v['terminal_relative'])} |" for name,v in last['stages'].items())
    energy_table='\n'.join(f"| {name} | {v['energy_change_J']:.6e} | {v['boundary_work_J']:.6e} | {v['metric_change_J']:.6e} | {v['constraint_kinetic_loss_J']:.6e} |" for name,v in c['cases']['gauss3-condensed-L3']['stages'].items())
    pairs='\n'.join(f"| {a:g} / {a/2:g} | {pct(v['raw_reaction_relative'])} | {pct(v['relative'])} | {pct(v['terminal_relative'])} |" for a,v in zip([500,250,125],c['pairs']))
    spatial='\n'.join(f"| {label} | {pct(v['reaction_relative'])} | {pct(v['stress_relative']['global'])} | {pct(v['stress_relative']['grip'])} | {pct(v['stress_relative']['interior'])} |" for label,v in [('v19 相容重建',s['records']['F45-r32-e0']),('150 个局部标量自由度',s['records']['F45-r32-e3']),('完整局部 FE，宽 0.125',r['records']['r32-w0.125']),('完整局部 FE，宽 0.1875',r['records']['r32-w0.1875']),('全部内部 FE 自由度（极限对照）',r['records']['r32-w0.25'])])
    losses=[c['cases'][f'gauss3-condensed-L{i}']['sums_J']['constraint_kinetic_loss_J'] for i in range(4)];maxhist=max(v['maxima']['history_commit_max'] for v in c['cases'].values());maxbudget=max(abs(v['closure_J']) for v in c['cases'].values());maxquad=max(max(v['native_mode_inertia_relative']) for v in q['records']);source=ROOT/'docs/ANISO_LITE_REACTION_MODES_LOCAL_DOF_ZH.md'
    report=r'''# v20：反力快速模式、移动惯性积分与夹持局部自由度

本轮完成了实现、对照试验和验收归档。研究分支仍限于受控 CPU float64 问题；生产默认未切换。时间精度与空间精度分别判定，不以反力接近代替应力验收。

本轮应力统一指第一 Piola–Kirchhoff 应力 P（PK1）。动态应力使用原材料点的体积加权 Frobenius 范数，时间差比较相同输出时刻：

\[
\|P\|_V^2=\frac{\sum_pV_p(P_p:P_p)}{\sum_pV_p},
\qquad
\eta_P^2=\frac{\sum_n\|P_{\Delta t}(t_n)-P_{\Delta t/2}(t_n)\|_V^2}
{\sum_n\|P_{\Delta t/2}(t_n)\|_V^2}.
\]

原始反力是该步中点冲量与末端约束冲量之和除以该步长；比较时不作额外平滑或时间平均。JSON 同时保留绝对差及参考 RMS，避免停载时较小的反力分母掩盖量级。空间比较是单独的线性静态问题。

## 结果与边界

- 完成四档 Gauss3 惯性＋材料不可见自由度静态消元的完整循环，以及原采样＋消元、Gauss3＋不消元两个同时间步完整循环：共 6 条、73600 步。另保留一条原投影实现的完整 3200 步对照。
- 完成 24 条同快照短轨迹，共 5600 步：卸载 t=0.85 与停载 t=1.4，同一输入分别比较固定/移动几何、原惯性/独立 Gauss3 惯性、62.5/31.25/15.625 微秒三档。初始一步没有删除，反力没有后处理平滑。
- 完成 4 份旧真实快照的反力模式分解、按移动插值分段的积分检查；另对最细候选循环的 4 份真实移动粒子快照重复检查。
- 完成 ISO/F0/F45/F90 × 两档重建网格 × 四档局部空间的 32 组去质量静态检查，以及 6 组更大局部空间/全部内部空间对照。38 组均保持正静态刚度。
- RELATED_TESTS 项相关测试通过，INDEPENDENT_AUDITS 份完整循环快照完成独立几何、应力、能量、边界约束、历史和无质量切线复算。没有宣称运行完整仓库测试或 CUDA 验收。

| 最细相邻档：125 / 62.5 微秒 | 原始反力差 | 全过程应力差 | 阶段末应力差 |
| --- | ---: | ---: | ---: |
PHASE_TABLE

| 相邻时间步（微秒） | 全循环原始反力差 | 全循环应力差 | 终态应力差 |
| --- | ---: | ---: | ---: |
PAIR_TABLE

时间反力门槛：REACTION_PASS；时间应力门槛：STRESS_PASS；约束损失逐档不增加：LOSS_PASS。空间应力仍未通过 2%，局部 Q3 应力参考自身也未取得完整 2% 认证，因此不能宣布整体精度通过。

[最终时间验收](results/lite-aniso-mainline/v20/cycle-acceptance.json) · [逐例最终数据路径及续算来源](results/lite-aniso-mainline/v20/completed-case-paths.json) · [独立快照复算](results/lite-aniso-mainline/v20/independent-audits.json)

![时间细化与约束损失](results/lite-aniso-mainline/v20/figures/time-convergence.png)

![逐步原始反力和应力](results/lite-aniso-mainline/v20/figures/raw-cycle.png)

## 1. 真正影响反力的快速模式

在同一冻结物理状态，令自由运动的刚度、惯性分别为 K、M，分析

\[
K_{ff}\phi_i=\omega_i^2 M_{ff}\phi_i,
\qquad \phi_i^T M_{ff}\phi_i=1.
\]

这里 M 来自同一 APIC 动能；充分积分证明的是这个离散动能泛函的积分精度，不是连续体惯性或空间特征频率已经精确。不加对角质量，也不把质量项混入静态刚度。为避免微小惯性在 M 的正规矩阵中被舍入淹没，实际采用刚度 Cholesky 白化，再对加权运动映射作 SVD。对重要模式的惯性还直接计算平方范数，避免小数相减。

右夹具单位位移的提升记为 ℓ。模式的反力贡献同时包含弹性力和惯性力：

\[
\delta R_i(t)=
\bigl(\ell^T KQ-\omega_i^2\ell^T MQ\bigr)\phi_i\,q_i(t).
\]

因此本轮按实际初始历史激发的有限时间窗反力 RMS 排序，而不是只排频率或应力。近简并模式合成一组，保留组内交叉项；组与组的 RMS 不能当作可直接相加的百分比。

在旧 t=1.4 快照，主要高频组约为 2.6665×10⁶、6.0883×10⁶、8.6164×10⁵ rad/s，周期约 2.36、1.03、7.29 微秒。刚度几乎全来自原稳定化项；这些方向的单位欧氏范数有效惯性只有约 10⁻¹⁴～10⁻¹³。稳定化按投影残差行归属分配后，较大份额落在物理体外的载体支撑上；这是离散行的归属，不能解释成物理体外存在真实材料储能。

受控几何缩放进一步核验了惯性退化：从同一 t=1.4 几何朝初始几何逐次缩小扰动一半，固定材料不可见方向的惯性每次约变为 0.24996～0.25000 倍；刚度不变，完全初始几何下惯性约为 10⁻³⁴。原采样与 Gauss3 得到同样的趋势。即

\[
J(\epsilon)Z=O(\epsilon),\qquad
m_Z=O(\epsilon^2),\qquad
\sqrt{k_Z/m_Z}=O(\epsilon^{-1}).
\]

卸载接近原几何时，这种表示方向可以更快，而不是更慢。这里验证的是固定方向的 Rayleigh 商缩放，不把它当作完整耦合特征分支或连续体频率。[12 组几何缩放诊断](results/lite-aniso-mainline/v20/ghost-inertia-scaling.json) 保留了全部惯性和刚度值。

这些快速方向几乎完全位于材料梯度看不见的子空间。充分惯性积分后它们仍然存在，所以根因不能简单归结为 Gauss 点不够。固定几何的实际短轨迹也验证了频率/相位解释：包含初始夹持冲量后，局部线性模式对逐步原始反力的预测相对误差约 0.0017%～0.0100%。移动几何存在额外模式变化，不能把冻结模式当作完整非线性解。

中点时间积分的离散相位为

\[
\theta_i=2\arctan(\omega_i\Delta t/2).
\]

当 ωΔt 很大时，每一步接近翻转半圈，且一个步长内的反力测量与真实高频振动并不等价。缩小时间步会改变测到的振幅和相位，不能保证相邻反力差立即单调下降。

![反力模式分组](results/lite-aniso-mainline/v20/figures/reaction-modes.png)

[模式完整分解](results/lite-aniso-mainline/v20/reaction-modes.json) · [相干分组及离散相位验证](results/lite-aniso-mainline/v20/reaction-clusters.json) · [同快照时间/几何对照](results/lite-aniso-mainline/v20/snapshot-acceptance.json)

## 2. 惯性积分：保持材料势能，按实际移动分段复核

材料势能、稳定化矩阵与原 192 个材料点保持不变；独立增加 1701 个正权重动能积分点。动能写成

\[
\mathcal K=\frac12\sum_q w_q\left(
|v_q|^2+\sum_j D_{qj}|C_{q,:,j}|^2\right),
\quad D_{qj}=h^2\{f_{qj}(1-f_{qj})+1/4\}.
\]

其中 f 是粒子在当前中心插值区间内的坐标。移动后，积分区间应按 x_j(X)=(k+1/2)h 的实际穿越位置切分。实现对分片三线性参考场采用嵌套扫描：沿积分线求所有物理坐标的分段交点，使用正 Gauss 权重，并比较三/四阶；若矩阵或目标模式惯性不收敛，再细分外层区间。保留参考体积权重，不以重设密度偷偷修正能量。

旧四份快照三/四阶分段积分已收敛；原固定 Gauss3 对重要反力模式的惯性误差最大约 0.0212%。最细候选循环实际移动粒子场的对应最大差为 RUNTIME_QUAD_ERROR。详细结果同时给出整体矩阵差，不能只挑某个模式。

必须区分“积分规则替换”和“相同离散速度”。旧快照的 x、v、C 在同一连续分片三线性场中插值到新动能点，原点值被精确保留；但新积分度量下，速度可能不再满足旧夹持投影约束。在 t=1.4 的 Gauss3 替换中出现约 3.469×10⁻⁷ N·s 的一次约束冲量：

\[
I_{\rm switch}=-\ell^TJ^TM_z
(\Pi_{\rm full}-\Pi_{\rm free})z_0,
\qquad R_{\rm first,extra}=I_{\rm switch}/\Delta t.
\]

这解释了为什么短快照试验中增加积分点可能造成很大的首步反力。全部原始样本仍计入验收；完整循环从一致的零速度、无应力状态开始，避免把这次迁移冲量混同于持续时间误差。

当前完整循环使用固定参考 Gauss3 动能点，移动分段规则是独立积分验证器；没有在每一步重划动能点，也没有声称解决任意粒子重采样的历史转移。

[旧快照移动分段积分](results/lite-aniso-mainline/v20/moving-quadrature.json) · [完整循环真实移动粒子复核](results/lite-aniso-mainline/v20/runtime-quadrature.json)

## 3. 原势能中的材料不可见自由度静态消元

令 B_j 为原材料梯度，Z 满足 B_j Z=0，且不改变夹持自由度。本例有 3 个标量方向，即 9 个矢量自由度。它们改变载体表示，却不改变材料 F。

原势能保持为

\[
U(Y)=U_m(BY)+\tfrac12\sum_aY_a^TK_sY_a.
\]

沿 Z 求这个原势能的精确静态驻点：

\[
Z^TK_sY=0,
\quad \mathcal R=I-Z(Z^TK_sZ)^{-1}Z^TK_s.
\]

这相当于消去没有独立材料变形意义的内部坐标。没有降低 K_s 的系数，没有增加质量、刚度或耗散。若 H 张成约束子空间，残差、切线分别由同一个 U 给出：Hᵀ∇U、HᵀKH。

测试验证原静态平衡、反力和势能最小值不变；材料 F、刚体运动、仿射场及二次弯曲场保持；线性/角动量冲量对偶成立。初始状态保留 216 个有限动态模式，最高频率约 1332 rad/s，且去质量刚度仍为正。全过程快照的频率与历史约束见诊断文件。

该时间候选仍沿用原材料空间；它没有自动修复原有静态空间应力差。原采样材料空间的 F45 反力对局部 Q3 差约 3.48%，全域应力对照差约 135.87%。下面的局部 FE 改进是另一条空间研究路线，不能把两者拼成一个已经同时通过时空精度的算法。

这是一个新的动态模型约束：变形后的 Z 并非严格零惯性，因此没有宣称它与旧完整动力学完全等价。旧快照若不满足 ZᵀK_sY=0 会被拒绝，不能直接投影后丢掉能量；正式候选循环从相容初始状态出发。原局部材料历史保持，新的约束仅作用于已识别的材料不可见方向。

[消元诊断与旧历史拒绝证据](results/lite-aniso-mainline/v20/condensation-diagnosis.json) · [冻结循环协议](results/lite-aniso-mainline/v20/fast-cycle/cycle-protocol.json)

完整同时间步 2×2 对照如下，均从同一零速度、无应力状态运行到 1.6 s，每条边只改变惯性积分或静态消元其中一个因素。频谱列是末端保持原始反力在 80% Nyquist 以上的功率占比，仅作模式归因，不是替代验收的平滑量。

| dt=125 微秒 | 高频功率占比 | 末端保持反力 RMS（N） | 终态应力 RMS（Pa） |
| --- | ---: | ---: | ---: |
ABLATION_TABLE

这个对照把改善来源分开了：只改为 Gauss3，末端保持高频功率占比从约 5.97% 降到 2.06%；在 Gauss3 上再加入静态消元，降到约 0.00364%。原采样上单独消元也降到约 0.00613%。消元前后相应终态材料应力几乎不变，累计约束动能损失还略有下降，因此主要的快速反力改善来自去除材料不可见的动态自由度，而不是增加耗散。惯性积分同时改变有限模式的惯性和相位；单凭两种积分的终态应力不同，不能判断哪一个连续体应力更准确。

[完整循环单因素对照](results/lite-aniso-mainline/v20/full-cycle-ablation.json) 同时列出各阶段应力和原始反力差。跨模型差异本身不是连续体误差估计；只有四档候选具有本轮独立的时间收敛验收。

## 4. 夹持附近的局部变形自由度

在连续 Q2 物理重建上添加标量局部形函数 Z_loc，每个空间分量使用同一算子：

\[
u_{FE}=A y+Z_{loc}a,
\quad U(y,a)=\int_\Omega\psi(I+\nabla u_{FE})\,dV
+\tfrac12\sum_b y_b^TK_sy_b.
\]

局部自由度在固定夹持区为零，允许夹持过渡区产生更细的局部形变。在线性静态试验中，从同一势能消去局部坐标，得到

\[
K_{\rm eff}=K_{yy}-K_{ya}K_{aa}^{-1}K_{ay}.
\]

补偿由明确的能量驻点给出，没有按目标反力调参数。另实现非线性能量、残差及精确切线作用，并通过有限差分和刚体转动客观性检查。

F45 相对同一局部 Q3 参考的结果如下。这里基线是 v19 相容重建候选；不能把它的 29.7% 与原采样算法约 3.48% 的反力偏差混为一谈。

| 空间，重建网格 32 | 反力差 | 全域应力差 | 夹持区应力差 | 内部应力差 |
| --- | ---: | ---: | ---: | ---: |
SPATIAL_TABLE

增加局部空间确实减轻过约束，但 150 个局部标量自由度仍不够。全部内部 FE 自由度是诊断极限，并非可直接替换的廉价算法；即使其反力通过 2%，应力仍明显不够准。相同势能下的稳定化最小额外反力约为 3.3482×10⁻⁵ N，说明该极限中的主要应力差不能再简单归因于稳定化系数过大。

独立复算还确认：全部内部 FE 放开后，材料位移与独立 Q2 解最大差小于 1.4×10⁻¹²，额外反力与稳定化最小反力之差小于 10⁻¹⁴ N。[空间极限与分量诊断](results/lite-aniso-mainline/v20/space-diagnosis.json) 给出了这项恒等式及纤维坐标中的应力差分解。

应力分量分解给出了更具体的空间优化方向：这些空间候选相对局部 Q3 的全域应力差平方中，约 98.9%～99.3% 落在纤维轴向分量。对 F45，

\[
\varepsilon_{aa}=\tfrac12(\varepsilon_{xx}+\varepsilon_{yy}+2\varepsilon_{xy}),
\quad
\delta P=2\mu\,\delta\varepsilon+\lambda\,\mathrm{tr}(\delta\varepsilon)I
+4k_f(a^T\delta\varepsilon a)aa^T.
\]

本例纯纤维轴向应变的系数为 λ+2μ+4k_f=840，而纯横向为 λ+2μ=40。因而很小的轴向应变误差也会被明显放大；99% 是应力差平方的占比，不是位移或应变误差的占比。下一步局部空间应重点表达沿斜纤维方向的伸缩，以及与横向收缩、剪切相协调的夹持过渡，而不是单凭总反力选择空间或系数。

局部 Q2/Q3 参考自身的应力差约为全域 12.80%、夹持区 18.69%、内部 4.47%，尚未形成完整 2% 应力认证。这些大幅空间偏差不能被解释成精确误差估计，但足以否定“反力接近即整体通过”。局部空间候选尚未接入完整非线性移动循环；先完成独立空间验收更合理。

![空间改进与剩余应力差](results/lite-aniso-mainline/v20/figures/local-space.png)

[32 组嵌套局部空间](results/lite-aniso-mainline/v20/local-space.json) · [6 组更大空间及极限对照](results/lite-aniso-mainline/v20/local-relaxation.json)

## 5. 能量、求解器及归档

延续边界速度与真实冲量共同更新，逐步核对

\[
\Delta E=W_{\rm mid}+W_{\rm end}
+\Delta K_{\rm metric}-D_{\rm constraint}
+\epsilon_{\rm solve}+\epsilon_{\rm AVF}.
\]

夹具停下时外部功为零；动能与弹性能仍可互换。卸载终点仍有应力并不单独证明历史被锁死：在没有材料阻尼的超弹性模型里，剩余动能与弹性能可以形成持续振动。必须结合全过程相位、时间收敛和能量账本判断，不能把强行压到零当作正确恢复。约束修正本身的小动能损失单独记录，没有把能量账本闭合当作严格无耗散。单调下降与趋于零也不是同一项结论；本轮四档尚不足以证明损失的无耗散极限。四档累计约束损失（J）为 LOSS_VALUES；六条正式循环最大总账本闭合误差为 MAX_BUDGET J，最大材料历史增量误差为 MAX_HISTORY。

最细档的分阶段总能量账本如下（J）。动能与弹性能各自的涨落还保存在逐步数据中；表中比较的是它们之和。

| 阶段 | 总能量变化 | 夹具外功 | 几何度量变化 | 约束动能损失 |
| --- | ---: | ---: | ---: | ---: |
ENERGY_TABLE

![能量与外功分项](results/lite-aniso-mainline/v20/figures/energy-ledger.png)

实现了数学等价的 Householder 投影应用与末端几何复用，避免反复形成高而密的正交矩阵。保留原 QR/SVD 秩阈值；测试比较了逐步真实卸载、停载与移动加载结果，并保留一条完整原实现对照。部分更慢的重复运行被明确标为 superseded，原数据保留。

某些轨迹在牛顿残差约 10⁻¹⁶ 时因能量差舍入停滞。续算仅在残差小于 10⁻¹²、能量变化落在 10⁻¹⁴ J 范围时允许使用“真实残差下降”作为线搜索判据；最终残差仍须 ≤10⁻¹⁷，逐步能量账本仍须满足 10⁻¹³ J。没有放宽最终收敛或物理门槛。独立快照审计还避开了一次分治 SVD 未收敛：所查约束矩阵有限且条件数约 1～1.11，审计改为直接重算所需稀疏几何，并用经典 SVD 投影复核；不修改已运行轨迹。失败日志保留。

求解失败记录保留，从已保存 x/Y/v/C 精确续算，预测器重新初始化；最终路径文件给出原失败与续算来源。

[相关测试](results/lite-aniso-mainline/v20/tests.json) · [替换计算记录](results/lite-aniso-mainline/v20/superseded-runs.json) · [最终完成清单](results/lite-aniso-mainline/v20/completion-check.json) · [成果 SHA256](results/lite-aniso-mainline/v20/artifact-sha256.json) · [交付源码](results/lite-aniso-mainline/v20/source-delivered.zip)

## 下一步

1. 以本轮完整循环及独立分段积分结果确定动态候选的适用范围，继续区分反力时间误差、约束损失与空间误差。材料不可见方向的消元在当前支撑下有明确能量依据，但任意支撑扩张仍需重新验证。
2. 把局部空间的误差估计用于选择更有针对性的自由度，在夹持过渡与高应力梯度区加密；继续分开验收全域、内部和夹持区应力，完善高阶参考。不要只根据总反力调稳定化系数。
3. 空间静态验收通过后，再为新增局部自由度定义一致的惯性、速度和历史更新，接入完整移动循环；本轮静态局部 FE 改善不等于这条动态路线已经完成。

## 实现入口

- [分离动能积分几何](../engine/aniso_phase1/separate_kinetic.py)、[移动分段积分](../engine/aniso_phase1/swept_quadrature.py)。
- [原势能静态消元与 AVF](../engine/aniso_phase1/integrated_avf.py)、[等价加速](../engine/aniso_phase1/fast_integrated_avf.py)、[舍入级线搜索修复](../engine/aniso_phase1/robust_integrated_avf.py)。
- [夹持局部空间及非线性势能](../engine/aniso_phase1/grip_enrichment.py)。
- 验收脚本位于 `benchmarks/aniso_v20_*.py`，上述实验没有改动原生产入口。

## 复核入口

读取本目录已有 JSON、NPZ 与源码归档即可复核结果。冻结实验目录禁止覆盖，重新执行完整 freeze/run 应在新目录或独立副本中进行。当前可直接重复执行相关单元测试：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_aniso_integrated_inertia tests.test_aniso_v20_variational tests.test_aniso_v20_projection tests.test_aniso_v20_roundoff tests.test_aniso_swept_adaptive tests.test_aniso_endpoint_boundary tests.test_aniso_carrier_driven tests.test_aniso_carrier_avf tests.test_aniso_compatible_carrier -v
```

数值结果来自受控 F45 盒体、h=0.125、材料 μ=10、λ=20、k_f=200、位移幅值 0.005、循环终点 1.6 s。它们不是一般几何、一般支撑变化、任意材料或生产 GPU 路径的精度证明。
'''
    values={'ENERGY_TABLE':energy_table,'ABLATION_TABLE':abtable,'RELATED_TESTS':str(tests['tests']),'INDEPENDENT_AUDITS':str(audits['total_snapshots']),'PHASE_TABLE':table,'PAIR_TABLE':pairs,'SPATIAL_TABLE':spatial,'REACTION_PASS':'通过' if c['raw_reaction_time_passed'] else '未通过','STRESS_PASS':'通过' if c['stress_time_passed'] else '未通过','LOSS_PASS':'通过' if c['constraint_loss_decreases'] else '未通过','RUNTIME_QUAD_ERROR':pct(maxquad),'LOSS_VALUES':', '.join(f'{v:.6e}' for v in losses),'MAX_BUDGET':f'{maxbudget:.3e}','MAX_HISTORY':f'{maxhist:.3e}'}
    for k,v in values.items():report=report.replace(k,v)
    source.write_text(report)
    intro=f"**最新验证进展（v20）：** 完成反力模式归因、运动状态分段惯性积分、6 条完整循环 73600 步和 38 组去质量空间检查。{tests['tests']} 项相关测试、{audits['total_snapshots']} 份独立快照复核通过。Gauss3＋材料不可见自由度静态消元的最细相邻原始反力差 {pct(last['raw_reaction_relative'])}、全程应力差 {pct(last['relative'])}；时间反力{values['REACTION_PASS']}、时间应力{values['STRESS_PASS']}。夹持局部自由度减轻过约束，空间应力仍未通过，默认未切换。参见 [v20 中文报告](docs/ANISO_LITE_REACTION_MODES_LOCAL_DOF_ZH.md)。"
    for name in ('README.md','RUNNING_RESTORED.md'):
        p=ROOT/name;content=p.read_text();lines=content.splitlines();lines.insert(2,intro+'\n');p.write_text('\n'.join(lines)+'\n')
    print(source)
if __name__=='__main__':main()
