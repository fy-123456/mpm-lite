"""Build the final Chinese report from completed v18 artifacts."""
import numpy as np
from benchmarks.aniso_v18_runs import ROOT,OUT,BASE,load

def pct(x):return f'{100*x:.4f}%'
def reaction_section():
    d=load(OUT/'reaction-diagnosis.json');b=load(OUT/'boundary-reflection.json');pair=d['pairs'][-1];last=d['records'][-1]['phases']['final_hold']
    parts=[r'''
### 4.1 完整验收为什么仍失败：约束反力的逐步反射

**不能只看应力通过。** 最细相邻档原始反力 RMS 差为 $RAW。把细档反力线性对齐到粗档相同中点后，差仍为 $ALIGNED，所以不是单纯的输出时刻偏差。以相同 2.5 ms 窗口比较总冲量/窗口时长，差为 $WINDOW，这里只作诊断，不能替代原始瞬时反力门槛。

在指定的单位右夹具虚速度场 \(l\) 下，独立拆分

\[
R=R_I+R_{\rm mat}+R_{\rm stab},\qquad
R_I=\frac{2}{\Delta t}\langle Jl,M_z(JW-z^n)\rangle,
\quad R_{\rm mat/stab}=\langle l,\bar f_{\rm mat/stab}\rangle.
\]

各分项依赖这个固定虚速度场约定，总反力不依赖自由方向的特解选择。末端独立复算为：

| Δt / μs | 惯性反力 / N | 材料反力 / N | 稳定化反力 / N | 总反力 / N |
|---|---:|---:|---:|---:|
''']
    for i,r in enumerate([r for r in d['independent_component_audits'] if abs(r['time']-1.6)<1e-10]):
        parts.append(f'| {d["records"][i]["dt"]*1e6:g} | {r["inertia_N"]:.7g} | {r["material_N"]:.7g} | {r["stabilization_N"]:.7g} | {r["total_N"]:.7g} |\n')
    parts.append(f'\n最细档末端保持的反力高频能量份额为 {pct(last["high_band_power_fraction"])}（去均值、单边 Parseval 权重、频率高于 Nyquist 的 0.8 倍），相邻样本相关系数 {last["lag1_correlation"]:.4f}。噪声集中在接近逐步交替的部分，惯性项变化远大于材料和稳定化项。40 份独立分项复算与原记录的最大反力差 {max(r["independent_error_N"] for r in d["independent_component_audits"]):.3g} N。\n')
    parts.append(r'''
还可直接从更新式验证反射机制。令

\[
b=(I-\Pi_{\sqrt{M_z}JQ})\sqrt{M_z}Jl,\qquad
p_b=b^T\sqrt{M_z}z_x.
\]

\(b\) 是单位右夹具运动中与自由速度空间正交的部分，不是普通总线动量。在每一步冻结的几何与动能度量下，当前新增加载边界的更新满足

\[
p_b^{n+1}+p_b^n=2v_D\,b^Tb.
\]

等式两侧都使用同一步的 b 和 M_z；下一步几何和投影改变时，该分量还可能被重新混合，不表示整条移动轨迹保持同一个动量幅度。

停载 \(v_D=0\) 时，该步已有的这部分动量会反号而不衰减：

\[
p_b^{n+1}=-p_b^n,\qquad
R_{I,\perp}=\frac{p_b^{n+1}-p_b^n}{\Delta t}=-\frac{2p_b^n}{\Delta t}.
\]

''')
    parts.append(f'40 份实际快照的该恒等式最大误差为 {max(r["reflection_identity_error"] for r in b["records"]):.3g}。最细末步的边界动量约从 +6.75e-8 变为 -6.75e-8，对应惯性反力约 -0.00432 N。**已确认反射关系存在，尚未完成移动几何、边界启动和投影误差各自激发量的隔离。** 停载时边界速度为零，反力即使振荡也不做功，所以良好的能量账本无法排除这类反力错误。\n\n![反力失败的定位](results/lite-aniso-mainline/v18/reaction-failure-diagnosis.png)\n')
    return ''.join(parts).replace('$RAW',pct(pair['raw_as_formal_relative'])).replace('$ALIGNED',pct(pair['fine_midpoint_interpolated_relative'])).replace('$WINDOW',pct(pair['common_window_impulse_rate_relative']))

def main():
    assert not (OUT/'completion-check.json').exists(),'preserve sealed v18 report'
    m=load(OUT/'moving-acceptance.json');c=load(OUT/'cycle-acceptance.json');s=load(OUT/'space-quadrature.json');f=load(OUT/'compatible-reference.json');q=load(OUT/'fixed-mode-quadrature.json');sens=load(OUT/'input-sensitivity.json')
    assert m['completed'] and c['completed'] and s['completed'] and f['completed']
    p=c['refinement']['cycle'][-1];passed='通过' if c['passed'] else '未通过';audits=m['independent_audits']+c['independent_audits'];allcases=list(m['cases'].values())+list(c['cases'].values());terminal=c['cases']['cycle-L3']['terminal']
    headline=f'**最新验证进展（v18）：** 移动粒子 8 条短窗轨迹 / 12000 步通过本轮 2% 门槛；四档完整循环 96000 步已完成，时间验收{passed}，最细相邻档全程应力差 {pct(p["relative"])}、终态差 {pct(p["terminal_relative"])}，原始反力差 {pct(p["reaction_RMS_relative"])}。21 项相关测试、56 份独立快照复核完成。同一稳定化主导形状的充分积分惯性约为原采样的 4.57 倍，空间应力精度仍未通过。完整循环使用新增的受限 CPU 加载边界实现，默认未切换。参见 [v18 中文报告](docs/ANISO_LITE_MOVING_CYCLE_SPACE_ZH.md)。'
    parts=['# v18：移动粒子、完整循环与稳定化主导模式的空间检查\n\n',headline.split('参见 ')[0],r'''

## 1. 本轮完成了什么，结论到哪里为止

本轮是验证版本与受限研究实现，不是正式发布号。保持原材料势能、二次场保持稳定化和局部历史，不调小稳定化、不增加耗散、不加质量或刚度移位。

移动短窗使用**未修改的原 v16 `CarrierAVFSolver`**，从同一份 t=1.1、1.6 s 快照各续算 0.05 s。四档完整循环则从无应力、零速度状态开始，使用新增 `DrivenAVF` 的加载边界规则。两种边界速度投影规则在下文明确区分，不能把完整循环当作原 v16 无改动的验证。

空间验证不使用动态轨迹的时间误差充当参考误差：先比较固定离散模型的积分，再用独立 Q1/Q2 连续体兼容投影检查应力形状。投影解不是自然振动频率真解，也不是完整非线性加载解。

## 2. 同一个势能，以及外部做功怎样记账

材料历史仍为

\[
F_p=(G_0Y)_pR_p,\qquad
U(Y)=\sum_pV_p\psi(F_p)+\frac12\sum_cw_c\|P_cY_{I_c}\|^2,
\quad f=\partial_YU,\quad K=\partial_Y^2U.
\]

\(R_p\) 保留输入局部历史；材料梯度算子没有替换成位置插值导数。Hencky 加纤维材料的势能为

\[
\psi(F)=\mu\sum_i(\log\sigma_i)^2+
\frac{\lambda}{2}\left(\sum_i\log\sigma_i\right)^2+
\frac{k_f}{2}\left[\operatorname{tr}(FAF^T)-1\right]^2.
\]

动能与原 APIC 联合历史一致：

\[
T=\tfrac12z^TM_z(x)z,\quad z=(v,C),\quad
D_{pj}=h^2[\xi_{pj}(1-\xi_{pj})+1/4].
\]

\(J\) 将载体速度映射到粒子速度及三个仿射速度列，\(E\) 将载体速度提升到网格；自由方向 \(Q\) 满足夹持约束 \((EQ)_D=0\)。令 \(W=(Y^{n+1}-Y^n)/\Delta t\)，完整循环解

\[
(EW)_D=\frac{d(t+\Delta t)-d(t)}{\Delta t},\qquad
Q^Ta=0,\quad a=2J^TM_z(JW-z^n)+\Delta t\,\bar f,
\]

\[
\bar f=\frac12\sum_{\alpha=1,2}f(Y^n+s_\alpha\Delta t W),
\qquad s_{1,2}=\frac12\mp\frac{\sqrt3}{6}.
\]

完整循环采用全 \(J\) 的加权正交投影：

\[
z^{n+1}=z^n+2(JW-\Pi_Jz^n),\qquad
W_{\rm ext}=\langle a,W_b\rangle,
\]

其中 \(W_b\) 是满足给定位移增量的任一特解，\(W=W_b+Qu\)。自由残差充分小后，外部功与特解选择无关。反力用单位右夹具速度的虚功计算，所以停载时仍能报告反力。无耗散的完整循环并不要求卸载后每个瞬间应力恰好为零。

原短窗保留的是 \(\Pi_{JQ}\) 对应的未解析速度；完整循环使用 \(\Pi_J\)，使给定运动的中点速度和边界功配对。这是必要的加载边界实现差异，不能仅称为提速。

逐步能量账本为

\[
\Delta(T+U)=W_{\rm ext}+\Delta T_{\rm metric}
+\epsilon_{\rm solve}+\epsilon_{\rm path},
\quad
\Delta T_{\rm metric}=\tfrac12(z^{n+1})^T[M_z(x^{n+1})-M_z(x^n)]z^{n+1}.
\]

两点高斯对非多项式 Hencky 势能不是数学上精确的 AVF 积分，故实际测量 \(\epsilon_{\rm path}=\Delta U-\bar f:\Delta Y\)，没有把它假定为零。移动后的惯性度量变化单独记账，不能因“账本闭合”就声称严格物理能量守恒。

## 3. 移动粒子时间细化

保持物理几何 [0.125,0.875] × [0.375,0.625]²、F45、μ=10、λ=20、纤维系数 200、密度 1，h=1/8、192 个粒子、物理体积 0.046875。CPU float64；每个接受步输出完整第一 Piola 应力张量。

误差使用所有共同真实时刻，按体积与时间梯形积分加权：

\[
e_P=\left[\frac{\int\|P_{\Delta t}-P_{\Delta t/2}\|_V^2dt}
{\int\|P_{\Delta t/2}\|_V^2dt}\right]^{1/2},\qquad
\|P\|_V^2=\frac{\sum_pV_pP_p:P_p}{\sum_pV_p}.
\]

不做相位平移后再验收，不只比较应力标量范数。终态另报张量差。这里“通过 2%”是相邻档门槛，不是连续时间真解的误差上界。

| 输入快照 | Δt 粗→细 / μs | 全过程张量差 | 终态张量差 |
|---|---:|---:|---:|
''']
    for label,rows in m['refinement'].items():
        for r in rows:parts.append(f'| {label} | {r["dt_coarse"]*1e6:g} → {r["dt_fine"]*1e6:g} | {pct(r["relative"])} | {pct(r["terminal_relative"])} |\n')
    parts.append(r'''
最细两档为 31.25 / 15.625 μs。早期快照的误差收缩较慢，尚不能据此证明二阶渐近收敛。最粗档与旧 v16 相同时间、相同输入结果复核一致；16 份独立快照重新计算了原映射、本构应力、材料/稳定化能、动能、速度更新、边界约束及去质量切线。

![移动粒子时间差](results/lite-aniso-mainline/v18/moving-time-convergence.png)

## 4. 四档完整加载—保持—卸载—末端保持

夹具位移沿用原物理协议：0–0.5 s 余弦加载至 0.005；0.5–0.6 s 保持；0.6–1.1 s 余弦卸载；1.1–1.6 s 再保持。时间步为 250、125、62.5、31.25 μs，共 6400+12800+25600+51200=96000 步。本轮最细完整循环为 31.25 μs，不冒充 15.625 μs 完整循环已运行。

| 相邻档 / μs | 全程应力差 | 终态应力差 | 全程反力 RMS 差 | 本对验收 |
|---|---:|---:|---:|---|
''')
    for r in c['refinement']['cycle']:parts.append(f'| {r["dt_coarse"]*1e6:g} → {r["dt_fine"]*1e6:g} | {pct(r["relative"])} | {pct(r["terminal_relative"])} | {pct(r["reaction_RMS_relative"])} | {"通过" if r["accepted"] else "未通过"} |\n')
    parts.append('\n最细相邻档分阶段结果：\n\n| 阶段 | 全阶段应力差 | 阶段末应力差 | 绝对应力 RMS 差 / Pa |\n|---|---:|---:|---:|\n')
    for k,r in p['stages'].items():parts.append(f'| {k} | {pct(r["relative"])} | {pct(r["terminal_relative"])} | {r["absolute_Pa"]:.6g} |\n')
    parts.append(f'\n最终判据同时要求全程、每个阶段及阶段末的应力差、全程反力差小于 2%。本轮完整循环时间验收**{passed}**。反力保存在各步中点，表中相邻档按共同步末的反力样本比较，含半步位置差；应力在共同步末直接比较。\n\n最细档停载能量账本：\n\n| 停载区间 | 起始总能 / J | 总能变化 / J | 外部功 / J | 惯性度量变化 / J |\n|---|---:|---:|---:|---:|\n')
    for k in ('hold','final_hold'):
        r=c['cases']['cycle-L3']['stages'][k];parts.append(f'| {k} | {r["start_energy_J"]:.7g} | {r["energy_change_J"]:.7g} | {r["boundary_work_J"]:.7g} | {r["metric_change_J"]:.7g} |\n')
    parts.append(f'\n最细档 1.6 s 应力 RMS 为 {terminal["stress_rms_Pa"]:.7g} Pa，动能 {terminal["kinetic_J"]:.7g} J，材料能 {terminal["material_J"]:.7g} J，稳定化能 {terminal["stabilization_J"]:.7g} J。该计算没有额外阻尼，终态是振动中的一个时刻，不能解释成塑性残余应力。\n')
    parts.append(f'\n全部 56 份独立快照的去质量静态刚度检查通过；有限差分切线与解析切线最大相对差 {max(a["fd_tangent_relative"] for a in audits):.3g}。108000 个正式步的最大历史闭合误差 {max(v["max_history"] for v in allcases):.3g}，最大求解功缺陷 {max(v["max_work_defect_J"] for v in allcases):.3g} J。质量没有计入静态验收，也没有添加对角移位。\n\n![完整循环与能量](results/lite-aniso-mainline/v18/full-cycle-response.png)\n')
    parts.append(reaction_section())
    parts.append(r'''
## 5. 稳定化主导模式：积分误差与空间形状误差分开

### 5.1 积分先收敛，再谈网格

在未变形、同材料和同物理边界下，h=1/8、1/10、1/12，各做二、三、四点高斯积分。所有积分单元沿 Lite 中心插值的分片边界切开。材料/稳定化刚度二点已精确；APIC 惯性中含更高次的 D 因子，需要三点。三点和四点的 K、M、有限频率相对差均约为机器精度，二到三点的 M 差为 4.98%、5.20%、6.67%。

这些是固定离散能量及 APIC 动能函数的积分收敛，不等于连续体惯性精度。三组网格仍有 9、15、15 个严格零惯性方向，均有正静态刚度；仅在离线特征值诊断中按 K 作 Schur 消元，没有给时间求解器加人工质量。

固定原模式 211 的同一载体位移，避免重新匹配模式混淆因果：

\[
\omega_R^2=\frac{\phi^TK_{\rm mat}\phi+\phi^TK_{\rm stab}\phi}
{\phi^TM_v\phi+\phi^TM_C\phi}.
\]

| 积分 | 总刚度 / 原值 | 总惯性 / 原值 | Rayleigh 频率 / rad/s |
|---|---:|---:|---:|
''')
    rr=[r for r in q['records'] if r['label']=='sampled-target211'];K0=rr[0]['material_stiffness']+rr[0]['stabilization_stiffness'];M0=rr[0]['translation_inertia']+rr[0]['affine_inertia']
    for r in rr:parts.append(f'| {r["scheme"]} | {(r["material_stiffness"]+r["stabilization_stiffness"])/K0:.6f} | {(r["translation_inertia"]+r["affine_inertia"])/M0:.6f} | {r["rayleigh_rad_s"]:.3f} |\n')
    parts.append(r'''
稳定化刚度几乎不变；材料刚度虽增大约 4.34 倍，但在该形状的总刚度中占比很小。频率大幅变化主要由同一 APIC 惯性函数的采样不足解释。原惯性只有三点结果约 21.9%，不能仅靠调小稳定化系数修饰频率。

重新求模态后，h=1/8 最佳应力形状匹配频率约 1340.3 rad/s、MAC=0.9973；h=1/10、1/12 的最佳匹配 MAC 只有约 0.398、0.401，不能把它们当作同一物理分支已收敛。模式编号也不跨网格继承。

![同一位移的刚度与惯性](results/lite-aniso-mainline/v18/same-mode-quadrature.png)

![网格改变后的形状匹配](results/lite-aniso-mainline/v18/space-mode-matching.png)

### 5.2 独立物理空间的兼容投影

只网格化物理材料，不包含外部网格支撑、MPM 稳定化或质量项。对原采样及三点积分得到的两类目标模式，求解

\[
u_* = \arg\min_{u\in V_D}\frac12\int_\Omega
(\nabla u-E_{\rm Lite}):\mathbb H:(\nabla u-E_{\rm Lite})\,dV.
\]

这里 H 是 F45 未变形解析本构切线：

\[
\mathbb H:E=\mu(E+E^T)+\lambda\operatorname{tr}(E)I+4k_f(A:E)A.
\]

V_D 在原物理夹持区施加零运动。比较 Q1 n32/n64、Q2 n16/n32/n48，并在 n48 的夹持过渡及外边界附近做两级局部加密。最终 Q2 网格 132793 个节点、242313 个自由分量。比较积分使用所有网格界面的并集，避免跨单元积分误差。制造解能恢复可精确表示的相容变形，原 Lite 梯度映射也独立核对通过。

下表是**能量意义的最近兼容投影**与目标应力之差；它不是应力 L2 范数的最小投影，不是完整加载空间误差百分比，也不是自然频率真解。夹持区定义为 x≤0.3125 或 x≥0.6875，内部为其余区域。

| 目标形状 | 全域应力差 | 夹持区应力差 | 内部应力差 |
|---|---:|---:|---:|
''')
    for r in f['comparisons'][f['finest']]:parts.append('| '+r['label']+' | '+' | '.join(pct(r['regions'][k]['stress_mismatch_relative']) for k in ('global','grip','interior'))+' |\n')
    parts.append('\n两级局部 Q2 参考自身的应力差：\n\n| 目标形状 | 全域 | 夹持区 | 内部 |\n|---|---:|---:|---:|\n')
    for r in f['comparisons']['q2-local1']:parts.append('| '+r['label']+' | '+' | '.join(pct(r['stress_vs_finest_regions'][k]) for k in ('global','grip','interior'))+' |\n')
    parts.append(r'''
模式 211 的两个目标在这次局部 Q2 参考比较中均低于 2%；同节点 Q1/Q2 应力差尚大，这仍不是跨阶次的最终参考认证。三点积分目标 139 的夹持区仍为约 2.056%，所以**所有目标的统一参考门槛未全部通过**。已有约 21% 全域、38% 夹持区、11% 内部的模式 211 差异不能用时间细化解释；但尚不能把全部差异归为稳定化本身，夹持表达、梯度相容性和动能离散仍需分开控制。

![独立兼容投影与参考加密](results/lite-aniso-mainline/v18/compatible-stress-reference.png)

## 6. 保留下来的未通过尝试与敏感性限制

加速路径在 12 步对照与独立单步方程检查中通过，但原容差下完整 400 步与旧保存路径的应力差约 0.0491%，超过预先采用的等价性桥接门槛。收紧两边残差至 1e-17 后，400 步探索对照仍未通过，保存的最大 Y 差约 2.29e-5、C 差约 0.0123。因此正式移动粒子轨迹回到原始求解器；初次八条加速轨迹、源码、协议和失败日志全部保留，没有替换成成功记录。

微小输入扰动对照继续使用未改动的原求解器，只沿允许的载体方向改变 Y，并实际测量初始应力差：

| 请求的 Y 扰动幅度 | 初始应力相对差 | 全短窗应力差 | 末端应力差 |
|---|---:|---:|---:|
''')
    for r in sens['records']:parts.append(f'| {r["amplitude_Y"]:.0e} | {r["initial_stress_relative"]:.3g} | {pct(r["whole_stress_relative"])} | {pct(r["terminal_stress_relative"])} |\n')
    parts.append(r'''
该实验只能说明被测输入的敏感程度，不能独自确定机制，更不能把短程通过说成长程数值路径严格相同。完整循环使用独立逐步残差/功检查及四档比较验收，不以加速路径长程等价为前提。

初次有限元跨网格比较采用的统一 n96 分片没有覆盖 Q1 n64 的全部界面，已保留原产物并改用网格界面并集重算；有限元方程求解本身未受该比较积分问题影响，原五组解按哈希复用。

## 7. 测试、产物与下一步

21 项相关测试通过：4 项新增加载边界/等价映射测试、4 项原 AVF 测试、8 项原共同历史测试、2 项新增空间参考测试、3 项原局部高阶参考测试。覆盖势能—力—切线、刚体转动、仿射/弯曲、移动支撑、原子失败、边界功、制造解等。本轮没有重跑整个仓库测试套件，没有运行 CUDA。

正式产物包括 12 条轨迹 / 108000 步 / 108012 份逐步应力、56 份独立物理与去质量刚度快照、9 组无时间误差的积分/模态检查、7 组 Q1/Q2 参考网格（每组四个目标），以及拒绝尝试与输入敏感性对照。原 v17 归档保持原样，生产默认未切换。

- [移动验收](results/lite-aniso-mainline/v18/moving-acceptance.json)
- [完整循环验收](results/lite-aniso-mainline/v18/cycle-acceptance.json)
- [同形状惯性与刚度](results/lite-aniso-mainline/v18/fixed-mode-quadrature.json)
- [高阶物理参考与分区差异](results/lite-aniso-mainline/v18/compatible-reference.json)
- [最终验收清单](results/lite-aniso-mainline/v18/completion-check.json)
- [产物 SHA256](results/lite-aniso-mainline/v18/artifact-sha256.json)
- [交付源码快照](results/lite-aniso-mainline/v18/source-delivered.zip)

相关测试命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_aniso_carrier_driven tests.test_aniso_carrier_avf tests.test_aniso_carrier_joint tests.test_aniso_v18_space benchmarks.aniso_local_reference.LocalReferenceTests
```

分两次实际执行的原日志保存在 tests.log、space-tests.log；新增 `benchmarks.aniso_v18_tests solver/space` 入口供空目录复现，拒绝覆盖已保存测试结果。正式运行入口为 `benchmarks.aniso_v18_runs`，先运行 solver 测试并 freeze，再 `run --group moving --jobs 8`，随后 `benchmarks.aniso_v18_analysis moving`，通过后 `run --group cycle --jobs 4` 和 `benchmarks.aniso_v18_analysis cycle`。这些脚本会拒绝覆盖已封存协议或结果；复现需在新的工作副本中保留原 v14/v16/v17 输入，用空 v18 输出目录并重新执行测试与冻结。空间先运行 `benchmarks.aniso_v18_space` 生成模式，再运行 space 测试；初次 Q1/Q2 求解入口为 `benchmarks.aniso_v18_compatible_reference`，保留初次产物后由 `benchmarks.aniso_v18_reference_refine` 追加局部加密。具体保留顺序以已保存协议和参考比较尝试目录为准；所有执行源码与日志随归档提供。

下一步先修正加载边界的端点速度与约束冲量更新，保住现有应力与能量改善，并使原始反力随时间步收敛；不能靠事后平滑反力宣布通过。同时改进已被同形状对照定位的惯性积分，同时保持能量、动量、刚体与仿射一致性；不能直接放大质量或调低稳定化来拟合频率。然后分开检查物理夹持表达、中心梯度相容性与稳定化支撑，使用收敛的局部高阶参考比较模态子空间。保留本轮完整循环作为时间验证基线，每项空间或惯性改动后重新检查去质量刚度、历史闭合和全循环时间差。即使本轮时间门槛通过，也不代表空间精度通过。
''')
    report=ROOT/'docs/ANISO_LITE_MOVING_CYCLE_SPACE_ZH.md';report.write_text(''.join(parts))
    for name in ('README.md','RUNNING_RESTORED.md'):
        path=ROOT/name;text=path.read_text();first=text.index('\n')+1
        if '**最新验证进展（v18）' in text:
            text='\n'.join(headline if line.startswith('**最新验证进展（v18）') else line for line in text.split('\n'));path.write_text(text)
        else:path.write_text(text[:first]+'\n'+headline+'\n'+text[first:])
    print(report,flush=True)
if __name__=='__main__':main()
