"""Render v19 evidence, separating accepted fixes from rejected candidates."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_v19_runs import ROOT,OUT,BASE,load
from benchmarks.aniso_v19_analysis import readrows

def pct(x):return f'{100*x:.4f}%'
def plots(cycle,space,reference):
    fig,axes=plt.subplots(1,2,figsize=(11,4));new=readrows(OUT/'cases/cycle-L3/steps.jsonl');old=readrows(BASE/'v18/cases/cycle-L2/steps.jsonl')
    for data,label in [(old,'v18'),(new,'v19 endpoint impulse')]:
        t=np.array([r['time'] for r in data]);v=np.array([r['reaction_N'] for r in data]);mask=(t>=1.59)&(t<=1.60001);axes[0].plot(t[mask],v[mask],label=label,linewidth=1)
    axes[0].set(xlabel='Time [s]',ylabel='Raw reaction [N]',title='Same dt = 62.5 microseconds');axes[0].legend()
    x=np.array([500,250,125]);pairs=cycle['pairs']
    for key,label in [('raw_reaction_relative','Reaction'),('relative','Whole-cycle stress'),('terminal_relative','Terminal stress')]:axes[1].loglog(x,[100*r[key] for r in pairs],'-o',label=label)
    axes[1].axhline(2,color='gray',ls='--');axes[1].invert_xaxis();axes[1].set(xlabel='Coarser dt of adjacent pair [microseconds]',ylabel='Relative difference [%]',title='No reaction smoothing');axes[1].legend();fig.tight_layout();fig.savefig(OUT/'boundary-time.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4));names=['sampled','gauss3','compatible32'];x=np.arange(3)
    for i,(key,title) in enumerate([('reaction_relative','F45 reaction vs local Q3'),('grip','F45 grip stress vs local Q3')]):
        vals=[reference['comparisons'][n]['reaction_relative'] if key=='reaction_relative' else reference['comparisons'][n]['stress_relative'][key] for n in names]
        axes[i].bar(x,100*np.array(vals),color=['#888888','#267fb5','#cc6a32']);axes[i].set_xticks(x,names);axes[i].set(ylabel='Relative difference [%]',title=title);axes[i].axhline(2,color='gray',ls='--')
    fig.tight_layout();fig.savefig(OUT/'space-candidates.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4))
    for level,dt in [(2,125),(3,62.5)]:
        data=readrows(OUT/'cases'/f'cycle-L{level}'/'steps.jsonl');t=np.array([v['time'] for v in data]);R=np.array([v['reaction_N'] for v in data]);mask=t>=1.1-1e-10;ax.plot(t[mask],R[mask],lw=.6,label=f'dt={dt} microseconds')
    ax.set(xlabel='Time [s]',ylabel='Raw reaction [N]',title='Entire final hold: remaining sensitivity is retained');ax.legend();fig.tight_layout();fig.savefig(OUT/'reaction-hold.png',dpi=180);plt.close(fig)


def main():
    c=load(OUT/'cycle-acceptance.json');s=load(OUT/'space-summary.json');r=load(OUT/'local-q3-reference.json');short=load(OUT/'short-moving-acceptance.json');inertia=load(OUT/'deformed-inertia.json');parts=load(OUT/'static-parts.json');last=c['pairs'][-1];plots(c,s,r)
    final=c['cases']['cycle-L3'];loss=[v['energy_sums_J']['constraint_kinetic_loss_J'] for v in c['cases'].values()]
    text=[f'''# v19：边界速度—冲量共同更新、惯性积分与夹持梯度实验

本轮完成四档完整循环 48000 步、三种空间离散各三档移动恢复共 2100 步、30 项相关测试，以及 28 份独立快照复核。**边界速度与实际冲量更新已实现，反力交替振荡减轻；各阶段应力通过 2% 时间门槛，但原始反力和空间精度仍未通过，默认求解器未切换。** 新实现是受限 CPU / float64 研究路径。

最细完整循环档为 125 / 62.5 μs：原始反力差 **{pct(last['raw_reaction_relative'])}**，全程应力差 **{pct(last['relative'])}**，终态应力差 **{pct(last['terminal_relative'])}**。两档用相同物理区间的总冲量比较时差 {pct(last['same_interval_impulse_relative'])}，此数只作补充，未替换原始反力门槛。v18 更细 62.5 / 31.25 μs 的原始反力差为 18.2291%；下文另给同一 dt 的波形对照，不能把不同时间步直接当作精确改善倍数。

实现入口：[边界共同更新](../engine/aniso_phase1/endpoint_boundary.py)、[相容位置与梯度](../engine/aniso_phase1/compatible_carrier.py)、[相容 AVF 实验](../engine/aniso_phase1/compatible_avf.py)。

末端保持的原始反力差仍高达 **{pct(last["reaction_stages"]["final_hold"]["raw_relative"])}**，按相同时间区间累计冲量后仍为 {pct(last["reaction_stages"]["final_hold"]["same_interval_relative"])}，因此不能仅归结为反力采样时刻错位，也不能用全程应力接近来宣布成功。另有一项没有解决：约束投影会损失极小的动能，四档损失没有单调趋零的证据。因此 `cycle-acceptance.json` 将应力时间通过、反力时间未通过与“损失递减”分开记录；保守综合状态为 **{'通过' if c['passed'] else '未通过'}**。不能据此宣称严格无耗散或完整算法已验收。

## 1. 边界究竟修正了什么

旧中点更新要求平均夹持速度正确，却没有同时要求步末速度正确。固定同一步的几何与度量后，某个夹持动量分量满足

\\[
p_b^{{n+1}}+p_b^n=2v_D\\|b\\|^2.
\\]

夹具停住时就变成每步翻号，反力中的惯性项又除以时间步，形成锯齿。这不是输出曲线需要平滑，而是下一步实际使用的速度状态不满足夹持运动。

设 \\(z=(v,C_1,C_2,C_3)\\)，\\(M_z\\) 为粒子 APIC 动能度量，\\(J\\) 把材料自由度速度映射为 \\(z\\)，\\(Q\\) 表示允许自由运动的材料方向，\\(\\ell\\) 为右夹具单位速度抬升。所有下式算子取**实际步末状态**。定义加权正交投影

\\[
\\Pi_A=A(A^TM_zA)^+A^TM_z,\\qquad z_D=J\\ell\\,\\dot d(t_{{n+1}}).
\\]

AVF 中点求解完成后，将速度和约束冲量一起更新：

\\[
z_{{n+1}}=(I-\\Pi_J)z_*+\\Pi_{{JQ}}(z_*-z_D)+z_D,
\\qquad I_c=J^TM_z(z_{{n+1}}-z_*).
\\]

由此 \\(Q^TI_c=0\\)：约束冲量不作用于当前允许的自由运动；完整映射的零空间历史也保留。满足的是可解析运动的夹持约束，不是把每个粒子的全部速度、C 任意清零。该方法是“AVF 中点求解＋步末约束冲量”，并非已经证明的完全单体变分积分器。

反力取实际施加的两个冲量之和：

\\[
R=\\frac{{\\langle I_{{mid}},\\ell_n\\rangle+\\langle I_c,\\ell_{{n+1}}\\rangle}}{{\\Delta t}}.
\\]

边界做功必须分别使用中点平均速度与步末速度。尤其不能简单用该 R 乘总位移来替代能量账本。步末修正严格满足

\\[
\\Delta K_c=\\dot d(t_{{n+1}})\\langle I_c,\\ell_{{n+1}}\\rangle
-\\tfrac12\\|z_{{n+1}}-z_*\\|_{{M_z}}^2.
\\]

后项是约束消除不相容速度带来的真实数值损失，本轮显式记录，没有可调耗散系数。单步能量总账为

\\[
\\Delta(K+U)=W_{{mid}}+W_c+\\Delta K_{{metric}}-D_c+\\epsilon_{{solve}}+\\epsilon_{{path}}.
\\]

材料和稳定化势能没有因这项边界修正而改变。

最终反力未通过后，另做了 28 份同状态投影替换诊断：在固定中点试速度的前提下，分别替换映射 J、动能度量和夹持子空间。保存时刻中，单独替换夹持表示的影响很小，例如最细档 t=1.4 s 的步末修正反力差约 2×10⁻¹⁵ N；变化主要出现在映射 J 替换后。但这只是步末修正的诊断，不是全反力根因证明。

进一步做实际反力误差的精确加法分解，125 / 62.5 μs 全程总反力 RMS 差为 3.3360×10⁻⁴ N，中点项为 3.3378×10⁻⁴ N，步末约束项为 5.1669×10⁻⁶ N，交叉项已记录。**剩余误差主要落在中点求解的反力，而非新增步末修正项。** 下一步应优先隔离中点动力学中的惯性与快速模式，不能继续把它简单归咎于夹持子空间变化；具体模式及成因尚未确认。

## 2. 完整循环与独立复核

几何、材料与 v18 相同：物体 \\([.125,.875]\\times[.375,.625]^2\\)，h=1/8，192 粒子、225 材料自由度位置，F45，μ=10、λ=20、kf=200。时序为加载 0–0.5 s、保持至 0.6 s、卸载至 1.1 s、末端保持至 1.6 s；最大夹持位移 0.005。所有循环从无应力、零 v/C 开始。

| 相邻 dt（μs） | 全程应力差 | 终态应力差 | 原始反力差 | 同区间冲量差 |
|---|---:|---:|---:|---:|
''']
    for i,v in enumerate(c['pairs']):text.append(f"| {500/2**i:g} / {250/2**i:g} | {pct(v['relative'])} | {pct(v['terminal_relative'])} | {pct(v['raw_reaction_relative'])} | {pct(v['same_interval_impulse_relative'])} |\n")
    text.append('\n| 最细相邻档阶段 | 全阶段应力差 | 阶段终态差 | 原始反力差 |\n|---|---:|---:|---:|\n')
    for key,title in [('ramp','加载'),('hold','保持'),('unload','卸载'),('final_hold','末端保持')]:v=last['stages'][key];text.append(f"| {title} | {pct(v['relative'])} | {pct(v['terminal_relative'])} | {pct(last['reaction_stages'][key]['raw_relative'])} |\n")
    text.append(f'''
![反力与时间差](results/lite-aniso-mainline/v19/boundary-time.png)

![完整末端保持原始反力](results/lite-aniso-mainline/v19/reaction-hold.png)

最细档末端应力 RMS 为 {final['terminal']['stress_rms_Pa']:.8g} Pa。原始反力去均值后的高频能量占比（超过 Nyquist 的 80%）由同 dt v18 的 {pct(c['same_dt_old_bridges']['cycle-L3']['old_frequency']['above_80pct_Nyquist_fraction'])} 降至 {pct(final['frequency']['above_80pct_Nyquist_fraction'])}。此频谱只作锯齿诊断，不参与应力验收，也不把物理振动当成误差清除。

四档约束损失依次为 {', '.join(f'{v:.6g}' for v in loss)} J。最细档损失为峰值总能量的 {final['loss_over_peak_energy']:.3g}；量级很小，但**没有证明随 dt 趋零**。粗档记录显示主要发生在卸载与末端保持；它与几何、度量、弱惯性方向变化的因果关系仍待隔离，不能仅凭秩统计认定根因。

28 份快照使用独立材料应力、能量、有限差分材料刚度和 SVD 投影重算，检查位移、历史闭合、总反力、边界功、损失、线动量和角动量。全部通过；全部去质量静态刚度保持正值。最大历史闭合误差 {max(v['maxima']['history_commit_max'] for v in c['cases'].values()):.3g}；最大单步账本差 {max(v['maxima']['budget_defect_J'] for v in c['cases'].values()):.3g} J。

## 3. 惯性积分：已落实的改善与适用范围

这次真正构造正权重的高斯材料点，用同一权重积材料能量和 APIC 动能，没有把某些质量条目乘经验系数。原始离散的动能矩阵为

\\[
M=\\int_{{\\Omega_0}}\\rho_0\\left[N^TN+\\sum_jD_jL_j^TL_j\\right]\\,dV_0,
\\qquad L_j=\\sum_kB_k(F^{{-1}})_{{kj}}.
\\]

积分单元切开原中心插值的折点。初始未变形时，原梯度方案 Gauss3 / Gauss4 的 M 差 {s['quadrature'][0]['M_relative']:.3g}，相容方案为 {s['quadrature'][1]['M_relative']:.3g}；对应 K 的差也约 10⁻¹⁵。前者 1701 / 4032 个积分点，后者 5184 / 12288 个积分点。

保持同一个原 mode211 的材料节点形状，原采样惯性为 1，充分积分后为 4.572769；Rayleigh 频率由 2834.77 降至 1335.94 rad/s，稳定化刚度不变。这明确表明原采样低估了这个形状的惯性。它仍是**该 APIC 动能泛函**的积分结果，不是连续体天然频率的准确性证书，也不是跨网格模式分支已经追踪成功。

变形后重新检查：

| 方案 | Gauss3 / 4 的 M 差 | Gauss4 / 5 的 M 差 |
|---|---:|---:|
''')
    for row in inertia['records']:text.append(f"| {row['name']} | {pct(row['M3_vs_M4_relative'])} | {pct(row['M4_vs_M5_relative'])} |\n")
    text.append('''
相容方案的变形后积分还没有显示单调收敛：移动网格折点及 \\(F^{-1}\\) 不再是初始状态的同一分段多项式。它需要继续按实际折点划分积分区间或做自适应积分，不能直接延用“初始 Gauss3 精确”的结论。

## 4. 夹持梯度相容：实现通过，F45 空间候选未通过

在物理区域建立连续 Q2 位移场，逐个空间分量求

\\[
u_h(Y)=\\arg\\min_{u_h}\\frac12\\int\\|\\nabla u_h-G_{{Lite}}Y\\|^2\\,dV,
\\qquad x_p=N_{{comp},p}Y,\\quad F_p=(\\nabla N_{{comp}})_pY.
\\]

夹持区采用只依赖夹持材料节点的单侧二次插值，避免自由节点的梯度泄漏进完全夹住的材料；位置和材料梯度取自同一个连续场。标量算子对三个空间分量相同，保持转动客观性及所有二次多项式，包括正常弯曲。力与切线仍由同一势能求导：

\\[
U(Y)=\\sum_pV_p\\psi(F_p)+\\tfrac12\\sum_cw_c\\|P_cY_c\\|^2,
\\quad f=\\partial_YU,\\quad K=\\partial_Y^2U.
\\]

原稳定化系数及历史能量没有调小。本候选采用固定材料夹持坐标和移动的相容积分点，是新的材料表示实验；并非把生产粒子—网格插值直接替换。它从给定相容初态启动，**未提供任意旧 Fp/Y 快照的无损迁移**。

ISO、F0、F45、F90 的 16 组静态矩阵均没有额外零刚度模式，没有加入质量或刚度平移。9 个精确零惯性方向仍存在，静态刚度检查没有用质量遮掩它们。刚体、仿射、二次场、能量—力—切线以及移动历史测试全部通过。

但这些条件不等于空间应力准确。使用原夹持条件的独立局部 Q3 参考，F45 结果为：

| 方案 | 反力差 | 全域应力差 | 夹持附近及夹持区应力差 | 内部应力差 |
|---|---:|---:|---:|---:|
''')
    for name in ('sampled','gauss3','compatible16','compatible32'):
        v=r['comparisons'][name];text.append(f"| {name} | {pct(v['reaction_relative'])} | {pct(v['stress_relative']['global'])} | {pct(v['stress_relative']['grip'])} | {pct(v['stress_relative']['interior'])} |\n")
    text.append(f'''
“夹持附近及夹持区”为 x≤0.3125 或 x≥0.6875，内部为两者之间；旧方案在完全夹持区域的梯度泄漏也计入误差。Q2 / Q3 局部参考之间的应力差仍为全域 {pct(r['comparisons']['Q2-local2-reference']['stress_relative']['global'])}、夹持区 {pct(r['comparisons']['Q2-local2-reference']['stress_relative']['grip'])}、内部 {pct(r['comparisons']['Q2-local2-reference']['stress_relative']['interior'])}，参考局部应力本身尚未达到 2%。本轮因此只报告差异，不能把任何一个粗参考称为真解。相容32 的反力偏差约 29.7%，远大于现有参考反力差约 0.13%，足以拒绝当前候选的推广。

![空间候选比较](results/lite-aniso-mainline/v19/space-candidates.png)

分解同一个静态平衡结果：

| 方案 | 材料能量 J | 稳定化能量 J | 完全夹持材料中的能量 J | 材料 / 稳定化反力 N |
|---|---:|---:|---:|---:|
''')
    for name in ('sampled','gauss3','compatible32'):
        v=parts['cases'][name];text.append(f"| {name} | {v['material_energy_J']:.7g} | {v['stabilization_energy_J']:.7g} | {v['physical_grip_material_energy_J']:.7g} | {v['material_reaction_N']:.7g} / {v['stabilization_reaction_N']:.7g} |\n")
    text.append('''
相容重建消除了夹持区泄漏，但相对于 sampled，新增反力约 99.92% 体现在材料反力差中，稳定化反力仅变化约 5.97×10⁻⁶ N。这里比较的是各自平衡解的力分解，不能当作单因素因果证明；它支持优先检查重建空间对局部变形的限制，而不是直接调低稳定化系数。内部应力差有所减小，夹持应力与总反力却更差，说明“位置和梯度相容”不足以保证正确的材料刚度。

## 5. 已验证细时间步的移动粒子检查

三种离散各跑 62.5、31.25、15.625 μs；同一解析材料节点小扰动、零初速，移动恢复 6.25 ms。误差使用每一步全部原生积分点的体积加权完整应力张量。

| 方案 | 最细相邻档全程应力差 | 终态差 | 2% 门槛 |
|---|---:|---:|---|
''')
    for name,v in short['cases'].items():a=v['pairs'][-1];text.append(f"| {name} | {pct(a['relative'])} | {pct(a['terminal_relative'])} | {'通过' if v['passed'] else '未通过'} |\n")
    text.append('''
这是各离散方案内部的时间收敛检查，不是三方案互为参考。相容重建同时改变位置基函数、材料梯度及夹持表示，不能把它的时间变化全部归因于某一个因子。高斯/相容方案尚未跑完整加载循环；本轮四档完整循环只验收边界共同更新。鉴于相容方案已经未通过静态 F45 精度，不应跳过该门槛推广完整组合。

## 6. 当前结论与下一步

1. 保留已实现的边界速度—实际冲量约束，优先解决卸载及末端保持的剩余反力敏感性，同时隔离极小约束损失为何未随 dt 单调减小，特别是中点动力学的弱惯性/快速模式与几何更新；不能称其严格无耗散。
2. 高斯积分解决了初始离散惯性欠采样。下一步应跟随运动后的真实折点改善积分，再对仅更换积分的版本做完整循环。
3. 相容梯度候选保住了能量一致、历史闭合和正静态刚度，却没有改善 F45 空间精度。优先放宽有明确能量依据的局部变形表示，并继续改进独立夹持参考。现阶段不调低稳定化系数，也不把接近的总反力当作应力准确。

复核测试命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest tests.test_aniso_endpoint_boundary tests.test_aniso_compatible_carrier tests.test_aniso_carrier_driven tests.test_aniso_carrier_avf tests.test_aniso_carrier_joint tests.test_aniso_v18_space benchmarks.aniso_local_reference.LocalReferenceTests
```

计算入口为 `benchmarks.aniso_v19_runs`、`aniso_v19_space`、`aniso_v19_short`、`aniso_v19_reference`、`aniso_v19_inertia`、`aniso_v19_static_parts`；汇总入口为 `aniso_v19_analysis`。运行目录拒绝覆盖；重跑应使用独立源码副本及新的结果目录。CUDA 和完整仓库测试本轮未重跑；没有非线性连续体参考证书。

证据：[完整循环](results/lite-aniso-mainline/v19/cycle-acceptance.json)、[移动短窗](results/lite-aniso-mainline/v19/short-moving-acceptance.json)、[空间积分及四方向静态](results/lite-aniso-mainline/v19/space-summary.json)、[局部 Q3 参考](results/lite-aniso-mainline/v19/local-q3-reference.json)、[变形惯性](results/lite-aniso-mainline/v19/deformed-inertia.json)、[同状态约束分解](results/lite-aniso-mainline/v19/constraint-projection-parts.json)、[实际反力误差分解](results/lite-aniso-mainline/v19/reaction-error-parts.json)、[30 项测试](results/lite-aniso-mainline/v19/tests.json)、[验收封存](results/lite-aniso-mainline/v19/completion-check.json)、[SHA256 清单](results/lite-aniso-mainline/v19/artifact-sha256.json)。旧 v18 结果和源码保留不变。
''')
    report=ROOT/'docs/ANISO_LITE_ENDPOINT_INERTIA_COMPATIBILITY_ZH.md';report.write_text(''.join(text))
    banner=f"**最新验证进展（v19）：** 边界速度与实际冲量共同更新完成四档 48000 步循环；最细相邻原始反力差 {pct(last['raw_reaction_relative'])}、全程应力差 {pct(last['relative'])}；应力通过 2% 时间门槛，原始反力未通过。30 项测试、28 份独立快照、9 条细步长移动恢复完成。约束损失极小但未证明趋零；充分积分消除初始惯性欠采样，相容梯度候选的 F45 反力仍偏硬约 29.7%，空间精度未通过，默认未切换。参见 [v19 中文报告](docs/ANISO_LITE_ENDPOINT_INERTIA_COMPATIBILITY_ZH.md)。\n\n"
    for file in ('README.md','RUNNING_RESTORED.md'):
        p=ROOT/file;old=p.read_text();assert '最新验证进展（v19）' not in old;i=old.index('\n')+1;p.write_text(old[:i]+'\n'+banner+old[i:].lstrip('\n'))
    print(report)
if __name__=='__main__':main()
