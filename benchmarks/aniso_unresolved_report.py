"""Generate the v13 report and standalone figures from completed artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_unresolved_history import ROOT,BASE,OUT,LEVELS,MODES,load
from benchmarks.aniso_unresolved_analysis import BUDGET
LABEL={'baseline':'v12 baseline','null':'exact null','weak':'weak transfer'}
ZH={'baseline':'v12 基线','null':'严格零空间','weak':'弱传递耗散'}


def pct(x):return f'{100*x:.4f}%'
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def figures(s):
    colors={'baseline':'#5b6573','null':'#ce7f28','weak':'#127c82'}
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),constrained_layout=True)
    for mode in MODES:
        rows=[json.loads(l) for l in (OUT/'cases'/f'{mode}-fourth'/'steps.jsonl').read_text().splitlines()]
        t=np.array([r['time'] for r in rows]);c=colors[mode]
        axes[0,0].plot(t,[r['right_force'] for r in rows],color=c,label=LABEL[mode])
        axes[0,1].plot(t,np.array([r['mechanical'] for r in rows])*1e6,color=c)
        axes[1,0].semilogy(t,np.maximum([r['kinetic'] for r in rows],1e-16),color=c)
        dts=[LEVELS[l] for l in list(LEVELS)[1:]]
        axes[1,1].loglog(dts,[100*r['P_relative'] for r in s['refinement'][mode]['ramp']['pairs']],'-o',color=c)
    for ax in axes.flat:
        ax.grid(alpha=.25)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        for a,b in ((.5,.6),(1.1,1.2)):ax.axvspan(a,b,color='#aaa',alpha=.12)
        ax.set_xlabel('time (s)')
    axes[0,0].set_ylabel('right reaction (N)');axes[0,0].legend()
    axes[0,1].set_ylabel('mechanical energy (micro J)')
    axes[1,0].set_ylabel('joint v/C kinetic energy (J)')
    axes[1,1].set_xlabel('finer timestep (s)');axes[1,1].set_ylabel('loading stress pair difference (%)');axes[1,1].axhline(2,color='black',ls=':',label='2% gate')
    fig.suptitle('F45 stretch / hold / unload: same static energy, selective kinetic dissipation')
    for ext in ('png','pdf'):fig.savefig(OUT/('cycle-comparison.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    names=['transfer','boundary','solve','final projection','moving metric','dissipation','reference rebuild']
    for j,h in enumerate(('hold','final_hold')):
        ax=axes[j]
        for i,mode in enumerate(MODES):
            a=s['phase_metrics'][mode+'-fourth'][h]
            ax.bar(np.arange(len(names))+(i-1)*.25,[a['stages_J'][k]*1e6 for k in BUDGET],width=.25,color=colors[mode],label=LABEL[mode])
        ax.set_xticks(range(len(names)),names,rotation=35,ha='right');ax.axhline(0,color='black',lw=.7);ax.set_ylabel('energy change (micro J)');ax.set_title('Peak hold' if j==0 else 'Hold after unload');ax.grid(axis='y',alpha=.25)
    axes[0].legend()
    for ext in ('png','pdf'):fig.savefig(OUT/('hold-energy-stages.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    from benchmarks.aniso_dynamic_space import stress
    protocol=load(OUT/'protocol.json')
    for mode in MODES:
        name=mode+'-fourth'
        with np.load(OUT/'cases'/name/'frames.npz') as z:
            F=z['F'];t=z['time'];P=stress(F.reshape(-1,3,3),protocol['configs'][name]).reshape(F.shape)
        norm=lambda a:np.sqrt(np.mean(np.sum(a*a,axis=(2,3)),axis=1))
        peak=int(np.argmin(abs(t-.5)))
        axes[0].plot(t,norm(P)/norm(P)[peak],color=colors[mode],label=LABEL[mode])
        axes[1].plot(t,norm(F-np.eye(3))/norm(F-np.eye(3))[peak],color=colors[mode])
    for ax in axes:
        ax.set_xlabel('time (s)');ax.grid(alpha=.25)
        for a,b in ((.5,.6),(1.1,1.2)):ax.axvspan(a,b,color='#aaa',alpha=.12)
    axes[0].set_ylabel('stress RMS / end-of-load value');axes[0].legend()
    axes[1].set_ylabel('deformation RMS / end-of-load value')
    fig.suptitle('Internal fields can remain large even when the total reaction is small')
    for ext in ('png','pdf'):fig.savefig(OUT/('residual-fields.'+ext),dpi=180)
    plt.close(fig)


def main():
    s=load(OUT/'summary.json');a=load(OUT/'artifact-check.json');screen={m:load(BASE/'v13-screen'/m/'summary.json') for m in ('none','null','weak')}
    figures(s)
    loadtable=table(['方案','加载应力差','加载末态应力差','加载 F 差','加载反力差','加载应力观测阶（两段）','全周期应力差'],[[ZH[m],pct(s['refinement'][m]['ramp']['pairs'][-1]['P_relative']),pct(s['refinement'][m]['ramp']['pairs'][-1]['P_terminal_relative']),pct(s['refinement'][m]['ramp']['pairs'][-1]['F_relative']),pct(s['refinement'][m]['ramp']['pairs'][-1]['reaction_relative']),', '.join(f"{r['P_order']:.3f}" for r in s['refinement'][m]['ramp']['pairs'][1:]),pct(s['refinement'][m]['whole']['pairs'][-1]['P_relative'])] for m in MODES])
    holdtable=table(['方案','峰值保持 ΔE / J','峰值保持应力漂移 / 满载应力','卸载后保持 ΔE / J','最终应力 / 满载应力','最终反力 / N'],[[ZH[m],f"{s['phase_metrics'][m+'-fourth']['hold']['mechanical_change_J']:.6e}",pct(s['phase_metrics'][m+'-fourth']['hold']['P_change_over_load_peak']),f"{s['phase_metrics'][m+'-fourth']['final_hold']['mechanical_change_J']:.6e}",pct(s['phase_metrics'][m+'-fourth']['residual']['P_over_load_peak']),f"{s['phase_metrics'][m+'-fourth']['residual']['reaction_N']:.6e}"] for m in MODES])
    budgettable=table(['环节（峰值保持 0.5–0.6 s）']+[ZH[m]+' / J' for m in MODES],[[k]+[f"{s['phase_metrics'][m+'-fourth']['hold']['stages_J'][k]:.6e}" for m in MODES] for k in BUDGET])
    screentable=table(['同一起点，保持 0.05 s','末态动能 / J','应力漂移（对末态归一）','有效区间 ΔE / J'],[[m,f"{r['final_kinetic']:.6e}",pct(r['stress_change']),f"{r['stages']['delta_mechanical']:.6e}"] for m,r in screen.items()])
    alltable=table(['方案','dt / s','峰值保持 ΔE / J','卸载后保持 ΔE / J','最终应力 / 满载应力'],[[ZH[m],f'{dt:g}',f"{s['phase_metrics'][m+'-'+l]['hold']['mechanical_change_J']:.6e}",f"{s['phase_metrics'][m+'-'+l]['final_hold']['mechanical_change_J']:.6e}",pct(s['phase_metrics'][m+'-'+l]['residual']['P_over_load_peak'])] for m in MODES for l,dt in LEVELS.items()])
    phasetable=table(['方案','比较区间','应力差','F 差','反力差','弹性能差','时空四项 ≤2%','末态 P/F ≤2%','观测阶筛查'],[[ZH[m],ph,pct(v['pairs'][-1]['P_relative']),pct(v['pairs'][-1]['F_relative']),pct(v['pairs'][-1]['reaction_relative']),pct(v['pairs'][-1]['energy_relative']),str(v['last_2pct_passed']),str(v['last_terminal_2pct_passed']),str(v['order_screen_passed'])] for m in MODES for ph,v in s['refinement'][m].items()])
    checks=table(['验收项','结果'],[['回归测试','103 / 103，含 8 项新增检查'],['完整循环',f"{s['trajectories']} 条，{s['steps']:,} 步"],['实现/物理检查',str(s['physical_checks_passed'])],['独立快照',f"{len(a['snapshots'])} 份；最大状态差 {a['max_state_error']:.3e}"],['独立反力差',f"{a['max_force_error']:.3e} N"],['同支撑历史闭合',f"{s['history_closure_max']:.3e}"],['同支撑历史差异率',f"{s['frozen_history_rate_max']:.3e} / s"],['移动支撑平均值差异率',f"{s['moving_history_rate_max']:.3e} / s"],['逐步能量分账闭合',f"{max(r['max_stage_budget_error'] for r in s['checks'].values()):.3e} J"],['弱传递方案全阶段时间门槛',str(s['weak_all_phase_time_2pct_passed'])],['弱传递方案全部保持段净能量不增',str(s['weak_all_holds_nonincrease'])],['整体空间/时间精度通过','False；未切换默认']])
    text=r'''# v13：停载能量定位、联合 v/C 耗散与四档循环验收

本轮完成三项工作：分环节核算停载能量；实现保护仿射运动的联合速度耗散；对 v12 基线、严格零空间对照、弱传递耗散运行完整四档拉伸—保持—卸载。保持 v11/v12 静态势能、材料历史、残差与切线。实现通过与整体精度通过分别报告，默认仍为 `velocity_dissipation=none`。

RESULT_LEAD

## 1. 通俗解释与能量来源

v12 保留了更多局部速度历史，使加载应力更稳定，但也保留了网格难以感知的运动。可以把它理解为：粒子内部仍在“动”，网格只看见其中一小部分。停载后，这些历史继续反馈，同时稳定化参考重建会改变离散弹性能，因而需要分别核算。

夹具速度为零时，外功为零。正常动能与弹性能交换满足 $\Delta K\approx-\Delta U$，所以应检查总机械能 $E=K+U$，不能仅把 $K$ 上升当成增能。这里的 $K$ 包含 $v$ 和 APIC 仿射速度 $C$：

\[
K=\frac12\sum_p m_p\left(\|v_p\|^2+\operatorname{tr}(C_pD_pC_p^T)\right),\quad
D_p=\sum_iT_{ip}(x_i-x_p)(x_i-x_p)^T.
\]

$T_{ip}$ 是粒子→中心→节点的复合权重。完整支撑下，$D_p$ 为对角阵，$D_{p,kk}=h^2[f_{p,k}(1-f_{p,k})+1/4]$，其中 $f=\operatorname{frac}(x_p/h-1/2)$。

重算旧 v12 最细档 0.500125–0.55 s 的 399 步：总能量增长 $2.11041\times10^{-7}$ J。参考重建贡献 $+1.71654\times10^{-7}$ J（约占净增长 81.3%）；传递往返贡献 $+5.71509\times10^{-8}$ J；网格求解与边界投影分别为 $-9.24144\times10^{-9}$、$-8.52218\times10^{-9}$ J。首个重启初始化步排除，区间端点能量与阶段求和一致。

粒子→中心约为 $-1.53908\times10^{-3}$ J，中心→网格约为 $-5.20719\times10^{-5}$ J，返回粒子约为 $+1.59121\times10^{-3}$ J。三者大量抵消，必须合并后判断净注入；不能单看返回粒子这一项。

新账本进一步把位置移动引起的动能度量变化与新增耗散分开：

\[
\Delta E=\Delta E_{\rm transfer}+\Delta E_{\rm boundary}
+\Delta E_{\rm solve}+\Delta E_{\rm final\ projection}
+\Delta K_{\rm moving\ metric}+\Delta K_{\rm filter}+\Delta E_{\rm rebuild}.
\]

其中 $\Delta K_{\rm moving\ metric}=K(x^{n+1},v^*,C^*)-K(x^n,v^*,C^*)$，$\Delta K_{\rm filter}=K(x^{n+1},v^+,C^+)-K(x^{n+1},v^*,C^*)$。加载时求解项包含外功；只有在两个零外功保持段，才把这张账本直接用于无外部输入的增能判断。

## 2. 只作用于弱传递运动的候选

把每个速度分量的状态写成 $z=(v,C_{:1},C_{:2},C_{:3})$，动能度量记为 $M$，节点质量阵为 $M_g$。定义

\[
y=M^{1/2}z,\quad K=\tfrac12\|y\|^2,\quad
j=Az,\quad W=M_g^{-1/2}AM^{-1/2},\quad K_g=\tfrac12\|Wy\|^2.
\]

$A$ 使用实际 P2G 权重和位置偏移，同时包含 $v$ 与 $C$。先用动能内积保护全局仿射速度子空间 $Q_a$：包括平移、刚体转动、拉伸和剪切的速度场；每个输出分量四个基，共十二个自由度。再分解 $W(I-Q_aQ_a^T)=U\Sigma V^T$，得到

\[
y=y_a+\sum_i a_i r_i+y_0,\qquad
 y^+=y_a+\sum_i\alpha_i a_i r_i+\alpha_0y_0.
\]

$y_0$ 是严格零空间，$\sigma_i^2$ 是非仿射模式传递到网格的能量比例。两个对照如下：

- `null`：$\alpha_i=1,\ \alpha_0=0$，每步完全清除严格零空间，保留每个节点的 P2G 动量。
- `weak`：$\alpha_i=\exp[-\gamma\Delta t\max(0,1-\sigma_i^2/0.1)^2]$，$\alpha_0=\exp(-\gamma\Delta t)$；$\gamma=\sqrt{\mu/\rho}/h$。传递比例不低于 10% 的模式保持不变；系数按物理时间定义，没有按反力调参。

因此，固定输入状态下，

\[
\Delta K=-\frac12\sum_i(1-\alpha_i^2)\|a_i\|^2
-\frac12(1-\alpha_0^2)\|y_0\|^2\le0.
\]

弱传递方案也可写为固定几何下非负二次耗散势的精确流：

\[
\mathcal R(y)=\frac{\gamma}{2}\left[\sum_i g_i\|a_i\|^2+\|y_0\|^2\right],\quad
g_i=\max(0,1-\sigma_i^2/0.1)^2,\quad \dot y=-\nabla_y\mathcal R.
\]

因此 $\dot K=-2\mathcal R\le0$。这是单独的速度耗散子步；静态弹性势能的残差与切线保持原样，并不声称整个分步算法来自一个新的统一隐式势能。

两种方案都保持总线动量和包含 APIC 内部项的角动量：

\[
p=\sum_pm_pv_p,\qquad
\ell=\sum_pm_p\left[x_p\times v_p+\sum_kD_{p,kk}\,e_k\times C_{p,:k}\right].
\]

这些量是速度状态与平移/转动仿射场的动能内积，所以保护对应子空间即可保持它们。`weak` 会改变弱模式的各节点动量；它不声称逐节点动量不变。固定几何下指数衰减具有时间可分性 $D_{\Delta t}^2=D_{2\Delta t}$；移动粒子后模态空间变化，因此仍需四档实际轨迹验证。

同一 v12 最细加载末态中，严格零空间只占动能约 5.7%；它以外还存在大量弱传递运动。这也是保留严格零空间对照的原因。10% 是预先固定的分辨能力筛选阈值，不能据此证明所有被衰减运动都没有物理意义。

滤波只在实际返回粒子之后改变 $v,C$；$x,F,L$、当前网格解和弹性势能不变。可能失败的分解在粒子提交前完成，失败不提交粒子或时钟。当前实现为小规模 CPU 稠密 SVD 原型，上限 512 个活跃节点、2048 个粒子，不适用于直接放大为生产大网格。

## 3. 相同输入下的短停载因果对照

三条轨迹从完全相同的 v12 最细加载末态开始，$x,F,v,C$ 均一致，dt=0.000125 s，保持 400 步。下面有效能量区间仍排除首个重启初始化步；应力漂移使用各自末态应力归一，不能与后面的峰值归一表混用。

SCREEN_TABLE

弱传递方案在这个区间的直接耗散为 $-3.56488\times10^{-6}$ J，但参考重建仍为 $+1.31607\times10^{-7}$ J，传递往返仍为 $+1.20854\times10^{-7}$ J。**改善主要来自新的选择性动能耗散，并未消除参考重建的增能机制。** 三种模式的九份实际快照另用 Gram 特征分解复算，与生产矩形 SVD 独立核对。

## 4. 完整四档拉伸—保持—卸载

F45、原有物理几何、材料与边界；网格 9、192 粒子。dt 为 0.001、0.0005、0.00025、0.000125 s。每条轨迹连续从 t=0 运行到 1.2 s，不在保持段重启：

\[
u(t)=\begin{cases}
0.0025[1-\cos(\pi t/0.5)],&0\le t<0.5,\\
0.005,&0.5\le t\le0.6,\\
0.0025[1+\cos(\pi(t-0.6)/0.5)],&0.6<t<1.1,\\
0,&1.1\le t\le1.2.
\end{cases}
\]

加载和卸载端点速度连续为零。速度混合仍为 $\beta_v=0.9^{\Delta t/0.001}$，仿射历史 $\beta_C=1$。每模式 18,000 步，三模式共 54,000 步。基线前 0.5 s 的四档结果与封存 v12 核对，反力、能量与逐粒子 F 差低于 $10^{-12}$。

最细相邻两档的加载和全周期差如下。加载统计区间为 0.05–0.5 s，全周期统计为 0.05–1.2 s；与 v12 一样剔除最初 0.05 s，完整仿真本身仍从 t=0 开始。应力差是匹配物理时刻、全部粒子的体积加权时空 RMS，材料参考粒子体积相等。相对误差分母为细档场范数；反力分母下限 0.001 N。$F$ 使用 $F-I$ 归一，防止单位阵掩盖误差。

\[
e_P=\frac{\|P_{\Delta t}-P_{\Delta t/2}\|_{V,t}}{\|P_{\Delta t/2}\|_{V,t}},\qquad
q=\log_2\frac{\|P_{2\Delta t}-P_{\Delta t}\|_{V,t}}{\|P_{\Delta t}-P_{\Delta t/2}\|_{V,t}}.
\]

LOAD_TABLE

![循环响应与加载应力时间差](results/lite-aniso-mainline/v13/cycle-comparison.png)

最细档的保持与卸载后残余响应如下。“满载应力”指 t=0.5 s、夹具达到最大位移时的应力场范数，不是全程应力最大值。材料仍是弹性材料；t=1.2 s 的“残余”只是有限观察时间的非零动态/离散响应，不能解释为塑性残余应变。满载应力归一能避免卸载后接近零的分母放大。

HOLD_TABLE

![内部应力与变形残余](results/lite-aniso-mainline/v13/residual-fields.png)

完整四档的停载能量和残余响应：

ALL_TABLE

峰值保持段的分环节能量如下（夹具外功为零）：

BUDGET_TABLE

![两个保持段的能量分账](results/lite-aniso-mainline/v13/hold-energy-stages.png)

所有阶段的最细相邻时间误差均保留，不仅验收反力：

PHASE_TABLE

## 5. 验证范围、静态保持与尚未解决的问题

CHECKS_TABLE

103 项检查包含新增 8 项：实际两层 P2G 对照；动能公式、线/角动量、可见模式；仿射与刚体场；固定几何时间可分性；同输入解与能量不变；移动粒子历史与预算；分解失败回滚；非法输入拒绝。耗散前数组必须独立复制，测试含 CPU 数组别名回归断言。另有 74 组支撑切换隔离检查：包括精确单元边界及边界两侧，模态秩覆盖 110–204，动能、线/角动量和仿射保持均通过。该检查的粒子位置是预先指定的，不能代替完整自由运动的跨单元能量验收。

静态能量相关五个源文件与 v12 完全相同；继承已封存的 24 组拉伸和 16 组梁结果，同时重新运行 ISO/F0/F45/F90 的实际 Warp 去质量切线及能量—残差—切线、仿射/刚体/二次场检查。本轮没有重跑全部 40 组静态扫描，也没有增加新的空间加密参考。此前 grid33 与局部 Q3 参考的 F45 内部空间应力差约 13.37%（与本轮 grid9 时间差分开），不能由本轮动力学耗散宣布解决。

历史率仍使用 $\eta_{\rm hist}=\|F_{\rm rebuild}-F_{\rm commit}\|_V/\Delta t$。同支撑重建与移动支撑平均值变化分别记录；后者不应被误称为逐粒子材料历史被清零。每步提交的 $F_p$ 与材料试算完全一致。即使同支撑历史闭合到舍入误差，重新构造稳定化参考仍可能改变离散势能，这两项检查不能互相替代。

仍应分别看三件事：加载时应力对 dt 的敏感性；零外功保持时的总能量账本；卸载后的响应。总反力相近并不代表应力准确。弱模式滤波提供了可证明的局部动能下降，但不保证整个求解与参考重建链条逐步总能量下降，也不保证移动模态空间下的时间精度。

下一步优先处理稳定化参考重建的离散能量改变，并保留严格零空间对照。对弱传递耗散，先分离“移动模态空间/求解与耗散分步”的时间误差，以及被耗散的真实局部变形运动；当前实验不能区分两者的贡献，不应继续单纯加大耗散。所有候选仍需同时通过局部历史、静态刚度、动量和能量—力—切线检查；再分别检验时间与空间应力精度，不能用调小静态稳定化或仅匹配反力代替。

## 6. 复现与归档

公共选项为 `--velocity-dissipation none|null|weak`，默认 `none`。20 步入口命令已实际验证：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso --device cpu --scene tensile --grid 9 --dt 0.001 --fiber-angle 45 --smooth-loading --history-consistency residual_center --stabilization selective_patch --boundary-impulse-transfer --apic-transfer incremental --affine-flip-ratio 1 --velocity-dissipation weak --reaction-force-atol 1e-7 --headless --steps 20
```

完整循环由专门验收运行器提供；普通演示的加载时序不等同于本报告的循环。使用新的输出目录，依次执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.aniso_unresolved_history freeze --output /tmp/mpm-v13-replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.aniso_unresolved_history tests --output /tmp/mpm-v13-replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.aniso_unresolved_history static --output /tmp/mpm-v13-replay
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.aniso_unresolved_history run --jobs 12 --output /tmp/mpm-v13-replay
```

正式结果：[冻结协议](results/lite-aniso-mainline/v13/protocol.json)、[机器汇总](results/lite-aniso-mainline/v13/summary.json)、[独立复算](results/lite-aniso-mainline/v13/artifact-check.json)、[短停载对照](results/lite-aniso-mainline/v13-screen/)、[完成清单](results/lite-aniso-mainline/v13/completion-check.json)。源码、文档另存 `source-delivered.zip`，原始轨迹与所有验收结果用 SHA-256 封存。CPU float64 已验证，CUDA 本轮未运行。

早期诊断试跑保留在 `v13-diagnostic-attempt` 与 `v13-screen-diagnostic-attempt` 并标为未验收：CPU `.numpy()` 视图随赋值改变，导致耗散误记为传递。修正为复制后重跑正式验收，旧试跑不计入 54,000 步或改善依据。
'''
    lead=f"结果需要分开判断：弱传递方案的最细加载应力差为 {pct(s['refinement']['weak']['ramp']['pairs'][-1]['P_relative'])}，加载末态为 {pct(s['refinement']['weak']['ramp']['pairs'][-1]['P_terminal_relative'])}；基线分别为 {pct(s['refinement']['baseline']['ramp']['pairs'][-1]['P_relative'])}、{pct(s['refinement']['baseline']['ramp']['pairs'][-1]['P_terminal_relative'])}。严格零空间方案的加载应力差为 {pct(s['refinement']['null']['ramp']['pairs'][-1]['P_relative'])}，兼顾性较好，但仍未达到 2%。选择性耗散的局部守恒证明和实现检查通过，弱传递方案的加载应力时间敏感性则变大，不能将其作为已经验收的替代方案。"
    text=text.replace('RESULT_LEAD',lead)
    for key,value in [('SCREEN_TABLE',screentable),('LOAD_TABLE',loadtable),('HOLD_TABLE',holdtable),('ALL_TABLE',alltable),('BUDGET_TABLE',budgettable),('PHASE_TABLE',phasetable),('CHECKS_TABLE',checks)]:text=text.replace(key,value)
    (ROOT/'docs/ANISO_LITE_UNRESOLVED_VELOCITY_ZH.md').write_text(text)
    weak=s['refinement']['weak']['ramp']['pairs'][-1]['P_relative'];baseline=s['refinement']['baseline']['ramp']['pairs'][-1]['P_relative']
    null=s['refinement']['null']['ramp']['pairs'][-1]['P_relative']
    intro=f'> **v13 停载能量与联合速度耗散（2026-09-29）：** [实现、公式与完整循环验收](docs/ANISO_LITE_UNRESOLVED_VELOCITY_ZH.md)。新增可选 `--velocity-dissipation weak`；保持静态势能与局部历史，103 项回归、三方案四档完整循环 54,000 步及 96 份独立快照复算完成。F45 最细加载应力差：基线 {pct(baseline)}，严格零空间 {pct(null)}，弱传递 {pct(weak)}；两种耗散改善停载能量，但均未达到应力 2% 门槛。参考重建仍存在正能量增量，空间应力精度尚未通过。**整体精度未通过，默认未切换。**\n\n'
    readme=ROOT/'README.md';r=readme.read_text();mark='> **v13 停载能量与联合速度耗散';start=r.find(mark)
    if start>=0:r=r[:start]+r[r.index('\n\n',start)+2:]
    a,b=r.split('\n\n',1);readme.write_text(a+'\n\n'+intro+b)
    running=ROOT/'RUNNING_RESTORED.md';r=running.read_text();mark='2026-09-29 v13';start=r.find(mark)
    if start>=0:r=r[:start]+r[r.index('<!-- v13 end -->',start)+len('<!-- v13 end -->\n\n'):]
    a,b=r.split('\n\n',1);running.write_text(a+'\n\n'+'2026-09-29 v13：'+intro[2:]+'完整循环时序、守恒证明、未通过门槛与复现命令见报告。公共选项仅启用速度耗散，完整循环使用专门验收运行器。\n\n<!-- v13 end -->\n\n'+b)
    print('report and figures generated')

if __name__=='__main__':main()
