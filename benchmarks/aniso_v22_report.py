"""Generate the v22 report and plots from completed, unsmoothed evidence."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_v22_common import *

def main():
    f=load(OUT/'final-spatial-acceptance.json');r=load(OUT/'reference-summary.json');a=load(OUT/'static-audits.json');b=load(OUT/'multiscale-audit.json');n=load(OUT/'nonlinear-audits.json');assert f['completed'] and r['completed'] and a['completed'] and b['completed'] and n['completed'];figures=OUT/'figures';figures.mkdir(exist_ok=True)
    candidates=[('q2',OUT/'space/q2'),('q3',OUT/'space/q3'),('q4',OUT/'space/q4'),('q4 + overlapping patches',OUT/'multiscale/space/q4')]
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':165})
    fig,ax=plt.subplots(figsize=(8.5,4.1))
    for label,folder in candidates:
        rows=list(load(folder/'summary.json')['records'].values());ax.plot([v['scalar_local_dofs'] for v in rows],[100*v['training_global_stress_relative'] for v in rows],'o-',label=label)
    ax.axhline(2,color='red',ls='--',lw=1);ax.set_xlabel('Added scalar functions (three displacement DOFs each)');ax.set_ylabel('Global stress difference [%]');ax.set_title('Same budget and original potential; v21 reference used for selection');ax.legend();fig.tight_layout();fig.savefig(figures/'space-budget.png');plt.close(fig)
    names=['v21-local144','q2-local144','q3-local144','q4-local144','q4-multiscale144','q4-full'];labels=['v21','Q2','Q3','Q4','Q4 + larger\npatches','Full Q4'];x=np.arange(len(names));fig,ax=plt.subplots(1,2,figsize=(11.4,4.1))
    for j,region in enumerate(('global','grip','interior')):
        for z,key in zip(ax,('stress_relative','fiber_strain_relative')):z.bar(x+(j-1)*.24,[100*f['cases'][name]['regions'][region][key] for name in names],width=.24,label=region)
    for z in ax:z.set_xticks(x,labels,fontsize=8);z.axhline(2,color='red',ls='--',lw=1);z.legend(fontsize=8)
    ax[0].set_ylabel('Stress difference [%]');ax[1].set_ylabel('Fiber strain difference [%]');fig.suptitle('All final fields compared with the SAME held-out v22 Q4 reference');fig.tight_layout();fig.savefig(figures/'regional-acceptance.png');plt.close(fig)
    names=['level1-q3-h','level1-q4-h','level1-p'];fig,ax=plt.subplots(1,2,figsize=(10.4,3.8));x=np.arange(3)
    for j,name in enumerate(names):
        for z,key in zip(ax,('stress_relative','fiber_strain_relative')):z.bar(x+(j-1)*.24,[100*r['pairs'][name]['regions'][region][key] for region in ('global','grip','interior')],width=.24,label=name)
    for z in ax:z.set_xticks(x,['Global','Grip','Interior']);z.axhline(2,color='red',ls='--',lw=1);z.legend(fontsize=8)
    ax[0].set_ylabel('Reference stress difference [%]');ax[1].set_ylabel('Reference fiber strain difference [%]');fig.suptitle('Reference h/p consistency; same physical grips, no corner exclusion');fig.tight_layout();fig.savefig(figures/'reference-checks.png');plt.close(fig)
    status=lambda yes:'通过' if yes else '未通过';pc=lambda x:f'{100*x:.4g}%';cases=f['cases'];ref=r['pairs']['level1-p'];best=cases['q4-multiscale144'];old=cases['v21-local144'];rows=[]
    for name,label in [('v21-local144','v21 Q2 局部空间'),('q2-local144','Q2 同预算控制'),('q3-local144','Q3 同范围局部空间'),('q4-local144','Q4 同范围局部空间'),('q4-multiscale144','Q4 重叠较大局部支撑'),('q2-full','Q2 全空间极限'),('q3-full','Q3 全空间极限'),('q4-full','Q4 全空间极限')]:
        v=cases[name];g=v['regions'];rows.append('| '+label+' | '+' | '.join([pc(v['reaction_relative']),pc(g['global']['stress_relative']),pc(g['grip']['stress_relative']),pc(g['interior']['stress_relative']),pc(g['interior']['fiber_strain_relative'])])+' |')
    reflines=[]
    for name in ('level0-q3-h','level0-q4-h','level0-p','level1-q3-h','level1-q4-h','level1-p'):
        v=r['pairs'][name]['regions'];reflines.append('| '+name+' | '+' | '.join(pc(v[k][m]) for k,m in [('global','stress_relative'),('grip','stress_relative'),('interior','stress_relative'),('grip','fiber_strain_relative'),('interior','fiber_strain_relative')])+' |')
    nonlinear=n['records']+[b['nonlinear']];maxte=max(v['tangent_relative_error'] for v in nonlinear)
    report=f'''# v22 夹持参考与局部高阶空间

本轮完成参考解两级夹持/边界加密、Q2/Q3/Q4 同预算局部空间、较大重叠局部支撑和分区应力验收。以同一最终 Q4 参考比较，v21 的全域应力差为 {pc(old['regions']['global']['stress_relative'])}，本轮 Q4 重叠支撑候选为 {pc(best['regions']['global']['stress_relative'])}；内部应力差从 {pc(old['regions']['interior']['stress_relative'])} 变为 {pc(best['regions']['interior']['stress_relative'])}。本轮候选的全区应力 2% 验收：{status(best['stress_passed'])}；参考全区应力及纤维应变自检：{status(r['all_stress_and_fiber_passed'])}。两项结果分别判定，不能互相替代。

范围仍为线性静态 F45 空间研究。物理盒体、硬夹持、右端位移 0.005、材料参数 μ=10、λ=20、k_f=200，以及原载体稳定化系数不变。没有质量项、对角刚度平移、额外耗散或应力平滑。新局部空间尚未接入移动动力学，生产默认未切换。ISO/F0/F90 用于一致性与静态刚度检查，尚未证明其空间精度同步改善。

| 同一最终参考下的方案 | 反力差 | 全域应力差 | 夹持应力差 | 内部应力差 | 内部纤维应变差 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

本轮归因对照显示：同范围 Q2→Q4 的全域应力差从 {pc(cases['q2-local144']['regions']['global']['stress_relative'])} 变为 {pc(cases['q4-local144']['regions']['global']['stress_relative'])}，内部差异仍接近；同样使用 Q4 并增加较大重叠支撑后，全域差异降至 {pc(best['regions']['global']['stress_relative'])}，内部降至 {pc(best['regions']['interior']['stress_relative'])}。主要收益来自扩大协调变形的范围。完整 Q4 空间的内部应力差为 {pc(cases['q4-full']['regions']['interior']['stress_relative'])}，说明当前有限局部空间仍有明显改进余地；这些数值均是相对当前参考的差异。

前五行均为 144 个新增标量函数，即 432 个局部位移自由度，另有原 225 个自由载体位移自由度。Q2/Q3/Q4 控制组使用同一物理网格、初始载体、54 个局部区域及每轮 8 个区域、共 6 轮的预算。阶次提高允许单元内部有更丰富的变形。各组按自己的当前残差重新选点，因此是同策略的阶次对照，不是强制同一组基向量。

Q4 较大支撑组在同一 54 个区域之外加入三个区域：x∈[0.25,0.5625]、[0.4375,0.75]、[0.34375,0.65625]，横截面为完整物理横截面。最大长度为自由段的 62.5%，没有一个区域覆盖整个自由段。它们仍通过同一评分竞争每轮的 8 个名额，总函数预算不变。实际六轮都选中了这三个较大区域，其余五个名额用于较小区域；48 次选区中有 18 次来自较大区域。人工截面上局部校正为零，物理横向表面保持自然边界。该对照检验较大范围的斜向伸缩、横向收缩和剪切协调，不能把新增基函数当成新增材料刚度。相同函数预算不代表相同计算成本：较大区域的校正求解更昂贵。本轮记录的耗时受到并行任务竞争影响，不作为公平性能基准。

![同预算空间对照](results/lite-aniso-mainline/v22/figures/space-budget.png)

**每个局部校正和最终平衡仍来自同一物理势能。** 在局部区域 A 求解

\\[
K_{{AA}}w_A=-r_A,\\qquad u=Ay+Z\\alpha,
\\]

\\[
U(y,\\alpha)=\\int_\\Omega\\psi(I+\\nabla u)\\,dV+\\tfrac12\\sum_b y_b^T K_s y_b,
\\quad r=\\frac{{\\partial U}}{{\\partial(y,\\alpha)}},\\quad K=\\frac{{\\partial^2U}}{{\\partial(y,\\alpha)^2}}.
\\]

取校正向量的三个分量函数组成标量空间，再让三个空间分量共享该空间，以保留任意有限刚体转动的表达。局部求解采用矩阵自由张量算子与可分离预条件器；预条件器不进入势能。小规模测试与独立全矩阵组装的主子矩阵逐项核对，区分物理自由面与人工局部边界。

选点分数仍为预计应力平方差减少量

\\[
G_A=-2r_\\sigma^Tw_A-w_A^TK_\\sigma w_A,\\qquad
K_\\sigma=\\int B^TH^THB\\,dV.
\\]

HᵀH 只评价应力差，平衡方程使用原 H。训练参考升级为上一轮已封存 Q4；新一轮最终参考不参与构造、排序或挑选最佳阶段。最终均取预先固定的第六轮。每轮保留能量、应力训练差、所选区域、局部残差和四方向静态结果。评分预测单次局部校正的收益，而加入多个函数后还要全局重新平衡，因此应力差不保证单调下降。本轮较大支撑组第四轮训练应力差约 14.83%，第五轮回升到约 17.63%，尽管物理势能仍在下降；完整曲线保留，最终不改选最有利阶段。能量最小化与应力差最小化使用不同的度量，不能互相替代。原局部函数以稀疏形式保存，另存基变换矩阵；重建的最终空间与计算所用空间核对一致。

**参考同时检查 h 与 p。** 从 v21 最细网格出发，每一级将自由段两端最近两个区间、两个横向方向的两端最近两个区间各二分。其他区间保留。刚性夹具体积的位移分别严格为零或常量，因此用单个区间表示夹具体积；物理区域和积分区域没有裁掉。最终 Q4 有 {r['cases']['level1-q4']['nodes']} 个节点，{r['cases']['level1-q4']['free_dofs']} 个自由位移未知量。

| 参考比较 | 全域应力 | 夹持应力 | 内部应力 | 夹持纤维应变 | 内部纤维应变 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(reflines)}

首次 level0 Q4 求解在保存结果之前触发线性残差与功恒等式的组合断言。原脚本没有持久化失败数值，不能从日志精确还原失败子项；原失败日志完整保留。后续协议保持同一网格和 1e-8 的残差/功恒等式门槛，收紧线性求解相对容差到 1e-13，必要时再到 1e-14，每次尝试先保存诊断和场，再决定是否接受。剩余三个固定物理问题并行求解，初值只用于加速，不定义参考解。见 [续算协议](results/lite-aniso-mainline/v22/reference-resume-protocol.json)。

h 表示同阶次相邻网格差；p 表示同网格 Q3/Q4 差。最终分区状态见 [参考汇总](results/lite-aniso-mainline/v22/reference-summary.json)。这仍是离散解的一致性检验，不是严格的连续体误差上界。硬夹持附近的应力梯度继续保留在验收中，没有排除角部。

![参考阶次和网格检查](results/lite-aniso-mainline/v22/figures/reference-checks.png)

**内部应力单独验收。** 分区沿用 v21：夹持区 x≤0.3125 或 x≥0.6875，内部为其余区域，深内部为 0.375<x<0.625；全域包括刚性夹具体积。纤维方向 a=(1,1,0)/√2，

\\[
\\varepsilon_{{aa}}=a^T\\operatorname{{sym}}(\\nabla u)a
=\\tfrac12(\\varepsilon_{{xx}}+\\varepsilon_{{yy}}+2\\varepsilon_{{xy}}),\\qquad
\\eta_{{P,r}}^2=\\frac{{\\int_{{\\Omega_r}}\\|P-P_{{ref}}\\|_F^2dV}}{{\\int_{{\\Omega_r}}\\|P_{{ref}}\\|_F^2dV}}.
\\]

评价在双方网格与分区的共同区间上使用五点 Gauss 积分，精确积分本轮分片多项式的应力差平方。新增张量分解只加速同一积分；已与原逐点梯度及独立分区积分比较，并核对非匹配 Q3/Q4 网格。完整有限元空间的对照用于识别网格表达限制，反力另加原载体稳定化的最小贡献；有限空间与全空间的百分比不能直接相减成为严格误差分解。

![最终区域验收](results/lite-aniso-mainline/v22/figures/regional-acceptance.png)

**实现与物理检查。** 40 项相关测试通过，其中 8 项本轮新增；不是完整仓库或 CUDA 验收。四组各 6 轮、四个材料方向，共 96 组去质量静态候选均保持正刚度和正局部 Schur 刚度；没有额外零模态。原空间中的刚体、仿射和二次弯曲场继续精确表达。这验证正常变形不被新增运动约束破坏，并不自动证明连续体弯曲刚度精确。

四个最终实际大空间另外通过非线性势能方向导数、内力与切线、有限刚体转动和平移检查，最大切线相对差为 {maxte:.4g}。这些测试使用同一非线性势能的原生高阶积分，不把线性静态平衡解称为非线性循环验收。

[最终区域数据](results/lite-aniso-mainline/v22/final-spatial-acceptance.json) · [原范围静态检查](results/lite-aniso-mainline/v22/static-audits.json) · [较大支撑检查](results/lite-aniso-mainline/v22/multiscale-audit.json) · [实际非线性候选检查](results/lite-aniso-mainline/v22/nonlinear-audits.json)

下一步按 Q3/Q4 应力差的空间分布细化整个夹持过渡层；本轮最近端点的二分使同阶次网格差缩小，但阶次差仍明显。局部空间则继续检查加入候选函数并重新平衡后的分区应力收益，并在新加载、材料方向上验证局部空间的适用性。完整空间对照仍有的差异，需要网格或高阶近似进一步改善；仅改变选点不能全部消除。接入移动粒子之前，还需为新增局部变量定义相容的惯性和历史更新，重新运行完整加载—保持—卸载验收。

实现入口：[高阶局部空间](../engine/aniso_phase1/high_order_space.py)、[张量区域积分](../engine/aniso_phase1/tensor_metrics.py)。实验入口为 `benchmarks/aniso_v22_*.py`。冻结协议和已归档目录不覆盖；重跑数值研究须使用新的输出目录或独立副本。相关测试可执行：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPM_LITE_DATA_ROOT=/dev/shm .venv/bin/python -m unittest tests.test_aniso_v22_metrics tests.test_aniso_v22_space tests.test_aniso_v21_reference tests.test_aniso_v21_local_space tests.test_aniso_v21_metrics tests.test_aniso_v21_gain tests.test_aniso_v21_quartic tests.test_aniso_compatible_carrier tests.test_aniso_v20_variational benchmarks.aniso_local_q3.CubicTests benchmarks.aniso_local_reference.LocalReferenceTests -v
```

[测试记录](results/lite-aniso-mainline/v22/tests.json) · [完成清单](results/lite-aniso-mainline/v22/completion-check.json) · [成果校验](results/lite-aniso-mainline/v22/artifact-sha256.json) · [源码归档](results/lite-aniso-mainline/v22/source-delivered.zip)
'''
    report_path=ROOT/'docs/ANISO_LITE_GRIP_HIGH_ORDER_ZH.md';report_path.write_text(report)
    intro=f"**最新验证进展（v22）：** 完成两级夹持参考加密、Q3/Q4 交叉验证及同预算高阶/重叠局部支撑。40 项相关测试、96 组去质量静态候选和 4 份实际非线性检查通过。最终 Q4 重叠支撑全域应力差 {pc(best['regions']['global']['stress_relative'])}、内部应力差 {pc(best['regions']['interior']['stress_relative'])}；全区应力验收：{status(best['stress_passed'])}，参考全区自检：{status(r['all_stress_and_fiber_passed'])}。生产默认未切换。参见 [v22 中文报告](docs/ANISO_LITE_GRIP_HIGH_ORDER_ZH.md)。\n\n"
    for name in ('README.md','RUNNING_RESTORED.md'):
        path=ROOT/name;text=path.read_text();marker='**最新验证进展（v22）：**'
        if marker in text:
            start=text.index(marker);end=text.index('\n\n',start)+2;text=text[:start]+intro+text[end:]
        else:
            first=text.index('\n')+1;text=text[:first]+'\n'+intro+text[first:]
        path.write_text(text)
    print('REPORT',report_path,flush=True)
if __name__=='__main__':main()
