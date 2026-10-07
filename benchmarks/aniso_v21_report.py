"""Chinese repository report, generated only from completed numerical evidence."""
from benchmarks.aniso_v21_common import *

def percent(x):return f'{100*x:.4g}%'

def main():
    assert load(OUT/'finish-status.json')['completed'];final=load(OUT/'final-spatial-acceptance.json');ref=load(OUT/'reference-self-checks.json');static=load(OUT/'static-audits.json');tests=load(OUT/'tests.json');co=load(OUT/'reference-completed-summary.json');cases=final['cases'];audits=load(OUT/'candidate-nonlinear-audits.json')['records']+load(OUT/'candidate-gain-nonlinear-audit.json')['records'];assert len(audits)==3 and all(a['passed'] for a in audits)
    names=[('相容重建基线','graded/baseline'),('v20 固定局部空间','v20-local150'),('初始应力选点加密空间','graded/F45-stress-40'),('逐轮几何选点','adaptive/geometric/round6'),('逐轮按误差大小选点','adaptive/stress/round6'),('逐轮按可修正误差选点','adaptive/gain/round6'),('加密 Q2 全空间极限','full-graded-Q2-limit')]
    table='\n'.join('| '+label+' | '+str(r['scalar_local_dofs'] if r['scalar_local_dofs'] is not None else '全量 FE')+' | '+' | '.join([percent(r['reaction_relative'])]+[percent(r['regions'][k]['stress_relative']) for k in ('global','grip','interior')]+[percent(r['regions']['global']['fiber_strain_relative'])])+' |' for label,name in names for r in [cases[name]])
    rt='\n'.join('| '+label+' | '+' | '.join(percent(v['regions'][r][k]) for r in ('global','grip','interior') for k in ('stress_relative','fiber_strain_relative'))+' |' for label,name in [('Q3 相邻网格','h_q3'),('Q4 相邻网格','h_q4'),('最细同网格 Q3 与 Q4','p_q3_q4')] for v in [ref['pairs'][name]])
    last=cases['adaptive/gain/round6'];sv=load(OUT/'space-validation.json')['cases'];av=load(OUT/'adaptive-validation.json')['cases'];gv=load(OUT/'gain-validation.json')['cases'];at='\n'.join('| '+name+' | '+percent(row['regions']['global']['stress_relative'])+' | '+percent(row['regions']['global']['fiber_strain_relative'])+' |' for name,row in [('初始几何选点 120',sv['graded/F45-geometric-40']),('逐轮几何选点 120',av['geometric/round5']),('逐轮收益选点 120',gv['gain/round5']),('逐轮几何选点 144',av['geometric/round6']),('逐轮误差大小选点 144',av['stress/round6']),('逐轮收益选点 144',gv['gain/round6'])])
    body=r'''# v21 纤维轴向变形与局部空间验收

本轮完成了参考解的局部加密与高阶交叉验证、按应力误差构造局部自由度、同预算对照和一致性检查。F45 空间应力明显改善，但仍未达到 2%；参考解的夹持区也仍有未通过的指标。v20 的时间算法与既有归档保留，生产默认未切换。

本轮是固定物理几何、原硬夹持和原材料切线下的线性静态空间研究。盒体为 [0.125,0.875]×[0.375,0.625]²，左右固定体积截止 x=0.25 与 x=0.75，右夹持位移 0.005，μ=10、λ=20、k_f=200。没有改变夹持物理模型、材料参数或稳定化系数。非线性能量、内力和切线另有实际候选检查；没有把线性平衡解称为已完成非线性加载验收。

应力是第一 Piola–Kirchhoff 应力 P 在初始状态的线性化。区域分为全域、夹持区 x≤0.3125 或 x≥0.6875、其余内部；深内部为 0.375<x<0.625。夹持分区包含刚性夹具体积，所有比较使用同一分区，不删除高应力角部。

| 最终独立高阶参考下的空间 | 新增标量函数 | 反力差 | 全域应力差 | 夹持应力差 | 内部应力差 | 全域纤维应变差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
FINAL_TABLE

一个标量局部函数对三个位移分量分别提供自由度，因此 144 个标量函数对应 432 个局部位移自由度；原有自由载体位移自由度为 225。完整 Q2 自由度极限用于判断候选网格自身的精度，不是低成本算法。该极限反力包含原载体稳定化的最小额外反力，材料应力不因该额外反力改变。

表中差异均相对同一个最终 Q4 参考。参考自身尚未完全认证，因此它们是离散解之间的对照差异，不是严格的连续体误差上界。反力仍由同一势能对夹具位移求导，另用物理有限元内力与原稳定化独立复算。应力未作平滑。

[最终分区验收](results/lite-aniso-mainline/v21/final-spatial-acceptance.json) · [候选选择及来源](results/lite-aniso-mainline/v21/final-candidate-selection.json) · [完整 Q2 空间极限](results/lite-aniso-mainline/v21/space-limit.json)

**参考解同时检查网格尺寸与多项式阶次。** 旧 Q2/Q3 应力差给出初始加密指标；每个坐标轴选取应力差积分最大的 1/8 区间，并包含夹持交界或物理边界的相邻区间。张量网格会把一个区间的细化延伸成整片网格，这一代价已明确记录。后续发现仅做局部 Q3 加密时内部变化很小，而 Q3/Q4 内部应力差仍有约 3.2%、纤维应变差约 5.1%，因此加入内部最大物理区间 1/64 的分辨率要求，并继续细化夹持区。

最终网格 Q4 有 FINAL_REF_NODES 个节点，实际自由位移未知量为 FINAL_REF_DOFS。对上一档与最终档分别进行 Q3、Q4 求解，再比较最细同网格的 Q3/Q4，结果如下。

| 参考比较 | 全域应力 | 全域纤维应变 | 夹持应力 | 夹持纤维应变 | 内部应力 | 内部纤维应变 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
REFERENCE_TABLE

最细两档 Q4 反力差为 REFERENCE_REACTION_GAP；反力收敛与局部应力收敛分别判定。完整参考应力及纤维应变自检：REFERENCE_STATUS。初期参考、第一次统一比较和最终参考均保留。初始保守规模估计在下一档触发后，依据实际矩阵自由求解内存评估完成了计划中的第二档；原估计记录与扩展协议都未覆盖。旧归档状态的复算以旧位移作为输入，零迭代结果只说明新算子确认旧残差；独立小规模组装解对照及随后七次高阶新求解共同验证新求解器。

求解器利用张量乘积和可分离预条件器加速，同一材料刚度算子保持不变。没有质量项、对角刚度平移或人为正则化。各次求解记录线性残差、反力和 U=Rd/2 的功恒等式。最终两档线性求解相对容差为 2×10⁻¹²；预条件器用于求解速度，不进入物理势能。

![参考网格和阶次检查](results/lite-aniso-mainline/v21/figures/reference-checks.png)

[初期参考结果](results/lite-aniso-mainline/v21/reference-summary.json) · [最终参考求解](results/lite-aniso-mainline/v21/reference-completed-summary.json) · [最终网格与阶次自检](results/lite-aniso-mainline/v21/reference-self-checks.json)

**纤维轴向应变单独验收。** 令纤维方向 a=(1,1,0)/√2，有

\[
\varepsilon_{aa}=a^T\operatorname{sym}(\nabla u)a
=\tfrac12(\varepsilon_{xx}+\varepsilon_{yy}+2\varepsilon_{xy}).
\]

它同时依赖横纵伸缩和剪切。材料切线为

\[
\delta P=2\mu\,\delta\varepsilon+\lambda\,\mathrm{tr}(\delta\varepsilon)I
+4k_f(a^T\delta\varepsilon a)aa^T.
\]

纯轴向应变对应的轴向系数为 840，纯横向对应系数为 40。局部空间需要协调这些分量；仅让总反力接近并不足够。体积积分误差定义为

\[
\eta_{P,\Omega_r}^2=
\frac{\int_{\Omega_r}\|P-P_{\rm ref}\|_F^2\,dV}
{\int_{\Omega_r}\|P_{\rm ref}\|_F^2\,dV},\qquad
\eta_{a,\Omega_r}^2=
\frac{\int_{\Omega_r}(\varepsilon_{aa}-\varepsilon_{aa}^{\rm ref})^2\,dV}
{\int_{\Omega_r}(\varepsilon_{aa}^{\rm ref})^2\,dV}.
\]

分区边界和两边网格的单元边界都纳入共同积分区间。Q4 比较采用五阶 Gauss 积分，足以精确积分这些分片多项式的应力差平方；JSON 同时保留绝对 RMS、梯度差及各区体积。误差评价代码有独立逐场梯度积分对照测试。

![纤维轴向应变与差异分布](results/lite-aniso-mainline/v21/figures/fiber-strain.png)

**局部函数由原材料残差产生，并保留完整分量表达。** 在一个有限支撑区域 A 中求解

\[
K_{AA}w_A=-r_A.
\]

这里使用原材料刚度与当前物理残差。取局部校正向量的三个分量函数组成标量空间，再让每个位移分量都使用这个空间。这样可以表达斜向拉伸、横向收缩和剪切，也使任意有限刚体转动后的状态仍留在同一空间。参考位移没有被直接复制为候选基函数。

整体位移和势能为

\[
u=Ay+Z\alpha,\qquad
U(y,\alpha)=\int_\Omega\psi(I+\nabla u)\,dV
+\tfrac12\sum_b y_b^TK_s y_b.
\]

残差及切线从这个相同势能求导。线性静态消去局部变量时，得到

\[
K_{\rm eff}=K_{yy}-K_{y\alpha}K_{\alpha\alpha}^{-1}K_{\alpha y}.
\]

所有候选都包含原有能精确表达刚体、仿射和二次弯曲场的空间。新增函数在夹具内部保持零迹，正交化仅更换基底。浮点 SVD 后个别基函数夹持行存在约 10⁻¹⁵ 的舍入残量；实际候选局部夹持位移最大 TRACE_ERROR，满足显式 10⁻¹² 的检查阈值。初次审计对浮点 SVD 要求位级严格零而失败，该日志保留，后续同时核验基函数残量和真实位移；求解模型没有因审计而改变。

**选点要考虑误差是否能被当前校正修正。** 初始应力误差排序在少量自由度时有效；反复选取误差最大的区域，会在局部校正收益已经很小时继续集中在那里。实际对照中，它的能量仍下降，但应力差可能停滞或回升。因此实现了校正收益排序。

定义应力差平方目标

\[
J_\sigma(u)=\int_\Omega\|H\nabla u-P_{\rm ref}\|^2\,dV,
\quad r_\sigma=K_\sigma u-g_{\rm ref},
\quad K_\sigma=\int B^T H^THB\,dV.
\]

对于由原物理残差得到的局部校正 w，预计应力差减少量为

\[
G_A=J_\sigma(u)-J_\sigma(u+w)
=-2r_\sigma^Tw-w^TK_\sigma w.
\]

这个公式在本轮线性静态离散问题上精确成立，并通过独立积分测试。它用于选择局部函数；选定空间后，实际位移仍由原势能 U 的平衡决定。每轮的全局重新平衡可能改变预测收益，所以仍逐轮验收真实应力差，不保证每个预算下收益排序都最优。

每轮最多加入 8 个区域、24 个标量函数，共 6 轮。以下同预算对照均使用未参与选点的中间 Q4 参考，参考在各行相同。

| 方案及新增标量函数数 | 全域应力差 | 全域纤维应变差 |
| --- | ---: | ---: |
ABLATION_TABLE

较大的改善来自更新当前残差所生成的校正方向；收益排序进一步避免了按误差大小反复选点的停滞。在部分预算下，几何选点仍更好。这说明选点指标需要结合实际效果评价，不能凭名称宣称普遍最优。候选空间依据旧 Q3 训练参考选择，中间和最终 Q4 参考都不参与基函数构造或选点。各族最佳阶段由训练误差预先确定，最终 144 函数阶段另作为同预算对照保留。

![同预算局部空间比较](results/lite-aniso-mainline/v21/figures/space-budget.png)

[初始空间对照](results/lite-aniso-mainline/v21/space-validation.json) · [逐轮几何与误差选点](results/lite-aniso-mainline/v21/adaptive-validation.json) · [收益选点独立验证](results/lite-aniso-mainline/v21/gain-validation.json)

**验收覆盖实现一致性和实际大空间。** RELATED_TESTS 项相关测试通过；STATIC_CHECKS 组 ISO/F0/F45/F90 去质量静态候选均保持正刚度，最小未平移特征值为 MIN_STATIC。该数值依赖坐标基的归一化，仅用于检查正定性，不作为跨空间的物理频率比较。三种最终真实大空间另外通过有限刚体转动和平移、势能方向导数、内力方向导数与切线比较；最大切线相对差为 MAX_TANGENT。ISO/F0/F90 的本轮检查主要验证静态正确性，空间精度主验收针对 F45。

这些结果没有证明空间应力达到 2%，也没有证明高阶参考是严格连续体解。新增局部函数尚未进入完整移动动力学；v20 的历史、边界冲量、惯性和时间算法保持原样。本轮没有重跑完整时间循环或 CUDA 验收，也没有修改生产默认。

[静态检查汇总](results/lite-aniso-mainline/v21/static-audits.json) · [真实空间非线性复核](results/lite-aniso-mainline/v21/candidate-nonlinear-audits.json) · [收益空间非线性复核](results/lite-aniso-mainline/v21/candidate-gain-nonlinear-audit.json) · [相关测试](results/lite-aniso-mainline/v21/tests.json)

下一步优先解决夹持过渡的参考收敛及候选网格自身的纤维应变误差。完整 Q2 空间极限仍有明显应力差，说明继续在同一网格内添加局部函数不能消除全部误差。可继续研究局部高阶空间或真正三维局部细化，减少张量网格整片加密的成本；同时按可修正应力差分配局部函数，并在不同加载与材料方向验证其泛化。通过独立空间应力验收后，再为新局部变量定义一致的惯性和历史更新，接入移动循环。

实现入口为 [高阶张量参考](../engine/aniso_phase1/tensor_reference.py)、[局部变分空间](../engine/aniso_phase1/stress_local_space.py)、[可修正应力收益](../engine/aniso_phase1/stress_gain.py)。各实验脚本为 `benchmarks/aniso_v21_*.py`。冻结协议和已归档结果禁止覆盖，重跑求解应使用新的输出目录或独立副本；直接复核测试可执行：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MPM_LITE_DATA_ROOT=/dev/shm .venv/bin/python -m unittest tests.test_aniso_v21_reference tests.test_aniso_v21_local_space tests.test_aniso_v21_metrics tests.test_aniso_v21_gain tests.test_aniso_v21_quartic tests.test_aniso_compatible_carrier tests.test_aniso_v20_variational benchmarks.aniso_local_q3.CubicTests benchmarks.aniso_local_reference.LocalReferenceTests -v
```

[完成清单](results/lite-aniso-mainline/v21/completion-check.json) · [成果 SHA256](results/lite-aniso-mainline/v21/artifact-sha256.json) · [交付源码](results/lite-aniso-mainline/v21/source-delivered.zip)
'''
    values={'REFERENCE_REACTION_GAP':percent(abs(load(OUT/'reference-extension/level0-q4.json')['reaction_N']-co['cases']['level1-q4']['reaction_N'])/abs(co['cases']['level1-q4']['reaction_N'])),'FINAL_TABLE':table,'REFERENCE_TABLE':rt,'ABLATION_TABLE':at,'REFERENCE_STATUS':'通过本轮相邻比较门槛' if ref['all_stress_and_fiber_passed'] else '尚未全部通过 2% 门槛','FINAL_REF_NODES':str(co['cases']['level1-q4']['nodes']),'FINAL_REF_DOFS':str(co['cases']['level1-q4']['free_dofs']),'TRACE_ERROR':f"{max(r.get('local_fixed_grip_displacement',0) for r in static['basis_checks']):.3e}",'RELATED_TESTS':str(tests['tests']),'STATIC_CHECKS':str(static['candidates']),'MIN_STATIC':f"{static['minimum_unshifted_stiffness']:.6g}",'MAX_TANGENT':f"{max(a['tangent_relative_error'] for a in audits):.3e}"}
    for a,b in values.items():body=body.replace(a,b)
    path=ROOT/'docs/ANISO_LITE_FIBER_SPACE_ADAPTIVITY_ZH.md';path.write_text(body)
    intro=f"**最新验证进展（v21）：** 完成纤维轴向应变和分区应力参考加密、Q3/Q4 交叉验证及局部自由度选取。{tests['tests']} 项相关测试、{static['candidates']} 组去质量静态候选和 3 份真实大空间非线性复核通过。144 个收益选取局部标量函数对最终独立参考的全域应力差 {percent(last['regions']['global']['stress_relative'])}、内部应力差 {percent(last['regions']['interior']['stress_relative'])}；空间精度仍未通过，参考也尚未完全认证。v20 时间算法保留，默认未切换。参见 [v21 中文报告](docs/ANISO_LITE_FIBER_SPACE_ADAPTIVITY_ZH.md)。"
    for name in ('README.md','RUNNING_RESTORED.md'):
        p=ROOT/name;lines=p.read_text().splitlines();existing=[i for i,line in enumerate(lines) if line.startswith('**最新验证进展（v21）：**')];assert len(existing)<=1
        if existing:
            i=existing[0];lines.pop(i)
            if i<len(lines) and not lines[i]:lines.pop(i)
        lines.insert(2,intro+'\n');p.write_text('\n'.join(lines)+'\n')
    print(path,flush=True)
if __name__=='__main__':main()
