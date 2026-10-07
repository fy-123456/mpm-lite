"""Render the delivered Chinese v12 report from verified numerical artifacts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/results/lite-aniso-mainline/v12'
def read(name):return json.loads((OUT/name).read_text())
def pct(v):return f'{100*v:.3f}%'

def main():
    result=read('summary.json');assert result['completed'] and result['physical_checks_passed']
    factorial=read('factorial-attribution.json')['modes'];b=factorial['baseline']['pairs'][-1];n=read('F45-comparison.json')['pairs'][-1]
    metrics=[('全程应力相邻差','P_relative'),('末态应力相邻差','P_terminal_relative'),('反力曲线相邻差','reaction_relative'),('全程变形 F−I 相邻差','F_relative'),('末态总弹性能相邻差','energy_relative')]
    table='| 指标 | v11 | v12 | 降幅 |\n|---|---:|---:|---:|\n'+'\n'.join(f'| {label} | {pct(b[k])} | {pct(n[k])} | {pct(1-n[k]/b[k])} |' for label,k in metrics)
    ablation='| 配置 | 粒子速度 βv | 仿射速度 βC | 全程应力差 | 末态应力差 | 反力差 |\n|---|---|---|---:|---:|---:|\n'
    for key,label,bv,bc in [('baseline','v11','β(dt)','β(dt)'),('velocity_only','仅取消速度平滑','1','β(dt)'),('affine_only','v12：仅取消仿射平滑','β(dt)','1'),('both','同时取消两项','1','1')]:
        r=factorial[key]['pairs'][-1];ablation+=f'| {label} | {bv} | {bc} | {pct(r["P_relative"])} | {pct(r["P_terminal_relative"])} | {pct(r["reaction_relative"])} |\n'
    trends='| 相邻 dt（s） | v11 全程应力差 | v12 全程应力差 | v12 末态应力差 | v12 反力差 |\n|---|---:|---:|---:|---:|\n'
    for pair,a,c in zip(('0.001 / 0.0005','0.0005 / 0.00025','0.00025 / 0.000125'),factorial['baseline']['pairs'],factorial['affine_only']['pairs']):trends+=f'| {pair} | {pct(a["P_relative"])} | {pct(c["P_relative"])} | {pct(c["P_terminal_relative"])} | {pct(c["reaction_relative"])} |\n'
    dirs='| 方向 | 全程 F 相邻差 | 全程应力相邻差 | 末态应力相邻差 | 反力相邻差 | 最细 2% 门槛 | 观测阶筛查 |\n|---|---:|---:|---:|---:|---|---|\n'
    for label in ('ISO','F0','F45','F90'):
        r=read(label+'-comparison.json');p=r['pairs'][-1];dirs+=f'| {label} | {pct(p["F_relative"])} | {pct(p["P_relative"])} | {pct(p["P_terminal_relative"])} | {pct(p["reaction_relative"])} | {"通过" if r["time_2pct_passed"] else "未通过"} | {"通过" if r["orders_passed"] else "未通过"} |\n'
    regional='| 分区 | v11 全程应力差 | v12 全程应力差 |\n|---|---:|---:|\n';rr=read('regional-time.json')['records']
    for key,title in [('whole','全域'),('near_grip','夹持过渡附近'),('interior','内部'),('deep_interior','更内部')]:
        a=next(r for r in rr if r['version']=='v11' and r['region']==key);c=next(r for r in rr if r['version']=='v12' and r['region']==key);regional+=f'| {title} | {pct(a["P_space_time_relative"])} | {pct(c["P_space_time_relative"])} |\n'
    a=read('artifact-check.json');e=read('public-api-equivalence.json');v=read('F45-comparison.json');old=factorial['baseline']
    source=ROOT/'docs/results/lite-aniso-mainline/v12-transfer-controls/same-input-attribution.json';same=json.loads(source.read_text());last=next(r for r in same['records'] if r['case']=='F45-fourth' and r['time']==.5)
    values={
      '@@MAIN_TABLE@@':table,'@@ABLATION@@':ablation,'@@TRENDS@@':trends,'@@DIRS@@':dirs,'@@REGIONS@@':regional,
      '@@ABS_OLD@@':f'{b["P_absolute"]:.8f}','@@ABS_NEW@@':f'{n["P_absolute"]:.8f}','@@ABS_DROP@@':pct(1-n['P_absolute']/b['P_absolute']),
      '@@P_ORDERS@@':', '.join(f'{r["P_order"]:.3f}' for r in v['pairs'][1:]),'@@R_ORDERS@@':', '.join(f'{r["reaction_order"]:.3f}' for r in v['pairs'][1:]),
      '@@REBUILD_OLD@@':pct(old['energy'][-1]['rebuild_over_elastic']),'@@REBUILD_NEW@@':pct(v['energy'][-1]['rebuild_over_elastic']),
      '@@C_EFFECT@@':f'{last["next_material_gradient_from_C"]:.6e}','@@V_EFFECT@@':f'{last["next_material_gradient_from_v"]:.6e}','@@EFFECT_RATIO@@':f'{last["next_material_gradient_from_C"]/last["next_material_gradient_from_v"]:.2f}',
      '@@HISTORY@@':f'{result["history_closure_max"]:.3e}','@@STATE@@':f'{a["max_state_error"]:.3e}','@@FORCE@@':f'{a["max_force_error_N"]:.3e}','@@EQUIV@@':f'{e["max_difference"]:.3e}',
    }
    report=r'''# v12：保留仿射速度历史，降低 F45 应力时间步敏感性

2026-09-28。CPU float64，Python 3.11.16，Warp 1.10.1。v12 是实验验收版本，包版本仍为 0.1.0。

本轮目标完成：保住 v11 静态改善，F45 最细相邻时间步的全程应力差降低约 54%，末态应力差降低约 61%，并通过独立单因素对照定位到 **APIC 仿射速度历史的返回环节**。应力差仍超过 2%，时间观测阶和空间精度尚未全部通过；默认配置保持原样。补充停载检查显示仿射动能和应力漂移增加，因此只作为可选研究候选保留。

## 1. 结果与比较口径

同一物理几何、材料、夹持、网格 9、每单元每轴 2 个粒子采样；192 粒子。ISO/F0/F45/F90 各运行 0.5 s，位移从 0 平滑增至 0.005 m。四档 dt 为 0.001、0.0005、0.00025、0.000125 s；每条均完整运行，没有用缩短加载、隐藏子步或外推代替细时间步。

下表比较最细两档 0.00025 / 0.000125 s。

@@MAIN_TABLE@@

全程绝对应力差 RMS 也从 **@@ABS_OLD@@ Pa 降至 @@ABS_NEW@@ Pa**，下降 @@ABS_DROP@@；改善并非只来自相对误差分母变化。这里应力是第一 Piola 应力 P，不是 Cauchy 应力。

采用与 v11 相同的体积、时间范数：

\[
\|Q\|_{V,t}=\left[\frac{1}{(T-t_0)\sum_p V_p}
\int_{t_0}^{T}\sum_p V_p\|Q_p(t)\|_F^2\,dt\right]^{1/2},\quad
 e_P(\Delta t)=\frac{\|P_{\Delta t}-P_{\Delta t/2}\|_{V,t}}{\|P_{\Delta t/2}\|_{V,t}}.
\]

取 t0=0.05 s、T=0.5 s，在 19 个共同保存时刻梯形积分。末态指标只取 0.5 s。F 比较使用 F−I，避免单位阵掩盖变形误差。反力在共同的粗档时间点积分；弹性能包含稳定化能量。相邻差衡量时间敏感性，不等于相对真解的误差。

@@TRENDS@@

![F45 时间步与归因对照](results/lite-aniso-mainline/v12/F45-time-attribution.png)

[可导出 PDF](results/lite-aniso-mainline/v12/F45-time-attribution.pdf)。图 D 的重建能量仍非零，不能把它解释成已经消除了全部历史误差。

## 2. 改了哪里，为什么会影响应力

把 C 理解成“粒子记住的局部速度如何变化”，把 F 理解成“材料已经怎样变形”。它们承担不同任务：C 影响下一步动量传递，材料 F 的更新始终使用物理梯度 L。

令 A=SH 为网格速度到粒子速度的两层插值，B=SD 为材料梯度采样；raw 表示夹持投影之前，+ 表示本步求解后的网格速度。原增量返回可写成：

\[
\begin{aligned}
v_p^{n+1}&=v_p^n+A(v_g^+-v_g^{raw})
 -(1-\beta_v)(v_p^n-A v_g^{raw}),\\
C_p^{n+1}&=C_p^n+(L_p^+-L_p^{raw})
 -(1-\beta_C)(C_p^n-L_p^{raw}),\\
F_p^{n+1}&=(I+\Delta t L_p^+)F_p^n,\qquad L_p^+=Bv_g^+.
\end{aligned}
\]

v11 令两个 β 相等，均为 β(dt)=0.9^(dt/0.001)。即使已按物理时间标定，这仍会持续平滑 C 中未被当前网格梯度表示的部分。

**v12 保留 βv=β(dt)，令 βC=1**，所以：

\[
C_p^{n+1}=C_p^n+(L_p^+-L_p^{raw}).
\]

这样保留局部仿射速度历史，并继续传回真实网格冲量。没有网格冲量时，C 完全保持；粒子速度仍保留原有平滑。C 随后进入粒子→中心→网格的动量传递，影响下一步 L，最终影响 F 和应力。

这属于取消一项数值平滑，会改变算法的耗散行为；不是对同一物理黏性参数的更精确积分。本轮没有加入材料黏性，也没有通过匹配反力选稳定化系数。

实现仅新增 `affine_flip_ratio`：默认 None，继续跟随原 `flip_ratio`。正式候选显式取 1；`overwrite` 模式不接受该参数，非法数值在提交状态之前被拒绝。相关代码：

- [独立 C 返回](../engine/aniso_phase1/affine_transfer.py)
- [参数与失败前校验](../engine/aniso_phase1/solver.py)
- [Config 与命令行入口](../demos/aniso.py)
- [新增 5 项回归测试](../tests/test_aniso_split_affine.py)

## 3. 改善来源：四组完整消融，而非只看相关性

固定同一输入初态和全部其他规则，对速度、仿射速度平滑分别开关。四组每组都完成四档完整加载；基线复用已封存 v11。

@@ABLATION@@

取消仿射平滑能获得主要改善；同时取消两项并没有进一步降低全程误差。仅取消粒子速度平滑，最细全程差仅小幅减少，末态差反而增加；前三档的全程差也较差。因此保留速度平滑，单独保留 C 历史。

另外完成 16 份 v11 快照的同输入诊断：

1. 固定求解前状态和已求得的网格速度，独立计算取消两项平滑分别增加的 dv、dC。
2. 当前步的 x、F、L、材料能量、稳定化能量、残差与切线均保持不变；真实内核单步测试也验证 x/F/v/L/网格速度/势能完全相同。
3. 在相同的下一步粒子位置上，将 dv 或 dC 分别传回网格，并置零夹持处的速度扰动，测量其材料梯度影响。

第四档末态，仿射项造成的下一步原始梯度扰动 RMS 为 @@C_EFFECT@@ s⁻¹，速度项为 @@V_EFFECT@@ s⁻¹，前者约 @@EFFECT_RATIO@@ 倍。该诊断尚未包含下一步平衡求解，不能把这个倍数直接当作应力误差贡献比例；完整四组加载才提供轨迹层面的干预证据。

稳定化参考重建规则在四组中完全相同。累积重建能量净变化/末态弹性能从 v11 的 @@REBUILD_OLD@@ 变为候选的 @@REBUILD_NEW@@。这发生在状态轨迹改变之后；没有据此声称“修改重建算法造成了改善”。重建误差仍是后续独立研究对象。

证据：[四组完整归因](results/lite-aniso-mainline/v12/factorial-attribution.json)、[同输入诊断](results/lite-aniso-mainline/v12-transfer-controls/same-input-attribution.json)、[新接口与先行消融结果一致性](results/lite-aniso-mainline/v12/public-api-equivalence.json)。正式接口与先行对照四档全程匹配，最大记录差 @@EQUIV@@。

## 4. 静态改善如何保住

v11 的材料势能、残差、切线、粒子 F 历史、参考坐标重建、二次多项式保持的稳定化全部保持原样。稳定化仍为：

\[
E_s=\sum_c\frac{\mu V_c}{2h^2\cdot64}\|P_c y_c\|_F^2,
\qquad P_c=I-Q_cQ_c^T.
\]

μ=10 Pa、系数 1；正常二次变形不受该能量惩罚。βC 只在网格平衡求解之后进入 C 返回，不进入静态势能、力、切线或参考重建公式。

本轮通过源码字节指纹验证相关能量与静态装配代码和 v11 一致，继承已封存的 24 组拉伸、16 组梁静态研究；**没有将它们计为本轮重新运行的 40 个静态算例**。本轮 95 项回归重新执行实际 Warp 去质量算子与静态矩阵的四方向比较，以及刚体转动、仿射/二次场、零模态、能量导数和历史闭合检查。

所以 v11 已获得的 F45 grid33 反力结果保持：相对同一旧参考偏差 −2.192%，相对局部 Q3 参考 −1.548%；局部 Q3 下内部应力空间差仍约 13.37%。这些是静态空间结果，不能与本轮 grid9 的时间差混为一谈。

[静态保持证明](results/lite-aniso-mainline/v12/static-preservation.json)、[v11 局部 Q3 比较](results/lite-aniso-mainline/v11/local-q3-reference-comparison.json)。

## 5. 验收范围与未通过的项目

- 完整回归 95/95，通过且无跳过；其中新增 5 项。
- 正式四方向、四档 16 条完整轨迹，共 30,000 步；另有 12 条归因对照，共 22,500 步。合计新增 52,500 步。
- 正式 64 份快照独立复算 x/v/C/L/F、中心历史、参考坐标、SVD 投影器、能量及夹持反力；另复算 48 份控制快照的传递公式。
- 正式最大冻结历史闭合误差 @@HISTORY@@，最大独立状态复算误差 @@STATE@@，最大夹持反力复算误差 @@FORCE@@ N；粒子试算/提交公式误差为零。
- 动量、正 Jacobian、夹持速度、收敛与失败回滚等物理/实现检查通过；正式四档结果与先行单因素对照匹配。
- CPU float64 已验证，CUDA 本轮未验证。所有旧归档保留并做 SHA256 复核。

另外增加 2 条最细档停载对照，各 400 步，共 800 步，并各做一次原末步重放以验证重启。停载诊断不计入上述 52,500 步四档加载，也不用于修改已冻结的时间误差门槛。

@@DIRS@@

F0 的最细反力差为 0.0138%、全程应力差为 0.0222%，数值虽小，但最后一次反力观测阶约 −0.299、应力观测阶约 0.073；本轮未通过观测阶筛查，不能笼统声称其他方向都已建立时间收敛。

F45 应力绝对相邻差的两个观测阶为 @@P_ORDERS@@；反力观测阶为 @@R_ORDERS@@。应力趋势比 v11 改善，但仍未同时满足最细 2% 和两次观测阶均至少 0.5 的要求。即使反力差已较小，也不能替代应力验收。

@@REGIONS@@

分区以原始材料坐标定义：夹持过渡距离 x=.25/.75 小于 .0625；内部 x∈(.3125,.6875)；更内部 x∈(.375,.625)。这些是固定采样上的时间敏感性，不是高阶参考下的空间误差。

本轮冻结目标“全程和末态应力差均至少降低 50%”通过。**整体精度验收仍未通过**：F45 时间应力差尚大，细网格内部应力空间差仍存在，夹持区高阶参考应力也未完全收敛。下一步优先处理第 6 节暴露的停载退化，再用同输入对照区分隐式时间积分、剩余速度传递与参考重建的误差；候选仍必须保持能量、力和切线一致。

## 6. 发现的代价：仿射动能和停载漂移

取消 C 平滑保留了更多局部速度信息，也保留了传递难以消除的部分。F45 最细档末态：

| 指标 | v11 | v12 |
|---|---:|---:|
| 仿射动能 | 4.714e−8 J | 3.907e−6 J |
| 总粒子动能/弹性能 | 0.0917% | 7.617% |
| 原始网格动能 | 7.135e−9 J | 3.762e−8 J |

粒子仿射动能的增长远大于网格动能增长，不能把保留下来的全部 C 都视为已经解析的物理速度。为此，在完全相同的各自末态后将夹具速度固定为零，补跑 0.05 s（每组 400 步）：

| 停载诊断 | v11 | v12 |
|---|---:|---:|
| 仿射动能变化 | −38.43% | +4.46% |
| 相对停载初始状态的应力变化 | 6.59% | 17.53% |

停载应力变化按 $\|P(0.55)-P(0.5)\|_V/\|P(0.55)\|_V$ 计算，分母取停载结束值。应力变化不是时间步误差，也不是对参考解的误差；它说明候选在停止加载后仍有明显演化。两组仍通过力平衡、正 Jacobian 和零夹持速度检查，末步重放最大差小于 4.36e−14；这不能替代长时间稳定性验证。[停载结果](results/lite-aniso-mainline/v12/hold/summary.json)。

因此本轮只确认原加载过程的应力时间敏感性显著减小，**没有确认停载表现改善，也不将候选切成默认**。下一步优先区分 C 中能传给网格的有效部分和传递未解析的剩余部分，研究只控制后者的能量/耗散机制；继续保持本轮静态势能和历史一致性，复核四档时间差，并加入停载及卸载检查。不能把减少时间步差异等同于全面提高物理精度。

## 7. 运行与复现

已实测的 20 步入口：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso --device cpu --scene tensile --grid 9 --dt 0.001 --fiber-angle 45 --smooth-loading --history-consistency residual_center --stabilization selective_patch --boundary-impulse-transfer --apic-transfer incremental --affine-flip-ratio 1 --reaction-force-atol 1e-7 --headless --steps 20
```

完整协议复现到新目录（不要覆盖封存目录）：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.aniso_split_history freeze --output /tmp/mpm-v12-repeat
.venv/bin/python -m benchmarks.aniso_split_history tests --output /tmp/mpm-v12-repeat
.venv/bin/python -m benchmarks.aniso_split_history static --output /tmp/mpm-v12-repeat
.venv/bin/python -m benchmarks.aniso_split_history run --output /tmp/mpm-v12-repeat --jobs 8
.venv/bin/python -m benchmarks.aniso_split_history analyze --output /tmp/mpm-v12-repeat
```

每档速度 βv 在冻结协议中按 dt 计算，βC 恒为 1。完整四档须使用协议，不能将 CLI 示例中的 βv=.9 原封不动用于所有 dt。

[协议](results/lite-aniso-mainline/v12/protocol.json)、[回归结果](results/lite-aniso-mainline/v12/tests.json)、[正式汇总](results/lite-aniso-mainline/v12/summary.json)、[独立复核](results/lite-aniso-mainline/v12/artifact-check.json)、[分区时间差](results/lite-aniso-mainline/v12/regional-time.json)。

先行消融使用修改前的 v11 接口，仅在独立进程的 C 返回时替换 β；两份 `source-frozen.zip` 和执行脚本随结果保存。它们的冻结源码应从对应归档读取，不能用新源码硬通过旧指纹。正式 v12 源码和文档另存 `v12/source-delivered.zip`，三份新归档的 `artifact-sha256.json` 与 `v12/completion-check.json` 记录最终校验。
'''
    for k,value in values.items():report=report.replace(k,value)
    assert '@@' not in report
    (ROOT/'docs/ANISO_LITE_AFFINE_HISTORY_TIME_ZH.md').write_text(report)
    p=ROOT/'README.md';text=p.read_text();intro='> **v12 仿射历史与时间应力改善（2026-09-28）：** [实现、公式与完整归因](docs/ANISO_LITE_AFFINE_HISTORY_TIME_ZH.md)。保留 v11 静态能量与刚度，新增可选 `--affine-flip-ratio 1`。95 项回归通过；正式四方向四档 30,000 步及归因对照 22,500 步完成。F45 最细全程应力差由 6.77% 降至 '+pct(n['P_relative'])+'，末态应力差由 6.56% 降至 '+pct(n['P_terminal_relative'])+'；主要改善来自保留 APIC 仿射速度历史。**应力尚未达到 2%，短停载漂移增大；整体精度未通过，默认未切换。**\n\n'
    assert 'v12 仿射历史' not in text;p.write_text(text.replace('\n\n','\n\n'+intro,1))
    p=ROOT/'RUNNING_RESTORED.md';text=p.read_text();intro='2026-09-28 最新实验为 [v12 仿射历史与时间应力改善](docs/ANISO_LITE_AFFINE_HISTORY_TIME_ZH.md)。保持 v11 静态改善，新增独立 `--affine-flip-ratio 1`，用于 `--apic-transfer incremental`。95 项回归和四方向四档完整加载通过实现/物理检查；F45 应力时间敏感性显著减小，但仍未达到 2%，短停载漂移增大，默认未切换。\n\n```bash\ncd /root/workspace/mpm-lite\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso --device cpu --scene tensile --grid 9 --dt 0.001 --fiber-angle 45 --smooth-loading --history-consistency residual_center --stabilization selective_patch --boundary-impulse-transfer --apic-transfer incremental --affine-flip-ratio 1 --reaction-force-atol 1e-7 --headless --steps 20\n```\n\n该命令已实际运行。完整四档配置、归因对照和剩余误差见报告。以下为历史记录。\n\n'
    assert 'v12 仿射历史' not in text;p.write_text(text.replace('\n\n','\n\n'+intro,1))

if __name__=='__main__':main()
