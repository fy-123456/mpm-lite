# 连续共享几何、热点与空间参考实施记录

日期：2026-10-05（UTC+8）。执行[MPM_LITE_NEXT_CONTINUOUS_GEOMETRY_AND_SPATIAL_REFERENCE_PLAN_20261005_ZH.md](MPM_LITE_NEXT_CONTINUOUS_GEOMETRY_AND_SPATIAL_REFERENCE_PLAN_20261005_ZH.md)。

开工最新BASE：`20261005T035524Z-startup-substeps`，SHA `3a0420805b250a81d05c757e7a6c639a621cff6734da04b41acba96efe13d1e1`；24源码/24快照/875产物及完整祖先通过。结果：`docs/results/continuous-geometry/20261005T045737Z-continuous-geometry`，真实路径`/root/autodl-tmp/mpm-lite-continuous-geometry/20261005T045737Z-continuous-geometry`。系统盘约7.9GiB，不触发迁移。

S0已冻结来源、物理参数、原全局步号/时间及预算。后续顺序执行，当前没有新增动态尝试。

S3：四种横向CPU网格已完成固定骨架参考，拓扑、守恒限制及RT0延拓检查通过；保留全部方向差异，不据此宣称三维空间收敛。

S3真实位移冻结算子：passed_scoped，完成8/8组静态积分，耗时438.1秒；实际动态空间精度仍未认证。

执行顺序调整：S1第10步因外部GPU进程抢占显存触发原保护；第9步持久化未受影响。一次重试在构造期再次受限，未新增动态调用。内存异常还阻止了进程内恢复时的几何校验，因此仅认定持久化提交安全，不宣称这次异常完成了进程内恢复。先顺序完成S3 CPU参考（440.15秒，8组真实冻结积分均通过），GPU空闲后从认证的第9步恢复。没有终止其他任务或放宽显存保护。

S1 连续后端B：14个真实步、25微秒独立进程恢复、50微秒失败回滚后重算，比较结果 `passed_scoped`；质量缺口1.33e-18m³，增量能量收支1.12e-16J。

S2 G1预评估：registered；即使把下载等待和CPU回缩全部去掉，整步乐观收益上界约6.63%。eligible for exactly one candidate

S2 G1实测两输入降幅[0.007518343804639449, 0.02411498380539079]，中位1.58%，极差1.66%；入选性能阶段=False。

S4：同一200微秒初态追加8/16步至300微秒，结果 `passed_scoped`；流体最坏预算比0.003278。新段Dnum为零，早期累计Dnum保留；不宣称0到300微秒全程时间/空间收敛。

S5/S6：参考与范围已分别记录；正式144局部函数、q7材料/M7质量保持原模型。默认252步日常场景只读加载，未重跑长周期。原始微步精度与实际空间收敛继续标为未认证。

## 本轮完成结论与边界

当前交付目录：[20261005T045737Z-continuous-geometry](results/continuous-geometry/20261005T045737Z-continuous-geometry/index.html)。发布身份及24步证据分别见[release.json](results/continuous-geometry/20261005T045737Z-continuous-geometry/release.json)和[requirement-audit.json](results/continuous-geometry/20261005T045737Z-continuous-geometry/requirement-audit.json)；开工直接父版本是上文 `3a042080…`，没有使用旧boundary-reference作为直接起点。当前工作目录没有Git元数据，本地发布内容哈希是版本依据。

| 工作 | 实际完成与验收 | 范围 |
|---|---|---|
| S1 连续共享几何B | 12.5→75微秒14步；逐状态、区域应力/纤维、六侧交换、能量、25微秒恢复与50微秒故障重算通过 | 同一新32单元网格，不宣称新连续加速率 |
| S2 四单元批处理G1 | 完整P回缩、尾批/零支撑/副本测试、静止/制造/真实状态与正反路径算子通过；4个真实单步等价通过 | 0.75%/2.41%降幅，中位1.58%<5%；输入0设置成本约29.06步才回收>16；候选保留为实验，未推荐，未执行G1连续分支 |
| S3 横向参考 | 32×1×1、32×2×1、32×1×2、32×2×2的4次固定骨架代数/谱参考；25/200微秒×两网格×7/9阶共8组真实冻结V/G/H通过 | 守恒限制和RT0延拓已核查；仅冻结位移与可比RT0子空间，实际三维动态空间收敛仍未认证 |
| S4 延长与历史 | 从同一BASE第28步200微秒初态追加8/16步到300微秒；压力、位移、速度、全场及区域应力、反力、六侧流量与收支通过；250微秒故障恢复及300微秒独立进程加载通过 | 新段θ=1/2，原早期Dnum保留，残差归一窗口仍为200微秒；不等同0→300微秒全程N/2N或完整周期 |
| S6 回归与展示 | 6个独立新增测试（先5项合同、后1项批处理），各执行一次；日常q5的252步终态只读验证；5个新物理帧，继承12帧日常动画 | HTTP 11项资源、GIF12帧解码、JS语法通过；查看研究诊断和300微秒实际场图；没有进行浏览器交互自动测试 |

实际新增动力学尝试 **47** 次：S1=17、S2=4、S4=26。其中 **44次提交、2次预设故障、1次外部显存竞争引起的动态失败**；另有1个仅构造失败的进程，不能误计成动态步，也未漏计其耗时。正常新轨迹为14＋4个单步对照＋24＝42步，另2步是故障后正常重算。GPU相关完整进程合计约399.44秒，CPU静态参考完整进程440.15秒；峰值RSS约5.56GiB。封存前结果和缓存约144.1MiB，低于3GiB上限。系统盘余量约7.81GiB，未触发迁移；新数据位于数据盘，工作区保持可加载链接。

延长窗口质量缺口最坏约5.38e-18m³、增量总能量收支最坏约6.45e-17J、最低detF约0.9999105、最低压力约0.002908Pa；这些是测得值，不升级为今后的强制容差。流体储能下降主要转为固体动能/弹性能，不能把总能量近乎不变误认为压力没有释放；图中已分开展示各项。

已知边界：原始首微步流量约12.45%的误差沿用上轮受限结论；横向冻结参考通过不证明144个局部函数的应力精度。正式空间、完整质量M7、材料q7、源项及边界不变，耦合q5与生产C/E均未新增资格。显存不足时，旧恢复路径的几何重验也可能被保护拒绝；本轮证明的是持久化提交可认证恢复，尚未给“显存饥饿下进程内恢复”新增保证。

下一步优先建立有界、可处理非连续张量切片的三维压力几何，再跑很短的两网格真实耦合对照；参考可信后才考虑重新安排144个局部函数。性能上需针对实际伴随计算/内存流量，而非继续扩大只省下载的批处理。显存异常的恢复可另立小范围事务改进，不修改本轮已封存祖先。

## 最新可视化CLI

```bash
cd /root/workspace/mpm-lite
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 --directory docs/results/continuous-geometry/20261005T045737Z-continuous-geometry
```

在IDE中转发8765端口，再打开 `http://127.0.0.1:8765/`。页面同时提供本轮真实结果与**明确标记为继承**的日常纯固体动画；日常动画位移放大10倍，研究300微秒物理场按原位移倍率1展示。

封存后的只读完整审计：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m benchmarks.research_continuous_geometry_next.publication --run docs/results/continuous-geometry/20261005T045737Z-continuous-geometry
```

本轮代码仅新增于 `benchmarks/research_continuous_geometry_next`、`engine/aniso_phase1/research_continuous_geometry_next`和`tests/research_continuous_geometry_next`；旧发布源码、快照、文档和轨迹保持原封存身份。
