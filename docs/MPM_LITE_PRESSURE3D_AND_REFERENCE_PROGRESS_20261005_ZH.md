# 三维压力几何与独立固体参考实施进度

2026-10-05 UTC。按[当前计划](MPM_LITE_NEXT_3D_PRESSURE_AND_SOLID_REFERENCE_PLAN_20261005_ZH.md)顺序实施。

## 本轮交付与版本

结果入口：[新运行目录](results/pressure3d/20261005T055744Z-pressure3d/index.html)。新源码位于 `benchmarks/research_pressure3d_next`、`engine/aniso_phase1/research_pressure3d_next` 和 `tests/research_pressure3d_next`。这里没有 Git 元数据，版本以封存发布、源码清单和 SHA-256 为准，不能冒称某个 Git commit。

开工及封存前两次扫描均确认，22个已有发布中最新有效父版本为 `continuous-geometry/20261005T045737Z-continuous-geometry`，发布时间 `2026-10-05T05:38:03.344113+00:00`，SHA-256 为 `a1780d7a4b5df88430f931b3119c3e0a070d27efc464d29da1744bf416d7c9fe`。开工前复现31源码、31快照、660产物及祖先审计。旧源码、旧计划、旧结果未修改。本轮发布身份见新目录内 `release.json`；其自身哈希用文末命令读取，避免文档与发布清单互相引用造成哈希循环。

| 已实施内容 | 验收结果与实际含义 |
|---|---|
| 有界三维压力几何D3 | 构造时共享权重和稀疏映射，按三轴局部块读取，不再为每个单元复制整域权重；完整P回缩保留。支持登记的32/64/128格拓扑，真实动态仅认证32与128格。元数据约152.5 MiB，含构造暂存约181.7 MiB，低于256 MiB上限。 |
| 真实粗细短窗 | 两格从同一物理初态分别推进18步，覆盖0→75微秒、启动步和theta切换。128格在25微秒独立进程重启。位移、速度、骨架/总应力、纤维、反力、压力、含量及六侧交换通过登记的工程预算。 |
| 独立固体参考R6 | 在最新R5上做一次实际局部h加密与非线性平衡；保留原稳定化、边界和材料，检查q7/q9充分性。R5→R6应力差进一步缩小，正式144函数保持。此项是F45/0.005 m纯固体静态参考。 |
| 稀疏转置候选G2 | 读取条目由501,706,024降到127,741,216，算子与两次真实单步等价，但整步中位收益仅0.627%。增加约26秒设置，约1232／2745步才能回收，未达到本轮16步要求，故不采用。 |
| 资源拒绝后的恢复 | 新 `SafePublication` 区分“已恢复但待设备重验”和“可推进”。只允许恢复已认证提交值；资源仍不足时禁止继续，解除后重算与正常第9步一致。新进程只读加载也通过。 |
| 回归与展示 | 6项合同测试通过；只读加载继承的日常252步终态，没有重跑长周期。新增6个物理帧；页面含实际短窗场图、压力截面、区域应力、六侧流量、能量和恢复曲线。日常12帧动画明确标为继承。 |

## 数值结果与范围

| 指标 | 32格 | 128格 |
|---|---:|---:|
| 0→75微秒累计质量缺口 / m³ | −1.490e−18 | 2.470e−19 |
| 累计能量账差 / J | 1.451e−16 | 6.925e−17 |
| 最小detF | 0.999993123 | 0.999993124 |
| 最低压力 / Pa | 0.003822981 | 0.003822635 |
| 最大真实残差／原预算 | 0.0006265 | 0.0012903 |

上述实测小误差不是新增的苛刻验收标准。实际使用登记的量纲预算：场景比较为“绝对下限＋5%参考尺度”，同时守住有限值、非负压力、detF、原始残差、质量/能量及事务完整性。

流体粗细比较最坏项为六侧流量，误差／允许预算为0.010876，即只使用约1.09%的误差预算；**这不是1.09%的相对空间误差**。75微秒最大位移分量约1.035e−7 m（103.5 nm），粗细最大位移差约8.60e−12 m。很多响应小于绝对工程下限，因此结论是“指定两格、指定短窗的工程一致性通过”，并非连续空间真解或严格5%相对精度证书。图中的中心线轴向位移约0.02 nm，是很小的单一分量，不能用它代表全场最大位移或把细小折线直接解释为整体失稳。

R6参考中，正式144函数相对R6的全场PK1误差约0.225%，内部PK1约0.297%，内部纤维应力约0.404%；全场R5→R6相邻PK1差约0.0341%。参考相邻变化已足够小，而正式区域误差未超本轮工程预算，未触发第二级参考或144函数重分配。上述百分比仅属于这个静态工况。

本轮保留以下限制：完整耦合周期、全部方向的连续空间收敛、动态固体真解、耦合q5及生产C/E接入均未认证。历史原始启动微步流量约12.45%的受限标签仍保留；已有h/half工程流量差约占预算34.8%，但不足以改变本轮工程结论，故没有新增细格36步h/2实验，也不声称已将极小空间差中的时间误差完全分离。

## 异常、资源与后续选择

本轮遇到并定位了身份元组/JSON列表不一致、初态predictor为None的比较器处理、三维探针统计维度及绘图变量遮蔽问题，均已修复。前两次性能步的物理提交有效，但后处理失败；它们计入6次S4尝试，保留在 `diagnostics/`，不参与四个公平计时结果。旧静态证据绑定的fixture只发生身份容器规范化，数值核未变；封存审计逐一列出这类历史源码例外，没有放宽当前轨迹的源码要求。

总计47次动态尝试，其中44次提交、2次预设故障、1次身份发布失败；分阶段为S2=39、S4=6、S5=2，低于90上限。另有1次新参考平衡、10组D3静态状态检查和3组G2算子检查。GPU相关完整进程合计约644.6秒，低于1200秒；测得峰值RSS约12.09 GiB，低于16 GiB。新结果及缓存约0.12 GB，位于数据盘。系统盘持续约7.8 GiB，未触发低于5 GiB的迁移条件。其他任务出现时没有终止它们；用于性能结论的四个进程均记录了前后独占情况。

G2没有通过性能选用门槛，故按计划保留D3，未加做其连续4步。局部伴随和P回缩合并计时，混合LU、材料计算及IO尚未分别细拆，不能把包含项和子项相加宣称热点总占比。恢复测试仅覆盖有效CUDA上下文中的受控分配拒绝；实际驱动上下文损坏仍应退出并由新进程恢复。

下一轮建议先按实际应用选择一个能产生可观察横向响应的非对称小载荷工况，确认三维接口在更有辨识力的输入下的表现；若应用重视启动峰值，再独立处理时间误差。性能方向应先细分局部伴随、P回缩与设置成本，优先考虑可复用元数据或缓冲，避免继续追逐只减少条目却几乎不减少整步时间的方案。当前没有证据支持立即重训练144函数或扩展耦合q5。

## 27步结案索引

完整带哈希证据见新结果目录的 `requirement-audit.json`。

| 步骤 | 结果 | 核心证据（相对新结果目录） |
|---|---|---|
| S0.1 | 最新父版本复核通过 | S0/preimplementation-version-audit.json；S6/latest-input-recheck.json |
| S0.2 | 同一物理模型与来源冻结 | S0/physical-contract.json；S0/source-states.json |
| S0.3 | 顺序与资源预算登记 | S0/experiment-budget.json；S0/resource-policy.json |
| S1.1 | 三维所有权/拓扑通过 | S1/partition-check.json |
| S1.2 | 构造与内存通过 | S1/construction-contract.json；S1/memory-preflight.json |
| S1.3 | 非连续读取及完整梯度通过 | S1/adjoint-check.json；S6/test-report.json |
| S1.4 | V/G/H、缓存与压力功通过 | S1/operator-equivalence.json；S1/cache-work-check.json |
| S1.5 | 指定两格进入动态 | S1/dynamic-entry-decision.json |
| S2.1 | 共同零时刻初态与18步协议 | S2/trajectory-protocol.json |
| S2.2 | 粗格对祖先轨迹等价 | S2/coarse-equivalence.json |
| S2.3 | 细格推进、重启、故障重算通过 | S2/transaction-check.json；cases/pressure128-D3/restart-8.json |
| S2.4 | 工程比较通过；h/2未触发 | S2/spatial-comparison.json；S2/time-entry-decision.json |
| S2.5 | 有限范围结论 | S2/spatial-scope-decision.json |
| S3.1 | 最新R5参考来源确认 | S3/solid-reference-audit.json |
| S3.2 | R6实际h加密与同一稳定化 | S3/refinement-protocol.json |
| S3.3 | 一次新平衡及参考比较通过 | S3/solid-reference-comparison.json |
| S3.4 | 保留144函数；训练未触发 | S3/allocation-entry-decision.json |
| S4.1 | 热点记录，细分粒度受限 | S4/hotspot-profile.json |
| S4.2 | 唯一G2候选算子通过 | S4/operator-check.json |
| S4.3 | 等价通过，性能选用未通过 | S4/paired-performance.json；S4/setup-cost.json |
| S4.4 | 保留D3，候选连续步未触发 | S4/backend-decision.json；S4/continuous-check.json |
| S5.1 | 恢复与设备重验分离 | S5/recovery-contract.json |
| S5.2 | 资源拒绝与重算通过 | S5/resource-fault-check.json |
| S5.3 | 新进程只读恢复通过 | S5/new-process-load.json |
| S6.1 | 6合同测试及默认终态加载通过 | S6/test-report.json；S6/load-check.json |
| S6.2 | 本地HTTP、图像、GIF及脚本语法核查 | S6/visual-assets-check.json；S6/visual-review.json |
| S6.3 | 当前源码、历史例外和祖先分别审计封存 | release.json；capability-matrix.json；S6/numerical-source-audit.json |

## 最新CLI

可视化入口已通过本地HTTP资源访问检查；图像实际查看，继承GIF的12帧均解码。未声称执行浏览器交互测试。以下命令启动新页面：

```bash
cd /root/workspace/mpm-lite
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 \
  --directory docs/results/pressure3d/20261005T055744Z-pressure3d
```

IDE转发8765端口后，打开 `http://127.0.0.1:8765/`。研究位移采用物理值，nm仅为显示单位；继承日常动画的位移放大10倍并单独注明来源。检查用临时HTTP服务已关闭，不宣称8765当前正在监听。

封存后完整审计与发布哈希：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_pressure3d_next.publication \
  --run docs/results/pressure3d/20261005T055744Z-pressure3d
sha256sum docs/results/pressure3d/20261005T055744Z-pressure3d/release.json
```

可选只读重验恢复后的压力检查点（需要可用GPU，不推进任何步）：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_pressure3d_next.inspect_pressure \
  --run docs/results/pressure3d/20261005T055744Z-pressure3d \
  --case S5/resource-fault
```

以上结果目录封存后禁止原地续写。下一轮计算应建立新的运行目录和来源身份。本轮没有修改正式日常场景入口，也没有扩展其q5适用范围。

## 分阶段实施记录

S0：最新BASE为a1780d7a4b5df88430f931b3119c3e0a070d27efc464d29da1744bf416d7c9fe；31源码/31快照/660产物及祖先审计通过。结果目录：`docs/results/pressure3d/20261005T055744Z-pressure3d`。系统盘约7.8GiB，迁移未触发；新数据位于数据盘。

S1 base：有界构造及两份真实位移CPU参考、制造路径压力功、缓存所有权通过；元数据152.5MiB，构造暂存峰值181.7MiB，耗时17.9s。

S1 yz：有界构造及两份真实位移CPU参考、制造路径压力功、缓存所有权通过；元数据152.5MiB，构造暂存峰值181.7MiB，耗时13.7s。

S2首步发布因身份shape元组/JSON列表不等而失败；初始持久化状态未改变。根因已定位，仅在fixture身份边界规范化JSON容器，不改数值模型。失败记录与旧源码保留于diagnostics/pre-canonical-identity；作为1次非预设动态尝试计入预算。

S2 base：已提交到第18步/75微秒，本进程18步，minJ=0.999993123。

S2 yz：已提交到第8步/25微秒，本进程8步，minJ=0.999999221。

S2 yz：已提交到第18步/75微秒，本进程10步，minJ=0.999993124。

S2 spatial-comparison：passed_scoped，流体最坏预算比0.01088；质量/能量硬条件分别为[True, True]。

S3：新增真实h参考R6完成，R5/R6参考分辨结果passed_scoped；新平衡1次，原Ks与充分q7/q9通过。正式144函数保持；结果仅针对F45/0.005m静态。

S4首对性能单步均已正确提交，但分区统计误将三维探针权重视为一维，统计阶段失败。已改为收缩全部空间轴并以常量场核查；保留2个有效提交与旧源码，不拿不完整计时作性能证据。四个公平对照仍在S4原8次预算内重做。

S4 G2：两输入整步降幅[0.009084215830991504, 0.003459643237714527]，中位0.63%，设置回收[1231.9787067451139, 2744.6670541674152]步；入选=False。

S5：提交前模拟资源拒绝后，以CPU认证恢复完整已提交状态；资源仍被拒绝时继续推进被阻止，解除后正常重算与原第9步等价。原显存保护未改，不宣称已覆盖真实驱动上下文损坏。

S6准备：6项独立合同测试通过；性能候选G2中位收益不足1%且设置回收超限，保留D3。新参考R6无需第二级加密；空间重分配、细格h/2及候选连续4步均未触发。正在核查继承日常终态、生成六帧来源的场图并封存27步证据。

S6结束：继承日常终态与新进程压力检查点均通过只读加载；六张诊断PNG已查看，本地HTTP资源/GIF/脚本检查通过；27步证据齐备。最终封存同时审计当前源码、明确登记的历史源码例外及全部祖先。
