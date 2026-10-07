# MPM-lite 时间相位、应力空间与暖运行：实施成果

已按 [本轮 30 步计划](MPM_LITE_NEXT_PHASE_STRESS_OPTIMIZATION_PLAN_20261001_ZH.md) 顺序执行。代码使用 FP64 与 Warp CUDA；没有并行实验，没有额外长周期扫描。

最新交付目录为 [phase-stress/20261001T072129Z-phase-stress](results/phase-stress/20261001T072129Z-phase-stress/release.json)。直接父版是 cost-phase/20261001T062026Z-cost-phase，父发布 SHA-256 为 `46a3dcd54a30508143185aac06a791e68fb65a2e62eba9245ad30872631fce48`。较晚的 070059 目录只有继承 smoke，不是有效发布。工作区没有可用 Git 元数据，因此版本由发布索引、完整源码、状态、配置与成果哈希界定，未虚构提交号。

原计划快照为本轮 `plan-frozen.md`；旧正式源码、成果及封存文档保持原哈希。逐步结果见 [requirement-audit.json](results/phase-stress/20261001T072129Z-phase-stress/requirement-audit.json)，当前范围见 [能力矩阵](results/phase-stress/20261001T072129Z-phase-stress/capability-matrix.json)。

| 阶段 | 实际工作与决定 | 验收证据 |
| --- | --- | --- |
| S0 | 完整来源核查；独立命名空间；四步继承并在两步后新进程恢复，状态差为 0 | S0/version-audit.json、compatibility.json |
| S1 | 实际模态贡献、受控线性相位、两个半步子窗；唯一 204 步候选未满足联合门槛，保留 dt=0.0125 s | S1/observable-modal-contribution.json、reference-resolution.json、window-comparison.json、time-decision.json |
| S2 | 重载 R4；独立坐标质量投影诊断；实际应力 Gram 六模式候选及真实非线性平衡；保留 swap6 | S2/reference-reload.json、span-diagnostic.json、candidates/stress-response6/operator-audit.json、space-decision.json |
| S3 | 采用减少中间计时屏障的暖路径；保留分配前和响应结束的内存检查，材料与可写缓冲仍私有 | S3/warm-profile.json、operator-equivalence.json、transaction-check.json、performance-decision.json |
| S4 | 小储存独立矩阵指数对照；显式时间数组真正传入求解、历史和恢复；四步三维固体耦合及完整事务 | S4/explicit-grid-check.json、coupled-checks.json、rollback-restart.json、publication-check.json |
| S5 | 最终数值源码冻结；唯一充分周期；主 q5 资格、故障分类／回退／恢复；两个新增敏感卸载窗 | S5/final-model-lock.json、qualification-final.json、fault-and-restart.json、sensitive-scope.json |
| S6 | 21 项回归、唯一日常周期、逐积分步对照、12 帧可视化、能力矩阵及发布封存 | S6/final-tests-result.json、final-scene.json、physical-review.json、cost-comparison.json |

**本轮真正采用的改动。**

1. 暖运行不再为每个诊断计时分区强制设备同步；依赖数据下载保持必要同步，分配预算仍提前检查，响应结束继续检查资源。新计数表示主机经过时间，不能再当成逐 GPU 内核耗时。没有新增切线缓存，没有修改数值核、质量、原 Ks、求解容差或阻尼。
2. 压力驱动拥有不可变显式时间数组及源项，并将其绑定到检查点身份。实际步长取相邻时间差；错误步长、不同时间表／源项和错误恢复索引会拒绝。全部固体／流体状态及累计交换量共同回滚；指针发布后异常只接受一次。
3. 材料资格使用最终真实轨迹，并补充 S1 观察到的应力敏感模态 501。0.0075 m 幅值新增 0.6—0.65 s 与独立 0.7—0.75 s 两个四步卸载窗，未将短窗资格扩成完整周期许可。

新增主入口为 `benchmarks.research_phase_stress_next.run`；核心实现见 `engine/aniso_phase1/research_phase_stress_next/warm.py` 和 `coupled.py`。原有代码路径保持封存。

**实测收益与稳定性。**

- 两个非零状态，各两组交替配对，每个变体独立进程：两步暖任务中位耗时 0.35047→0.29815 s，下降 **14.9%**；构建中位 9.897→9.922 s；峰值 RSS 比值约 1.00058。能量、力、切线、状态和场输出未发现差异。
- 日常 q5 完整周期：128 步、1.6 s、12 帧，构建 12.820 s，推进＋场＋发布 18.037 s，总计 **30.931 s**，RSS 3.424 GiB。父版为 32.892 s，总耗时下降约 **6.0%**；推进与发布部分下降约 **20.2%**。这是一对实际完整运行记录，不是重复统计的全周期基准。
- 本次单次构建比父版慢，而配对构建中位基本相同；保留这项波动，不隐藏或重新择优取样。日志显示 GPU 模块均缓存命中，未出现新增 JIT；其余主机／IO 初始化波动未独立定位，不宣称启动收益。
- 最终 q7 周期：128 步，构建 6.761 s，推进与发布 25.788 s，总计 32.625 s。两条完整周期仅各运行一次。完整周期没有单独拆开显示和 IO；两步配对任务单独测得推进／IO／场中位约 0.24765／0.03975／0.01075 s，不外推为完整周期分账。
- q5 周期对父版的 q、速度和预测器差为 **0**。对同空间同时间表 q7，最大分区差：位移 1.79e-9 m、速度 9.25e-8 m/s、PK1 1.03e-5 Pa、纤维 PK1 1.02e-5 Pa、区间反力 4.24e-8 N；全部满足工程预算。
- q5 最小材料采样 det(F)=0.993563，q7 为 0.991589；采样位置随积分规则不同，不能把这项差解释成几何改善。日常材料哨兵 3/3 通过，无自然回退；人为注入故障的回退另行通过。自由阶段总能量约 3.162e-7 J，没有明显增长。
- 小储存 S=0.0002 时，旧 h=0.01 s 的中点更新矩阵存在负元素。固定固体诊断采用 h≤2 min(Cii/Lii) 的 0.2 倍：两单元 h=3.75e-5 s，四单元 h=9.375e-6 s。四步与矩阵指数最大压力差约 2.62e-5／2.58e-5 Pa。
- 实际小储存两单元耦合只运行至 0.00015 s；在第 2 步退出后由新进程续至第 4 步，与连续轨迹的全部状态／历史差为 0。最小 det(F)≈0.999999，累计含量闭合约 8.76e-15 m³，压力交换功及耗散通过。正常储存封闭／排水仍各四步、dt=0.01 s。压力不做裁剪。

**保留的问题与停止条件。**

时间候选的主要频段速度误差在两个窗分别改善约 72.4%、44.5%，但分区场与事件时刻未同时通过，故未采用。代表模态 4 不足以描述实际输出；周期约 0.03243 s 的模态 72 有更大的诊断能量贡献，模态 501 的应力响应也较大。线性模态能量仅用于解释，不冒充非线性总能量。`temporal_accuracy=false`。

R4 投影到 R3 的内部应力损失约 0.001124 Pa；当前生产空间对 R4 的内部应力误差约 0.027248 Pa。应力 Gram 候选为 0.027252 Pa，未超过参考可分辨尺度取得收益，因此不换正式空间。仍使用约 0.003487 Pa 的保守经验参考尺度；第二候选与 0.00425 m 保留幅值条件未触发。`spatial_accuracy=false`，未达到长期 2% 应力目标。

完整质量投影曾遇到已知载体零空间导致的奇异矩阵；原因已明确，改用原有消零后的独立自由坐标，未添加物理正则项。压力汇总曾出现 `abs(list)` 类型错误，只修正报告计算，旧检查点及身份保持原样。未发生需要掩盖的场景不稳定或无法定位的数值异常。

压力结论限于 x 向半单元阻力、横向封闭的两／四单元夹具。固体虽为三维，当前压力空间不是一般三维认证。固定固体无源的正性不能推广成带源、骨架运动的一般正性。`pressure_spatial_accuracy=false`、`production_C_E_integration=false`。详见 S4/three-dimensional-next-scope.md。敏感幅值完整周期 q5 仍未认证，主入口越界使用充分规则。

下一轮应先改变固定函数预算下的支撑／删除策略，确认空间瓶颈；时间方向继续围绕实际主导观测设计，而不是只追踪模态 4。一般三维压力需要单独定义面朝向、张量迁移率、边界功与制造解，不在本轮直接并入生产场景。

**资源与最新可视化 CLI。**

开工系统盘约余 6.16 GiB，结束约余 6.15 GiB；没有触发低于 5 GiB 的迁移条件。所有新结果和 Warp 缓存直接放在 `/root/autodl-tmp/mpm-lite-phase-stress/20261001T072129Z-phase-stress`，工作区结果路径为链接，数据盘结束约余 116.17 GiB。未删除旧封存输入或活动检查点。

直接展示已生成的结果，无需重新计算：

```bash
cd /root/workspace/mpm-lite
.venv/bin/python -m http.server 8000 --bind 127.0.0.1 \
  --directory docs/results/phase-stress/20261001T072129Z-phase-stress
```

浏览器访问 `http://127.0.0.1:8000/visualization/daily-q5-retry-dt0125/index.html`；压力图访问 `http://127.0.0.1:8000/S4/visualization/index.html`。远程 IDE 使用端口转发。图中位移显示放大 10 倍，HTML 可调；计算数据、应力与反力未放大或平滑。

若需要重新渲染，请输出到封存目录外：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_phase_stress_next.visualize \
  --run docs/results/phase-stress/20261001T072129Z-phase-stress \
  --case daily-q5-retry-dt0125 \
  --output /root/autodl-tmp/mpm-lite-phase-stress-preview
```

完整审计命令：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_phase_stress_next.publication audit \
  --from-release docs/results/phase-stress/20261001T072129Z-phase-stress
```

本轮已实际执行数值入口、恢复入口和渲染入口；上述 HTTP 命令供用户启动，本次没有留下后台服务器。封存后请通过 `publication fork --from-release <本轮发布目录>` 创建新运行，不继续写入旧发布。
