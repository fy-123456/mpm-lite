# MPM-lite 五项顺序优化实施记录

2026-10-01（UTC）。已按 [执行计划](MPM_LITE_POST_RELEASE_FIVE_PRIORITY_PLAN_20261001_ZH.md) 顺序完成 S0 → S1 → S2 → S3 → S4 → S5 的有界实施和验收。数值任务串行运行。本轮采用计划允许的“参考受限、候选不晋级”分支；原 144 函数空间继续使用。

当前交付为 `research_post_release`，默认发布案例 `daily-q5-retry-dt0125`：float64、Warp CUDA 分段算子、`dt=0.0125 s`、q5 材料积分和整步 q7 回退、完整质量 M5、探针几何缓存，128 步至 1.6 s，12 个显示帧。未增加阻尼、人工质量或压力罚项。场景稳定、材料规则对照、事务和恢复通过限定验收；时间精度、空间精度和三维物理耦合仍未认证或实现。

## 1. 版本身份和资源

| 项目 | 实际依据 |
| --- | --- |
| 开工父发布 | [20260930T122305Z-next-practical/release.json](results/sequential-next/20260930T122305Z-next-practical/release.json)，封存时刻 2026-09-30 14:55:40 UTC |
| 父发布文件 SHA-256 | `ae8a1d2151056ab5472f8287d9bfbb950c78cb59a69f3b0f7763f0b3250d5296` |
| 数学包 SHA-256 | `d20f40a21c893831f9bf22504dea43a99eac29242b111e41e920e185cf66896e` |
| 本轮成果 | [20261001T013007Z-post-release](results/post-release/20261001T013007Z-post-release)，实际位于 `/root/autodl-tmp/mpm-lite-post-release/20261001T013007Z-post-release` |
| 本轮发布索引 | [release.json](results/post-release/20261001T013007Z-post-release/release.json) |
| 开工输入锁 | [parent-release-lock.json](results/post-release/20261001T013007Z-post-release/parent-release-lock.json) |
| 原始计划身份 | `d957a40185b8a80280c9d64d1f522ea4fbf87957ae861a2daa7d8e65f97b7a80`；原文保存在 [plan-frozen.md](results/post-release/20261001T013007Z-post-release/plan-frozen.md) |
| 本轮最终源码 | [final-source-sha256.json](results/post-release/20261001T013007Z-post-release/final-source-sha256.json)，18 个新增 Python 文件及独立快照；最终案例另绑定 8 个新数值入口／适配器文件及父版身份 |
| 步骤核对 | [requirement-audit.json](results/post-release/20261001T013007Z-post-release/requirement-audit.json)，原计划 35 项步骤逐项记录完成或条件分支 |
| 能力边界 | [capability-matrix.json](results/post-release/20261001T013007Z-post-release/capability-matrix.json) |

工作区没有 `.git`，因此版本通过发布文件、内容哈希、输入锁和案例协议识别，不编造 commit 或远端 HEAD。开工和最终场景验收均复核父版 39 项源码、39 项快照、1,946 项成果、683 项历史绑定及 25 项输入，全部保持一致；这些清单存在重叠，不能将数量相加当作独立文件数。

开工系统盘可用约 6.32 GiB，最终场景验收时约 6.29 GiB；没有触发低于 5 GB 的迁移条件。新结果、检查点和 Warp 缓存直接放到数据盘，验收时数据盘剩余约 127.89 GiB。没有删除历史失败记录或重要输入。最终运行峰值 RSS 约 1.35 GiB，CUDA 池进程高水位约 2.34 GiB，在本轮预算内。

## 2. 各阶段实际完成内容

| 阶段 | 改动与理由 | 验收和决定 |
| --- | --- | --- |
| S0 版本及入口 | 新建独立命名空间，显式继承最终父发布，锁定源码、输入和计划；为新运行分配数据盘目录 | 父版全量哈希复核；明确时间表的真实退出／重启通过 |
| S1 时间相位 | 保留材料、空间、质量和稳定化，比较卸载及保持短窗；选择统一 0.0125 s | 两窗模态速度误差分别降低 75.52% 和 76.58%；128 步 q7 完整周期稳定，时间精度仍未认证 |
| S2 真实参考 | 实现环境网格局部 h 加密及 p=5 嵌入，构造完整质量和交叉项，保留原 Ks | h 局部相邻差收缩；p 补充仍改变内部 PK1 约 5.86%，全域参考保持受限 |
| S3 函数预算 | 实现含交叉作用的条件能量评分，构造替换 6／12 函数的两个 144 函数设计并检查质量秩 | 参考不足，两个候选均不晋级；正式空间仍为 original144，无需跨空间动态状态迁移 |
| S4 q5 日常回退 | 主入口接入三种策略、资格绑定、q5 整步失败后 q7 重算、规则能量差、阶段哨兵及失败历史 | 同状态 q5/q7/q8 算子检查、故障注入、整步回滚、发布失败和真实新进程恢复通过 |
| S5 性能与物理边界 | 依据实际热点缓存固定探针几何；补独立二维混合系统闭合及三维接口说明 | 几何缓存数值等价且有效；最终完整场景稳定；不启用无收益证据的切线缓存或改动预条件器；三维耦合未冒进 |

### S1：时间相位

在父版已提交的 1.0 s 和 1.2 s 状态启动两个短窗，分别覆盖 `[1.0,1.2]`、`[1.2,1.4]`。比较旧 0.025 s、0.0125 s 和局部更细的 0.00625 s；对共同时间点恢复物理场，对反力按相同时间区间比较冲量平均，保留原加载断点。

相位事件最大偏移从 0.01875 s 减至 0.009375 s／0.00625 s。模态速度 RMS 差下降约 75.52%／76.58%。部分分区物理场仍未达到计划的工程比较预算，因此只确认局部相位改善，不把小步长结果当作已收敛真解，也不以添加阻尼消除剩余振荡。

选定统一 0.0125 s，避免在加载阶段继承未认证的粗时间相位。`full-q7-dt0125` 完成 128 步、1.6 s 和 12 帧，最小采样 det(F)=0.990075；后续作为同时间表的材料积分对照。

依据：[time-decision.json](results/post-release/20261001T013007Z-post-release/S1/time-decision.json)、[restart-check.json](results/post-release/20261001T013007Z-post-release/S1/restart-check.json)。

### S2：局部 h/p 参考及其局限

实际 Q4 环境节点由 217×65×65 增至 225×65×65、241×65×65；原载体及原局部函数精确嵌入，新函数增加真实表达能力。构造新的完整质量、载体与局部交叉项，检查质量阶次、秩、真实切线作用及原稳定化语义。两个 h 包和一个 p 包均在新进程中重新加载，并完成仿射场和刚体材料响应检查。

两级 h 相邻内部 PK1 差约为 6.18×10⁻⁶ 相对值，说明所选夹持邻域局部改善逐渐接近；但该加密没有解决内部表达的不确定性。按预算追加一次内部 p=5 研究，PK1 仍改变约 5.86%，且只有一级 p 结果，不能声称全域参考可靠。

因此采用 [reference-decision.json](results/post-release/20261001T013007Z-post-release/S2/reference-decision.json) 的 `reference_limited` 总结。局部 `h-result.json` 的通过仅代表其限定范围。没有为使参考“通过”改变原 Ks 或增加人工质量。

包与检查：[h1](results/post-release/20261001T013007Z-post-release/S2/h1/space-package.json)、[h2](results/post-release/20261001T013007Z-post-release/S2/h2/space-package.json)、[p1](results/post-release/20261001T013007Z-post-release/S2/p1/space-package.json)、[重载](results/post-release/20261001T013007Z-post-release/S2/reload-check.json)、[仿射／刚体](results/post-release/20261001T013007Z-post-release/S2/affine-rigid-check.json)。

### S3：144 函数的交叉作用评分

用完整自由切线矩阵的块消元评价每组 6 个函数。对于块 b，采用

\[
S_b=\big[(K_{ff}^{-1})_{bb}\big]^{-1},\qquad
\Delta E_b=\tfrac12 a_b^T S_b a_b.
\]

该评分允许其余自由度重新松弛，包含重叠函数的交叉影响。与完整线性松弛直接计算进行核对，通过后形成 `interaction-swap6` 和 `interaction-swap12`，各保留 144 函数并验证独立质量秩。

参考不足使“换函数能提高物理精度”尚无可靠判据，因此只交付设计和诊断，不运行候选长周期，不宣称未见场景泛化。正式空间未变，S3.5 的新空间动态资格属于未触发的条件步骤。

依据：[block-scores.json](results/post-release/20261001T013007Z-post-release/S3/block-scores.json)、[space-decision.json](results/post-release/20261001T013007Z-post-release/S3/space-decision.json)。

### S4：q5 入口、适用范围及回退

实际入口 `benchmarks.research_post_release.run` 支持：

- `full_only`：使用充分规则 q7。
- `q5_fixed`：固定 q5，失败直接报告，不自动切换。
- `q5_with_full_retry`：q5 可分类的数值失败后恢复整步起点，使用 q7 重算；成功后保持 q7，并保存失败历史。

在最终保留空间及 S1 选定时间表上，以 0、0.5、1.1 s 状态以及混合／敏感方向比较能量、组装内力、弱应力矩和切线；q7 另与 q8 比较，质量始终保持 M5。[资格证书](results/post-release/20261001T013007Z-post-release/S4/qualification.json) 绑定空间、材料、质量、边界和规则身份。

资格仅覆盖当前 F45、峰值位移 0.005 m、最大步长 0.0125 s 的限定路径。不支持的材料／加载声明直接拒绝，避免参数被静默忽略；时间表超出 q5 步长资格时，入口记录原因并采用 q7。基础 `configure` 默认仍是 `full_only` 且缓存关闭；要使用本次推荐组合，须显式给出后文选项。

运行时在 0.5、0.6、1.1 s 做 q5 与充分规则能量／内力哨兵检查。切换规则的势能偏移单列，不伪装成物理外功。失败后位移、速度、预测器、子状态和账本一并恢复；两种规则都失败时保持原已提交状态。`KeyError` 等程序错误不伪装成可重试的材料误差。

检查点采用已提交代次指针：指针发布前的磁盘失败恢复内存并忽略未提交孤立代次；指针发布后不重复推进。真实新进程恢复检查包含已切换 q7 的控制器历史；开启几何缓存后的 q7 回退、显示帧和恢复也单独通过，恢复后的 q/v/predictor 与充分规则两步对照一致。

依据：[故障与事务](results/post-release/20261001T013007Z-post-release/S4/fault-results.json)、[新进程恢复](results/post-release/20261001T013007Z-post-release/S4/restart-check.json)、[缓存加回退恢复](results/post-release/20261001T013007Z-post-release/S5/cache-retry-check.json)。受控故障用于验证恢复路径；最终自然运行没有发生规则回退。

### S5：按实测热点优化

先测普通和卸载两个窗口，发现固定探针场恢复成本明显，Krylov 迭代为 0。新增 `CachedProbes`，保存固定几何的值／梯度映射及边界偏置，避免每帧重建约 91.7 万个环境节点；缓存只依赖几何和空间，仍按实际状态计算物理场，并核查空间／边界身份。缓存有 256 MiB 上限，本场景占 19,132,344 字节，约 18.25 MiB。

两个窗口各做三组交替性能对照，状态完全一致，物理场差为浮点舍入量级：

| 测量 | 原方式 | 几何缓存 | 口径 |
| --- | --- | --- | --- |
| 单次场恢复中位数 | 0.22384 s | 0.01020 s | 约 22 倍场恢复加速 |
| 两步加一帧及检查点中位数 | 0.40253 s | 0.19024 s | 此短任务总成本降低约 52.7% |
| 128 步／12 帧日常成本估计 | 13.307 s | 10.888 s | 包括约 0.144 s 缓存构造，估计降低约 18.2%；不是全周期配对实测 |

最终实际完整运行器耗时约 18.43 s，其中模型等构造 5.82 s、推进及场恢复／检查点合计 12.53 s。上述计时口径不同，不能把短任务 52.7% 当作所有场景的整体收益。切线缓存继续关闭，预条件保持 `original`，因为当前热点没有支持优先改动它们。

依据：[performance-decision.json](results/post-release/20261001T013007Z-post-release/S5/performance-decision.json)。

物理耦合方面，实际运行既有二维 Q2/P0/RT0 混合模型，4×4 网格、4 个变步长小步；188×188 系统满秩，采用一般线性求解器，检查压力功抵消、质量、耗散、能量分账和完整子状态回滚。最大压力功抵消差约 4.87×10⁻²¹ J；不直接沿用固体 SPD/CG 假设。

三维 C–E 真实耦合还需要共同空间、压力功和流体交换的一致离散。已写 [耦合接口说明](MPM_LITE_POST_RELEASE_COUPLING_INTERFACE_20261001_ZH.md)，明确候选自由能、变量单位、功共轭关系与未来验收；本轮没有把二维独立通过当作三维耦合完成。

依据：[coupling-check.json](results/post-release/20261001T013007Z-post-release/S5/coupling-check.json)。

## 3. 最终场景和测试

最终 `daily-q5-retry-dt0125` 绑定当前 8 个新数值文件和父版身份。128 步、129 个状态代次、12 帧，完整运行至 1.6 s；最小路径／端点采样 det(F)=0.992281，真实残差最大约 5.23×10⁻⁸，账本闭合误差最大约 5.86×10⁻²¹ J。三个材料哨兵全部通过，没有自然触发 q7 回退。

逐一对照同时间表 q7 的全部 129 个共同状态和同区间反力；在全域、夹持、过渡、内部各区域恢复相同探针物理场。下表是所有时刻／分区中最大的加权 RMS 差，反力为同时间区间的绝对差，并非连续域逐点最大误差：

| 量 | 最大比较差 |
| --- | --- |
| 位移 | 1.8624×10⁻⁹ m |
| 速度 | 9.5523×10⁻⁸ m/s |
| PK1 | 9.9930×10⁻⁶ Pa |
| 纤维方向 PK1 | 9.9184×10⁻⁶ Pa |
| 反力 | 4.3304×10⁻⁸ N |

全部通过原计划的实用混合绝对／相对预算，没有为通过而提高预算。该对照只说明同空间、同时间表的积分规则差异很小，不能消除两者共有的时空离散误差。

已实际查看最终总览图，加载／卸载、能量和形变没有明显运行异常，采样点未见翻转。卸载后保留小幅无阻尼振荡；相位仍有研究空间。展示位移放大 10 倍，颜色为实际 PK1 分量。q5/q7 的最小 det(F) 位于不同材料积分点集合，两个最小数值不能直接当作同点误差。

11 项针对性测试全部通过，覆盖时间表、入口参数、受控规则失败、程序错误区分、双失败回滚、规则能量差、检查点发布前／后失败、封存目录禁止配置／推进。实际场景恢复、缓存等价和二维闭合另外用真实算子检验，未仅依赖模拟夹具。

依据：[最终场景结果](results/post-release/20261001T013007Z-post-release/S5/final-scene-result.json)、[分区全周期对照](results/post-release/20261001T013007Z-post-release/S5/final-field-comparison.json)、[11 项测试](results/post-release/20261001T013007Z-post-release/S5/final-tests-result.json)。

## 4. 发现的问题及处理

| 问题 | 根因与处理 | 后续影响 |
| --- | --- | --- |
| 首轮 CLI 提前停止未生效 | `stop_after` 未传到运行器；修复后新建 v2 案例，真实中断再重启 | q/v/predictor 及除计时外数值账本与不中断对照一致；首轮记录保留 |
| h 包 GPU 上传报只读 CSR 异常 | 新插值 CSR 未排序却提前冻结，上传器需要原地排序 | 在新适配器内先规范化副本再冻结，受影响研究重跑通过；旧实现保持原样 |
| 局部 h 收缩被误当全域可靠参考的风险 | 加密位置未覆盖内部试验空间不确定性 | 补内部 p 研究，明确参考受限，阻止候选晋级；普通接口问题没有跳过 |
| 已完成案例可能被再次配置或推进覆盖 | 增加封存目录写保护和“从封存发布创建新运行”入口 | 防覆盖测试通过；数值控制源码因此变化，最终新建案例重新做了一次 128 步／12 帧验证，旧验证保留 |

没有进行无界根因搜索或大量长帧扫描。S2 在预定 h/p 预算内保留未解决的参考覆盖问题，后续仅跳过依赖可靠参考的空间晋级；q5、缓存及事务等独立工作继续完成。

## 5. 最新 CLI 和可视化

现成成果：[交互预览](results/post-release/20261001T013007Z-post-release/visualization/daily-q5-retry-dt0125/index.html)、[总览图](results/post-release/20261001T013007Z-post-release/visualization/daily-q5-retry-dt0125/scene-summary.png)。

以下命令只读取已提交帧，在封存目录外生成 HTML、GIF 和 PNG，不重新运行仿真：

```bash
cd /root/workspace/mpm-lite
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_post_release.visualize \
  --run docs/results/post-release/20261001T013007Z-post-release \
  --case daily-q5-retry-dt0125 \
  --output /root/autodl-tmp/mpm-lite-post-release/preview-latest
```

未来要修改或重跑，先创建新运行，避免改变封存结果。下面是推荐组合的完整命令；新运行入口验证应用发布源码／成果及继承的材料资格，并保留原数学父包和应用发布两层来源：

```bash
cd /root/workspace/mpm-lite
mpm_post_run=$(OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_post_release.provenance \
  --from-release docs/results/post-release/20261001T013007Z-post-release)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_post_release.run configure \
  --run "$mpm_post_run" --case daily \
  --dt 0.0125 --rule-policy q5_with_full_retry --field-cache
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_post_release.run cycle \
  --run "$mpm_post_run" --case daily
```

短程验证可在 `configure` 增加 `--end 0.025`，或在 `cycle` 增加 `--stop-after 2`；后者保留完整时间表，再次执行相同 `cycle` 会从已提交代次恢复。已有封存目录禁止 `configure/cycle`，其可视化必须指定归档外输出。

## 6. 下一步的明确边界

当前可日常使用的是上述固定 F45 场景的 q5 回退入口、可靠提交／恢复和场恢复缓存。研究上最值得继续的是内部区域参考的覆盖与收敛：先在预算内再验证内部 p 或 h 的相邻级别，参考可信后才能决定 144 个函数怎样重新分配。时间方向继续针对无阻尼卸载／保持的相位，不用阻尼掩盖误差。

新材料、几何、载荷、更大变形或新的空间需要重新绑定并验证积分资格；当前证书不自动泛化。三维孔弹性按照接口文档先落实共同空间与离散功／质量交换，再进入真实求解。只有新的场景显示切线调用成为热点时，才重新考虑切线缓存和预条件优化。
