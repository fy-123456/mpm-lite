# MPM-lite 成本、相位与压力瞬态：实施记录

2026-10-01（UTC），按 [本轮计划](MPM_LITE_NEXT_COST_PHASE_PRESSURE_PLAN_20261001_ZH.md) 顺序实施。P0—P6 的 29 个步骤均已落实或按预登记条件作出不晋级决定；没有将候选不通过改写成通过。

本轮发布：[20261001T062026Z-cost-phase/release.json](results/cost-phase/20261001T062026Z-cost-phase/release.json)。目录时间是运行标识，实际封存 UTC 见 release.json。实体为 `/root/autodl-tmp/mpm-lite-cost-phase/20261001T062026Z-cost-phase`。发布前完成本文，随后由 seal 生成源码、成果及本文哈希并审计；封存后禁止覆盖本轮输入、源码、成果和文档。

**版本与实际默认值**

开工核实的最新正式父版为 `spatial-phase/20261001T043604Z-spatial-phase`，release SHA-256 为 `b9b458890a170998b57202fd1a926a2477d044e48f2a05909ba714d45d258cb6`。较晚的 `20261001T055244Z-spatial-phase` 没有 release.json，只是继承检查。工作区没有 Git 元数据，以内容哈希和发布链确认身份，不虚构提交号。

本轮继续使用 nonlinear-modes-swap6：144 个局部标量函数、216 个自由标量坐标／648 个自由向量分量，完整 M7、原 Ks、原材料与边界；float64／Warp CUDA、分段算子、场缓存。dt=0.0125 s，128 步至 1.6 s，每例 12 帧。主场景 F45／0.005 m 使用 q5_with_full_retry，充分材料 q7 并核查 q7/q8。没有新增阻尼、人工质量或网格传递。

新增代码位于 `benchmarks/research_cost_phase_next`、`engine/aniso_phase1/research_cost_phase_next`、`tests/research_cost_phase_next`，共 19 个 Python 文件。祖先源码和已封存文档逐项核对，旧入口未被修改。原始开工计划保存在 [plan-frozen.md](results/cost-phase/20261001T062026Z-cost-phase/plan-frozen.md)，当前计划更新状态不会改写该快照。

**阶段结论**

| 步骤 | 实际工作与决定 | 验收／证据（本轮结果目录内） |
| --- | --- | --- |
| P0.1—P0.3 | 最新版及全部祖先审计；新入口显式绑定空间／质量；4 步兼容与新进程恢复；锁定预算与数据盘 | P0/version-audit.json、compatibility.json；q/v/predictor 与父版差为 0 |
| P1.1—P1.2 | 定位双模型重复载入，q5/q7 共享不可变 reduction；响应、布局、试算与事务状态分别拥有 | baseline-profile.json、shared-resource-check.json、operator-equivalence.json；错误来源／设备拒绝，共享数组不可写 |
| P1.3—P1.5 | 解压仍为热点；采用原始 CSR 无压缩数值归档，保持全部数组和物理算子；两状态交替配对与回退验证 | compact-representation.json、transaction-check.json、performance-decision.json；最终独立进程内存见 P6/cost-comparison.json |
| P2.1—P2.4 | 核验复用父版两窗三档；分析相位／激励；拟议分段时间表重复已有未通过项，保留 0.0125 s | P2/reuse-audit.json、failure-breakdown.json、phase-cause-analysis.json、time-decision.json；不追加第四档 |
| P3.1 | 在两个既定局部区域进一步 h 加密，构造一个 R4，参考差继续收缩 | P3/R4/result.json、reference-decision.json；108.98 s，峰值 RSS 12.09 GiB |
| P3.2—P3.4 | 两个候选各保持 144 函数，检查完整算子／充分积分并求非线性静态平衡；无可分辨收益，均不晋级 | P3/design-protocol.json、candidates、space-decision.json；新保留幅值 0.00425 m 未开启 |
| P3.5 | 空间未改变，继承相同物理空间的动力学资格与时间表；未重跑无必要前缀 | selected-space.json、P3/dynamic-qualification.json |
| P4.1—P4.3 | 复现一致 RT0 过冲；推导半单元守恒面阻力；分别验证半离散与中点时间正性 | P4/transient-baseline.json、discretization-derivation.md、monotonicity-and-time.json |
| P4.4—P4.5 | 三维固体加两压力单元，封闭／排水各四步，四单元敏感性、局部守恒、混合秩、回滚及新进程恢复 | P4/limits-and-closure.json、coupling-decision.json、rollback-check.json、publication-check.json；全部限定检查通过 |
| P5.1—P5.3 | 最终配置 q7/q8、q5/q7 重新核查；资格绑定实际时间网格；故障分类、整步回退和发布事务 | P5/qualification-final.json、qualification-scope.json、fault-results.json、retry-restart.json |
| P5.4 | F45／0.0075 m 用充分规则从静止到 0.5 s，仅在 0.5—0.55 s 四步比较 q5/q7 | P5/sensitive-scope.json：qualified_window=true，full_cycle_q5=false |
| P6.1—P6.3 | 17 项回归；最终源码恢复；一组 q7/q5 完整周期及图像检查；29 步、能力和来源封存 | P6/final-tests-result.json、final-scene.json、physical-review.json、requirement-audit.json、release.json |

每个 P1 简写文件名均位于 P1/。不同步骤的交付名按实际实现记录在 requirement-audit.json；不能只根据计划中的拟定文件名判断是否完成。

**采用的成本优化与代价**

共享只读 reduction 将配对构建从 18.93 s 降到 12.66 s，代表两步暖任务从 0.351 s 到 0.348 s。之后原始 CSR 无压缩归档的另一次配对中，构建 13.69→9.84 s，暖任务 0.348→0.350 s，满足预登记的暖任务预算。两次配对收益不相乘宣称整体加速。GPU 规则布局和响应没有共享，失败回退检查差为 0。

最终各规则各用一个独立进程跑完整周期，与同物理空间的已封存父版比较：

| 实际成本 | 父版 q5＋回退 | 本轮 q5＋回退 | 父版 q7 | 本轮 q7 |
| --- | ---: | ---: | ---: | ---: |
| 构建／初始化（s） | 19.73 | 10.21 | 9.44 | 7.06 |
| 推进＋场重建＋发布（s） | 22.16 | 22.61 | 33.83 | 33.79 |
| 总计（s） | 41.99 | 32.89 | 43.34 | 40.93 |
| 进程峰值 RSS（GiB） | 4.83 | 3.42 | 3.42 | 3.42 |

本次日常完整场景总时间减少约 **21.7%**，构建减少约 **48.3%**，峰值 RSS 减少约 **29.2%**。这是已有 Warp 编译缓存、未清空 OS 页缓存条件下的单次观测，不是统计加速保证；收益主要在启动，暖推进没有显著加速。P1 配对进程同时保留多个模型，累计 RSS 不能当作独立变体内存；此处独立进程解决了该口径问题。

数值归档从 679,232,141 字节增至 1,457,275,248 字节（约 679 MB→1.46 GB），以磁盘换取免解压和少重复载入。归档生成约 2.24 s，文件仍逐次校验内容，不依赖 mtime 跳过完整性核查。M、Ks、组合映射、raw CSR 数组和完整轨迹保持相同；没有删除小系数或减少质量交叉项。详细成本及 GPU 内存口径见 [cost-comparison.json](results/cost-phase/20261001T062026Z-cost-phase/P6/cost-comparison.json)。

**时间与空间：取得的证据和未解决部分**

时间：两窗三档的共同初态、充分材料和轨迹来源已核验。中点格式在线性模式的数值频率为

\[
\omega_{num}=\frac{2}{h}\arctan\left(\frac{\omega h}{2}\right).
\]

高频在细档仍可能分辨不足，加载转换也会激励这些模态；这是诊断线索，不是全部非线性误差的根因证明。拟从 0.6 s 细化的方案与既有局部窗口没有新可检验差别，仍不能满足联合场／反力／事件要求，所以复用证据并保留 dt=0.0125 s。没有用阻尼、平移曲线或滤波掩盖差异，temporal_accuracy=false。

空间：R4 的主静态真实残差约 1.49e-8 N。内部 R2/R3 PK1 差约 0.003487 Pa，R3/R4 差约 0.001115 Pa；相邻差收缩，登记的保守经验不确定性仍取较大者 0.003487 Pa。该参考仅支持 F45／0.005 m 的有限静态比较，不是连续精确解。

两个候选的内部 PK1 误差分别约 0.027251 Pa、0.068064 Pa。六模式重构没有可分辨增益，十二模式退化；质量、算子和材料充分性检查已通过，不能将退化简单归为积分错误。主状态不合格，故按协议不访问新保留幅值、不另造动力学前缀。正式空间继续 swap6，约 6.52% 应力／8.84% 纤维误差的数量级仍高于长期 2% 目标，spatial_accuracy=false。

**压力瞬态改进的实际范围**

固定固体、无源、初压 0.01 Pa／右储库 0.002 Pa 下，一致 RT0 的 2／4／8 单元半离散解都出现过冲，最大约 3.00e-4／3.36e-4／3.15e-4 Pa。新方案使用沿 x、横向零流量子空间内的半单元阻力积分：

\[
A=\frac{k}{\mu}JF^{-1}F^{-T},\qquad
R_f=\sum_{K\sim f}\int_{K,f}\frac{(A^{-1})_{xx}}{A_f^2}\,dV,
\qquad C\dot p+BH^{-1}B^Tp=b.
\]

不能把有效传输系数误写成 Axx，也不能将新对角面阻力声称为原一致 RT0 的等价删项。相同排水条件下，新半离散解这三档的过冲为 0；正常储存 0.2、dt=0.01 s 的两／四单元中点推进通过正性、守恒和耗散检查。小储存的大步长可能使中点更新失去正性，本轮只检查其混合可解性，没有授予通用单调性。

恢复三维固体后，封闭和排水四步、四单元对照均通过。逐单元含量闭合最大约 5.85e-17 m³，原始能量闭合最大约 3.22e-15 J；q/v/predictor、压力、通量、含量、累计源和边界历史在新进程恢复后差为 0。提交前失败保持旧提交，指针发布后观察异常仅接受一次。最终源码又完成一次两步退出、恢复到第四步的检查。

这是独立研究入口，默认日常场景仍为纯固体；一般三维压力离散、连续压力精度、通用单调性及生产级 C/E 接入均未认证。

**材料资格、最终场景与异常处理**

主资格使用实际充分轨迹 0、0.5、1.1 s，另用独立 0.8 s 检查；同状态能量、组装内力、弱应力矩和混合／敏感方向切线通过。最终证书另存 qualification-final.json，早期案例引用的旧证书不被覆盖。时间资格绑定实际时间网格，不能只凭最大步长相同放行新路径；超出范围采用充分规则，非法身份拒绝。

F45／0.0075 m 用 q7 从静止推进 40 步至 0.5 s，随后四步短窗通过 q5/q7 对比。该结果只记为窗口资格，主入口完整敏感周期仍 full_only。跨角度、跨材料、跨几何以及耦合 q5 没有自动获得资格。

最终两个场景各 128 步／12 帧，17 项回归通过。日常与父版全部 128 个已推进状态的 q/v/predictor 差为 0。q5/q7 的最大分区差为：位移 1.79e-9 m、速度 9.25e-8 m/s、PK1 1.03e-5 Pa、纤维 1.02e-5 Pa；同区间反力差 4.24e-8 N。三次预登记材料哨兵全部通过，日常全周期未触发自然回退；受控故障独立验证过回退路径。

保存帧数值有限，边界位移／速度约束差为 0。q5 最小 det(F)=0.99356，q7=0.99159；两者采样点不同，不将不同点上的最小值差当作场误差。日常原始单步能量差最大 1.57e-11 J，分账后缺口约 5.63e-21 J；后者不冒充原始误差。1.1—1.6 s 总能量约 3.16198e-7—3.16211e-7 J，未见异常增长。原始反力与 det(F) 的卸载后振荡保留，时间精度仍未认证。已实际核看场景图与耦合图，显示形变倍率明确为 10 倍。

主要决定均有证据：时间方案无联合收益时保留原步长；空间候选无可信收益时不晋级；压力从一致 RT0 改为独立有推导的受限离散；更大载荷只授予短窗资格。没有因未找到一般高频根因而放弃其他已授权工作，也没有通过放宽来源、事务或几何条件取得通过。

**资源与下一步**

开工系统盘约余 6.16 GiB，最终约余 6.16 GiB，数据盘约余 116.7 GiB；全程未触发低于 5 GiB 的迁移条件。新大归档、实验与 Warp 缓存均在数据盘。最高参考进程 RSS 12.09 GiB，低于预登记 16 GiB 上限。

后续优先顺序：进一步分离实际激励与高频时间分辨率问题；用改善后的参考寻找真正降低应力误差的空间表示；在新热点数据支持下再优化暖推进；另立预算研究小储存／一般三维压力及敏感载荷全周期资格。本轮不追加这些实验，也不增加长周期或显示帧。

**最新 CLI 与可视化**

以下读取本轮已提交结果；外部输出避免修改封存目录。HTML 支持逐帧与倍率调整，同时输出 PNG、GIF；渲染不重新运行数值场景。

```bash
cd /root/workspace/mpm-lite

# 当前正式发布及全部祖先审计
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_cost_phase_next.publication audit \
  --from-release docs/results/cost-phase/20261001T062026Z-cost-phase

# 日常纯固体：128 个积分步、12 个已保存显示帧
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_cost_phase_next.visualize \
  --run docs/results/cost-phase/20261001T062026Z-cost-phase \
  --case daily-q5-retry-dt0125 \
  --output /root/autodl-tmp/mpm-lite-cost-phase/preview-latest

# 四步独立固液耦合：压力／通量／能量和总应力
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
  -m benchmarks.research_cost_phase_next.coupling_study render \
  --run docs/results/cost-phase/20261001T062026Z-cost-phase \
  --output /root/autodl-tmp/mpm-lite-cost-phase/preview-coupled-latest
```

已封存预览：[纯固体 HTML](results/cost-phase/20261001T062026Z-cost-phase/visualization/daily-q5-retry-dt0125/index.html)、[PNG](results/cost-phase/20261001T062026Z-cost-phase/visualization/daily-q5-retry-dt0125/scene-summary.png)、[GIF](results/cost-phase/20261001T062026Z-cost-phase/visualization/daily-q5-retry-dt0125/cycle.gif)、[耦合 HTML](results/cost-phase/20261001T062026Z-cost-phase/P4/visualization/index.html)。

如需新场景，先运行 `publication fork --from-release` 并使用其输出的新目录，再用本轮 `run configure`／`run cycle`。不要对正式发布执行 configure/cycle。fork 继承优化后的实际归档和最终资格；较晚的无 release.json 验证目录不成为最新正式版。
