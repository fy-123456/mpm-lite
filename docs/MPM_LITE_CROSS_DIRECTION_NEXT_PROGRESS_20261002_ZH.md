# MPM-lite 跨方向空间与压力网格实施记录

本轮有界实施、数值验收与可视化检查已完成。按29步方案单进程顺序推进；条件未满足的分支明确保留为未触发。最终发布身份、逐项状态与哈希以本目录 `release.json`、`requirement-audit.json` 为准。

## 版本与输入

- 直接父版：`observable-pressure/20261001T154616Z-observable-pressure`。
- 父版 release SHA-256：`0ee1da8b6973bf6757d565149372addbded5c02ac16f2f4478e48412affa9812`。
- 本轮结果：`docs/results/cross-direction/20261001T170848Z-cross-direction`。
- 实际数据位置：`/root/autodl-tmp/mpm-lite-cross-direction/20261001T170848Z-cross-direction`。
- 正式新空间：`cross-direction-snapshot6`，144函数，完整M7，原稳定化；空间包SHA-256：`885a2870a9908f5fd5a0904173a5c1244695e4b7e5a80bf953ff445fda644a30`。
- 仓库没有Git元数据。版本依据为发布清单、源码和成果哈希；开工完整审计包含父版28源码、28快照、3482成果及全部祖先。
- 文档日期沿用20261002；结果目录采用环境实际UTC。日期不替代版本身份。
- 原计划及旧发布保持不变；本轮新代码位于 `benchmarks/research_cross_direction_next`、`engine/aniso_phase1/research_cross_direction_next`、`tests/research_cross_direction_next`。

## 已完成的工作

| 阶段 | 本轮结果与验收 | 主要证据 |
| --- | --- | --- |
| S0 | 新入口四步及新进程恢复，与父版q/v/预测器/反力/能量一致 | `S0/compatibility.json` |
| S1 | 仅构造一个144函数新空间；保留138列，用F45/F60参考的质量加权快照重建6列；真实新方向、完整质量、CPU/GPU算子、q7/q8和实际平衡通过 | `S1/training-comparison.json`、`S1/conditioning-review.json`、候选`operator-audit.json` |
| S1预留验证 | 52.5°只验收一次设计，不看后再调；R4/R5、原空间、新候选四个平衡解通过；八步从静止动力学及公共点场对照通过 | `S1/reserved-direction-check.json`、`S1/dynamic-smoke.json`、`S1/smoke-field-comparison.json` |
| S2 | 固定原BASELINE，保留两支各自1.1s状态，新增64/128步到1.125s；公共物理输出通过，继续细步有收益；正式252节点未改 | `S2/propagation-comparison.json`、`S2/time-scope-decision.json` |
| S3 | 实现真实2/4/8单元几何和RT0，8压力/41面通量；固定固体的压力和累计边界体积均由增广矩阵指数对照；早期4→8排水差超预算 | `S3/topology-operator-check.json`、`S3/fixed-grid-comparison.json` |
| S3条件分支 | 新网格实际耦合与相应恢复未触发；保留原两单元限定研究入口 | `S3/coupled-grid-check.json`、`S3/pressure-scope-decision.json` |
| S4 | 唯一性能候选为显式3×3行列式；同状态场与流体收支一致；两个代表状态公平比较的完整一步成本中位下降19.44% | `S4/paired-performance.json`、`S4/runtime-check.json` |
| S5 | 冻结数值实现、空间和252节点；从本轮实际q7轨迹签发新主工况q5许可；7类真实越界请求拒绝 | `S5/numeric-lock.json`、`S5/scope-boundaries.json`、`S6/qualification-final.json` |
| S5敏感窗口 | 新空间不继承原0.0075m、0.6–0.7s窗口许可；本轮未重建敏感前缀或敏感完整周期 | `S5/window-runtime-final.json` |
| S6 | 26项针对性回归通过；q7和q5各唯一252步周期到1.6s；7类故障检查和新进程恢复通过；完成视觉、GIF、HTTP及零步加载检查 | `S6/final-tests-result.json`、`S6/final-scene.json`、`S5/fault-and-restart.json`、`S6/physical-review.json` |

### 空间改善及其边界

F60内部PK1误差从0.06633 Pa降至0.003232 Pa，纤维PK1从约0.06554 Pa降至0.003226 Pa，均约降低95%。F45内部PK1误差约0.001227 Pa。材料参数和原稳定化保持原含义。

这里改变了真实函数子空间，而不是换名称或可逆旋转。完整算子按 `M_new=Tᵀ M_parent T`、`K_new=T₃ᵀ K_parent T₃` 重建；质量仍包含载体—局部交叉项，没有人工对角质量。新空间质量的归一化条件数约7.33e6，与原空间约7.10e6相近。

52.5°预留方向的内部PK1差约0.02819 Pa，纤维PK1差约0.02811 Pa；后者相对误差约14.7%。它通过的是本轮有绝对项的工程预算 `0.02 Pa + 5%参考范数`，不能称为统一2%精度。该方向以后属于已见证据，不再称隐藏样本。

新旧空间八步公共点对照的最大分区差：位移7.13e-7 m、速度4.15e-5 m/s、PK1 0.001374 Pa，均通过。

### 时间与动力学仍需进一步认证

本轮192步传播研究只属于原BASELINE，不转移给新空间。正式新空间的252步时间表虽保持不变，动态相位仍须单独认证。

完整新旧空间的八个观测时刻对照中，位移最大差3.32e-6 m、PK1最大差0.004815 Pa，均通过；9项分区速度比较略超原解比较预算，最大约2.69e-4 m/s。原空间不是精确动力学参照，不据此宣称新空间更准确，也不将这9项隐藏为通过。

质量内积模态对照显示：原模态4与新模态4的重叠平方约0.995，频率增加约0.484%；原72/523的最佳单模态重叠约0.399/0.422，存在模态混合，不能照搬旧编号。频率变化与速度相位差相符，但线性诊断不构成非线性时间精度证明。未增加阻尼或平移曲线。

### 压力问题定位

旧2/4单元共同终点累计边界体积差5.5749%已复现。新2/4/8采用同一7.594308e-6 s终点、12个时间步，没有改储存、迁移率、初压或边界压。

增广线性系统同时积分压力和累计边界体积，避免把粗时间积分误差当作空间误差。与矩阵指数相比，时间压力差最大约1.62e-5 Pa。4→8的终点压力、含量和排水量均通过，终点排水差约2.13%；但早期排水差最高约11.87%，前5个公共采样节点未通过。因此未启动新网格实际耦合。

初压0.01 Pa与边界压0.002 Pa的跳变激发早期快速排水，对压力网格敏感；这是与现有证据一致的解释，不是一般三维空间收敛证明。现有压力网格仍仅沿x划分，新固体空间也尚未接入压力研究。

### 性能采用范围与计时修正

当前剖析显示，消除重复F计算的收益上界仅约0.12%，没有采用。通用行列式约占21%总成本，于是只试显式3×3公式，仍使用当前F和完整迁移率张量。

初次发现基线带cProfile、候选不带，约27.4%的初始比例作废为采用依据。按用户允许定位普通问题后修复的要求，额外补2个无剖析器基线步；候选没有重跑。性能实际接受步总计6，这一小幅预算修正记录于 `S4/timing-correction-protocol.json`。

统一计时后：两状态基线约13.66/20.80 s，候选约11.13/16.58 s；含构造、实际步进、场重建和提交，中位下降约19.44%。这是两单元压力入口的限定结果，不是固体完整周期或八单元耦合加速结论。没有增加切线缓存。

## 完整场景与可靠性

- 新空间q7：252步、1.6 s、12显示帧，约54.15 s。
- 新空间q5：252步、1.6 s、12显示帧，约42.93 s；正常周期无回退，3次材料哨兵均通过。
- q5/q7公共物理点最大分区差：位移1.75e-9 m、速度8.81e-8 m/s、PK1 1.03e-5 Pa、反力4.23e-8 N。
- 两条轨迹均满足有限值、边界、原残差及能量账；q5最大能量平衡差约5.01e-12 J，账本闭合约6.20e-21 J。
- q7/q5原记录的min(detF)分别约0.989664/0.991940；两规则采样点不同，不能直接相减当作变形差。同一物理探针复核的最大J差仅约2.21e-7。
- 控制性材料失败可整步回退到充分规则；提交前和指针前失败完整回滚，指针后故障只接受一次；新进程恢复验证通过。
- q5只改变材料积分，完整M7保持不变。主许可仅覆盖新空间、F45、0.005 m、本轮252时间节点与绑定来源；敏感完整周期及耦合q5未获许可。

## 已定位并处理的问题

1. F45旧报告多一层 `errors` 包装：修正新分析器，首次错误发生在候选求解前。
2. 未归一化Gram的条件数警告：质量归一化后正交残差约2.61e-13，确认至少3个明显新增方向；没有改候选或添加质量。
3. 52.5° R4旧F45搜索矩阵慢收敛：保留首次12次未收敛记录，增加慢收敛切换当前真实切线GMRES条件；一次复验后R4/R5残差约2.34e-9 N。未调候选或更换验证角度。
4. 性能计时口径不同：已作废初始采用依据，并以上述两步有界修正解决。
5. q7只读检查点入口原先假定步进器自带validate：新入口兼容模型验证器；q7/q5零步加载均通过。

## 最新CLI

直接查看本次已经保存的交互场景、原始反力和能量，不重新计算：

```bash
cd /root/workspace/mpm-lite
.venv/bin/python -B -m http.server 8765 --bind 127.0.0.1 \
  --directory docs/results/cross-direction/20261001T170848Z-cross-direction
```

执行后打开 `http://127.0.0.1:8765/`。远程IDE将8765端口转发到本机。固体页可播放、选择帧及调整显示倍率；默认形变显示10倍，计算数据和应力不放大。

只读检查最终状态，不推进、不覆盖：

```bash
.venv/bin/python -B -m benchmarks.research_cross_direction_next.inspect \
  --run docs/results/cross-direction/20261001T170848Z-cross-direction
```

只读加载当前限定两单元压力入口：

```bash
.venv/bin/python -B -m benchmarks.research_cross_direction_next.pressure_runtime inspect-current \
  --run docs/results/cross-direction/20261001T170848Z-cross-direction
```

封存后复核完整清单与祖先：

```bash
.venv/bin/python -B -m benchmarks.research_cross_direction_next.publication audit \
  --from-release docs/results/cross-direction/20261001T170848Z-cross-direction
```

PNG已实际查看，12帧GIF已逐帧解码，6个HTTP页面/图片资源返回200且内容哈希一致。未声称做过真实浏览器按钮交互测试。后续实验须另开结果目录，封存目录禁止原地续写。

## 存储与后续优先级

结果、场景和Warp缓存写数据盘。各检查点系统盘均高于5 GiB，本次未触发迁移；当前约5.027 GiB，数据盘约91.13 GiB。没有删除唯一成果或修改旧封存目录。

下一轮建议依次：在新空间重建卸载／保持段的时间参照；改善跨方向应力泛化并另登记验证集；针对早期压力边界层建立更可靠空间参照。压力与新固体空间的集成、敏感窗口许可分别重新验收，不继承本轮未通过或未触发分支。
