# C：相容动力学的隔离 CPU 原型

本目录基于 2026-09-30 核验的 v22 源码：488 项源码和 131 项成果全部匹配。遵循 `docs/parallel_research_v22_20260930/` 的目录所有权；生产入口、公共接口、其他方向和旧归档不变。

## 模型与适用范围

`legacy.legacy_v20()` 直接调用归档的 `aniso_v20_fast_runs.setup('gauss3-condensed')` 和 `FastIntegratedAVF`，恢复原材料积分、稳定化、Gauss3 惯性、不可见方向约束、移动几何与末端边界算法。它与下面的小模型是两个显式配置，不能用关闭 α 冒充 legacy。

`Model` 是一个可完整检查矩阵的参考模型：原物理盒体、原硬夹持、μ=10、λ=20、k_f=200、F45、ρ=1。20 个 Q1 标量载体，增加 3 个标量局部模式，三分量共享每个标量基；共 69 个向量分量，其中 21 个自由分量。局部基在自由跨内为 `4t(1-t)(2t-1)^k`，`t=(X_x-.25)/.5`、`k=0,1,2`，夹持体内为零；支撑端的梯度按分片积分解释。系数均为米。

这个小型相容有限元空间没有旧载体的沙漏模式，因此稳定化为零。它不是 v22 的 225+432 自由度候选，也不替代 A 的空间研究。专项测试用 v22 `HighOrderPotential` 独立核对同一小空间的非线性能量与力；转换旧接口时只对载体加一次参考坐标。

## 运动学与惯性

位移自由度 `q=(u_y,α)`；载体总位置 `Y0+u_y`。同一个基定义：

```
x = X + N q
F = I + D q
v = N qdot
L = D F^{-1}
C = L qdot
M = Nᵀ diag(ρ dV) N
```

`M` 包含完整的载体—局部交叉块和局部质量块。这里选定的是参考连续体的点惯性：`C` 是派生的空间速度梯度，没有独立的 APIC 微惯性。对固定参考基，粒子可以移动，但 `M` 严格恒定；不需要遗漏或补记所谓“几何质量变化”。这不等于已经求解几何相关 APIC 度量的动态问题。独立预应力、不可恢复 F、塑性历史和任意独立 C 不能直接接入这个状态模型。

材料和质量使用正参考体积积分。默认 `(5,3,3)` 阶；局部基最高四次，x 向五点积分精确覆盖质量多项式。禁止用对角质量补偿秩缺陷。`tangent_action` 使用原 Hencky＋纤维本构的精确切线；显式 K 与矩阵作用分别检查。

## 时间与真实边界冲量

以 `W=(q1-q0)/dt` 为中点广义速度，固定夹持速度由整段位移差给定，路径平均力为 `fbar=∫grad U(q0+a dt W) da`。自由方向求解：

```
Qᵀ [2M(W-v0) + dt(fbar-f_external)] = 0
v_pre = 2W-v0
impulse_mid = 2M(W-v0) + dt(fbar-f_external)
```

默认三点路径积分，独立比较其他阶次的功误差。Newton 的常刚度矩阵只用作搜索方向；停止条件始终是实际非线性残差。需要时重新组装精确残差切线，不增加质量、刚度或阻尼。

末端速度用同一质量内积施加夹具瞬时速度。先指定固定分量 `delta_fixed`，再解 `M_ff delta_free=-M_fc delta_fixed`；`impulse_end=M delta` 的自由分量为零。分别保存末端做功和约束动能损失，验证 `ΔK=W_end-loss`。反力是 `(impulse_mid+impulse_end)` 对单位右夹具提升的对偶除以 dt；保持阶段也输出反力，不平滑，不删首步。

每步保存材料、稳定化、动能、材料/惯性/末端反力、外功、路径功误差、真实残差、detF、约束、动量、传递能量变化和失败状态。账本闭合与能量守恒、时间误差分开判定。外力接口目前只接受一个时间步内冻结的广义力；E 的状态相关力及切线需要后续单独接入。

## 移动网格与事务

`transfer.encode` 把 `(v,C)` 按仿射 P2G 写入移动粒子所覆盖的 Q1 网格，再显式保留 G2P 无法表达的 `residual_v/residual_C`。解码必须包含这些残余，随后从完整速度场恢复实际提交的广义速度；不能直接保留旧系数绕过回传检查。位置/F 仍由权威 `q` 给出。

这是“网格＋残余历史”的完整表示，动力学求解仍在参考广义空间中。原始网格和残余之间未声明动能正交，不能把二者动能简单相加。网格原点改变与实际跨单元事件单独记录。该实现验证 C7 的历史保存，不是已经完成 Eulerian 网格上的力学求解器。

`AVF.step` 在局部试探对象上计算。材料子状态从深拷贝开始，由 `prepare_children` 返回一个新字典；校验回调也只收到拷贝。直到求解、材料、约束、传递、子状态和账本全部成功，才一起提交 q、速度、时钟、预测器、网格包及子状态。回调不得修改外部资源；真正 B/E 外部对象的提交需要未来适配。失败仅追加拒绝诊断，已提交状态不变。

`migration.migrate` 在共同物理探针上拟合位置、F、速度及梯度，检查动量、约束和材料/动能跳变，只返回可接受的候选。嵌套扩充和可逆缩减可保持历史；丢失已激活模式的粗化被拒绝。空间与积分规则的迁移分开记录。目前它是离线迁移审计，尚未接入 A/B 的动态自适应链。

## 重放

从仓库根目录使用 `.venv/bin/python`。输出目录必须尚不存在；大轨迹建议放在 `/root/autodl-tmp/mpm-lite/research_c/`，在 C 成果目录中保留链接。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/research_c -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.research_c.run diagnostics --output /root/autodl-tmp/mpm-lite/research_c/new-diagnostics
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.research_c.run legacy --output /root/autodl-tmp/mpm-lite/research_c/new-legacy
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.research_c.run cycles --output /root/autodl-tmp/mpm-lite/research_c/new-cycles
```

`cycles` 先运行单因素短窗和四档 dt 短窗，通过后才跑 1.6 秒完整四阶段。dt 为 500/250/125/62.5 微秒，实际步数 3200/6400/12800/25600。全过程 PK1 保存在 `PK1.npy`，原始逐步账本在 `steps.jsonl`，共同阶段端保存 q、速度、x、F、v、C、PK1 和纤维应变快照。协议在计算前冻结，结果包含源码、输入与成果哈希。不能覆盖已有输出。

数值检查采用有量纲的绝对下限与相对条件。新模型求解容差为 `1e-11+1e-8*scale`，不照搬旧案例的 `1e-17`；时间比较沿用方案的 2%，近零反力/应力分别使用预先声明的 `1e-5 N`、`1e-4 Pa` 绝对下限。失败不事后放宽协议。

本轮是 CPU 小模型的阶段交付。完整 v22/A 空间、B 压缩与重分组、变几何 APIC 度量、Eulerian 网格力学求解、CUDA 及 A×B×C 联合空间精度均需各自验收，不能从本轮小模型推断通过。
