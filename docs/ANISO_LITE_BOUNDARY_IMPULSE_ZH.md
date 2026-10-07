# 原 Lite：边界冲量回传修复与非均匀梯度测试

本轮落实第一批工作：可关闭的边界冲量修复、实际传递内核的单次冲量测试、独立粒子动量验收，以及非均匀解析速度场测试。使用 CPU float64；梯度算法保持原实现。本轮短轨迹用于验证修复，不替代完整加载、空间精度或 CUDA 验收。

## 1. 修改与适用范围

`AnisotropicLiteImplicitSolver(..., boundary_impulse_transfer=True)`、`Config(boundary_impulse_transfer=True)` 和命令行 `--boundary-impulse-transfer` 均可启用修复。默认关闭，保留原有基线行为。新选项限定中心积分路径；在其他积分路径启用会在推进前报错。

C2G 内核可额外输出施加边界投影前的 `grid_v_raw`。原始速度在质量归一化后、边界投影前保存；每步刷新，按活动块数分配，避免稀疏块编号变化复用旧速度。开启时额外保存每个活动网格节点一个三维速度；关闭时不分配该缓冲。原 `MPMSolver` 的包装接口原有调用方式保持兼容，默认数值路径不启用捕获。

隐式求解仍使用边界投影后的 `grid_v`。仅在 G2C 计算 FLIP 增量时，将参考速度从 `grid_v` 换为 `grid_v_raw`：

\[
\Delta v_i=v_i^{new}-v_i^{raw}
=(v_i^{new}-v_i^{bc})+(v_i^{bc}-v_i^{raw}).
\]

PIC 速度、中心速度梯度、位置推进与材料更新公式均保持原实现。后续步的解会因粒子速度修复而改变，这是修复动力学的预期效果。

在本轮完整插值支持、均匀密度的传递条件下，旧路径满足

\[
p_p^{n+1}=p_g^{new}-\beta(p_g^{bc}-p_g^{raw}),
\]

修复后预期 \(p_p^{n+1}\simeq p_g^{new}\)。这只涉及初始边界投影的冲量，不代表丢失了全部边界作用力的 \(\beta\) 倍，也不构成任意截断支持、任意密度场下的守恒证明。

## 2. 独立粒子验收

能量账本额外记录步前、步后实际粒子总动量，以及粒子终态与最终网格的总动量差。标准无重力双夹具拉伸增加：

\[
e_p=\left\|\boldsymbol R_L+\boldsymbol R_R
-\frac{p_p^{n+1}-p_p^n}{\Delta t}\right\|_2.
\]

`particle_momentum_balance_error_norm` 是三维范数；`particle_momentum_balance_error` 保留 x 分量，另有 x/y/z 独立字段。夹具力从网格离散内力与原始到最终网格的惯性项计算；粒子动量直接读取粒子数组，因此不会用网格通过代替粒子通过。

`particle_grid_momentum_gap_norm` 的单位为 kg·m/s；粒子合力误差单位为 N。两者不能混用。旧 `momentum_balance_error` 网格 x 分量诊断保留。

该夹具验收沿用现有拉伸函数的几何及无重力假设，不是所有边界、体力和阻尼场景的通用外力汇总接口。

## 3. 测试设计与冻结协议

完整测试清单和源码哈希见 [protocol.json](results/lite-aniso-mainline/v3/protocol.json)。正式实验前冻结，运行器拒绝覆盖已有案例或使用已变化的冻结源码。

新增 8 项测试（含本轮发现的快照保存回归），加上上一轮 39 项检查，共 47 项：

- 实际 P2C/C2G/G2C/C2P 内核单次投影：\(\beta=0,0.9,1\)，逐一比较修复开关；额外用独立主机重建核对原始网格速度。隔离试验没有材料求解与位置推进。
- 静止夹持、移动夹持、法向滑移边界；滑移测试同时确认切向分量不被新增冲量改变。
- 无边界投影时开关结果逐位相同；纯 PIC 的粒子速度逐位相同。
- 4 步 F45 检查独立粒子合力、网格合力与粒子/网格动量；之后人为耗尽 Newton 次数，验证粒子 x/v/F 与时钟不提交。
- 不支持的积分模式在推进前拒绝。
- 实际运行器保存的初始 F 必须为独立单位阵快照，并与非零变形终态不同。
- 常量与全三维仿射场，包含两种粒子相对网格偏移。
- 正弦非均匀场，两种波数、两种偏移、三档网格，检查误差下降及独立插值导数。

短程对照为 ISO/F45 × dt=0.001/0.0005 s × 开关，共 8 条、1200 步。均使用上一轮原速加载曲线，T=0.5 s、loading_speed=0.01 m/s，只运行前 0.1 s，未缩短加载命令周期。非线性反力容差保持 1e-7 N。

预先固定的门槛：网格 x 合力误差 ≤1e-7 N；修复开启时粒子三维合力误差 ≤1e-7 N、粒子/网格总动量差 ≤1e-14 kg·m/s；夹具网格速度误差 ≤1e-12 m/s；平均粒子夹具位移误差 ≤0.001 m；F 的行列式为正；Newton 达到有效残差目标。关闭修复的粒子误差作为负对照保留，不要求它通过修复目标。

关闭开关的轨迹还与 v2 相同案例的前 0.1 s 比较反力和重合时刻的 x/F 快照，门槛 1e-12，避免基线悄然变化。F 的旧基线比较只用 t>0 的有效帧，原因见下一段。

首次批次的快照核对发现：旧 `aniso_mainline.run_case` 保存初始 F 时没有 `.copy()`，CPU 下持有实时数组视图，结束时第 0 帧 F 变为终态 F。后续帧早已使用拷贝，反力、位置、逐步标量与 t>0 的 F 帧均不受影响。本轮修复初始保存并补充回归；首次分析的失败结果、完整日志、轨迹和冻结协议保留于 [pre_snapshot_fix](results/lite-aniso-mainline/v3/pre_snapshot_fix/summary.json)，随后重新冻结和执行全部检查与短轨迹。v1/v2 原文件保持原样，但其第 0 帧 F 不应当作为初态使用；可以由本轮明确的无预应变配置重建初态单位阵，不能把终态数组伪装成原始初态记录。

## 4. 非均匀解析场：一致性与精度分别测量

给定网格速度场

\[
v_x=a\sin(kx),\quad v_y=v_z=0,\qquad
G_{xx}^{exact}=ak\cos(kx),\quad a=0.01\ \mathrm{m/s}.
\]

在 1 m 域内分别使用 1、2 个周期，单轴 8/16/32 个单元；粒子采用中心插值坐标的小数偏移 0.13、0.73，覆盖夹具位置附近与单元内部，支持完整。此制造解试验直接指定速度，不施加真实夹具约束、不运行材料方程，因此不能代替夹持问题的空间收敛。

生产 G2C/C2P 内核提供中心梯度、粒子梯度和 PIC 速度；独立主机程序对中心速度插值求导，并用中心差分再次校验该导数。比较三项：

\[
G_p-G^{exact}(x_p),\qquad
\nabla v_{PIC}(x_p)-G^{exact}(x_p),\qquad
G_p-\nabla v_{PIC}(x_p).
\]

仿射场中三者应一致；非均匀场中既看一致性，也看相对解析真值的误差。不能把三者差异一律当作代码错误，不能仅因梯度与位置插值导数一致就认定更准确。

## 5. 实测结果

最终 47 项测试全部通过，无跳过。最终 8 条短轨迹共 1200 步完成，冻结的本批验收全部通过；首次 1200 步及其快照比较失败记录另行归档。这不是完整主线精度验收。

| 案例 | dt (s) | 修复关闭：粒子合力最大误差 (N) | 修复开启：粒子合力最大误差 (N) |
|---|---:|---:|---:|
| ISO | 0.001 | 0.00959662 | 6.89186e-10 |
| ISO | 0.0005 | 0.0110718 | 9.5395e-10 |
| F45 | 0.001 | 0.00838936 | 6.581e-10 |
| F45 | 0.0005 | 0.00958083 | 5.71111e-10 |

修复开启的最大粒子/网格总动量差为 **3.795e-19 kg·m/s**。全部网格夹具速度检查与正 J 检查通过。关闭开关的反力序列与 v2 对应前缀逐值相同；每例 5 个重合位置帧和 4 个有效 F 帧也逐值相同（旧第 0 帧 F 已排除并明确标注）。

![边界冲量与粒子合力对照](results/lite-aniso-mainline/v3/boundary-impulse-comparison.png)

短程 t=0.01..0.1 s 的粗细时间步反力 RMS 差（分母为细步 RMS，底限 0.001 N）为：

| 案例 | 关闭 | 开启 |
|---|---:|---:|
| ISO | 12.47% | 3.49% |
| F45 | 10.53% | 9.57% |

ISO 在该短区间显著改善；F45 仍有约 9.57% 的时间差。这里没有执行第三档时间步或完整加载，不能用短程数字替代 v2 的 t=0.05..0.5 s 验收，也不能宣称全局精度已经收敛。反力幅值本身也明显改变，是粒子速度回传影响后续动力学的结果；没有连续参考解时，不能将幅值下降直接等同于精度提高。

正弦制造解在偏移 0.13 时的粒子梯度相对 RMS 误差及幅值增益如下；另外偏移 0.73 的完整数值一并保存在 JSON 中。

| 域内周期数 | 单轴单元数 | 梯度相对 RMS 误差 | 幅值增益 |
|---:|---:|---:|---:|
| 1 | 8 | 5.932% | 0.941239 |
| 2 | 8 | 21.389% | 0.790735 |
| 1 | 16 | 1.505% | 0.984982 |
| 2 | 16 | 5.894% | 0.941493 |
| 1 | 32 | 0.378% | 0.996222 |
| 2 | 32 | 1.504% | 0.984990 |

解析插值导数与有限差分的最大偏差为 3.857e-11 1/s；生产 PIC 速度与独立主机插值的最大差为 0.000e+00 m/s。常量、仿射场通过。两种非均匀场的粒子梯度误差在每次加密后均下降到上一级的 40% 以下，显示本测试中的明显平滑及随加密改善。

![非均匀梯度精度](results/lite-aniso-mainline/v3/analytic-gradient-convergence.png)

这些正弦场中，直接取 PIC 位置插值的导数反而比当前插值中心梯度具有更大的解析误差。例如 8 单元、1 周期、偏移 0.13 时，两者 RMS 误差分别为 0.013552 与 0.002311 1/s。因此本批保留原梯度算法；下一候选需要兼顾精度与离散功的一致性，不能只把两种梯度强制改成相等。

完整数值见 [summary.json](results/lite-aniso-mainline/v3/summary.json)、[analytic-gradients.json](results/lite-aniso-mainline/v3/analytic-gradients.json)、[artifact-check.json](results/lite-aniso-mainline/v3/artifact-check.json)。


## 6. 使用与复现

在仓库根目录运行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso \
  --headless --scene tensile --grid 9 --device cpu --steps 100 \
  --dt 0.001 --smooth-loading --fiber-angle 45 \
  --reaction-force-atol 1e-7 --boundary-impulse-transfer
```

删除最后一个开关即运行旧传递基线。现有 GUI 重建配置也会携带命令行选项；本轮验证了实际 headless CLI 两步，没有声称浏览器交互验证。

重新运行整批时使用新的输出目录，依次执行：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.aniso_boundary_impulse freeze --output /tmp/aniso-boundary-repeat
.venv/bin/python -m benchmarks.aniso_boundary_impulse tests --output /tmp/aniso-boundary-repeat
.venv/bin/python -m benchmarks.aniso_boundary_impulse probes --output /tmp/aniso-boundary-repeat
.venv/bin/python -m benchmarks.aniso_boundary_impulse run --output /tmp/aniso-boundary-repeat --jobs 2
.venv/bin/python -m benchmarks.aniso_boundary_impulse analyze --output /tmp/aniso-boundary-repeat
```

复现运行器的关闭基线检查需要仓库保留的 v2 成果。最终批次每条轨迹保存逐步 JSONL、CSV、21 帧独立 x/F 和运行状态；解析场保存 16 份数值记录。修改前生产文件存于 v3/before，v1/v2 原始成果均保留。

## 7. 后续边界

本批修复线动量回传，不自动保证能量守恒、角动量守恒、静态刚度精度或完整加载反力精度。PIC/FLIP 混合及中心历史压缩仍会影响能量与应力。

下一步应先对修复开启的 ISO/F45 做完整 T=0.5 s 与第三档时间步对照，再决定是否扩展全部方向与慢加载。梯度候选改法必须同时通过仿射再现、非均匀解析精度、历史传递及材料功的一致性检查。直接把中心 F 覆盖回粒子或只改反力定义，都不属于本轮修复方案。
