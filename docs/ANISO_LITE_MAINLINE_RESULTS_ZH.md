# 原 MPM Lite 各向异性主线：实施与验收记录

2026-09-27。执行依据：[交付计划](ANISO_LITE_MAINLINE_DELIVERY_PLAN_ZH.md)。

**本轮实施和有限验收已完成，但固定预设整体未通过。** 35 项必要检查通过；八条轨迹共 6000 步完整完成，沿加载方向的纤维效应可分辨。三组反力时间敏感性超出 5%，部分动量绝对误差也超出预设门槛。

## 推荐入口与复现

复用 `demos.aniso.Scene` 和 `AnisotropicLiteImplicitSolver`。后者继承原 `MPMSolver`，默认旧入口没有被替换；不是独立 `SpatialMPM`，也不使用粒子或 group4x8 材料积分。

在仓库根目录，用现有虚拟环境执行单组完整无头轨迹：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m demos.aniso \
  --scene tensile --device cpu --grid 9 --dt .001 \
  --kf 200 --fiber-angle 0 --loading-speed .01 --loading-time .5 \
  --smooth-loading --history-mode particle_resample --direction-model mean_tensor \
  --quadrature center --stabilization none --flip-ratio .9 --linear-solver pcg \
  --headless --steps 500 --csv /tmp/lite-aniso-F0.csv
```

ISO 使用 `--kf 0 --fiber-angle 0`；F45/F90 使用 `--kf 200 --fiber-angle 45/90`。细步长使用 `--dt .0005 --steps 1000`，物理终点仍为 0.5 s。这里只运行平滑加载的单调半周期，没有保持或卸载段。

完整验收入口（新目录，避免覆盖本次证据）：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.aniso_mainline freeze --output /tmp/lite-aniso-reproduce
.venv/bin/python -m benchmarks.aniso_mainline tests --output /tmp/lite-aniso-reproduce
.venv/bin/python -m benchmarks.aniso_mainline run --output /tmp/lite-aniso-reproduce
.venv/bin/python -m benchmarks.aniso_mainline analyze --output /tmp/lite-aniso-reproduce
```

`run --case F0` 可以只运行一组的两个步长；缺少其余组时总验收仍不通过。源码与冻结哈希不一致时拒绝运行；已有结果不覆盖。`analyze` 返回 2 表示完成分析但未通过验收，不能把运行成功当成验收通过。

已有 Viser 查看入口：将单组命令中的 `--headless --steps 500 --csv ...` 换为 `--viser-port 8080`，打开打印的地址。物理配置一致；点击“播放”执行现场模拟。这不是已保存轨迹的逐帧回放；超过 0.5 s 后既有加载函数会进入卸载，超出本次验收范围。离线代表帧另见 `frames.npz` 和场景图。

修改方向、刚度和加载设置后，点击“应用参数并重置”。既有 `Demo.rebuild` 释放旧场景并创建新的 `Config/Scene`，不延续旧历史。GUI 控件本轮未改，不额外声称浏览器交互自动化已验证。显示位移放大只改点位置，纤维使用真实 `F a0 / |F a0|`；本次离线场景图位置放大为 1。

最小 Python 创建入口：

```python
from demos.aniso import Config, Scene
scene = Scene(Config(scene='tensile', grid=9, dt=.001, kf=200.,
                     fiber_angle=45., smooth_loading=True, linear_solver='pcg'), 'cpu')
assert scene.step()
print(scene.loading_rows[-1])
```

## 固定协议与支持边界

[protocol.json](results/lite-aniso-mainline/v1/protocol.json) 在正式验收前冻结。四组两档仅 `kf`、`fiber_angle`、`dt` 不同。单位约定 m、s、kg、Pa、N。

- 3D、单材料、弹性、均匀单纤维参考方向；μ=10 Pa、λ=20 Pa、k_f=0/200 Pa。
- center 积分、mean_tensor、particle_resample、variational、无稳定化，带曲率保护/正定近似回退的 PCG 和原势能线搜索。
- 网格 9³、dx=0.125 m；192 个粒子、每格每轴 2 个采样；体积 0.046875 m³、密度 1 kg/m³、质量 0.046875 kg。
- APIC 开、FLIP=0.9、重力 0、damping=1；无额外人工阻尼。
- x≤0.25 m 与 x≥0.75 m 的网格夹具。右端命令 `u=.5*.01*.5*(1-cos(pi*t/.5))`，终点 0.005 m。夹具是固定空间的网格速度约束，不能等同精确材料面夹持。
- CPU float64；唯一可见 A100 启动时利用率 100%，本次不占用该 GPU，不声明本轮 CUDA 验证通过。
- 工作区无 Git 元数据，以协议中的 SHA-256 标识执行源码；修改前 solver 副本保存在 `before/solver.py`。

时间敏感性预设：在 0.05–0.5 s 的共同粗步物理时刻插值细步反力，以梯形积分定义 RMS；相对误差分母为 `max(R_fine_RMS, 0.001 N)`，门槛 5%。角度或开关的细步曲线差异须大于两条曲线绝对时间扰动之和的 2 倍；至少一对可分辨即可，允许横向纤维与 ISO 接近。两档时间步不证明收敛阶。

附加预设：动量平衡误差 ≤1e-7 N，网格夹具速度误差 ≤1e-12 m/s，夹具区域粒子平均相对位移与命令的绝对差 ≤0.001 m，Newton 残差 ≤1e-10，粒子及中心 detF>0。粒子平均位移并非精确边界约束，单独报告其偏差。

本模型二次纤维能在压缩时也响应，不能称为仅受拉纤维。不承诺任意混合方向、多材料、各向异性塑性、损伤、断裂、空间收敛、长期稳定或性能提升。

## 执行链与局部修复

`Config.params` 将 XY 角度转为方向，`AnisotropicMaterialParams` 归一化并构造 A0。`Scene` 显式选中中心积分子类并播种粒子 F=I、参考 A0、速度和夹具。

`_transfer_to_centers` 调用各向异性 P2C，继而 `resample_history` 用粒子体积权重构造本步中心 F/A0。中心快照每步重建；Newton/PCG 内不重新访问粒子历史。CPU 势能、应力与设备切线使用同一 μ、λ、k_f、A0 和 trial F。这里直接计算各向异性应力，避开旧 Lite 的各向同性应力反演。

C2G 使用原 Lite 传递；Newton 内执行变分残差、精确切线，负曲率时仅将求解方向改为正定近似，再用原势能线搜索。成功后调用原 `lite_g2c_kernel`、`lite_c2p_kernel`；内部材料记录为 `Material.elastic`，无旧塑性返回映射。随后提交中心状态，最后递增步数/时间。

失败时不更新粒子 x/v/F/G/A0、模拟时间、成功步数和能量历史行。推荐模式在 Newton 前重采样得到的中心快照是由粒子导出的本步工作状态，并非独立、永久不变的粒子历史；失败 trial 回滚到该快照。不能把它与上一成功步、尚未重采样的中心数组逐元素等同。

本轮局部修复仅在各向异性子类：播种前验证方向、F、速度梯度、密度及体积；非法输入不再先追加粒子。明确拒绝非零材料索引和继承的额外 `add_material` 调用。原 `MPMSolver` 文件不改。保留研究模块及其成果。

## 必要检查与预设容差

既有测试直接复用，新增缺口集中在 `tests/test_aniso_mainline.py`。完整逐项清单与源码哈希见协议；[tests.json](results/lite-aniso-mainline/v1/tests.json) 记录 **35 项通过、0 项失败、0 项跳过**。数值误差另见 [numerical-evidence.json](results/lite-aniso-mainline/v1/numerical-evidence.json)：材料能量导数最大归一化误差 7.52e-10；生产势能梯度与精确切线最小差分误差分别为 6.40e-11、2.72e-11；精确切线对称误差 1.73e-16。重复生产残差/切线求值后，粒子历史及冻结中心 F/A0 保持不变。

| 检查 | 证据位置 | 既定门槛/说明 |
|---|---|---|
| 初态、无向性、归一化、非法 J、纤维切线 | test_aniso_phase1_material | 初态绝对 1e-12；纤维切线 rtol 2e-5、atol 2e-7 |
| CPU Warp 残差、切线及重复求值无副作用 | test_aniso_phase1_warp | 残差 rtol/atol 1e-10；切线 rtol 1e-8、atol 1e-9 |
| 生产势能/精确 Hessian | test_aniso_variational 中选定用例 | 归一化差分分别 <1e-7、<1e-8；对称误差 <1e-12 |
| 曲率回退、原势能 Armijo、非法状态拒绝 | 同上 | 负曲率被识别，修改切线正定，原势能下降，残差 <1e-9 |
| 能量导数、解析 1:1/4:0 纤维增量、PK1 客观性 | test_aniso_mainline | 导数误差/max(1,abs(P:H)) <2e-7；纤维增量绝对 1e-11；客观性 atol 1e-10 |
| k_f=0 材料和推荐路径方向无关 | test_aniso_mainline | 三步 x/v/F/反力，rtol 1e-10、atol 1e-12 |
| 推荐模式仿射一步、失败回滚 | test_aniso_mainline | F 绝对 5e-8；粒子历史和时钟精确保持；trial 回到重采样快照 |
| 跨稀疏块历史、旋转传输 | test_aniso_history_reference 两个用例 | F/A/位置误差 <1e-5；其他研究用例不运行 |
| 旧路径特殊初态与旧模式重编号/回滚 | test_aniso_phase1_solver 四个用例 | 指定一步等价 atol 1e-12；不能代替推荐模式一般等价证明 |
| 加载、反力/动量、显示放大与方向 | test_aniso_tensile、test_aniso_demo | 动量误差 <1e-7；显示方向绝对 2e-6 |

CPU Warp 检查不等于 GPU 检查。CUDA 本轮为“未运行”；不运行要求不存在 cuda:1 的测试，不把跳过算通过。

## 正式轨迹与验收结论

八条正式轨迹共 **6000 个成功时间步**，全部运行至 0.5 s；未发生失败步或非法提交。四组粗步各 500 步、细步各 1000 步。

**整体结论：未通过本轮固定预设的全部验收门槛。** 必要功能检查通过，沿加载方向的纤维效应可分辨；但 ISO、F45、F90 的总反力时间敏感性未达到 5%，另有四条轨迹超过预设动量平衡绝对门槛。

| 组别 | k_f (Pa) / θ | 时间相对差 | 绝对 RMS 差 (N) | 细步惯性/总反力 RMS | 5% 门槛 |
|---|---|---:|---:|---:|---|
| ISO | 0 / 0° | 12.066% | 0.00128208 | 60.13% | 未通过 |
| F0 | 200 / 0° | 0.794% | 0.00240409 | 2.51% | 通过 |
| F45 | 200 / 45° | 15.137% | 0.00249113 | 34.86% | 未通过 |
| F90 | 200 / 90° | 12.877% | 0.00154088 | 54.41% | 未通过 |

![反力—位移及分量](results/lite-aniso-mainline/v1/reaction-displacement.png)

图中 ISO/F0/F45/F90 的角度与 k_f 如上表；虚线为粗步、实线为细步。横轴是命令位移，所有组采用相同物理坐标。

![小反力组细节](results/lite-aniso-mainline/v1/reaction-detail.png)

| 细步曲线对 | 反力差 RMS (N) | 预定分辨门槛 (N) | 可分辨 |
|---|---:|---:|---|
| ISO-F0 | 0.294532 | 0.00737233 | 是 |
| ISO-F45 | 0.0066921 | 0.00754643 | 否 |
| ISO-F90 | 0.00155877 | 0.00564591 | 否 |
| F0-F45 | 0.287904 | 0.00979044 | 是 |
| F0-F90 | 0.292997 | 0.00788993 | 是 |
| F45-F90 | 0.00514477 | 0.00806402 | 否 |

F0 与 ISO、F45、F90 均可分辨；其他配对尚不能据此宣称已分辨。真实块体存在侧向自由变形与边界影响，不应套用材料点固定 F 的 1:1/4:0 纤维应力增量比例。

![相同坐标与显示比例的物理场景](results/lite-aniso-mainline/v1/scene-comparison.png)

场景图为细步轨迹 t=0.125/0.25/0.5 s 的 XY 投影，蓝点为真实粒子位置，橙箭头为归一化真实 F a0；位置放大为 1。各组保存含初态在内的 21 帧 x/F/a0，未只保留末帧。

| 轨迹 | 步数 | 最小中心 detF | 最小粒子 detF | 最大求解残差 | 最大动量差 (N) | 最大夹具粒子位移差 (mm) |
|---|---:|---:|---:|---:|---:|---:|
| ISO-coarse | 500 | 0.99998493 | 0.99996766 | 2.564e-12 | 6.961e-09 | 0.1515 |
| ISO-fine | 1000 | 0.99997793 | 0.99997325 | 4.144e-11 | 2.467e-07 | 0.1499 |
| F0-coarse | 500 | 0.99999988 | 0.99999981 | 6.211e-11 | 2.991e-07 | 0.1753 |
| F0-fine | 1000 | 0.99999962 | 0.99999939 | 5.332e-11 | 2.729e-07 | 0.1753 |
| F45-coarse | 500 | 0.99997064 | 0.99996316 | 3.424e-11 | 4.784e-08 | 0.1410 |
| F45-fine | 1000 | 0.99996776 | 0.99995346 | 9.997e-11 | 9.092e-08 | 0.1391 |
| F90-coarse | 500 | 0.99998935 | 0.99997624 | 3.357e-11 | 5.132e-08 | 0.1575 |
| F90-fine | 1000 | 0.99998339 | 0.99998006 | 4.144e-11 | 2.467e-07 | 0.1561 |

所有步的网格夹具速度误差为 0；夹具区域粒子平均位移差最大 0.1754 mm，小于预定 1 mm。这是网格夹持与传递后的粒子运动核对，不是精确材料面边界的证明。

ISO-fine、F0-coarse、F0-fine、F90-fine 的动量平衡绝对误差超过 1e-7 N，最大 2.9911e-7 N；门槛没有放宽。其量级与自由节点力残差一致：如 F0-coarse 在 t=0.444 s 的 Newton 残差为 6.1651e-11，除以 dt 后为 6.1651e-8 N 的自由节点力残差范数；多个节点的合力可大于该范数。求解成功不能代替更严格的合力门槛。

反力时间扰动分解见 [reaction-components.json](results/lite-aniso-mainline/v1/reaction-components.json)。ISO 的弹性反力差 RMS=9.2265e-5 N、惯性反力差 RMS=1.2411e-3 N，后者主导总反力的时间差。当前应按动态响应解释；F0 惯性相对较小仍不构成整组准静态证明。

加载功、各传递阶段能量变化、机械能减加载功，以及每步残差/迭代数，均保留在各轨迹 CSV/JSON 中。没有删除未达标组，也未切换求解器或增加阻尼来掩盖差异。

机器摘要见 [summary.json](results/lite-aniso-mainline/v1/summary.json)：`run_completed=true`、`required_checks_passed=true`、`direction_effect_resolved=true`，但 `time_sensitivity_passed=false`、`trajectory_checks_passed=false`、`accepted=false`。空间精度未验证；CUDA 未运行。

运行记录：四步启动冒烟通过；五项局部修复检查通过；随后冻结协议并执行 35 项必要测试和八条正式轨迹。没有失败修复重跑或扩展扫描。`analysis` 退出码 2 是预设验收未通过，不是数据生成中断。

## 最终验收总表

| 编号 | 结论 | 证据/适用范围 |
|---|---|---|
| A1/A2 | 通过 | 冻结协议、配置差异核对、实际 Lite 6000 步 |
| A3/A4 | 通过（CPU） | 材料、方向、客观性、精确切线及数值误差记录 |
| A5/A6 | 通过 | 关闭纤维方向无关、推荐历史、仿射及失败回滚 |
| A7 | 完整运行通过；附加动量门槛未通过 | 八条终点/步数/逐步状态及上表 |
| A8 | 通过预定“至少一对”判据 | F0 与其余三组可分辨；其他配对未分辨 |
| A9 | 未通过 | 三组时间相对差 >5% |
| A10/A11 | 通过 | 旧默认未修改，指定旧初态回归及入口/复现说明 |
| A12 | 未运行 | 不声明本轮 GPU 验证 |
| A13 | 未运行 | 本轮只验收单调加载半周期 |
| A14 | 未证明/不属本轮 | 空间收敛、长期稳定、性能提升 |

按计划的规模边界结束本轮，交付完整失败结论与可复现结果。若另行开展下一轮，应先针对加载/惯性与时间离散、求解残差对应的反力精度制定新协议；本轮不追加参数或网格扫描。

## 数值限制

中心材料压缩不保证非均匀场下 F_center=F_particle；均值、本构和传递不交换，不能把中心与粒子差异直接视为材料公式错误。均匀参考方向以外的精确混合能量不在本轮范围。

反力是当前离散方程的 actuator-on-body 力，含 C2G 投影与惯性分量。`R/u` 不是静态模量，零位移邻域应跳过。加载功与机械能的差同时包含原传递和边界离散影响；保存这些诊断不代表能量守恒。

历史研究入口保留在 README 后续索引。本轮结论只适用于明确给出的场景与参数，不替代空间精度研究，也不依赖独立 SpatialMPM 的通过结果。
