# 分离 APIC 仿射系数与材料梯度，并验证增量回传

**满足本轮 F45 预定数值门槛，但尚未建立可靠时间收敛。** 已完成 APIC 系数 C 与材料速度梯度 L 的独立存储、可选增量回传、61 项回归、420 次有效冻结传递、三档完整 F45 及一条旧方式复现，共 4000 个加载步。新模式最细相邻总反力差为 2.525%，绝对相邻差缩减比 rho=0.9951；生产默认未切换。

## 1. 状态含义和实际改动

| 状态 | 用途 | 更新方式 |
|---|---|---|
| `ptc_C` | 携带局部仿射速度，供下一步 P2C/APIC 使用 | 可选 overwrite 或 incremental |
| `ptc_L` | 本步求解速度场在粒子位置的梯度，供材料历史更新 | 对求解后中心梯度插值 |
| `ptc_F` | 材料变形历史 | $(I+\Delta t L_p)F_p$ |

`ptc_C` 是旧存储 `ptc_G` 的兼容别名；`ptc_L` 是独立数组。中心积分路径成功提交后保存 L；种子追加分别初始化并保留状态。`ptc_G` 对外含义明确为 APIC 系数，材料诊断改读 L。内存统计计入新数组。

新增配置 `apic_transfer='incremental'`，命令行 `--apic-transfer incremental`；默认仍为 `overwrite`。增量模式要求 APIC、中心积分和边界冲量修复同时开启。新状态仅在 Newton 成功后提交，失败步不改变 C/L/F。位置和粒子速度的原更新公式、中心材料积分、残差、切线和反力停止条件保留。

## 2. 回传公式与一致性依据

以本步粒子旧位置和同一组插值权重 S 定义

\[
L_p^{raw}=S\nabla v_g^{raw},\qquad L_p^{new}=S\nabla v_g^{new}.
\]

这里的梯度先在中心计算，再插值给粒子。raw 是施加本步初始边界投影之前的网格速度，因此差值包含边界冲量造成的梯度变化。

APIC 的两种候选值为

\[
C_p^{FLIP}=C_p^n+(L_p^{new}-L_p^{raw}),\qquad C_p^{PIC}=L_p^{new}.
\]

新回传采用与粒子速度相同的 beta：

\[
C_p^{n+1}=\beta C_p^{FLIP}+(1-\beta)C_p^{PIC}
=L_p^{new}+\beta(C_p^n-L_p^{raw}).
\]

材料使用独立的当前速度梯度：

\[
F_p^{n+1}=(I+\Delta t L_p^{new})F_p^n.
\]

因此修改 APIC 数值传递强度不会直接把混合后的 C 当成材料梯度。它仍会通过下一步前向传递改变后续速度和应力，这正是本轮需要完整加载验证的反馈。

在固定几何、没有网格冲量的隔离条件下，若原 PIC 传递算子为 T0，新 C/v 联合更新为

\[
y^{n+1}=[\beta I+(1-\beta)T_0]y^n,\qquad y=(v,C).
\]

采用 beta=exp(-kappa dt) 后，整个 C/v 状态更新趋近单位映射：

\[
T(dt)=I+\kappa\Delta t(T_0-I)+O(\Delta t^2).
\]

旧方式只有速度混合随 dt 减弱，C 每步直接覆盖。新方式为两种传递状态共同设定时间尺度；这不保证有限 dt 下严格满足半步组合恒等，也没有把一般的中心往返 W S 变成单位矩阵。

## 3. 前置检查与实验记录

61 项必要回归全部通过、无跳过，其中 6 项新增检查覆盖：C/L 独立、F 使用 L、独立 NumPy 增量公式、beta=0/0.9/1、移动/滑移边界冲量、常量与三维仿射保持、纯 FLIP 零冲量保持、种子追加、非法组合和失败不提交。原有力停止与梯度诊断也纳入回归。

首批冻结探针被独立公式拒绝：测试运行器将 C 的混合比例留在默认 0.9，而速度已使用各档 beta。只修复探针参数赋值；该批协议、输出和当时运行器原样保存在 `pre_probe_beta_fix/`，未进入完整加载。随后重新冻结协议、重跑 61 项回归及 420 次探针。

最终探针的独立递推最大差 1.388e-16，最大总动量误差 4.278e-20 kg·m/s，位置/F 始终精确不变。

冻结探针依旧使用 v4 中段 F45 几何，位置/F 不动、关闭受力与边界，CPU float64。每个场以 20/40/80 次传递代表同一标称 0.02 s，实际内核 dt=0。这是传递检查，不是加载模拟。

| 场 | 次数 | 速度范数保留 | C 范数保留 | L 范数／初始梯度范数 |
|---|---:|---:|---:|---:|
| sine_x1 | 20 | 0.924734 | 0.898673 | 0.846171 |
| sine_x1 | 40 | 0.923525 | 0.895494 | 0.840351 |
| sine_x1 | 80 | 0.922910 | 0.893873 | 0.837374 |
| sine_x2 | 20 | 0.529923 | 0.483244 | 0.341700 |
| sine_x2 | 40 | 0.527581 | 0.477482 | 0.333902 |
| sine_x2 | 80 | 0.526417 | 0.474618 | 0.330065 |
| sine_45 | 20 | 0.959396 | 0.817512 | 0.763707 |
| sine_45 | 40 | 0.958863 | 0.814731 | 0.759607 |
| sine_45 | 80 | 0.958594 | 0.813334 | 0.757557 |

三种场的速度/C 相邻差缩减比均约 0.50–0.51。新旧模式的直接比较及动能、角动量数据见 [frozen-comparison.json](results/lite-aniso-mainline/v7/frozen-comparison.json)。

![新旧传递的相邻时间差](results/lite-aniso-mainline/v7/frozen-comparison.png)

## 4. 完整 F45 三档

沿用 v5：网格 9、dx=0.125 m、192 粒子、mu=10、lambda=20、kf=200、纤维角 45°；center/particle_resample/variational/mean_tensor、无稳定化、边界冲量修复开启，反力停止容差 1e-7 N。

加载 0.5 s，loading_speed=0.01 m/s，平滑余弦终点位移 0.005 m。三档 dt=0.001/0.0005/0.00025 s，beta=0.9/0.9486832980505138/0.9740037464252967。新增增量模式 500+1000+2000 步，另以 overwrite 重跑粗档 500 步，共 4000 步。

旧三档直接读取 v5 不可变记录；新增粗档 overwrite 用于检验单纯拆分 C/L 后行为不变。全部记录采用 CPU float64，最多两个单线程模拟进程，CUDA 未运行。v1–v6 归档文件哈希全部保留；改动前源码备份见 `source-before.zip`。

反力比较统一到粗档物理时刻，在 0.05–0.5 s 做梯形积分 RMS。相对差以较细曲线 RMS 为分母、底限 0.001 N。预定验收要求最细相邻总反力差 ≤5%，且绝对相邻差下降；同时满足物理和诊断检查。各分量百分比不能相加。

| 模式 | 分量 | 粗→细相对差 | 细→最细相对差 | 粗→细 RMS (N) | 细→最细 RMS (N) | rho |
|---|---|---:|---:|---:|---:|---:|
| overwrite | 总反力 | 9.299% | 10.189% | 0.00157051 | 0.00191549 | 1.2197 |
| overwrite | 弹性 | 7.783% | 8.757% | 0.00129087 | 0.00159152 | 1.2329 |
| overwrite | 惯性 | 28.529% | 29.032% | 0.000285294 | 0.00033523 | 1.1750 |
| incremental | 总反力 | 2.602% | 2.525% | 0.000368056 | 0.000366269 | 0.9951 |
| incremental | 弹性 | 2.431% | 2.399% | 0.000351034 | 0.000354673 | 1.0104 |
| incremental | 惯性 | 2.022% | 1.412% | 2.02183e-05 | 1.41208e-05 | 0.6984 |

![完整三档反力](results/lite-aniso-mainline/v7/affine-reaction.png)

| 轨迹 | 最大粒子合力误差 (N) | 最小粒子 J | 终点反力 (N) | 传递动能净变化 (J) |
|---|---:|---:|---:|---:|
| incremental-coarse | 6.433e-10 | 0.99999154 | 0.01892901 | -6.02538e-06 |
| incremental-fine | 8.006e-10 | 0.99998858 | 0.01923320 | -7.32574e-06 |
| incremental-finest | 5.562e-10 | 0.99998652 | 0.01953476 | -8.69689e-06 |
| anchor-coarse | 6.581e-10 | 0.99998192 | 0.02004116 | -1.02104e-05 |

物理检查：通过；梯度及历史诊断：通过。传递动能净变化由 P2C/C2G/G2P 三段相加，不是反力误差占比，完整分段及载荷功见 artifact-check.json。

每条轨迹保存 21 帧，初始 F 独立保留；四条轨迹共 16 份原始快照。离线从旧粒子位置、原始/求解后网格速度和旧 C 重算 C/L/v/F/x，避免仅验证相同实现自身。

16 份快照独立复算最大差 4.857e-17；旧方式粗档反力最大差 0.000e+00 N，位置/F 全帧最大差分别为 0.000e+00/0.000e+00。历史文件 568 个哈希保持不变。

## 5. 影响与下一步

相较已标定 FLIP 的旧回传，新模式最细两档绝对反力差变化幅度为降低 80.88%；相对差从 10.189% 变为 2.525%。这是时间步间一致性的比较，不是连续真解误差比例。新模式观测阶 p=0.007，只反映当前三档趋势。

总反力绝对相邻差仅下降约 0.486%，rho 约 0.995、观测阶接近 0；弹性分量的绝对相邻差反而增加约 1.04%。因此虽然严格按预定条件通过，本轮还没有建立可靠的渐近时间收敛，不能将反力差小于 5% 解读为连续真解误差小于 5%。下一步优先补第四档时间步，并继续定位中心材料历史重采样；随后以空间加密和其他纤维方向确认适用范围。

本轮建立了 C/L 状态职责分离与一种可关闭的联合增量回传。默认继续使用 overwrite；需要显式选择 incremental。完成时间一致性验收也不等于已证明反力接近连续真解；空间网格、粒子数、中心材料压缩、其他方向与加载速度还需要独立验证。

动量验收针对总线动量。补充冻结 45° 场检查中，新模式最大角动量变化约 7.89e-8 kg·m²/s，旧模式约 1.27e-6 kg·m²/s；新模式仍未严格保持角动量。新 C 会改变 APIC 仿射动能，能量账本使用 C；没有给它附加严格耗散或守恒保证。应同时观察分段能量与载荷功，避免只凭反力曲线接近就认定所有传递性质均已修复。

## 6. 文件与复现

实现：[affine_transfer.py](../engine/aniso_phase1/affine_transfer.py)、[solver.py](../engine/aniso_phase1/solver.py)。测试：[test_aniso_affine_transfer.py](../tests/test_aniso_affine_transfer.py)。运行器：[aniso_affine_consistency.py](../benchmarks/aniso_affine_consistency.py)。

[冻结协议](results/lite-aniso-mainline/v7/protocol.json)、[汇总](results/lite-aniso-mainline/v7/summary.json)、[独立复算](results/lite-aniso-mainline/v7/artifact-check.json)、[逐快照检查](results/lite-aniso-mainline/v7/independent-snapshots.json)、[隔离探针](results/lite-aniso-mainline/v7/frozen-summary.json)。

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m benchmarks.aniso_affine_consistency freeze --output /tmp/affine-repeat
.venv/bin/python -m benchmarks.aniso_affine_consistency tests --output /tmp/affine-repeat
.venv/bin/python -m benchmarks.aniso_affine_consistency frozen --output /tmp/affine-repeat
.venv/bin/python -m benchmarks.aniso_affine_consistency run --jobs 2 --output /tmp/affine-repeat
.venv/bin/python -m benchmarks.aniso_affine_consistency analyze --output /tmp/affine-repeat
```

分析退出码 2 表示验收未通过；0 表示本轮预定验收通过。输出不覆盖已有案例。独立检查脚本 `v7/check_results.py` 可复制到新输出目录，以 `PYTHONPATH=.` 从仓库根目录运行。

单独运行新模式粗档：

```bash
.venv/bin/python -m demos.aniso --headless --scene tensile --grid 9 \
  --device cpu --dt 0.001 --steps 500 --fiber-angle 45 \
  --smooth-loading --loading-time 0.5 --loading-speed 0.01 \
  --reaction-force-atol 1e-7 --boundary-impulse-transfer \
  --flip-ratio 0.9 --apic-transfer incremental
```
