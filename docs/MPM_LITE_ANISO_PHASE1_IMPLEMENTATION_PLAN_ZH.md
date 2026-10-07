# MPM Lite 阶段一实现计划：固定参考方向的横向各向同性弹性

本轮最终定义、数据与限制见[第二轮定量验收结果](ANISO_QUANTITATIVE_RESULTS_ZH.md)；本文件中的早期阶段记录不替代最终验收。

## 实施进度（2026-09-20）

### 追加：Viser 交互式场景

新增 [`demos/aniso.py`](../demos/aniso.py)，将固定边界纤维块体、均匀三维仿射场和材料点方向响应接入同一 Viser 界面。支持播放/暂停、单步、重置、纤维角度/刚度修改、参考位置和当前纤维方向显示、显示位移放大、0°/45°/90° 应力曲线及材料点卸载。动力学和显示共用无头入口；新增显示/本构对照测试，已验证 CPU 场景和 CUDA 20 步场景，真实 HTTP/WebSocket 控件交互通过。启动方式与验证数据见 [`ANISO_VISER_DEMO_ZH.md`](./ANISO_VISER_DEMO_ZH.md)。下文 40 项测试为阶段一求解器原始验收记录，新 demo 另增加 3 项测试。

### 阶段一求解器实施记录

本轮已开始实施，当前工作区保留用户已有未提交修改，未创建独立 worktree；因此 `e00aabf` 仅作为记录的基线提交，不把当前目录宣称为干净对照分支。
当前运行环境将主 checkout 的 `.git` 视为只读，直接创建分支、写入 index 和提交都会收到只读文件系统错误；主工作区因此继续保留文件级隔离。
已在可写数据盘建立独立实验镜像 `/mnt/0c18569c-b839-4255-bae0-6f48c9fc835b/yin/tmp/mpm-lite-aniso-phase1`，分支为 `feature/aniso-phase1`（基于实验提交 `85bc060`，当前实现快照 `392219b`）；该镜像使用同一套 40 项测试通过。主工作区仍保留用户已有修改。

已完成第一批基础设施：

- 新增 `engine/aniso_phase1/` 独立实验包，提供材料参数、参考方向归一化和结构张量 `A0`；
- 实现固定参考方向二次纤维能量、第一 Piola/Kirchhoff 应力和矩阵自由方向导数；
- 实现中心 committed/trial/commit/rollback 生命周期，CG 可只读 trial 状态；
- 实现结构张量加权平均和 Warp CPU stress/commit kernel；
- 增加 Warp CPU 中心 trial、应力、残差和矩阵自由 matvec kernel，并与 NumPy 参考算子逐项对照；
- `AnisotropicCenterBuffer` 已直接封装上述设备算子，状态数组与 committed/trial 生命周期保持同一所有权；
- 增加设备端粒子 (A_{0,p}) 到中心 (A_{0,c}) 的加权结构张量累加/归一化 kernel，并验证 $a_0$ 与 $-a_0$ 等价；
- 增加结构张量混合度诊断；纯方向的混合度为零，正交粒子方向的中心混合度可检测为正值；
- 独立适配器增加冻结状态 PCG/Newton 参考驱动，收敛步提交中心状态，失败步回滚；
- 新增 `AnisotropicLiteImplicitSolver`，通过独立 sparse-grid kernel 完成中心 P2C、trial 应力、残差、PCG matvec、G2C/C2P 和成功步状态提交；
- 新增 `engine/aniso_phase1/implicit.py` 中心隐式参考算子：trial (F_c)、中心残差和冻结状态矩阵自由作用；
- 新增 `engine/aniso_phase1/adapter.py` 独立中心适配器，完成粒子方向到中心 `A0` 的结构张量传递，以及 trial/commit/rollback 状态门控；
- 增加 `k_f=0` 与旧 `lite_implicit` 的位置、速度和粒子 (F) 逐量回归；
- 新增 `benchmarks/aniso_phase1_isotropic_regression.py`，在 `k_f=0` 下同时比较粒子位置/速度/F、中心速度/应力、残差、Newton/CG 次数和墙钟时间；旧路径仅增加 `last_legacy_implicit_stats` 统计字段。
- 增加均匀仿射中心 (F_c) 解析更新检查和中心/粒子 (F) 差异统计接口；
- 增加无边界三维均匀仿射场端到端收敛回归，并支持逐粒子初速度/速度梯度初始化；
- 增加固定边界纤维增强块体 CPU 冒烟回归，检查 Newton/CG、粒子更新和中心状态有限性；
- 增加端到端中心 P2C 的质量、体积守恒断言；
- 设备端中心 buffer 回滚同时清空 trial Kirchhoff 应力 scratch，避免失败步残留材料状态；
- 失败 Newton 步现在不更新粒子位置/速度、不提交中心历史，并保持 `sim_steps/sim_time` 不变；
- 设备端 trial 更新检测非正 Jacobian，直接失败并回滚粒子/中心状态；
- 修复边界和 legacy implicit residual kernel 的设备参数遗漏，支持 `cuda:N` 非零 ordinal，避免输入数组落在错误设备。
- 稀疏块前缀和每步可能改变块编号；各向异性中心状态现在按全局块坐标设备端重排，保留连续块的 `A0/F` 历史，并记录离开后重新出现的块重激活事件；新增跨块移动回归。
- 求解器运行统计已拆分为 P2C、CG、材料算子、传输、matvec 次数和各向异性状态内存，并覆盖失败步；
- 求解器暴露 `cg_tol`、`cg_atol` 和 `max_cg_iters`，可在长期稳定性实验中提高线性解精度，同时保留旧路径兼容的默认容差；
- 新增材料点、结构张量、有限差分、刚体左旋客观性和状态生命周期检查。
- 增加简单剪切、弹性卸载和负 Jacobian 输入拒绝检查。
- 新增材料点诊断脚本，输出 $0^\circ/45^\circ/90^\circ$ 单轴应力/单位面积力、工程应变、简单剪切能量和卸载应力。
- 新增固定边界纤维增强块体长期探针，记录收敛步数、$\det F_c$、粒子速度、中心/粒子 $F$ 差异及质量体积误差。
- 新增全局中心残差 Jacobian 诊断脚本，检查有限差分误差、重复 matvec 一致性和冻结状态路径。
- Jacobian 诊断增加 16 个随机方向的正定性探针；当前最小二次型为 `4.78944`，负值数量为 `0`，与有限差分和重复 matvec 结果一起记录。
- 稀疏中心 P2C 只在首次激活时初始化 $A_{0,c}$，连续时间步保持参考方向固定，并有跨步方向改变回归测试。
- 增加中心占用离开/重新激活生命周期：重新激活时显式重置 $A_{0,c}$ 与 committed $F_c$，并累计记录 `reactivations`。
- 扩容各向异性状态数组时保留累计 `reactivations`，并有专门的扩容回归测试。
- 重新激活占用标记使用 `uint8`，并按 Warp dtype 的实际字节数统计各向异性状态内存，避免低显存设备上的无谓开销。
- 新增实验前资源守卫：总剩余空间低于 5 GiB 时暂停；系统盘低于 2 GiB 时将 Warp 缓存迁移到 `MPM_LITE_DATA_ROOT` 指定的数据盘。
- 新增 `auto` 设备选择器：通过 `nvidia-smi` 选择已用显存最低的 GPU，驱动不可用时回退 CPU；PPC、网格和固定边界探针共用该选择器，并记录 GPU 测试前后显存（CPU 回退时为 `na`）。
- 网格/时间步探针额外输出全程最小中心 Jacobian 和最大粒子速度，长期稳定性记录不再依赖外部脚本推断。

当前验证结果：

- 当前 CPU 实验环境记录为 Linux、Python 3.11.15、NumPy 2.3.5、Warp 1.10.1、`USE_FLOAT64=True`；
- NumPy 材料点与状态断言：通过；
- Warp 1.10.1 CPU kernel launch：通过（使用 `/tmp/mpm-lite-warp-cache` 缓存）；
- `git diff --check` 和 Python 编译检查：通过；
- 低显存回归断言：重新激活占用数组为 `uint8`，阶段一状态预算低于 590 MiB；
- `python -m unittest -v tests.test_aniso_phase1_material tests.test_aniso_phase1_implicit tests.test_aniso_phase1_adapter tests.test_aniso_phase1_warp tests.test_aniso_phase1_solver`：36 项执行，CPU 环境 1 项 CUDA ordinal 测试跳过，其余通过；
- `python -m unittest discover -v` 全量发现并执行 40 项测试：CPU 环境通过（1 项 CUDA ordinal 测试跳过）；
- CUDA 可见环境 `python -m unittest discover -q`：40 项全部通过（14.39 s），包含 `cuda:1` 非零 ordinal 边界回归；跨块状态重排回归在 `cuda:2` 单独运行通过（0.53 s）。
- 旧 `lite_explicit` 单粒子 CPU 一步冒烟：通过，确认新增包未改变旧路径。
- 旧 `lite_implicit` 单粒子 CPU 一步冒烟：通过（1 次 Newton、10 次 CG 迭代），确认旧矩阵自由路径仍可运行。
- `k_f=0` 各向同性回归探针：CPU 位置/速度/粒子 F/中心速度最大差均为零，中心应力最大差 `1.01e-28`，两边均 1 次 Newton/10 次 CG；CUDA `cuda:2` 同样位置/速度/粒子 F/中心速度为零，中心应力最大差 `2.02e-28`，两边均 1 次 Newton/10 次 CG，新增路径耗时 `0.0455 s`（旧路径 `0.2231 s`，含首次运行差异）。
- 旧 demo 低规模 CPU 无头冒烟：`noodles` 播种 739 粒子并正常启动/停止 Viser，`snow` 播种 1930 粒子并进入边界初始化，均无 traceback；`wheel` 在 8 秒窗口内仍处于网格/网格资源初始化，未把超时解释为求解失败。
- 原始 demo 的固定配置、环境和 headless 结果记录在 [`HEADLESS_BENCHMARK_ZH.md`](./HEADLESS_BENCHMARK_ZH.md)。

各向同性退化回归复现命令：

```bash
PYTHONPATH=. MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache \
  .venv/bin/python benchmarks/aniso_phase1_isotropic_regression.py --device auto
```

实现约定：`mu`、`lambda`、`k_f` 使用与压力和应力一致的材料单位，参考方向 `fiber_direction` 是无量纲参考坐标向量，`vol0` 使用当前 demo 的体积单位；初始化时方向归一化为 $a_0$ 并保存 $A_0=a_0\otimes a_0$。一次成功隐式步的 kernel 顺序为：粒子到中心 P2C（含质量、体积、速度梯度和 $A_0$）→ committed 应力 → 中心到网格 C2G → Newton trial $F_c$ / 残差 → 固定 trial 状态的 PCG matvec → 收敛后 G2C/C2P → 中心 commit；失败路径只执行完整 trial rollback。

端到端 `AnisotropicLiteImplicitSolver` 的固定网格 CPU PPC 探针（第二步稳态近似，1 次 Newton）结果如下。四组粒子都放在同一个网格单元内，因此活跃中心/节点 footprint 固定，主要改变 $N_p/N_c$；`memory_mib` 只统计独立各向异性状态数组，不是整个 Python/Warp 进程的常驻内存；重新激活占用标记使用 `uint8`，使该数组预算只增加约 2 MiB；运行时间随机器负载变化。

| 粒子数 | $N_p/N_c$ | 活跃中心 | 活跃节点 | 单步时间（s） | P2C（s） | CG（s） | CG 迭代 | 材料算子（s） | 状态内存（MiB） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 8 | 27 | 0.235070 | 0.046722 | 0.101020 | 10 | 0.002359 | 586.001 |
| 32 | 4 | 8 | 27 | 0.232096 | 0.046230 | 0.099528 | 10 | 0.002415 | 586.002 |
| 128 | 16 | 8 | 27 | 0.232888 | 0.046883 | 0.098654 | 10 | 0.002454 | 586.009 |
| 512 | 64 | 8 | 27 | 0.236026 | 0.047434 | 0.101752 | 10 | 0.002543 | 586.035 |

该表是 CPU 小网格正确性/趋势探针，不代表 GPU 性能结论；matvec 只访问中心状态。复现命令：

```bash
PYTHONPATH=. MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache \
  .venv/bin/python benchmarks/aniso_phase1_ppc.py --device auto
```

均匀三维仿真的网格/时间步 CPU 探针使用 64 个粒子、固定三维速度梯度、无边界场，并比较一步后的中心变形梯度与解析解。九组运行均收敛，`affine_F_max_error` 是中心 $F_c$ 的最大 Frobenius 误差，`memory_mib` 仍只统计独立各向异性状态数组：

| 网格 | $\Delta t$ | 活跃中心 | 活跃节点 | 单步时间（s） | CG 迭代 | 仿射 $F_c$ 最大误差 | 状态内存（MiB） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8³ | 0.0005 | 343 | 512 | 0.378432 | 20 | 1.51e-10 | 586.004 |
| 8³ | 0.0010 | 343 | 512 | 0.332737 | 20 | 1.21e-09 | 586.004 |
| 8³ | 0.0020 | 343 | 512 | 0.331072 | 20 | 9.66e-09 | 586.004 |
| 12³ | 0.0005 | 512 | 1000 | 0.343001 | 20 | 7.25e-10 | 586.004 |
| 12³ | 0.0010 | 512 | 1000 | 0.334808 | 20 | 5.80e-09 | 586.004 |
| 12³ | 0.0020 | 512 | 1000 | 0.420186 | 30 | 4.64e-08 | 586.004 |
| 16³ | 0.0005 | 512 | 1728 | 0.451994 | 20 | 7.49e-10 | 586.004 |
| 16³ | 0.0010 | 512 | 1728 | 0.464647 | 20 | 5.99e-09 | 586.004 |
| 16³ | 0.0020 | 512 | 1728 | 0.462172 | 20 | 4.79e-08 | 586.004 |

复现命令：

```bash
PYTHONPATH=. MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache \
  .venv/bin/python benchmarks/aniso_phase1_grid_dt.py --particles 0
```

为避免固定 64 个粒子在网格加密后只覆盖越来越稀疏的区域，额外使用 `--particles 0` 让每个分辨率的内部中心各放一个粒子。GPU `cuda:2` 的 `dt=0.001` 结果为：$8^3/12^3/16^3$ 活跃中心/节点 `216/343`、`1331/1728`、`2744/3375`，最大 $F_c$ 误差 `1.31e-9/3.22e-9/5.99e-9`，可比较的物理中心采样误差分别为 `2.58e-20/6.32e-20/7.31e-20`；中心采样保持机器精度，边缘最大误差来自不同分辨率下的稀疏支持边界。脚本现在还输出排除两层稀疏支持边界后的 `affine_F_interior_max_error`；CPU 同参数三组为 `1.00e-14/6.12e-14/2.11e-13`，因此全局最大值与内部一致性分开记录。三组均 20 次 CG，状态内存为 `586.015/586.069/586.188 MiB`。

同一脚本的早期 CPU 长期探针中，`--grid 8 --dt 0.001 --steps 100` 的 100 步均收敛，总耗时 31.81 s，最终中心 $F_c$ 最大误差为 $1.25\times10^{-4}$，最小中心 $\det F_c=1.00009$，最大粒子速度为 0.0661。最新 CUDA 运行见下方 GPU 表：100 步耗时 13.24 s，最大误差为 $2.33\times10^{-4}$；该误差随步数累积，因此仍需更长时间和更大场景验证。

将同一仿射场扩大到 $16^3$ 网格、$\Delta t=10^{-3}$ 并运行 20 步时也全部收敛；早期 CPU 记录为 6.85 s，最新 CUDA 记录为 2.575 s，活跃中心/节点为 512/1728，最终中心 $F_c$ 最大误差为 $9.29\times10^{-6}$，最后一步残差为 $6.09\times10^{-11}$。
单组复现命令为：

```bash
PYTHONPATH=. MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache \
  .venv/bin/python benchmarks/aniso_phase1_grid_dt.py --grid 16 --dt 0.001 --steps 20
```

进一步使用 256 个粒子和 $24^3$ 网格运行 5 步时全部收敛；最新 CUDA 活跃中心/节点为 2048/3208，总耗时 0.807 s，最后一步残差为 $9.69\times10^{-14}$，中心 $F_c$ 最大误差为 $9.90\times10^{-5}$，各向异性状态内存为 586.018 MiB：

```bash
PYTHONPATH=. MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache \
  .venv/bin/python benchmarks/aniso_phase1_grid_dt.py \
  --grid 24 --dt 0.001 --particles 256 --steps 5
```

补充 100 步同场景运行时，100 步全部收敛，总耗时 31.81 s，中心 $F_c$ 最大误差为 $1.25\times10^{-4}$，最小中心 $\det F_c=1.00009$，最大粒子速度为 0.0661；没有出现负 Jacobian 或 NaN。误差的长期累积仍是后续稳定性工作的重点。

```bash
PYTHONPATH=. MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache \
  .venv/bin/python benchmarks/aniso_phase1_grid_dt.py --grid 8 --dt 0.001 --steps 100
```

GPU 恢复后的探针使用 `--device auto`，会按当时 `nvidia-smi` 的已用显存选择设备。首次复查时 `cuda:0/1/2` 均约 15 MiB、`cuda:3` 约 13313 MiB，PPC 选择 `cuda:0`；随后其他进程改变占用，网格探针选择了当时较低占用的 `cuda:2`（约 4371 MiB）。PPC GPU 单步结果为：

| 粒子数 | $N_p/N_c$ | GPU | 采样已用显存（MiB） | 单步时间（s） | P2C（s） | CG（s） | CG 迭代 | 状态内存（MiB） |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | cuda:2 | 18 | 0.013648 | 0.000197 | 0.009643 | 10 | 586.001 |
| 32 | 4 | cuda:2 | 18 | 0.012954 | 0.000168 | 0.009259 | 10 | 586.002 |
| 128 | 16 | cuda:2 | 18 | 0.013228 | 0.000147 | 0.009599 | 10 | 586.009 |
| 512 | 64 | cuda:2 | 18 | 0.012881 | 0.000173 | 0.009179 | 10 | 586.035 |

早期固定 64 粒子的 GPU 网格一步探针在 `cuda:2`（采样已用显存约 4393 MiB）下，`dt=0.001` 的 8³/12³/16³ 单步分别为 0.2161/0.1875/0.2216 s；当前脚本使用更严格的 CG/粒子布局，最新长期数据见下表。

GPU 长期和扩展探针结果如下：

| 场景 | GPU | 步数 | 总耗时（s） | 最后残差 | $F_c$ 最大误差 | 最小 $\det F_c$ | 状态内存（MiB） |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8³ 仿射 | cuda:2 | 100 | 13.24 | 4.64e-15 | 2.33e-4 | 1.00009 | 586.004 |
| 16³ 仿射 | cuda:2 | 20 | 2.575 | 6.09e-11 | 9.29e-6 | 1.00009 | 586.004 |
| 24³ / 256 粒子 | cuda:1 | 5 | 0.807 | 9.69e-14 | 9.90e-5 | 1.00008991 | 586.018 |
| 固定边界块体 | cuda:0 | 100 | 21.99 | — | — | 0.9999999997 | — |

固定边界 CUDA 100 步的最大粒子速度为 `0.04482`，最大中心/粒子 $F$ 误差为 `4.87e-5`，质量和体积误差均为零；`cuda:1` 非零 ordinal 边界回归、`cuda:2` 跨块状态重排回归和 CUDA 全量 40 项测试均通过。

材料点和 CPU kernel 可复现命令（Warp 编译缓存放在可写临时盘）：

实验探针可设置 `MPM_LITE_DATA_ROOT=/mnt/<data-disk>`；系统盘低于 2 GiB 时，探针会把 Warp 缓存迁移到该目录，总剩余空间低于 5 GiB 时直接暂停。

```bash
cd /home/yin/mpm-lite
export PYTHONPATH=.
mkdir -p /tmp/mpm-lite-warp-cache
MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache .venv/bin/python -m unittest -v tests.test_aniso_phase1_material
MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache .venv/bin/python - <<'PY'
import warp as wp
wp.config.kernel_cache_dir = "/tmp/mpm-lite-warp-cache"
from engine.aniso_phase1 import AnisotropicCenterBuffer
print("center-state module import: PASS")
PY
```

材料点方向/剪切/卸载诊断：

```bash
PYTHONPATH=. .venv/bin/python benchmarks/aniso_phase1_material_curve.py
```

当前默认参数在工程应变 `0.1` 下输出 `P11(0°/45°/90°)=19.0865/5.2265/0.6065`（单位面积力相同），简单剪切能量为 `0.124354`，卸载能量和应力 Frobenius 范数均为零。

中心残差 Jacobian 诊断在非零 Newton 速度、$\varepsilon=10^{-4},10^{-5},10^{-6}$ 下相对 $L^2$ 误差分别为 `3.42e-12/5.53e-12/3.35e-11`，重复 matvec 的最大绝对差为零：

```bash
PYTHONPATH=. .venv/bin/python benchmarks/aniso_phase1_jacobian_check.py
```

同一诊断的 16 个随机方向正定性探针最小二次型为 `4.78944`，负值数量为 `0`。

固定边界块体 20 步探针（$k_f=200$, $8^3$ 网格）20 步全部收敛，耗时 6.12 s，$\det F_c$ 最小值为 `0.9999999997`，最大粒子速度为 `0.04482`，最大中心/粒子 $F$ 误差为 `3.19e-6`，质量误差为 `7.11e-15`、体积误差为零：

同一探针在最新 `cuda:0` 上 20 步耗时 8.98 s，最终内部能量为 `1.3414015e-4`，相邻步最大能量跳变为 `1.2946537e-5`；收敛步数、Jacobian 下界、守恒误差和中心/粒子 $F$ 误差与上面的物理检查一致。

同一固定边界场景延长到 100 步时仍全部收敛，CPU 早期探针耗时 32.02 s；最新 CUDA 探针耗时 21.99 s。两者的 $\det F_c$ 最小值均为 `0.9999999997`，最大粒子速度均为 `0.04482`，最大中心/粒子 $F$ 误差均为 `4.87e-5`，质量和体积误差均为零。

```bash
PYTHONPATH=. MPM_LITE_WARP_CACHE=/tmp/mpm-lite-warp-cache \
  .venv/bin/python benchmarks/aniso_phase1_fixed_block.py
```

当前包不修改旧 `MPMSolver` 的默认 `lite_explicit`/`lite_implicit` kernel；`AnisotropicLiteImplicitSolver` 通过独立类和 sparse-grid kernel 接入，避免在旧路径中静默把各向异性材料当作各向同性材料处理。三个规模探针默认使用 `--device auto`，显式传入 `cpu` 或 `cuda:N` 可固定设备；设置 `MPM_LITE_DATA_ROOT` 或传入 `--data-root` 后，资源守卫可在系统盘不足时迁移 Warp 缓存。

资源记录（最终复查）：系统盘可用约 18 GB，尚未触发“低于 2 GB”迁移阈值；数据盘可用约 45 GB，系统盘与数据盘合计约 63 GB，未触发“总计低于 5 GB”暂停条件。PCIe 和 `/proc/driver/nvidia` 可见 4 张 RTX 3090、驱动模块 `580.178.04`；普通受限 shell 仍没有 `/dev/nvidia*`，但获准的沙箱外实验环境可用 CUDA 设备，Warp 枚举 `cuda:0..3` 并完成本轮 GPU 探针。

步骤 9–12 已同时覆盖 CPU/CUDA 固定边界 100 步、$16^3$ 网格 20 步、256 粒子/$24^3$ 网格 5 步和 $N_p/N_c=1,4,16,64$；旧 `elastic` 路径继续保持不变。

阶段一 CPU 与 CUDA 验证均已执行；均匀三维仿真、固定边界长期稳定性、失败步回滚、混合方向检测、P2C/CG/material/memory 分项、多 GPU ordinal 设备一致性和跨块状态重排均已有证据。阶段一仍只覆盖固定参考方向横向各向同性弹性，不包含后续扩展边界。

当前验收闭环：

| 验收组 | 当前证据 | 状态 |
| --- | --- | --- |
| 状态与本构 | 40 项测试、材料点诊断、结构张量混合度、重新激活计数（含跨块重排）、资源守卫 | CPU/CUDA 通过 |
| 隐式求解 | 非零 Newton 状态下 Jacobian 有限差分诊断、重复 matvec 差为零、CUDA 全量回归 | CPU/CUDA 通过 |
| 物理与守恒 | 刚体客观性、方向刚度、卸载、仿射场、固定边界 100 步、质量/体积误差 | CPU/CUDA 通过 |
| 扩展性 | $N_p/N_c=1,4,16,64$，512 粒子 PPC 探针，256 粒子/$24^3$ 网格探针 | CPU/CUDA 通过 |
| 隔离与复现 | 独立分支 `feature/aniso-phase1`、基线 headless 记录、全部复现命令 | 已记录 |
| GPU 显存与吞吐 | `cuda:0/2` 自动选择、显存前后读数、PPC/网格/固定边界探针 | 已通过 |

> 本文最初是实现前的设计、步骤和验收标准；上方“实施进度”记录了当前实验代码和验证结果。

目标是建立一个与原始 MPM Lite 隔离的实验版本，实现

$$
\psi(F,A_0)=\psi_{\mathrm{iso}}(F)+\psi_{\mathrm{fiber}}(I_4),
\qquad
I_4=A_0:C,
\qquad
C=F^{\mathsf T}F,
$$

其中 $A_0=a_0\otimes a_0$ 是参考构形中的固定纤维结构张量。阶段一只处理固定参考方向的横向各向同性弹性，不包含纤维旋转、塑性、损伤、方向硬化、正交各向异性、CK-MPM 和自适应 quadrature。

## 1. 阶段一目标和非目标

阶段一要实现：

1. 中心保存 $F_c$ 和参考方向 $A_{0,c}$；
2. 不再通过混合应力 $\tau_c$ 反演各向同性的 $F_c^{\mathrm{base}}$；
3. 使用固定中心 quadrature 计算各向异性应力；
4. 继续使用网格端矩阵自由隐式求解；
5. 实现解析各向异性方向导数；
6. 在 CG 迭代中不重新遍历全部粒子；
7. 为后续纤维旋转、塑性和损伤保留状态接口。

明确不做：

- 方向随当前构形或塑性旋转更新；
- 正交各向异性和多纤维族；
- 塑性、损伤、断裂和方向硬化；
- 每个粒子的独立各向异性历史变量；
- 接触、摩擦、多相材料、自适应网格；
- 自动微分和隐式伴随。

阶段一要回答的问题是：

> 固定参考方向的各向异性状态，能否作为中心状态接入 MPM Lite 隐式流程，同时保持正确性、收敛性和粒子数扩展性？

## 2. 源码隔离方案

### 2.1 推荐使用独立 worktree

不要直接修改当前原始 checkout。实现开始前，在原始目录执行检查：

~~~bash
git status --short
git log -1 --oneline
~~~

选择干净的基线提交 BASE_COMMIT 后，创建独立实验目录：

~~~bash
git worktree add ../mpm-lite-aniso-phase1 \
  -b feature/aniso-phase1 BASE_COMMIT
~~~

建议保持：

~~~text
/home/yin/mpm-lite/                 # 原始基线，只作对照
/home/yin/mpm-lite-aniso-phase1/    # 各向异性阶段一实验分支
~~~

如果当前原始工作区有未提交修改，应先明确这些修改是否属于基线，不能直接把它们当成论文对照版本。本文不执行这些 Git 命令。

### 2.2 分支保护要求

实验分支必须满足：

- 原始 lite_explicit 和 lite_implicit 仍可运行；
- 各向同性材料结果可以和基线比较；
- 新材料通过独立材料或本构类型选择；
- 新状态不改变旧 kernel 的语义；
- 旧 API 的默认参数保持兼容。

推荐新增而不是覆盖的文件：

~~~text
engine/aniso_phase1/
    types.py
    constitutive.py
    kernels.py
    implicit.py
    state.py
tests/aniso_phase1/
    test_material_point.py
    test_tangent.py
    test_objectivity.py
    test_solver.py
demos/aniso_bar.py
~~~

如果需要复用稀疏网格函数，可以新增独立的各向异性 d3 kernel 文件。

## 3. 必须先固定的设计不变量

1. 参考方向使用参考构形坐标；
2. $A_0=a_0\otimes a_0$ 是无向结构张量，$a_0$ 和 $-a_0$ 等价；
3. 中心的 $F_c$ 是各向异性隐式积分的有效状态；
4. $A_{0,c}$ 在阶段一中固定；
5. Newton 外迭代可以产生 trial 状态，CG 内迭代不改变历史状态；
6. 失败时间步不能提交 trial 状态；
7. 各向同性路径保持原始行为；
8. P2C、残差、切线和状态提交使用一致的中心权重和体积。

## 4. 数学模型

### 4.1 各向同性部分

复用当前 Lite 的 Hencky-StVK 能量：

$$
\psi_{\mathrm{iso}}(F)
=
\mu\sum_i(\log\sigma_i)^2
+
\frac{\lambda}{2}
\left(\sum_i\log\sigma_i\right)^2.
$$

### 4.2 纤维部分

设

$$
A_0=a_0\otimes a_0,
\qquad
I_4=A_0:C.
$$

第一版使用光滑二次能量：

$$
\psi_{\mathrm{fiber}}(I_4)
=
\frac{k_f}{2}(I_4-1)^2.
$$

总能量为

$$
\psi(F,A_0)
=
\psi_{\mathrm{iso}}(F)
+
\psi_{\mathrm{fiber}}(I_4).
$$

只拉伸纤维的模型

$$
\psi_{\mathrm{fiber}}(I_4)
=
\frac{k_f}{2}\langle I_4-1\rangle_+^2
$$

放到后续步骤，因为 $I_4=1$ 处需要分段或光滑切线。

### 4.3 应力和方向导数

第一 Piola 应力为

$$
P=P_{\mathrm{iso}}+P_f,
\qquad
P_f=2\psi_4FA_0,
$$

其中

$$
\psi_4=
\frac{\partial\psi_{\mathrm{fiber}}}{\partial I_4}.
$$

Kirchhoff 应力为

$$
\tau=PF^{\mathsf T}.
$$

由于

$$
\delta I_4=2(FA_0):\delta F,
$$

纤维方向导数为

$$
\delta P_f
=
2\psi_4\,\delta F A_0
+
4\psi_{44}
\big[(FA_0):\delta F\big]FA_0.
$$

总方向导数为

$$
\delta P=\delta P_{\mathrm{iso}}+\delta P_f.
$$

实现时只需要给定 $\delta F$ 后返回 $\delta P$，不需要显式保存完整 $9\times9$ 切线。

## 5. 中心状态设计

### 5.1 状态所有权

中心状态定义为

$$
z_c^n=(F_c^n,A_{0,c},V_c,m_c).
$$

阶段一采用以下所有权：

- $F_c$ 是隐式中心 quadrature 的有效变形状态；
- $A_{0,c}$ 是固定参考结构张量；
- 粒子 $F_p$ 可继续用于粒子更新和诊断，但不能用混合应力覆盖中心 $F_c$；
- 中心应力由 $P(F_c,A_{0,c})$ 计算，不用于恢复中心状态。

### 5.2 建议增加的字段

~~~text
center_F_aniso       : mat33
center_A0_aniso      : mat33
center_state_valid   : int 或 bool
center_state_volume  : real，可选
ptc_A0               : mat33 或 vec3，可选
~~~

建议增加独立的各向异性 scratch 类型，至少包含：

~~~text
F_trial
A0
tau
~~~

### 5.3 中心状态初始化

中心状态有三种情况：

1. 首次初始化：
   $$
   F_c=I;
   $$
2. 连续占据：保留上一个时间步的 $F_c^n$；
3. 重新激活或材料占据改变：进入明确的重新初始化路径并记录事件。

阶段一不应在一般情况下直接平均不同粒子的 $F_p$。若必须用粒子初始化中心，只允许在初始或近均匀场景使用，并报告初始化误差。

## 6. 具体实施步骤

### 步骤 0：冻结对照基线

**做什么：**

- 选择原始 baseline commit；
- 记录 Python、Warp、设备、精度和依赖；
- 运行 wheel、snow、noodles 的最小 headless 配置；
- 保存运行时间、粒子位置摘要、最大速度、残差和失败情况。

**为什么：**

没有固定基线，无法判断实验分支变化是否来自各向异性扩展。

**验收：**

- 原始目录仍可独立运行；
- 三个 demo 有固定配置；
- 基线输出和环境信息已保存；
- 实验分支可以重复同一组对照测试。

### 步骤 1：增加独立材料入口

**做什么：**

新增独立材料或本构类型，例如 transverse_isotropic_elastic，不改变现有 elastic 的含义。参数至少包括：

~~~text
mu, lambda
k_f
fiber_direction，可选的统一参考方向
~~~

空间变化的方向还需要粒子或中心方向数组。

**为什么：**

独立入口可以避免污染旧材料分支，并为多纤维族扩展留出接口。

**验收：**

- 原始材料枚举和默认参数不变；
- 新材料可显式选择；
- 零长度方向报错；
- 方向归一化；
- 参数单位和坐标系记录清楚；
- 各向同性材料不分配多余的各向异性状态。

### 步骤 2：初始化方向和结构张量

**做什么：**

支持统一方向和逐粒子方向，转换为

$$
A_0=a_0\otimes a_0.
$$

初始化中心变形为

$$
F_c^0=I.
$$

**为什么：**

材料方向必须来自参考构形，不能固定为世界坐标中的临时方向。

**验收：**

- 统一方向的粒子和中心方向一致；
- 零向量报错；
- 非单位方向被归一化或拒绝；
- $a_0$ 与 $-a_0$ 得到相同 $A_0$；
- 初始无变形材料无非物理应力。

### 步骤 3：粒子到中心的方向传递

**做什么：**

使用结构张量平均：

$$
\widetilde A_c
=
\frac{\sum_p\omega_{pc}A_{0,p}}
{\sum_p\omega_{pc}},
\qquad
A_{0,p}=a_{0,p}\otimes a_{0,p}.
$$

第一版可以先实现中心内统一方向的快速路径，再实现一般方向平均。

**为什么：**

不能直接平均角度或方向向量。结构张量平均能够保持无向纤维性质，并暴露中心内的多方向情况。

**验收：**

- 统一方向传递误差为零或在机器精度内；
- $a_0$ 与 $-a_0$ 结果一致；
- 中心张量对称半正定；
- 质量和体积守恒；
- 多方向混合能够被检测。

### 步骤 4：停止各向异性应力反演

**做什么：**

各向异性 P2C 流程为：

1. 聚合中心质量、体积和速度梯度；
2. 保留或初始化中心 $F_c$；
3. 保留或更新中心 $A_{0,c}$；
4. 直接计算

   $$
   P_c=P(F_c,A_{0,c}),
   \qquad
   \tau_c=P_cF_c^{\mathsf T};
   $$

5. 跳过

   $$
   \tau_c\longrightarrow F_c^{\mathrm{base}}
   $$

   的各向同性谱反演。

**为什么：**

各向异性应力不能唯一决定 $(F_c,A_{0,c})$。

**验收：**

- 各向异性路径不调用应力到 stretch 的谱反演；
- 零速度连续时间步中 $F_c$ 不改变；
- 同一 $F_c$ 下改变 $A_0$ 会改变方向应力；
- 各向同性路径结果不变。

### 步骤 5：实现中心 trial 状态

**做什么：**

在 Newton 外迭代中计算

$$
F_c(v)
=
\left(I+\Delta t\,G_c(v)\right)F_c^n.
$$

分开保存：

- 已提交的 $F_c^n$；
- 当前 trial $F_c(v)$；
- trial 应力。

CG 内层固定 $F_c^n$ 和 $A_{0,c}$。

**为什么：**

不同 CG 搜索方向必须对应同一个 Jacobian。

**验收：**

- 同一 Newton 外迭代内状态不随 CG 改变；
- zero velocity 时 trial 等于 committed；
- 失败步不提交；
- 成功步只提交一次；
- 重复运行结果一致。

### 步骤 6：实现各向异性应力

**做什么：**

实现本构接口：

~~~text
P(F, A0, params)
tau(F, A0, params)
~~~

其中

$$
P=P_{\mathrm{iso}}+2\psi_4FA_0.
$$

第一版只使用光滑二次纤维能量。

**为什么：**

先隔离应力公式，避免同时处理不可微正部函数、塑性和损伤。

**验收：**

- 材料点应力与手算一致；
- $F=I$ 时无预应变应力为零；
- 方向改变时应力改变；
- 纯刚体旋转满足客观性；
- 没有 NaN、Inf 或未处理的负 Jacobian。

### 步骤 7：实现矩阵自由各向异性切线

**做什么：**

实现

~~~text
dP_aniso_apply(F, A0, dF, params) -> dP
~~~

并使用

$$
\delta P_f
=
2\psi_4\,\delta F A_0
+
4\psi_{44}
\big[(FA_0):\delta F\big]FA_0.
$$

**为什么：**

MPM Lite 的矩阵自由求解只需要 Hessian-vector product，不需要完整四阶张量。

**验收：**

对随机 $F$ 和 $\delta F$ 检查

$$
\frac{P(F+\varepsilon\delta F)-P(F)}{\varepsilon}
\approx
\delta P(F)[\delta F].
$$

应在逐渐减小的 $\varepsilon$ 中出现收敛区间。各向同性部分也要通过同一测试。

### 步骤 8：接入残差和矩阵自由算子

**做什么：**

中心残差使用

$$
r_i(v)
=
m_i(v_i-v_i^n)
-\Delta t\,m_i g
+
\Delta t
\sum_cV_cP_c(F_c(v),A_{0,c})\nabla w_{ic}.
$$

搜索方向使用

$$
\delta F_c
=
\Delta t(\nabla p_c)F_c^n,
$$

$$
(Ap)_i
=
m_ip_i+
\Delta t
\sum_cV_c\delta P_c\nabla w_{ic}.
$$

第一版继续使用 PCG 和块对角预条件器，但记录收敛和正定性。

**为什么：**

这一步验证新材料是否真正保留 MPM Lite 的隐式矩阵自由结构。

**验收：**

- residual 和 matvec 不访问粒子级各向异性本构；
- $A p$ 与有限差分残差 Jacobian 一致；
- 重复调用 $A p$ 结果一致；
- 阶段一材料上的 PCG 收敛；
- 残差和迭代数有日志。

### 步骤 9：提交中心状态和同步粒子

**做什么：**

隐式速度收敛后计算

$$
F_c^{n+1}
=
\left(I+\Delta tG_c^{n+1}\right)F_c^n.
$$

使用单独 commit kernel 写入中心状态，保持 $A_{0,c}$ 不变，然后执行原有 G2C/C2P 速度和位置更新。

阶段一可保留粒子 $F_p$ 更新，但把它定义为粒子侧派生状态，不能用它静默覆盖中心 $F_c$。同时统计两者差异。

**为什么：**

中心状态必须跨时间步持久化，否则下一步会退回应力反演或重新初始化。

**验收：**

- 成功步提交中心 $F_c$；
- 失败步不提交；
- 静止场景不漂移；
- 均匀仿射变形中中心 $F_c$ 与解析解一致；
- 中心和粒子状态差异可统计。

### 步骤 10：各向同性回归

**做什么：**

让新路径在 $k_f=0$ 时退化为各向同性模型，比较：

- 粒子位置；
- 中心速度；
- 应力；
- 残差；
- Newton/CG 迭代数；
- 每步耗时。

**为什么：**

必须证明新增状态和 kernel 没有破坏原始各向同性路径。

**验收：**

- 原始各向同性路径不回归；
- $k_f=0$ 新路径与参考在容差内一致；
- 原有 demo 无需修改即可运行；
- 差异都有记录和解释。

### 步骤 11：材料点和小型结构验证

**做什么：**

按以下顺序测试：

1. 单轴拉伸；
2. $0^\circ/45^\circ/90^\circ$ 纤维方向；
3. 简单剪切；
4. 纯刚体旋转；
5. 拉伸和卸载；
6. 均匀三维仿射变形；
7. 固定边界纤维增强块体。

记录应力、力、应变、能量、残差和迭代数。

**为什么：**

复杂 demo 会同时引入接触、边界和粒子稀疏问题，难以定位状态或切线错误。

**验收：**

- 方向刚度满足

  $$
  E(0^\circ)\neq E(45^\circ)\neq E(90^\circ);
  $$

- 刚体旋转不产生虚假应力；
- 卸载曲线符合弹性模型；
- 均匀仿射变形随网格加密收敛；
- 能量和残差没有无法解释的跳变。

### 步骤 12：效率和粒子数扩展性

**做什么：**

固定场景、网格、材料、时间步和收敛条件，改变

$$
N_p/N_c\in\{1,4,16,64\}.
$$

记录

$$
T_{\mathrm{step}},
\quad
T_{\mathrm{P2C}},
\quad
T_{\mathrm{CG}},
\quad
T_{\mathrm{material}},
\quad
N_{\mathrm{CG}},
\quad
M_{\mathrm{memory}}.
$$

如果有粒子 quadrature 各向异性基线，使用相同本构、网格和收敛标准。

**为什么：**

目标不是声称各向异性版本每一步都更快，而是验证隐式阶段没有退化为

$$
O(N_pN_{\mathrm{CG}}C_{\mathrm{mat}}).
$$

中心状态方法应尽量保持

$$
O(N_c\,n_{\mathrm{support}}C_{\mathrm{aniso}})
$$

的增长趋势。

**验收：**

- matvec 时间不随 $N_p$ 线性增长；
- P2C/C2P、材料更新和 CG 时间分开报告；
- CG 不调用粒子级各向异性本构；
- PPC 增加时误差和收敛质量可解释；
- 低 PPC 下若更慢，报告性能交叉区间。

### 步骤 13：完善可复现实验文档

**做什么：**

补充材料参数和单位、参考方向坐标系、中心状态生命周期、kernel 调用顺序、残差和切线公式、测试命令、运行环境、性能表格和已知限制。

**为什么：**

各向异性结果对参考构形、方向定义和状态初始化敏感。

**验收：**

- 新用户可从干净 worktree 运行材料点测试；
- 应力和切线可复现；
- 文档明确区分原始实现和实验分支；
- 不把阶段一描述成已支持塑性、损伤或纤维旋转。

## 7. 推荐执行顺序

1. 冻结原始基线；
2. 创建独立 worktree 和分支；
3. 先写材料点能量、应力和方向导数测试；
4. 增加材料参数和方向状态类型；
5. 实现中心 $A_0$ 初始化与传递；
6. 实现持久中心 $F_c$ 生命周期；
7. 实现中心 trial $F_c$；
8. 接入中心应力和残差；
9. 接入矩阵自由 $\delta P$；
10. 做有限差分和全局 Jacobian 检查；
11. 做刚体旋转和 $0^\circ/45^\circ/90^\circ$ 测试；
12. 做各向同性退化回归；
13. 做 PPC、网格和时间步 benchmark；
14. 最后再增加复杂 demo。

不要先从 wheel、snow 或接触场景开始，它们包含边界、塑性或复杂运动，会把中心状态问题和其他问题混在一起。

## 8. 阶段一完成的判定条件

### 状态和本构

- 中心拥有持久 $F_c$ 和 $A_{0,c}$；
- 各向异性路径不执行 $\tau_c\to F_c^{\mathrm{base}}$ 的各向同性反演；
- 应力、Kirchhoff 应力和方向导数通过材料点测试；
- 参考方向定义、单位和初始化方式明确。

### 隐式求解

- trial $F_c$ 只在当前 Newton 状态中更新；
- CG 过程中材料状态冻结；
- $A p$ 与有限差分 Jacobian 一致；
- 阶段一材料稳定收敛；
- 失败步不会提交中心历史状态。

### 物理性质

- $0^\circ/45^\circ/90^\circ$ 方向刚度正确；
- 刚体旋转不产生虚假应力；
- 均匀仿射变形可收敛；
- 体积和质量传递没有异常漂移。

### 性能和隔离

- 原始各向同性分支可以独立运行；
- 隐式 matvec 不访问粒子级各向异性本构；
- PPC 扩展实验可复现；
- 结果同时报告精度、迭代数、耗时和显存；
- 实验代码位于独立分支或 worktree。

## 9. 阶段一之后的扩展边界

阶段一通过后，再按以下顺序扩展：

1. 只拉伸纤维和光滑正部函数；
2. 多纤维族和正交各向异性；
3. 方向随当前构形旋转；
4. 纤维塑性和有限塑性旋转；
5. 方向硬化；
6. 纤维损伤和断裂；
7. 自适应中心 quadrature；
8. 各向异性预条件器和非正定求解器。

每一步扩展都应保留阶段一的材料点、切线、客观性和各向同性回归测试。

## 10. 阶段一论文表述范围

阶段一完成后，比较稳妥的表述是：

> 我们在 MPM Lite 的固定中心 quadrature 和矩阵自由隐式框架中，引入了保存参考结构张量的横向各向同性弹性状态，并推导了方向相关的矩阵自由切线。该方法不再通过混合应力重建各向同性伸长，同时在隐式 CG 迭代中不重新遍历粒子。

阶段一不应声称：

- 已支持一般各向异性；
- 已支持纤维旋转；
- 已支持塑性、损伤或方向硬化；
- 对所有粒子数都比传统各向异性 MPM 更快；
- 对任意材料和任意时间步无条件稳定。

核心验收是：

$$
\boxed{
\text{在增加有限本构和方向状态成本的同时，}
\text{隐式求解没有退化为每个 CG 迭代依赖全部粒子。}
}
$$

能量诊断、拉伸试验、真实稀疏算子与公平基线的后续工作见 [第二轮量化验收进度](./ANISO_VALIDATION_PROGRESS_ZH.md)。
