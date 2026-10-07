# 固定参考方向横向各向同性弹性：Viser 交互演示

> 最新静态诊断：[group4x8 的夹持、加载与插值一致性](ANISO_BOUNDARY_CONSISTENCY_ZH.md)。精确材料面约束能消除夹持遗漏并改善梁位移，但不能解释全部偏软；全局 MLS 候选暴露新的积分不足，未替换生产路径。新增两个网格的位移/反力数据及 `demos.aniso_boundary` 查看器。

> 本轮进展：[group4x8 可选材料积分与稳定化组合](ANISO_GROUPED_QUADRATURE_ZH.md)。已接入冻结采样势能、残差和切线，完成去质量静态核对及短程试验；默认未变。试验中不需旧稳定化消除零模态，但相对独立梁参考仍有空间误差，保留实验选项。下面“未接入”的表述指上一阶段。

最新配对采样对照：`.venv/bin/python -m demos.aniso_snapshot --data docs/results/joint-sampling --port 8083`。可切换最多 8/16 个代表点和 2/4/8 组条件矩，查看同一冻结快照的积分点、材料能、中心平均应力与采样成本。说明见[配对采样报告](ANISO_JOINT_SAMPLING_ZH.md)。它是离线查看器，没有对应的生产 `--stabilization` 选项。

新增固定快照诊断查看器：`.venv/bin/python -m demos.aniso_snapshot --port 8083`，打开 `http://127.0.0.1:8083`。可切换原粒子、单中心、固定八点和空间矩八点，查看相同状态的材料能及应力差异。此模式有意不推进时间，无需 GPU；数据生成与说明见[快照报告](ANISO_MATERIAL_SNAPSHOT_ZH.md)。

当前推荐试用二次重建：梁命令改用 `--stabilization quadratic`；`material_quadratic` 为材料转动模板对照。运行 `.venv/bin/python -m demos.aniso_rotation --modes corotated quadratic material_quadratic --grid 17 --axis z --port 8081`，在浏览器打开 `http://127.0.0.1:8081`，可直接比较改进前后的朝向能量曲线。完整公式、结果、适用范围与命令见[二次重建报告](ANISO_QUADRATIC_RECONSTRUCTION_ZH.md)。下面保留早期演示说明，最新模式以新报告为准。

当前新增可选生产共旋模式：将下面梁或拉伸命令中的 `--stabilization supplemental` 改为 `--stabilization corotated`。新旧旋转对照可运行 `.venv/bin/python -m demos.aniso_rotation --modes supplemental corotated --axis y --port 8081`，改为 `--axis z` 可查看尚未消除的固定网格误差。[实现、结果与完整命令](ANISO_COROTATED_STABILIZATION_ZH.md)。默认 PCG 带检查及回退；`--linear-solver pcg_projected` 可从开始使用修改切线。

最新[旋转/耗散/慢速混合纤维报告与完整命令](ANISO_ROTATION_DISSIPATION_ZH.md)已加入三模式旋转回放：`.venv/bin/python -m demos.aniso_rotation --port 8081`，在浏览器打开 `http://127.0.0.1:8081`（远程需转发端口）。实时 demo 新增 `--flip-ratio` 与 `--quadrature particle`；粒子积分逐粒子计算纤维，不要同时指定 `--direction-model fourth_moment`。当前 supplemental 仅建议用于已验证的小变形、小转动展示。

本轮新增可选空间稳定化、四阶方向矩及预弯曲梁释放，适用范围和实验结果见[最新报告](ANISO_STABILIZATION_MOMENTS_ZH.md)。小变形梁演示：

```bash
.venv/bin/python -m demos.aniso --scene beam --grid 17 --device auto \
  --dt .0005 --stabilization supplemental --direction-model fourth_moment --play
```

混合方向拉伸：

```bash
.venv/bin/python -m demos.aniso --scene tensile --grid 9 --device auto \
  --dt .005 --loading-time .08 --loading-speed .025 --loading-cycles 2 \
  --smooth-loading --fiber-field crossed --direction-model fourth_moment \
  --stabilization supplemental --play
```

两者默认访问 `http://127.0.0.1:8080`；可用 `--viser-port` 修改端口。`--stabilization none/hourglass` 和 `--direction-model mean_tensor` 用于对照；`--fiber-field smooth` 显示平滑方向场。稳定化仍默认关闭，并限定小变形、小转动，不代表通用有限应变方案已经验收。

最新默认使用 `particle_resample` 历史模式。迁移修复、独立力学参考及已知梁欠积分限制见[最新结果](ANISO_HISTORY_REFERENCE_RESULTS_ZH.md)。

历史迁移对照回放（无需 GPU，浏览器打开 `http://127.0.0.1:8081`）：

```bash
.venv/bin/python -m demos.aniso_validation --scene directions --port 8081
```

`--scene translate` 为预拉伸平移，`--scene rotate` 为预变形旋转。左侧旧历史、右侧重采样；这是规定运动的保存数据回放。

实时加载—卸载—重新加载（`http://127.0.0.1:8080`）：

```bash
.venv/bin/python -m demos.aniso --scene tensile --grid 9 --device auto \
  --dt .005 --loading-time .08 --loading-speed .025 \
  --loading-cycles 2 --smooth-loading --play
```

余弦加载模式下 `loading-speed` 为单程平均速度；最大位移为 `loading-speed × loading-time`。两循环结束后加载端回零保持，可手动暂停。可用 `--history-mode grid_locked` 复现旧模式；旧模式会丢失迁移历史，不建议用于新实验。

入口为 [`demos/aniso.py`](../demos/aniso.py)。在项目根目录运行：

```bash
cd /home/yin/mpm-lite
.venv/bin/python -m demos.aniso --device auto
```

也可以使用 `uv run -m demos.aniso --device auto`。浏览器打开终端打印的地址，默认是 `http://127.0.0.1:8080`；如果端口被占用，Viser 会选择下一个可用端口。默认暂停，点击“单步”或勾选“播放”开始。

远程服务器建议通过 IDE 端口转发，或在本地执行 `ssh -L 8080:127.0.0.1:8080 用户名@服务器`。需要修改监听地址或端口时使用 `--viser-host`、`--viser-port`。

## 三个场景

| 场景 / CLI 参数 | 行为 | 默认设置 |
| --- | --- | --- |
| 固定边界纤维块体 / `--scene fixed` | 左侧网格节点固定，块体带初始拉伸速度，使用各向异性隐式积分 | 64 粒子、8³ 网格、初始速度梯度 `diag(0.08, 0, 0)` |
| 均匀三维仿射场 / `--scene affine` | 内部格点布置粒子，施加初始三维速度梯度，随后自由演化 | 8³ 网格、216 粒子；网格 16³ 时为 2744 粒子 |
| 材料点方向响应 / `--scene material` | 用滑条指定均匀拉伸/剪切，展示变形、能量与应力，可卸载 | 直接指定 `F`，不进行 MPM 时间积分 |

例如直接启动较密的仿射场：

```bash
.venv/bin/python -m demos.aniso --scene affine --grid 16 --device auto
```

三个场景统一使用 `mu=10, lambda=20`，初始 `k_f=200`、参考纤维方向沿 X。通过 XY 平面的角度滑条比较 0°、45°、90°，将 `k_f` 调为 0 可观察各向同性退化。此统一参数与原材料点 benchmark 的参数不同，数值不应直接混用。

固定边界场景在默认网格下沿用已有 benchmark 的几何与初始速度；仿射场采用网格填充粒子布局。Demo 使用更严格的 Newton/CG 容差，因此结果不应与旧 benchmark 的宽松容差记录逐项视为相同。

## 界面操作与显示含义

- **播放 / 单步 / 应用参数并重置**：切换场景自动暂停并重建；修改方向、刚度、网格、时间步后需重置才生效。当前已应用参数显示在状态区。
- **每次显示的仿真步数**：控制一帧前执行多少个积分步；不会改变单步时间步长。
- **灰点**：参考位置；**彩色点**：当前粒子，蓝到橙表示真实位移从 0 到 0.01（超过后饱和）；**金色点**：固定网格节点。
- **黄线**：使用粒子传输得到的 `F_p a_0` 归一化绘制当前纤维方向。材料参考方向 `a_0` 固定；随物体变形的空间方向可以改变。粒子 `F_p` 是显示用的传输量，中心 `F_c` 才是求解器材料状态。
- **显示位移放大**：仅改变点的显示位置 `x_display = X + scale * (x - X)`，不改变仿真、统计值或黄线的真实方向。动力学场景默认 20 倍，材料点模式默认 1 倍。设为 1 可查看真实比例。
- **方向响应对照**：展开后显示同一已应用刚度下 0° / 45° / 90° 的应力曲线。这里指定 `F=diag(1+strain,1,1)`，横轴工程应变、纵轴第一 Piola 应力 `P11`，侧向伸长固定为 1；曲线并非仿真粒子应力的实时历史。
- **材料点卸载**：将应变、剪切同时归零，恢复参考形状、零能量与零应力。

状态区显示真实仿真时间、步数、位移、当前最小中心 `det(F)` 和 Newton/CG 次数。求解未收敛时回滚并暂停；粒子接近计算域边界时暂停，避免长时间播放跑出有限网格。暂停信息保留在界面，可调整参数后重置。

## 资源与无头验证

`--device auto` 在启动时按已用显存选择 GPU，CUDA 不可用时回退 CPU；也可明确指定 `--device cpu` 或 `--device cuda:N`。切换场景前先释放旧求解器，GPU 操作和控件请求在主线程串行处理。

启动前以及交互运行时每 10 秒检查磁盘。此机器默认数据目录为 `/mnt/0c18569c-b839-4255-bae0-6f48c9fc835b/yin/tmp`，其他机器可通过 `--data-root` 或 `MPM_LITE_DATA_ROOT` 指定。系统盘低于 2 GiB 时迁移 Warp 缓存；系统盘与数据盘总剩余低于 5 GiB 时暂停。

```bash
# 运行同一场景构建与积分路径，不启动服务器；失败返回非零退出码。
.venv/bin/python -m demos.aniso --headless --scene fixed --steps 20 --device auto
.venv/bin/python -m demos.aniso --headless --scene affine --grid 16 --steps 20 --device auto
.venv/bin/python -m demos.aniso --headless --scene material --device cpu

# 显示数学与本构对照测试。
.venv/bin/python -m unittest tests.test_aniso_demo -v

# 有限时长的服务冒烟；平常使用不需要 --duration。
.venv/bin/python -m demos.aniso --scene fixed --device auto --play --duration 15 --viser-port 8090
```

## 本轮验证记录（2026-09-20）

- 新增 3 项测试通过：显示放大不修改仿真数据、纤维方向使用真实 `Fa0`、材料点卸载、三方向曲线及各向同性极限（合并在 3 个测试中）。
- CPU 全量发现 43 项测试，42 项通过、1 项 CUDA 专用测试跳过，耗时 13.58 秒。
- CPU：固定块体 3 步、仿射场 8³ / 20 步和材料点模式通过。
- CUDA 自动选择 `cuda:0`：固定块体 20 步全部完成，最终最大真实位移约 `8.94948e-4`，当前最小中心 `det(F)=1.0014102`。
- CUDA 自动选择 `cuda:0`：仿射场 16³ / 2744 粒子 / 20 步全部完成，最终最大真实位移约 `2.20821e-3`，当前最小中心 `det(F)=1.0013888`。
- 本机 Viser 1.1.1 HTTP 返回 200，使用真实 WebSocket 协议完成摄像机握手、单步、播放/暂停、重置、两个动力学场景切换、材料点场景切换与卸载归零验证。此项为协议交互测试，不替代浏览器画面截图验收。

## 第二轮新增：能量与定量拉伸

```bash
# 原块体/仿射场：在浏览器看能量曲线，下载 CSV
PYTHONPATH=. .venv/bin/python -m demos.aniso --device auto
# 位移由加载速度积分，加载后按相反速度卸载
PYTHONPATH=. .venv/bin/python -m demos.aniso --scene tensile --grid 9 --device auto --loading-speed .01 --loading-time .2
# 无头同一路径导出完整曲线
PYTHONPATH=. .venv/bin/python -m demos.aniso --scene tensile --grid 9 --device auto --dt .002 --steps 200 --loading-time .2 --headless --csv output/tensile.csv
```

拉伸模式使用 x≤.25 的静止夹具与 x≥.75 的加载夹具，要求网格 9 或 17（夹具位置对齐节点）。界面提供速度和单程加载时长，参数应用需重置。默认密度为 1、参考块体积 .046875；该场景与初始速度驱动块体的密度/加载不同，不能混用反力验收。右端反力包括惯性与初始边界投影冲量，CSV 同时列出弹性反力、惯性反力、实际粒子夹具位移、有效刚度和加载功。橙色为加载，绿色为卸载。

能量面板同时画两层 APIC 动能、中心弹性能和机械能，CSV 包括各自变化量、边界投影及状态重建等分项。界面的 CSV 下载包含完整模拟历史，不受粒子显示放大影响。曲线含同步诊断开销，性能测试须使用独立 benchmark。

求解器默认采用统一增量势能残差、带曲率检查的 PCG，以及原势能线搜索；必要时用正定近似切线重新求解，原残差与能量保持不变。旧 Kirchhoff 离散仅用于显式对照，继续使用 BiCGSTAB/GMRES。状态栏中的计数表示线性迭代数，实际方法见 JSON 的 `linear_solver`。推导、谱验证、新旧能量与反力对照见[统一增量势能结果](ANISO_VARIATIONAL_RESULTS_ZH.md)。
