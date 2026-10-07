> 2026-09-26：按新的适度时长要求已完成[独立短程比较与一致传递验证](ANISO_PRACTICAL_VALIDATION_ZH.md)，保留完整网格、缩短至 0.125 ms；本页旧 3 ms 数据保留，不表示进程仍在当前机器内存中。

> 当前状态：已按用户要求暂停全部本次实验进程（SIGSTOP），内存状态保留。详见 results/nonuniform-reference/v1/pause-session.json。

# 等成本的非均匀 Y 参考加密

状态更新：CUDA 已恢复，包含真实 GPU 力和短程释放的 17 项专项回归全部通过（0 跳过）。已启动局部 Y24／较细 Y48 的四条时间对照轨迹；尚未产生新增终态结论。实时进度由队列每五分钟更新。下文的 CUDA 阻碍记录属于恢复前阶段。前轮见[分轴参考报告](ANISO_AXIS_REFERENCE_ZH.md)。本轮只改变比较基准的网格分配，保留候选历史增量／整体形状余量算法、原初态、本构、求解器与严格恢复检查。

## 预先固定的比较协议

基线为 (88,22,22)。在已指定的 Y 条带 [0.453125,0.46875] 内，二分两个完整基线单元，得到非均匀 (88,24,22)。全部基线 Y 节点和旧历史界面保留。与均匀 (88,24,22) 相比，两者均为 46,464 单元、151,800 个自由位移分量、5,808,000 个材料积分点，因此该主对照的离散预算相同；不以共享设备上的墙钟耗时判断效率。

保留前轮的 X=(96,22,22)、Z=(88,22,24) 和均匀 Y 对照，复用前先检查初态指纹、完整轴坐标、积分规则及数值验收。三方向加密仍均为 46,464 单元，但 X 的自由度略多。没有同时叠加局部 Y 与 X/Z，本轮不能判断其交互项。

另增 (88,48,22) 作为较细 Y 比较网格，预算单列，不属于等成本主对照，也不视为已经收敛的真解。它用于检验局部与均匀 Y 在同一个更细基准下的表现，避免把“更接近粗基线”误当成改善。它仍保持原 X/Z 分辨率，因此不能认证整体空间参考已经可信。

每个新增网格先比较 0.1220703125 与 0.06103515625 μs，释放到共同的 3 ms。共计划 4 条新轨迹、147,456 步。如果新网格时间门槛失败，则继续减半共同时间步并补齐必要对照，不放宽门槛。采样按新旧分界公共切分，材料 Gauss 5、质量 Gauss 2，积分精度另行复核。

## 诊断量

在共同材料坐标上使用体积加权 RMS。固定候选增强与未增强的差异尺度 S_f，检查

\[
T_f=\frac{\|f_{2dt}-f_{dt}\|}{S_f}<0.05,\qquad
R_{i,f}=\frac{\|f_i-f_{\mathrm{base}}\|}{S_f}.
\]

同时检查 f=F,P,v，保留 x。空间目标仍为 R<0.1，运行完成不等于达到空间目标。

对等成本主对照，定义与较细 Y 网格的差异比

\[
G_f=\frac{\|f_{\mathrm{localY24}}-f_{\mathrm{Y48}}\|}
          {\|f_{\mathrm{uniformY24}}-f_{\mathrm{Y48}}\|}.
\]

G<1 只说明更接近该较细 Y 网格，不证明更接近精确解。分别给出整梁、预先指定条带（体积 12.5%）及条带外结果，防止局部改善掩盖其他区域变差。还比较粗细时间步下两种差异向量及改善量的变化，判断时间误差是否足以改变排序。

非均匀单元上的插值、梯度和一致质量均使用实际轴间距 h_i；例如一维质量的对角与非对角分别为 (h_{i-1}+h_i)/3 与 h_i/6。位置和梯度保持同源，不改写历史。

## 运行、验证与资源

入口为 `benchmarks.aniso_nonuniform_reference` 与 `benchmarks.aniso_nonuniform_reference_queue`。运行前验证非均匀仿射再现、质量、能量／残差／切线、CPU/CUDA 短程释放一致性及比较指标。完整实验结束后复核独立高阶积分和沿梁三方向剖面。

[实时进度](results/nonuniform-reference/v1/PROGRESS.md)每五分钟刷新；[固定协议](results/nonuniform-reference/v1/protocol.json)保存完整轴坐标和预算。系统盘 <2 GiB 时迁移可重建缓存；系统与数据盘剩余总和 <5 GiB 时暂停。启动时约为 5.6/42 GiB，未触发阈值。

## 当前已核验的数据（不含新增终态）

- 两个新增 Y 节点：0.45738636363636365、0.4630681818181818，均落在预先指定条带内。
- 六组网格初态重建全部通过，最大位置误差 5.56e−16、最大梯度误差 1.45e−13（向上取整）。
- [CPU 回归](results/nonuniform-reference/v1/tests-cpu.txt)：15 项通过、2 项 GPU 测试跳过；此前显式开启 GPU 测试时，两项因 CUDA 初始化失败而报错，不能算作通过。
- [保留对照](results/nonuniform-reference/v1/retained-controls.json)：base/X/Y/Z 的时间门槛继续全部通过，最大 T=0.029677；三方向空间仍未全部达标。这些是旧轨迹核验，不是本轮新增结果。
- [底层驱动证据](results/nonuniform-reference/v1/cuda-probe.txt)、[队列退出日志](results/nonuniform-reference/v1/queue.log)与[设备预检](results/nonuniform-reference/v1/gpu-preflight.json)已保存。没有重置驱动或终止其他用户任务。

GPU 恢复后先运行真实 GPU 回归，成功后恢复队列：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ANISO_TEST_CUDA=1 .venv/bin/python -m unittest tests.test_aniso_nonuniform_reference tests.test_aniso_directional_reference tests.test_aniso_axis_reference -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpm-lite-mpl-cache .venv/bin/python -u -m benchmarks.aniso_nonuniform_reference_queue --devices cuda:2,cuda:0,cuda:3
```

队列会跳过完整且通过检查的既有轨迹，完成新增网格的时间对照、独立积分审计和剖面报告。若时间门槛失败，使用更小的 `--fine-dt` 补齐全部必要对照。当前仍缺新增终态、T 检查、局部／均匀差异比 G、时间排序稳定性及终态独立积分审计。

## 本次收尾：初态积分复核通过，完整比较仍受阻

两种新网格都完成真实尺寸的初态 Gauss 5/7 复核：

| 网格 | 能量相对差 | 力相对差 | 切线作用相对差 |
|---|---:|---:|---:|
| 局部 Y24 | 2.55e−16 | 5.58e−13 | 6.20e−15 |
| 较细 Y48 | 2.68e−15 | 9.69e−13 | 6.32e−15 |

均低于 1e−6 门槛；这是初态积分检查，不能替代终态复核或时间收敛。详见[原始积分报告](results/nonuniform-reference/v1/initial-quadrature.json)和[当前检查汇总](results/nonuniform-reference/v1/preflight-summary.json)。最终 CPU 回归共 17 项：15 通过、2 项真实 GPU 测试因环境不可用而跳过；新增测试还验证了 GPU 不可用时队列不会启动任何轨迹。

当前没有在后台运行的新参考轨迹。GPU 恢复后按前述命令继续；运行队列会每五分钟更新进度，并在停止／完成时再次更新。当前系统／数据盘剩余约 5.56/41.34 GiB，未触发迁移或容量暂停条件。

## CUDA 恢复与继续运行

真实设备验证见[恢复回归日志](results/nonuniform-reference/v1/tests-cuda-recovery.txt)：17 项通过、0 跳过。四条新增轨迹已启动，设备 0/1/2/3 各运行一条；运行入口、PID 和开始时间见[恢复记录](results/nonuniform-reference/v1/recovery-session.json)，队列日志为[queue-recovery.log](results/nonuniform-reference/v1/queue-recovery.log)。系统／数据盘约 7.1/41.3 GiB，未触发阈值。旧的 preflight-summary.json 保存阻碍阶段快照，当前状态以 status.json 与 PROGRESS.md 为准。

### 运行中隔离 GPU 1

GPU 1 的局部 Y 粗时间步停在 128/24576，设备利用率返回 N/A，其他三条仍正常推进。已仅终止该故障任务，保留其余三条进程及已完成 CPU 检查，由接管队列继续管理。局部 Y 粗时间步在屏蔽 GPU 1 的 GPU 0 上重启；GPU 0 的新上下文和内存分配验证通过。当前 GPU 0 同时运行两条约 4.6 GiB 任务，GPU 2/3 各运行一条较细 Y 任务；不据共享设备耗时评价等成本效率。

故障记录见[handoff.json](results/nonuniform-reference/v1/handoff.json)、[隔离后 CUDA 检查](results/nonuniform-reference/v1/cuda-after-isolation.txt)，当前队列为[handoff-queue.log](results/nonuniform-reference/v1/handoff-queue.log)。此前 queue-recovery.log 仅保留首次启动记录。原 GPU 1 的短日志带 gpu1-stalled 后缀保存，不计入验收轨迹。
