# A 第二阶段运行入口

本目录只新增 A 的 stage2 实现。父包为 `20260930T054100Z-common-inputs`，589 项绑定源码及 29 项成果保持只读。`CaseSpec` 真实控制方向、刚度、三维几何、夹持和加载；线性空间精度与 Hencky 有限变形算子检查分开。

## 启动及继续

在仓库根目录用现有 `.venv/bin/python` 执行。每个运行身份只能冻结一次，修复代码后使用新身份，不改写旧结果。

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.research_a.stage2.run --run-id YOUR_RUN_ID --phase prepare
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.research_a.stage2.run --run-id YOUR_RUN_ID --phase kickoff
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.research_a.stage2.run --run-id YOUR_RUN_ID --phase budgets
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.research_a.stage2.run --run-id YOUR_RUN_ID --phase ablation
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.research_a.stage2.run --run-id YOUR_RUN_ID --phase audit
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.research_a.stage2.run --run-id YOUR_RUN_ID --phase generalization
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m benchmarks.research_a.stage2.run --run-id YOUR_RUN_ID --phase audit
```

`prepare` 恢复共同源码快照并叠加 A 新代码；正式子进程从数据盘独立源码副本导入。历史数据按摘要复核，只读挂载。`kickoff` 包含七项针对性测试、九例构造检查和全物理域 Q2/Q3/Q4 参考可行性求解。构造网格不会被算作正式生产预算曲线。

`budgets` 在原生产网格上分别运行 72/144/288 个局部标量函数、六轮、每轮 4/8/16 个支撑。`ablation` 包含四种支撑、固定选区 Q2/Q3 对照、允许重新选区的阶次对照以及四种评分。`generalization` 在基准 144 空间的有限变形审核通过后，运行原八例；其中七例额外尝试复用 F45 基，几何变更例拒绝直接复用。

每个候选的六轮字段、稀疏原始基和变换都保存在 `arrays/candidates/`，对应结果保存内容摘要。成功阶段继续执行时跳过；失败阶段保持失败记录，修复需另建 run_id。检查点用于诊断和后续恢复开发，目前不提供跳过失败轮的自动恢复。

## 资源和可宣称范围

所有新数组和源码副本位于 `/root/autodl-tmp/mpm-lite-research-a/stage2/`，工作区仅保存小型协议、日志与结果链接。每个子进程最多 3600 秒和 24 GiB RSS，根分区和数据盘都保留 5 GiB；同时只运行一个 A 大任务。当前耗时属于共享机器上的探索测量，没有三次独占重复的加速比结论。

新数值实现检查采用工程容差，保留夹持、有限值、原势能、力符号、detF 和静态正定要求。原冻结证据的门槛不变。科学空间目标仍单独记录为应力/纤维 2%、反力 1%，不能由算子测试或图像稳定替代。

当前参考模块提供完整域、共形 p 收敛预检和大参考资源设计。局部非共形 hp 方法、新的独立最终保留场、A10 最终空间验收、A11 三次公平成本测量和 A12 可推荐空间发布尚未实现。现有 v21/v22 场都是已知开发数据，不能称为新的隐藏测试。

`progress` 阶段写入带时间戳的进度清单；每项能力都默认未认证。新结果不会自动覆盖共同输入，未通过或尚未执行的工作会原样保留。
