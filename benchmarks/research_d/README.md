# D：隐式求解器与 GPU 扩展

这是 v22 上的可选研究路径，不改变生产求解器、默认参数或旧归档。

## 运行

从仓库根目录运行，所有 `--out` 必须是尚不存在的新目录：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m benchmarks.research_d.run --out docs/results/parallel-v22/D/<run_id>
.venv/bin/python -m benchmarks.research_d.local_space --out docs/results/parallel-v22/D/<run_id>/local-space
.venv/bin/python -m benchmarks.research_d.performance --out docs/results/parallel-v22/D/<run_id>/performance
```

正确性实验和正式计时应串行使用 GPU。性能入口运行三个独立进程，每个场景均交替 AB/BA，记录构造、首次求解、复用求解和资源采样。CPU A 使用既有 SciPy CG 与 v22 可分离预条件；GPU B 使用同一数学问题、float64 和真实残差停止。CUDA 编译/加载另列；GPU 竞争会使性能验收失效。

失败的阶段保留原输出，修复后可用 `--resume --phases nonlinear` 等显式续算，仅写尚未存在的文件，并新增源码指纹。若阶段已有输出，应改用新目录。

## API

```python
from engine.aniso_phase1.research_d.tensor import FrozenTensor
from engine.aniso_phase1.research_d.gpu import TensorGPU, ResidentPCG

problem = FrozenTensor(edges, degree=2, H=material_hessian)
backend = TensorGPU(problem, device='cuda:0')
solver = ResidentPCG(backend, maxiter=2500, check_every=8)
u_free, info = solver.solve(problem.rhs, rtol=1e-7, atol=1e-12)
if not info.converged:
    raise RuntimeError(info.status)
field = problem.field(u_free)  # 自由跨距上的位移，节点排列 component,x,y,z
solver.close()
backend.close()
```

`q` 是位移：`x=X+u`、`F=I+grad(u)`。硬约束通过自由子空间消元；非零提升只用于原两端夹持，局部人工边界使用零提升。`FrozenTensor` 继承既有 Qp 和 `BoxElastic`，不添加物理刚度或质量。空间几何、材料、边界、阶数或资源身份改变后必须重建；对象不允许并发跨流复用。

张量路径目前支持固定、常切线材料的 Cartesian Qp 空间。异构材料明确使用独立组装的 CSR 控制，不能把平均材料预条件器当作物理算子。局部块及粗空间是固定 SPD 预条件；不定或非对称问题不得直接使用 PCG。负曲率、非有限值、迭代上限和真实残差失败均有独立状态。

`diagnostic_true_residuals=True` 会逐批重算真实残差，适合诊断。常驻计时默认仅在最终接受前重算；每次迭代的递推残差另存。Graph 捕获失败回到同一带保护的 launch 路径，诊断记录回退原因。

## 合同和范围

`engine/aniso_phase1/research_contracts.py` 定义 schema 1 的元数据、求解回调及 C 所有的事务协议；`tensor.solver_callbacks` 是真实静态算子的适配器。集成入口要求输入、源码和验收证据全部匹配封存摘要，不自动读取其他方向的最新结果：

```bash
.venv/bin/python -m benchmarks.research_d.integration.check path/to/package.json
```

动态接入必须另加 `--dynamic` 并具有 C 的动态验收和事务接口。当前新增张量 GPU 后端尚未接入移动 Lite／APIC；生产场景的 CPU/GPU 检查是既有求解器的回归证据。v22 全域空间精度未通过的问题也不会因求解加速自动解决。

系统盘阈值使用较保守的 5 GiB。D 缓存可迁移至独立数据盘 `/root/autodl-tmp/mpm-lite-research-d`；不移动活跃的其他研究方向数据。
