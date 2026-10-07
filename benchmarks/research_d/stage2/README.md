# D 第二阶段入口

本目录只扩展 `research_d/stage2`。父包、旧源码、依赖、生产入口保持原样。

固定输入：共同父包 `55682a7b…c7c`，A 空间 `7422b2b0…332`，219×3 个自由位移和 369×3 个完整位移。材料显式使用六阶，新增状态用七阶复核；质量仍为五阶、密度 1 kg/m³。

从项目根运行（沿用 `.venv`，四个 CPU 线程；大型副本和 Warp 缓存在数据盘）：

```bash
.venv/bin/python -m benchmarks.research_d.stage2.bootstrap /root/autodl-tmp/mpm-lite-research-d/stage2/checkout
cd /root/autodl-tmp/mpm-lite-research-d/stage2/checkout
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
/root/workspace/mpm-lite/.venv/bin/python -m benchmarks.research_d.stage2.pilot
/root/workspace/mpm-lite/.venv/bin/python -m benchmarks.research_d.stage2.gpu_pilot
/root/workspace/mpm-lite/.venv/bin/python -m benchmarks.research_d.stage2.validate
```

`pilot` 在求值前封存协议和状态；已有协议不允许被改写。`validate` 保存三个状态、两个方向、三档差分、完整刚度谱、预条件比较和 CPU/GPU 静态再平衡。中断后保留检查点和原始日志，不把中途退出当作通过。重新正式运行应使用新的结果目录，并保留旧记录。

性能入口 `performance <结果目录> --index 0|1|2` 需要正确性先通过、GPU 空闲。三个独立进程分别按 AB/BA/AB 运行冷、暖和重复求解。首次编译、共同包加载、GPU 构造和搜索矩阵构造单独记录。搜索矩阵由完整真实 GPU 初始切线生成；CPU/GPU 只将其用作相同的迭代辅助，分别重算原始能量和残量。这种复用范围不认证“从头构造的纯 CPU 算法”或一般生产性能。

公共合同在 `engine/aniso_phase1/research_d/stage2/contracts.py`：固体位移与 E 的二维位移—压力混合块分别有显式类型；状态、材料、积分、质量、边界和扩展代码分别绑定。`force` 为正势能梯度，机械内力为负；映射伴随不自带 dV。混合算子不能声明固体 PCG 能力。父包的动态快照语义不变，D 不定义时间推进。

`integration.inspect` 只发现其他方向的交接包。没有明确封存和固定身份时不会导入正在修改的其他方向 stage2 实现。

扩展加载用 `verify_extension(..., expected_sha256=...)`，它同时验证父包全部闭包和本次扩展源码、成果。动态 GPU、连续体空间精度和生产默认采用均需单独验收。

封存与独立验证入口：`python -m benchmarks.research_d.stage2.seal`；在新的独立副本导入并运行 `consumer <结果目录> --sha256 <固定摘要> --receipt <外层回执路径>`。消费方必须保存可信的固定摘要，不能以包内声明自动代替信任来源。`source-extension.zip`只包含新增源码；父包快照、全部父成果和v22源码清单仍须按`bootstrap`规则恢复。D的性能总成本估算复用了单次测得的搜索矩阵构造时间，独占资源的多进程从头构造仍待执行。
