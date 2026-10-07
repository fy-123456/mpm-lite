# B：冻结材料空间中的正权积分压缩

这是 v22 基线上的独立研究实现。生产入口、公共本构、质量/惯性、夹持、旧归档均保持原样。新增内容只位于 B 所有的目录。完整进展和原始证据见 `docs/results/parallel-v22/B/` 下各次运行。

## 可调用接口

C 接入时推荐从 `binding.BoundMaterialFamily` 构造充分积分、压缩与局部回退算子，再通过其 `session` 创建事务。它拒绝其他参考材料来源的提案，即使零应变下能量跳变恰为零。低层 `MaterialSession` 只检查事务/能量预算，直接使用时调用方必须自己验证方向场和材料身份来源。

- `rules.MaterialRule`：不可变参考位置、正参考体积、二/四阶方向矩、互不重叠材料分区及内容签名。`compress` 同时研究实际代表点和条件矩质心；选择不使用材料响应。材料界面必须由输入的 `partition` 显式分开，较大的方向跳变会强制拆组。空间矩约束仅声明体积和一阶位置矩，不能据此宣称非均匀变形精确。
- `operator.LinearMaterialOperator`：`evaluate(q, direction)`、`energy`、`pk1`、`tangent_action`、分区 `weak_moments`。同一势能产生能量、PK1、内力和精确切线。`force` 表示 `dU/dq`，进入运动方程时由调用方采用其负号。没有正定替代切线。
- `operator.MaterialSession`：`trial` / `commit(accepted_step=True)` / `rollback`。Newton/线搜索中规则冻结，最后一次非法试探使之前的成功试探不可提交。`propose` 同状态双算规则变化，`commit_rule` 只接受未过期的提案；单次能量跳变和累计绝对跳变分别受预算控制。C/调用方拥有整步提交和几何、速度、F 历史；B 的当前超弹性模型没有塑性等物理内部变量。
- `rules.local_fallback`：只替换失败材料分区，不叠加体积。回退是步间显式请求，不会在 Newton 内偷偷改采样。
- `tensor.V22Space` / `TensorMaterialOperator`：读取已封存的 Q4 重叠支撑第六轮，通过稀疏局部基和张量延拓计算真实空间中的材料能量、组装内力和切线。材料 Gauss 积分按不重叠物理单元组织，与重叠空间支撑无关。
- `gpu_adapter.RepresentativeGPUOperator`：可选复用 D 的 `MaterialGPU`，只接受实际单方向代表点规则。一般条件 A4 会明确拒绝；未把 A4 错换为平均 A2 的外积。

位移约定是 `x = X + u(q)`、`F = I + grad(u)`；`q` 为 `(标量基数量, 3)`，每行对应同一个标量基的 xyz 系数。`gradient_map(X)` 的维度为 `(积分点数, 标量基数量, 3)`。位置用米、体积用立方米、能量用焦耳、PK1 用帕。参考 F 为单位阵，后续 F 来自调用方冻结的运动学映射；压缩规则不改参考历史。

PK1 是**压缩样本上的材料响应**；本实现没有压缩逐点应力重建器。精度证据是同空间组装内力、分区弱应力矩及切线差，不能以同一 F 重算同一本构作为压缩精度证明。

## 复现

在仓库根目录运行，输出目录必须不存在：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m unittest tests.research_b.test_material_compression tests.test_aniso_joint_sampling tests.test_aniso_v22_space -v
.venv/bin/python -m benchmarks.research_b.run --output docs/results/parallel-v22/B/NEW_RUN --stage all
.venv/bin/python -m benchmarks.research_b.diagnostics --output docs/results/parallel-v22/B/NEW_RUN
.venv/bin/python -m benchmarks.research_b.gpu_probe --output docs/results/parallel-v22/B/NEW_RUN
```

GPU 命令需要 D 的材料后端及 CUDA；CPU 命令不依赖 D。所有源文件和输入必须与相应协议的指纹一致。隐藏复核在候选和输入协议封存后执行；正式隐藏结果不能用于重新选择同一版本的规则。

主扫描保留三档预算的成功及失败记录。小场景另有真实约束平衡的加载—保持—卸载—末保持检查；它是准静态检查，不是时间积分。真实 v22 检查是固定状态材料压缩检查，没有把旧 v20 动力学接到新 Q4 空间。B9 必须等待与所测空间一致的 C 动态接口及时间验收；B10 的材料 CPU/CUDA 等价也不替代完整循环或端到端性能验收。

小型多项式空间按源码定义的固定坐标排列比较广义力：前三个系数为无量纲仿射分量，二次项系数单位为 1/m；报告中的小场景 `reaction` 是与轴向应变系数共轭的广义响应，不是原硬夹持梁的牛顿反力。真实 v22 空间保留全部夹持力分量。
