# A 独立空间研究入口

基于 v22 已交付源码和结果，只向 A 命名空间及新结果目录写入。此版本完成首批可运行实现，整个 A1—A11 研究尚未完成。生产默认不切换。

在仓库根目录执行（建议显式限制线程）：

```bash
OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m benchmarks.research_a.suite --run-id <新的运行编号>
```

首批套件依次执行：输入/源码冻结、相关测试、真实 v22 Q4 重叠空间重新平衡、新参考资源预估、三组固定 144 函数六轮候选、真实空间非线性审核、训练/保留参考指标、空间包导出与重放、报告和指纹封存。支持族分别为原 v22 重叠和较宽重叠；选区对照使用无需参考的修正应力指标和确定性几何顺序。不会在保留参考上挑选中间轮次或修改默认候选。

单阶段入口均使用 `--run-id`：`protocol`、`baseline`、`reference_certification`、`support_ablation`、`nonlinear_audit`、`acceptance`、`export`、`validate_package`、`report`。具体参数见各入口的 `--help`。单阶段与套件使用同一实现；已存在的结果不会覆盖，修改源码后应创建新编号。

`support_ablation` 允许 `--budget 72/144/288`、`--degree 2/3/4`、四个预设支撑族与四个在线指标。额外预算或材料泛化未运行前不算完成。`reference_certification` 当前只负责整个过渡带加密的资源预估，不会自动启动数千万节点求解。

新数组会在预计系统盘余量不足 5 GiB 时直接写入 `/root/autodl-tmp/mpm-lite-research-a/<run_id>/scratch`；结果目录保留链接。若系统盘已低于 5 GiB，优先校验并迁移本组可再生 scratch。不会删除其他组文件或重写旧归档。移走/备份交接包时应解引用符号链接，同时保留 `space-package.json` 的校验摘要。

交接接口为 `engine.aniso_phase1.research_a.export_adapter.FixedSpace` 和 `package_loader.load_package`。q 的形状是 `[自由载体标量数+局部标量数, 3]`，表示位移；提供位置/F、JVP、VJP、原能量/力/切线，整个 Newton 内固定空间。动态惯性、状态迁移与历史提交由 C 负责，公共契约由 D 负责。

最终报告同时列出运行稳定性、相对参考差异和参考自身认证状态。当前参考全域/夹持区未认证，不能用本轮工程检查通过替代 1%/2% 空间目标，也不能据静态通过声称动态通过。

能量方向导数使用与旧 v22 相同的绝对差判据（默认 1e-8）；近零方向导数的相对差作为诊断单独保存。所有实际接受阈值由协议传入审核函数。首批旧协议中相对字段与绝对代码判据的差异和一项相对阈值超出已保留在对应运行的 `NUMERICAL_NOTES_ZH.md`，没有覆盖原始证据。

最新已交付结果：[首批实验及当前源码复核](../../docs/results/parallel-v22/A/20260930T0455Z-A-delivery-review/REPORT_ZH.md)。完整 A1—A11、严格空间精度、动态与联合验证的未完成项已逐项列出。
