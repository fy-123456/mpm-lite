"""A11 first-delivery evidence, explicit pending stages, source/artifact seal."""
import argparse
from pathlib import Path
from .protocol import ROOT, BASE, run_path, read_json, write_json, verify_baseline, sha, source_manifest


def figures(run, candidates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from engine.aniso_phase1.research_a.baseline_adapter import read_field
    from engine.aniso_phase1.tensor_metrics import sampling, evaluate_gradient
    from engine.aniso_phase1.tensor_reference import apply_axis
    from benchmarks.aniso_v21_common import hessian
    folder = run / "figures"
    folder.mkdir(exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name in candidates:
        records = [read_json(run / "candidates" / name / f"round{i}.json") for i in range(1, 7)]
        ranks = [r["scalar_local_dofs"] for r in records]
        axes[0].plot(ranks, [r["materials"]["F45"]["reaction_N"] for r in records], "o-", label=name.replace("q4-", ""))
        axes[1].plot(ranks, [r["materials"]["F45"]["energy_J"] for r in records], "o-")
    axes[0].set(xlabel="Added scalar functions", ylabel="Reaction (N)")
    axes[1].set(xlabel="Added scalar functions", ylabel="Original potential (J)")
    axes[0].legend(fontsize=7)
    fig.savefig(folder / "round-curves.png", dpi=150)
    plt.close(fig)
    paths = [BASE / "v22/multiscale/space/q4/round6.npz", run / "candidates" / candidates[0] / "round6.npz"]
    points = [np.linspace(.125, .875, 121), np.linspace(.375, .625, 45), np.array([.5])]
    X, Y = np.meshgrid(points[0], points[1], indexing="ij")
    samples = []
    for path in paths:
        edges, p, u = read_field(path)
        U = u.reshape(*(p*(len(e)-1)+1 for e in edges), 3)
        for k in range(3):
            U = apply_axis(sampling(edges[k], p, points[k]), U, k)
        gradient = evaluate_gradient((edges, p, u), points)
        stress = (gradient.reshape(-1, 9)@hessian("F45").T).reshape(gradient.shape)
        samples.append((U[:, :, 0], np.linalg.norm(stress[:, :, 0], axis=(-2, -1))))
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), layout="constrained")
    vmax = max(float(v[1].max()) for v in samples)
    for ax, (u, stress), title in zip(axes, samples, ["v22 archived overlap", "A reference-free overlap"]):
        color = ax.pcolormesh(X, Y, stress, shading="nearest", cmap="viridis", vmin=0, vmax=vmax)
        for j in range(0, len(points[1]), 4):
            ax.plot(X[:, j]+10*u[:, j, 0], Y[:, j]+10*u[:, j, 1], color="white", lw=.55)
        for i in range(0, len(points[0]), 8):
            ax.plot(X[i]+10*u[i, :, 0], Y[i]+10*u[i, :, 1], color="white", lw=.55)
        ax.axvline(.25, color="red", ls="--", lw=.7)
        ax.axvline(.75, color="red", ls="--", lw=.7)
        ax.set(title=title+"; displacement lines x10; raw stress at z=0.5", xlabel="x (m)", ylabel="y (m)")
        ax.set_aspect("equal")
    fig.colorbar(color, ax=axes, label="Stress Frobenius norm (Pa)")
    fig.savefig(folder / "scene-comparison.png", dpi=150)
    plt.close(fig)


def seal(run, candidates):
    if (run / "completion-check.json").exists():
        raise FileExistsError("Run is already sealed")
    baseline = verify_baseline()
    write_json(run / "baseline-after.json", baseline)
    reproduction = read_json(run / "baseline/reproduction.json")
    tests = read_json(run / "tests.json")
    acceptance = read_json(run / "held-out-acceptance.json")
    rows = {}
    for name in candidates:
        folder = run / "candidates" / name
        summary, audit = read_json(folder / "summary.json"), read_json(folder / "nonlinear-audit.json")
        metric = acceptance["candidates"][name]
        rows[name] = dict(summary=summary, nonlinear_passed=audit["passed"], metric=metric)
    package = read_json(run / "candidates" / candidates[0] / "package-validation.json")
    stable = (baseline["passed"] and reproduction["passed"] and tests["passed"] and package["passed"]
              and all(v["summary"]["all_static_passed"] and v["nonlinear_passed"] for v in rows.values()))
    completion = dict(first_delivery_implementation_completed=True, first_delivery_tests_passed=stable,
        all_A1_to_A11_completed=False, spatial_accuracy_passed=False, reference_certified=False,
        stages={"A1": "completed", "A2": "completed: same archived basis freshly equilibrated",
                "A3": "resource plan only; no new certified reference", "A4": "archived controls plus new nonlinear quadrature check",
                "A5": "144-scalar six-round pilot; full support/cost sweep pending", "A6": "reference-free pilot and geometric control",
                "A7": "actual candidate statics and nonlinear audits completed", "A8": "144-only evaluation; 72/288 sweep pending",
                "A9": "unseen material/geometry/load accuracy not tested", "A10": "static package exported and replayed; joint tests pending",
                "A11": "first delivery sealed, entire research program ongoing"},
        production_defaults_changed=False, dynamic_scene_validated=False, cuda_validated=False,
        baseline_488_sources_131_artifacts_unchanged=baseline["passed"], candidates=rows,
        integration_status="B/C/D/E not loaded or modified; combined tests pending")
    figures(run, candidates)
    table = []
    old = acceptance["archived_controls"]["q4-multiscale144"]
    def metric_row(name, metric):
        return f"| {name} | {100*metric['reaction_relative']:.3f}% | " + " | ".join(
            f"{100*metric['regions'][region]['stress_relative']:.3f}%" for region in ("global", "grip", "interior", "deep_interior")) + " |"
    table.append(metric_row("v22 Q4 重叠（归档数值）", old))
    table.extend(metric_row(name, row["metric"]) for name, row in rows.items())
    plan = read_json(run / "reference-plan.json")
    peaks = [level["cases"]["q4"]["predicted_peak_bytes"]/1024**3 for level in plan["levels"]]
    text = f"""# A 首批实施与验证报告

基线为已核验的 v22 源码归档。此次已实施可独立运行的空间适配、有限支撑族、无需参考场的在线指标、原势能全局再平衡、实际候选非线性审核和静态空间交接包。首批稳定性检查状态：**{'通过' if stable else '未通过'}**。

这是按计划开工后的首批交付；A1—A11 全部研究尚未完成，不能将本报告理解为全域 2% 精度或动态场景已经通过。旧源码、默认求解器与旧成果均未改变。

## 结果

144 个新增标量函数、六轮固定预算。原 225 个自由载体位移自由度保留，新增 432 个位移自由度；每一轮均实际重新平衡。以下为相对同一 v22 最终离散参考的差异，含全部夹持区域，无应力平滑。

| 空间 | 反力差 | 全域应力差 | 夹持应力差 | 内部应力差 | 深内部应力差 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(table)}

绝对应力 RMS、纤维应变和积分区域体积见 `held-out-acceptance.json`。归档对照明确标注为旧数值；最新 Q4 重叠基底另外完成了重新组装和求解，见 `baseline/reproduction.json`。

![场景对照](figures/scene-comparison.png)

![逐轮反力和能量](figures/round-curves.png)

## 正确性与稳定性

- 新增接口与原有相关回归测试见 `tests.json`、测试日志。
- 每个实际候选六轮均检查 ISO/F0/F45/F90 的无质量静态刚度、局部刚度与 Schur 补；没有通过额外刚度、质量或耗散消除零模态。
- 验证硬夹持、自然自由面、人工边界、二次多项式与弯曲表达、嵌套空间能量下降。
- 完整选区测试禁止文件读取与 `numpy.load`；线上指标只接收残差、局部修正与同一材料的局部算子。最终参考只在所有候选冻结后评价。
- 实际 144 函数空间检查三个扰动尺度的能量/力/切线，以及有限旋转和平移；另记录非线性积分加密差异。用户要求的工程稳定性与严格空间精度分别报告。
- 静态空间包经过独立加载后重放位置、F、能量、力和切线。q 表示位移，使用 `x=X+u(q)`、`F=I+grad(u)`，Newton 内空间固定。

## 资源与限制

新参考对整个夹持过渡带及横向边界带预估了两级 Q3/Q4 加密；Q4 预测峰值约 {peaks[0]:.1f}/{peaks[1]:.1f} GiB。超过本轮预先冻结的节点/内存预算时不启动；新参考尚未认证。原参考的全域/夹持区认证也仍未通过，因此没有作全域空间精度通过声明。

新数组优先存放在数据盘的本组 scratch，通过相对链接保持 A 结果目录可访问。源文件和归档保留原位。磁盘、线程、并发负载在协议中留档；本轮耗时是实测运行成本，不能当作独占设备的公平加速比。每个候选的基构造、局部求解、全局平衡成本分别在逐轮 JSON 中。

尚待推进：A3 大参考实际 h/p 认证、A4 完整网格/阶次归因、A5 完整支撑与成本匹配、A6 指标泛化、A8 的 72/288 收敛曲线、A9 未见材料/几何/加载、A10 与 B/C/D 的联合验证。A 不负责动态历史推进；本次没有将静态测试冒充动态稳定性验收。

## 复现与交接

入口说明见仓库 `benchmarks/research_a/README.md`。协议、输入指纹、运行源码快照、原始逐轮场与指标均保留。`completion-check.json` 单独区分首批完成、完整研究完成和空间精度通过。`source-sha256.json` 与 `artifact-sha256.json` 用于核查。
"""
    (run / "REPORT_ZH.md").write_text(text)
    write_json(run / "completion-check.json", completion)
    write_json(run / "source-sha256.json", source_manifest())
    files = {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*")) if p.is_file()
             and p.name != "artifact-sha256.json" and p.suffix != ".log" and "scratch" not in p.relative_to(run).parts}
    # Live process logs are intentionally excluded; per-stage closed logs can be
    # sealed by copying after the suite returns. JSON and numeric evidence are sealed.
    write_json(run / "artifact-sha256.json", files)
    return completion


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--candidate", action="append", required=True)
    a = p.parse_args()
    seal(run_path(a.run_id), a.candidate)
