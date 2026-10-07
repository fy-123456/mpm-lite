"""Seal scoped B evidence and schema-v1 handoff, without claiming C dynamics."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import zipfile
from .run import ROOT, sha, write, source_hashes, verify_baseline


def storage_record(output):
    before=shutil.disk_usage('/').free
    record=dict(threshold_bytes=5*2**30,threshold_convention='conservative 5 GiB',
                before_available_bytes=before,migrated=False)
    # User authorized migration only below the threshold. Keep pip lookup
    # paths usable; never move active Warp caches or archived research input.
    if before < 5*2**30:
        source=Path('/root/.cache/pip')
        destination=Path('/root/autodl-tmp/mpm-lite/storage-migrations/B-20260930-pip-cache')
        if source.is_dir() and not source.is_symlink() and not destination.exists():
            manifest={str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()}
            size=sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(destination))
            source.symlink_to(destination,target_is_directory=True)
            if any(sha(destination/n)!=digest for n,digest in manifest.items()):
                raise RuntimeError('migrated cache verification failed')
            record.update(migrated=True,source=str(source),destination=str(destination),
                          verified_files=len(manifest),bytes_moved=size,original_path_preserved='symlink')
        else:
            record['migration_note']='eligible pip cache already absent/migrated; no other data touched'
    record.update(after_available_bytes=shutil.disk_usage('/').free,
                  data_available_bytes=shutil.disk_usage('/root/autodl-tmp').free)
    write(output/'storage.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve()
    load=lambda name:json.loads((out/name).read_text())
    baseline=verify_baseline()
    protocol=load('protocol.json');small=load('small-scan.json');tensor=load('v22-scan.json')
    hidden_small=load('hidden-small-result.json');hidden_tensor=load('hidden-v22-result.json')
    gpu=load('final-backend-revalidation/cuda-equivalence.json');replay=load('guard-revalidation/finite-state-replay.json')
    feedback=load('feedback-scene.json');cost=load('cpu-cost.json');space=load('v22-space.json')
    assert hidden_small['passed'] and hidden_tensor['passed'] and gpu['passed'] and replay['passed']
    assert tensor['reference_certified'] and feedback['scene_passed']
    for filename,count in [('guard-regression-tests.log',20),('material-origin-tests.log',2)]:
        log=(out/filename).read_text();assert f'Ran {count} tests' in log and '\nOK\n' in log
    # Added binding/sealing helpers do not alter any independently reviewed
    # numerical source. Record that relationship instead of rewriting history.
    hidden_protocol=load('hidden-v22-protocol.json')
    for name,digest in hidden_protocol['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in gpu['source_sha256'].items():assert sha(ROOT/name)==digest,name
    files=source_hashes()
    for name,digest in protocol['reused_source_sha256'].items():
        assert sha(ROOT/name)==digest;files[name]=digest
    files['engine/aniso_phase1/research_d/material.py']=gpu['source_sha256']['engine/aniso_phase1/research_d/material.py']
    inputs=dict(protocol['input_sha256'])
    for name in ('v22-selected-rule.json','snapshots.npz','hidden-v22-inputs.npz','hidden-small-protocol.json'):
        path=out/name;inputs[str(path.relative_to(ROOT))]=sha(path)
    from engine.aniso_phase1.research_contracts import HandoffMetadata, validate_package
    contract=ROOT/'engine/aniso_phase1/research_contracts.py'
    capabilities=('static','energy','sample_PK1_linear_adapter','exact_tangent','tensor_weak_stress',
                  'positive_reference_rule','C_owned_rule_transactions','representative_CUDA_material_equivalence')
    metadata=dict(schema_version=1,baseline_sha256=baseline['source_zip_sha256'],producer='B',
        code_sha256=files,input_sha256=inputs,units=protocol['units'],dtype='float64',device='CPU; optional D CUDA representative adapter',
        coordinate_system='reference Cartesian',q_convention='displacement',material_parameters=protocol['material'],
        boundary_conditions=protocol['boundary'],volume_convention='positive reference volumes; disjoint material partitions; mass/inertia unchanged',
        random_seed=220930,capabilities=capabilities)
    HandoffMetadata(**metadata).validate()
    chosen=tensor['selected_order'];rows=[r for r in tensor['rows'] if r['order']==chosen]
    summary=dict(producer='B',baseline_sha256=baseline['source_zip_sha256'],code_sha256=files,input_sha256=inputs,
        passed=True,validated_capabilities=list(capabilities),independent_material_stage_completed=True,full_B_plan_completed=False,
        complete_dynamic_acceptance=False,continuum_spatial_acceptance=False,formal_end_to_end_speedup=False,
        selected_small=small['selected'],selected_v22_order=chosen,
        v22_samples=rows[0]['samples'],reference_samples=rows[0]['full_samples'],
        material_evaluation_reduction=1-rows[0]['samples']/rows[0]['full_samples'],
        development_max_errors={k:max(r['errors'][k]['relative'] for r in rows) for k in rows[0]['errors']},
        hidden_small_max_errors=hidden_small['max_errors'],hidden_v22_max_errors=hidden_tensor['max_errors'],
        related_tests_passed=22,hidden_state_direction_pairs=64,
        valid_state_guard_replay_bitwise_unchanged=replay['bitwise_unchanged'],
        provenance_binding='BoundMaterialFamily/BoundMaterialSession tested independently after hidden freeze; original MaterialSession is low-level and requires caller source checks',
        B9='not accepted: C currently provides a different small Q1+Q4 model; no certified matching dynamics for the archived v22 225+144 scalar coefficient space',
        B10='CPU and representative material CUDA equivalence passed; full GPU tensor operator, dynamic cycles and end-to-end timing not accepted',
        candidate_origin='v1 main scan; v1.1 finite guards with bitwise finite-state replay and one-shot hidden review; added material-origin binding leaves reviewed numerical core unchanged',
        limitations=['No compressed pointwise stress reconstruction','No new E material law acceptance',
                     'No runtime guarantee outside sampled deformation/direction families',
                     'Tensor det/finite validation and linear-adapter singular-value floor differ; integration must declare common valid domain'])
    write(out/'acceptance.json',summary)
    package=dict(metadata=metadata,acceptance=dict(path=str((out/'acceptance.json').relative_to(ROOT)),sha256=sha(out/'acceptance.json')),
        integration_entry='engine.aniso_phase1.research_b.binding.BoundMaterialFamily',
        contract_source_sha256=sha(contract),dynamic_capability=False)
    validate_package(package,ROOT,baseline['source_zip_sha256'])
    dynamic_rejected=False
    try:validate_package(package,ROOT,baseline['source_zip_sha256'],require_dynamic=True)
    except ValueError:dynamic_rejected=True
    assert dynamic_rejected
    write(out/'handoff-package.json',package)
    write(out/'handoff-validation.json',dict(static_accepted=True,dynamic_correctly_rejected=True,contract_sha256=sha(contract)))
    storage=storage_record(out)
    timing={o:statistics.median(r['seconds'] for r in cost['interleaved'] if r['order']==o) for o in (chosen,6)}
    h=hidden_tensor['max_errors'];hs=hidden_small['max_errors']
    report=f'''# B 材料压缩：独立阶段交付

完成时间：{datetime.now(timezone.utc).isoformat()}。基于当前 v22 归档，交付前复核 **488 项源码、131 项成果全部匹配**。没有 Git 元数据；身份采用 SHA256。源码包：`{baseline['source_zip_sha256']}`。

**结论：B 的独立材料压缩阶段通过，尚未完成与 C 的真实 v22 动力学全循环及端到端性能验收。** 公共求解器、默认算法、夹持、材料参数、质量/惯性和 v1–v22 归档未改。

## 可复核结果

| 项目 | 结果 |
|---|---:|
| 真实固定空间 | Q4 重叠支撑第六轮；225 个载体标量系数 + 144 个局部标量系数，xyz 共 1107 个总系数，含夹持系数 |
| 基重建相对差 | {space['reconstruction_relative_error']:.3g} |
| 选定正权 Gauss 规则 | 每材料单元四阶；六阶充分积分以七阶自检 |
| 材料求值点 | 2,985,984 → 884,736，减少 70.37% |
| 相对旧 v22 五阶点数 | 1,728,000 → 884,736，减少 48.8%；不是旧生产路径整体加速承诺 |
| 隐藏能量差 | {100*h['U']['relative']:.6f}% |
| 隐藏组装内力差 | {100*h['force']['relative']:.6f}% |
| 隐藏分区弱应力矩差 | {100*h['weak_moments']['relative']:.6f}% |
| 隐藏切线作用差 | {100*h['tangent_action']['relative']:.6f}% |
| 隐藏最小 detF | {hidden_tensor['min_detF']:.6f} |

全部方向和失败工况保存在 `v22-scan.json`。三阶在局部切线作用上误差 2.638%，因此没有选它；四阶开发集最大切线差 1.437%。没有删除困难夹持区域或平滑应力。体积和低阶空间矩按不重叠材料单元验证；重叠的是运动学支撑，不是材料体积。

小问题中，条件矩质心压缩未满足全部开发预算；实际位置—方向配对的正权代表点规则以每分区 8 组通过，典型点数 1536 → 516。三个独立隐藏整族覆盖新角度、混合方向、连续转向和出平面方向，12 状态 × 5 切线方向；最大能量/内力/弱矩/切线差分别为 {100*hs['U']['relative']:.4f}% / {100*hs['force']['relative']:.4f}% / {100*hs['weak_moments']['relative']:.4f}% / {100*hs['tangent_action']['relative']:.4f}%。实际 v22 另有 2 新状态 × 2 独立方向。隐藏协议先封存、只打开一次，没有用隐藏结果调参。

小型约束平衡加载—保持—卸载—末保持共 17 帧全部收敛，完整材料点检查未翻转；轴向广义响应相对差 {100*feedback['raw_reaction_relative_error']:.5f}%。这里的 `reaction` 与仿射应变系数共轭，不能冒称原硬夹持梁的牛顿反力，也不是动态时间验收。小空间前三个仿射系数无量纲、二次系数为 1/m；误差范数固定在声明的同一基和系数顺序中。

![积分误差与准静态响应](validation-overview.png)

## 步骤、稳定性和能力边界

| 步骤 | 状态与证据 |
|---|---|
| B1–B2 | 实际 v22 空间、原势能接线、小问题独立公式、相邻阶积分认证通过；见 baseline-wiring / reference-certification / supplemental-diagnostics |
| B3–B6 | 整族划分、三档预算、正权分区、能量/内力/弱矩/精确切线通过；保留所有失败规则 |
| B7 | B4 已得到可行候选，因此未启动响应经验拟合；独立隐藏复核已通过 |
| B8 | 固定规则、同状态提案、局部充分积分回退、单次及累计绝对能量账本、接受/拒绝/回滚通过；C 仍拥有整步提交 |
| B9 | 未验收：当前 C 新增的是不同的小空间模型，不能把它或旧 v20 拼接为此 Q4 空间动力学 |
| B10 | CPU 与 D 后端的实际代表点 CUDA 材料/组装内力/精确切线等价通过；一般条件 A4、完整 Q4 GPU、动态总成本未验收 |
| B11 | 本报告、算子、规则、原始数据、来源、失败记录、机器验收及交接包已封存 |

相关回归 **22 项通过**（20 项材料/联合采样/v22 回归，加 2 项材料来源约束），不是全仓库测试。非法 detF、非有限材料响应、失败试探、过期提案与回滚均有检查。保护修复前源码和结果没有覆盖；合法状态重放逐位不变，最终数值核心与隐藏复核指纹一致。

独立复核指出低层 `MaterialSession` 不独自认证方向场来源。交付另加 `BoundMaterialFamily` / `BoundMaterialSession` 作为 C/B 推荐入口，只允许同一充分积分源产生的压缩及回退规则，零应变态换方向也会被拒绝。这个来源包装有单独两项测试，没有改动已打开隐藏集的规则。直接使用低层事务必须由调用方保证来源。当前材料为超弹性，无塑性内部变量；E 新材料需重新验收。

没有逐点压缩应力重建器；PK1 接口给出压缩样本材料响应，应力精度只声明弱形式矩。Tensor 与小适配器的近奇异状态下有效域保护还需在 C/D 接入时统一；本报告不保证所有未采样的大变形场景。原 v22 连续体空间精度仍未通过，B 压缩通过不能替代该结论。

## 成本与磁盘

真实空间基准备 {space['build_seconds']:.3f} 秒，基数组 {space['basis_array_bytes']/2**20:.1f} MiB，CPU 主运行峰值 RSS {cost['peak_process_RSS_bytes']/2**20:.1f} MiB。暖机后交错三次能量+内力评估的中位数：四阶 {timing[chosen]:.3f} 秒，六阶 {timing[6]:.3f} 秒。同机存在 C/D/其他任务竞争，以上只作诊断耗时，不宣称正式加速；没有计入完整 Newton/Krylov/动力学成本。GPU 型号与 UUID、驱动、Warp、首次编译日志见 cuda-equivalence 与日志文件；不能以材料 kernel 等价推出 GPU 更快。

系统盘最后检查可用 {storage['after_available_bytes']/2**30:.3f} GiB。按保守 5 GiB 阈值检查；{'已将闲置 pip 下载缓存迁移到数据盘，哈希核对并保留原路径符号链接，详见 storage.json。' if storage['migrated'] else '未低于阈值，本轮没有迁移或删除项目数据。'}

## 重放与交接

常规使用与新运行命令见 [实现说明](../../../../../engine/aniso_phase1/research_b/README.md)。新运行必须使用新目录。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m benchmarks.research_b.run --output docs/results/parallel-v22/B/NEW_RUN --stage all
.venv/bin/python docs/results/parallel-v22/B/{out.name}/replay_hidden.py --workspace /tmp/NEW_B_HIDDEN_REPLAY
```

隐藏重放脚本从原 v22 源码包及 `source-v1.1-hidden-reviewed.zip` 建立隔离目录，链接只读原输入并复制冻结协议；后加的来源包装不会让旧隐藏复核的源码集合检查失效。重放是旧证据复现，不是新的独立隐藏集。`source-v1-before-guard.zip` 保留初次开发扫描实现；`source-final.zip` 保存完整交付。

`handoff-package.json` 使用 D 的 schema 1，已检查静态能力接受、动态请求正确拒绝。引用固定输入/代码/验收散列，禁止自动读取未验收的“最新”其他组输出。`artifact-sha256.json` 封存全部本轮成果，`changed-files.json` 列出本任务新增/修改文件。主协议里早期 C/D 未就绪及候选待隐藏状态属于历史记录；当前状态以 acceptance.json 为准。
'''
    with open(out/'REPORT_ZH.md','x') as f:f.write(report)
    plan=ROOT/'docs/parallel_research_v22_20260930/B_MATERIAL_COMPRESSION_PLAN_ZH.md'
    original=plan.read_text()
    note=f'\n> 2026-09-30 实施更新：B 独立材料阶段已完成并通过固定空间和独立隐藏复核；B9 完整动力学、B10 全链路 GPU/性能仍未验收。见[本轮交付报告](../results/parallel-v22/B/{out.name}/REPORT_ZH.md)。以下原计划和门槛保留作为历史协议。\n'
    if note not in original:plan.write_text(original.replace('\n', '\n'+note,1))
    owned=[p for folder in ('engine/aniso_phase1/research_b','benchmarks/research_b','tests/research_b')
           for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    owned.append(plan)
    write(out/'changed-files.json',dict(new_files=[str(p.relative_to(ROOT)) for p in owned if p!=plan],
          modified_files=[str(plan.relative_to(ROOT))],new_results_prefix=str(out.relative_to(ROOT)),
          preexisting_baseline_source_modified=False,other_research_namespaces_modified=False))
    with zipfile.ZipFile(out/'source-final.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in owned:z.write(p,str(p.relative_to(ROOT)))
    with zipfile.ZipFile(out/'source-D-dependencies.zip','x',zipfile.ZIP_DEFLATED) as z:
        for dependency in (ROOT/'engine/aniso_phase1/research_d/material.py',contract):
            z.write(dependency,str(dependency.relative_to(ROOT)))
    write(out/'source-final-sha256.json',{str(p.relative_to(ROOT)):sha(p) for p in owned})
    for name in ('research_b_v1','research_b_cuda','research_b_guard_cuda','research_b_guard_replay','research_b_diagnostics','research_b_final_cuda'):
        source=Path('/tmp')/(name+'.log')
        if source.exists():shutil.copy2(source,out/(name+'.log'))
    write(out/'baseline-final-check.json',baseline)
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'}
    write(out/'artifact-sha256.json',manifest)
    assert all(sha(out/n)==digest for n,digest in manifest.items())
    print('SEALED',out,'artifacts',len(manifest),'source_files',len(owned),'dynamic_acceptance',False)


if __name__=='__main__':main()
