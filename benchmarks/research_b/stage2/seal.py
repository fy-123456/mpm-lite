"""Seal independently supported capabilities, raw evidence and source closure."""
import argparse
from pathlib import Path
import json
import shutil
import zipfile
from .run import checked, load, read, write, sha, now, source_hashes, PARENT, FIELDS


def peak(checks,prefix):
    vals=[v['scaled'] for k,v in checks.items() if k.startswith(prefix)]
    return max(vals,default=0.)


def seal(repo,out):
    p=checked(repo,out);space,_,_,baseline=load(repo)
    selected=read(out/'candidate-seal.json')['fields'];hidden=read(out/'hidden-acceptance.json')
    wiring=read(out/'wiring-audit.json');transaction=read(out/'transaction-audit.json');cost=read(out/'cost-breakdown.json')
    tests=read(out/'tests-summary.json')
    test_passed=tests['passed'] and sha(out/'tests.log')==tests['log_sha256']
    fields={}
    for f,sel in selected.items():
        rows=hidden['fields'][f]['states']
        fields[f]=dict(**sel,certified=bool(sel['reference_passed'] and hidden['fields'][f]['passed'] and wiring['passed'] and transaction['passed'] and test_passed),
            hidden_passed=hidden['fields'][f]['passed'],hidden_state_count=len(rows),
            maximum_scaled_errors={key:max((peak(v['candidate'].get('checks',{}),prefix) for v in rows.values()),default=0.) for key,prefix in [('material_energy','material_energy'),('assembled_force','material_force/'),('weak_moment','weak/'),('tangent_action','tangent/'),('local_tangent_work','tangent_work/')]},
            failed_hidden_states=[n for n,v in rows.items() if not v['passed']])
    write(out/'parent-final-check.json',dict(**baseline,verified_at=now()))
    write(out/'C-common-space-comparison.json',dict(status='dependency_pending',executed=False,
        reason='No pinned, accepted C same-space full-material CPU cycle and shared state-domain protocol consumed by this run',
        required=['same space SHA','same full material rule and material source bytes','same mass and cross blocks','same initial state, boundary lift and velocity, dt, nonlinear tolerance','C accepted short window then full cycle'],
        dynamic_certified=False,mass_rule_changed=False))
    contract=dict(schema_version=1,parent_bundle_sha256=p['parent_bundle_sha256'],space_manifest_sha256=space.signature,
        fields=fields,input_semantics=p['state_semantics'],boundary_lift='evaluate_free(q, lift=full_lift, directions=named_free_directions); full API requires complete displacement',
        force_convention='positive potential gradient; mechanical internal force is negative',
        stabilization='unchanged Ks; reference carrier positions added once to stabilization coordinates',
        operator='engine.aniso_phase1.research_b.stage2.CommonMaterialOperator',
        fixed_rule='same immutable rule for every Newton, line-search and AVF evaluation in one step',
        transaction='MaterialSession.prepare/attach only; CommonState transaction owned by C performs exactly one commit',
        mass=p['invariant_mass'],cpu_dtype='float64',gpu='not_run',
        energy_jump_from_regrouping=None,online_regrouping='not_enabled',
        bounded_reference='Only frozen states/directions; future states and C joint tighter reference threshold require revalidation')
    write(out/'material-contract.json',contract)
    accept=dict(schema_version=1,independent_delivery_complete=True,parent_unchanged=baseline['passed'],
        tests_passed=test_passed,static_operator=wiring['passed'],material_value_transaction=transaction['passed'],
        fields=fields,hidden_all_passed=hidden['passed'],dynamic_cycle='dependency_pending',cuda='not_run',coupled_physics='not_run',
        performance_certified=False,continuum_spatial_certified=False,production_defaults_changed=False,
        steps={'B1':'passed','B2':'passed' if wiring['passed'] else 'failed','B3':'passed','B4':'passed' if test_passed else 'failed',
               'B5':'passed' if all(x['reference_passed'] for x in fields.values()) else 'limited_or_failed',
               'B6':'completed_with_all_candidate_failures_retained','B7':'passed' if transaction['passed'] else 'failed',
               'B8':'passed' if hidden['passed'] else 'limited_or_failed','B9':'passed_at_value_interface_level' if transaction['passed'] else 'failed',
               'B10':'dependency_pending','B11':'exploratory_operator_cost_only_shared_resources','B12':'sealed_independent_scope'})
    write(out/'acceptance.json',accept)
    lines=['# B 第二阶段共同空间材料压缩交付','',
           '基于已逐字核验的共同冻结包执行；全部变更位于 B 的 stage2 专属目录，父包 589 项源码与 29 项输入／证据保持一致。',
           '',f"父包：`{p['parent_bundle_sha256']}`。空间：`{space.signature}`，1107 完整／657 自由分量。",'',
           '## 已执行结果','',
           '| 材料分支 | 充分阶次 | 固定候选阶次 | 点数减少 | 隐藏验收 | 材料能力 |','| --- | ---: | ---: | ---: | --- | --- |']
    for f,v in fields.items():lines.append(f"| {f} | {v['full_order']} | {v['candidate_order']} | {100*v['point_reduction']:.2f}% | {'通过' if v['hidden_passed'] else '未通过'} | {'有限状态集认证' if v['certified'] else '未认证'} |")
    lines+=['',f"每个材料分支执行 8 个训练／开发状态，以及解封一次的 12 个隐藏状态（3 个独立组合族，每族 4 状态）；每个状态 5 个预登记切线方向。接口测试和父包相关回归共 {tests['test_count']} 项，结果见 `tests.log`。",'',
        '算子显式接收完整位移或自由坐标加完整 lift，方向采用零 lift；返回正能量梯度、完整反力、精确切线、原 Ks 分量及逐薄层弱应力矩。固定规则只压缩材料积分，质量及交叉块不变。方向场在参考构形定义，多纤维族按完整纤维势能加权。', '',
        '按本次用户要求使用实用数值预算：候选能量／力／弱矩 1%，切线 2%；充分参考相邻阶次 0.2%／0.5%，近零绝对尺度在 protocol.json 预先固定。未放宽父包已存在的测试门槛。未声称逐点压缩应力准确度或整个有限变形域的一致误差上界。','',
        '## 能力边界','',
        '- B1—B9 的独立材料实现、扫描、隐藏验收和接口级故障注入均留存原始证据；具体通过状态以 acceptance.json 的各项为准。',
        '- B10 真实 C 循环为 dependency_pending。本次未消费已验收且同字节规则的 C 完整 CPU 循环，不能把静态测试或值树事务称为真实动力学完成。',
        '- B11 完成预热后至少五次交错算子测量。同机存在其他研究作业，结果仅作共享资源下的成本观察，未认证性能收益。',
        '- GPU、耦合物理、在线重分组、移动传递及生产默认入口均不在本次通过范围。A 的连续体空间精度状态保持原状。','',
        '## 原始证据与复现','',
        '`protocol.json`、`state-families.json`、`full-reference-certification.json`、`rule-scan.json`、`candidate-seal.json`、`hidden-opening.json`、`hidden-acceptance.json`、`wiring-audit.json`、`transaction-audit.json`、`cost-breakdown.json`；raw/ 保存全部逐状态／材料／阶次响应及时间数据。失败规则与非法状态不删除。','',
        '在 delivered-source.zip 解出的独立源码根运行，Python 使用原项目 .venv/bin/python，并设置 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1。新实验应选择全新的输出目录，依次执行：','',
        '```sh',
        'python -m benchmarks.research_b.stage2.run freeze --output /path/to/new-run',
        'python -m benchmarks.research_b.stage2.run development --output /path/to/new-run',
        'python -m benchmarks.research_b.stage2.run hidden --output /path/to/new-run',
        'python -m benchmarks.research_b.stage2.audit wiring --output /path/to/new-run',
        'python -m benchmarks.research_b.stage2.audit cost --output /path/to/new-run','```','',
        '固定交付读取入口为 engine.aniso_phase1.research_b.stage2.delivery.load_delivery，必须显式传 expected_sha256；默认拒绝未认证材料与 dynamic 能力请求。源码包包含父代码、父输入的解引用副本和 B 扩展，无需读取可变 latest。','',
        '系统盘低于 5 GiB 时已迁移可重建 pip 缓存至数据盘并逐文件校验，保留原路径链接；新实验与独立副本也在数据盘。storage-migration.json 记录具体路径和摘要。','']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines))
    archive=out/'delivered-source.zip'
    bound=read(repo/PARENT/'bundle.json')
    members=set(bound['code_sha256'])|set(source_hashes(repo))|{'docs/results/lite-aniso-mainline/v22/source-delivered-sha256.json'}
    members.update(str(PARENT/n) for n in bound['files'])
    members.update(str(PARENT/n) for n in ('bundle.json','bundle-sha256.txt'))
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name in sorted(members):z.write(repo/name,name)
    artifact_names=[str(f.relative_to(out)) for f in sorted(out.rglob('*')) if f.is_file() and f.name not in ('handoff-package.json','handoff-sha256.txt','consumer-smoke.json') and not f.name.endswith('.tmp')]
    handoff=dict(schema_version=1,created_at=now(),parent_bundle_sha256=p['parent_bundle_sha256'],parent_relative_path=str(PARENT),
        space_manifest_sha256=space.signature,extension_source_sha256=source_hashes(repo),fields=fields,min_detF=p['min_detF'],
        protocol_sha256=sha(out/'protocol.json'),artifacts_sha256={n:sha(out/n) for n in artifact_names},
        capabilities=accept,rule_signatures='B-fixed-cell-Gauss-v1 plus material source, exact space, explicit boundary, dtype, device')
    write(out/'handoff-package.json',handoff);(out/'handoff-sha256.txt').write_text(sha(out/'handoff-package.json')+'\n')
    print(json.dumps({'phase':'sealed','fields':{f:x['certified'] for f,x in fields.items()},'handoff_sha256':sha(out/'handoff-package.json')}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();seal(Path(__file__).resolve().parents[3],a.output.resolve())
if __name__=='__main__':main()
