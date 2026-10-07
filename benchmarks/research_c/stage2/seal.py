"""Seal C's blocked-mass stage without granting unexecuted dynamic capabilities."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
from .run import ROOT, BUNDLE, TRUSTED, PARENT, load
from engine.aniso_phase1.research_c.stage2.checkpoint import source_manifest
from engine.aniso_phase1.research_d.identity import sha, write_json


def seal(out, *, final=False):
    if final:
        smoke=json.loads((out/'consumer-smoke.json').read_text())
        if smoke.get('passed') is not True:
            raise ValueError('independent consumer did not pass')
        name='handoff-package.json'
    else:
        _,_,_,post=load()
        write_json(out/'baseline-postcheck.json',post)
        mass=json.loads((out/'mass-audit.json').read_text())
        derivative=json.loads((out/'potential-derivative-audit.json').read_text())
        log=(out/'tests.log').read_text()
        if 'Ran 16 tests' not in log or not log.rstrip().endswith('OK'):
            raise ValueError('expected targeted regression checks were not successful')
        write_json(out/'tests-summary.json',dict(passed=True,tests=16,new_tests=10,
            original_C_tests=6,tests_log_sha256=sha(out/'tests.log'),
            scope='new ODE prototype on coupled algebraic oracle plus original C small-space regressions; not real-space dynamic certification'))
        write_json(out/'dynamic-contract.json',dict(schema_version=1,status='prototype_blocked_singular_mass',
            space_manifest_sha256=post['space_manifest_sha256'], full_shape=[369,3],free_shape=[219,3],
            q='full displacement: x=X+Nq; F=I+grad_X(Nq)',
            velocity='all carrier and local velocity coefficients',
            boundary='q=S q_free+L(t), v=S v_free+Ldot(t); derivative directions have zero lift',
            force='positive gradient of original material energy plus original Ks potential',
            mass='fixed rho=1 continuum point inertia, Gauss5, full carrier/local and boundary cross terms',
            residual='r_free=[2 M(W-v0)+dt*(integral_0^1 grad U(q0+a*dt*W) da-fext)]_free',
            exact_search_tangent='2 Mff+dt^2 integral_0^1 a Hessian U(q0+a*dt*W) da',
            endpoint='Mff delta_v_free=-Mfb delta_v_fixed; rejected here because Mff is singular',
            reaction='rigid right-grip virtual translation paired with full interval impulse / dt',
            transaction='trial-owned q/v/time/step/predictor/rule/ledger/child values; sole commit last',
            child_protocol='prepare(candidate_copy, child_value_copy) -> owned immutable proposal; no external object writes',
            checkpoint='source/input/model/dt/path/tolerance identity and full committed state digest',
            B_substitution='replace material only after shared adequate rule/state-domain certification; mass unchanged',
            D_contract_ownership='C private semantics only; research_d/stage2/contracts.py not modified',
            dynamic_gpu=False,compressed_dynamic_cycle=False,B_E_external_atomicity=False))
        acceptance=dict(schema_version=1,status='blocked_singular_mass',
            stage_delivery_complete=True,common_space_dynamics_passed=False,
            parent_verified=post['passed'],static_operator=derivative['passed'],
            bounded_material_reference='inherited static checks only; no new dynamic states',
            free_mass_full_rank=mass['passed'],dynamic_cycle=False,cuda=False,coupled_physics=False,
            continuum_spatial_certified=False,asymptotic_time_convergence=False,
            independent_consumer_required=True,
            milestones=dict(C1='passed',C2='equilibrium_and_boundary_passed_dynamic_admission_blocked',
                C3='failed_rank_216_of_219',C4='potential_and_rest_tangent_passed_modes_blocked',
                C5='prototype_only_real_space_blocked',C6='prototype_only_real_space_blocked',
                C7='value_oracle_tests_passed_real_space_blocked',C8='not_run_mass_gate_failed',
                C9='not_run',C10='not_run',C11='private_contract_and_recompute_delivered',C12='sealed'))
        write_json(out/'acceptance.json',acceptance)
        disk=dict(system_free_bytes=shutil.disk_usage('/').free,data_free_bytes=shutil.disk_usage(TRUSTED).free,
            threshold_bytes=5*1024**3,pip_cache_path='/root/.cache/pip',
            pip_cache_resolved=str(Path('/root/.cache/pip').resolve()),
            migration_by_this_C_run=False,
            explanation='Before C migration started, parallel work had already moved pip cache to the data disk; C rechecked available space and did not move historical evidence.')
        write_json(out/'disk-audit.json',disk)
        own=source_manifest(ROOT)
        with zipfile.ZipFile(out/'source-snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            parent=json.loads((BUNDLE/'bundle.json').read_text())
            names=set(parent['code_sha256'])|set(own)|{'docs/results/lite-aniso-mainline/v22/source-delivered-sha256.json'}
            for path in sorted(names):archive.write(ROOT/path,path)
        write_json(out/'extension-source-sha256.json',own)
        report(out,mass,disk)
        name='preliminary-handoff.json'
    own=source_manifest(ROOT)
    artifacts={str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file()
        and p.name not in {'handoff-package.json','handoff-package-sha256.txt',
                           'preliminary-handoff.json','verification.json'}}
    manifest=dict(schema_version=1,producer='research_c_stage2',parent_bundle_sha256=PARENT,
        parent_bundle_path=str(BUNDLE),trusted_parent_data_root=str(TRUSTED),
        space_manifest_sha256=json.loads((out/'baseline-check.json').read_text())['space_manifest_sha256'],
        extension_source_sha256=own,artifacts_sha256=artifacts,default_changed=False,
        capabilities=dict(static_potential_checks=True,dynamic_cycle=False,cuda=False,coupled_physics=False))
    write_json(out/name,manifest)
    if final:(out/'handoff-package-sha256.txt').write_text(sha(out/name)+'\n')


def report(out,mass,disk):
    report=f'''# C 第二阶段交付：完整质量秩阻塞已定位

本轮按当前冻结共同包实施，只新增 C 的 stage2 文件。父共同包 589 项源码、29 项输入／证据，以及原 v22 的 488 项源码、131 项成果均保持一致。没有修改 A/B/D/E、公共合同、依赖、生产默认或旧归档。

## 结论与实际范围

**当前冻结空间不能通过计划 C3 的普通动力学准入。** 自由质量秩为 **216 标量／648 分量**，要求为 219／657。完整 369 标量矩阵秩为 {mass['full_scalar_rank']}，其中 18 个零对角约束载体位于物理积分域外。自由空间另有 3 个独立零质量方向；它们源自 75 个自由载体基仅有 72 个线性独立，与 144 个局部模式的积分精度无关。

五阶／六阶完整质量相对差约 {mass['full_order5_vs6_relative']:.3e}；恢复矩阵与原 PointInertia 作用相对差约 {max(mass['action_recovery_relative']):.3e}。尺度化秩阈值为 {mass['rank_threshold']:.3e}，把阈值缩小或放大 100 倍，秩仍为 216。载体基的独立 SVD 也得到三个零方向，其实际节点位移最大值低于 1.7e-16，而原 Ks 在这些单位系数方向上的能量约 0.099 J。因此这是零惯性而非零刚度的结构性问题，放松数值误差门槛无法修复。

按照 C3 失败处理，未加人工质量、未用伪逆、未消去局部或载体自由度，也未偷偷替换 A 空间。真实空间 C5—C10 和完整四档循环均未运行；资源不足不是本轮停止原因。下一步应由 A/D 审阅 [独立零空间证据](mass-nullspace-audit.json) 和 [可重算方向](mass-null-directions.npz)，另立独立空间版本，或另行推导保留代数自由度的动力学协议。后者超出当前满秩质量协议。

## 已完成和未完成

| 计划 | 本次状态 |
| --- | --- |
| C1 | 严格来源核验通过，父输入和方向场身份保存 |
| C2 | 零夹持平衡、完整 q/v、F/PK1 和边界历史保存；静态快照未改 |
| C3 | 完整矩阵与交叉项恢复通过，质量满秩验收失败，独立根因复核完成 |
| C4 | 零态／归档态势能导数和切线、对称性通过；广义模态因奇异质量未运行 |
| C5—C7 | 完整质量冲量、AVF、账本、事务和检查点原型已实现；小型耦合测试通过，真实空间受准入阻塞 |
| C8—C10 | 真实代表性步吞吐、短窗、终态归因、48,000 步四档循环均 not_run |
| C11 | 私有语义合同、状态重算和来源校验工具交付；不声明 B/D/E 联合完成 |
| C12 | 来源、失败记录、矩阵、零空间、初态、测试与扩展源码封存 |

零夹持初态的 detF=1，自由残差约 1.93e-15，PK1 最大值约 4.44e-14 Pa。零态稳定化能量约 -2e-15 J，属于原 Ks 浮点求和的零附近误差，未更改势能定义。归档已加载态稳定化能量约 9.55e-7 J，稳定化力范数约 4.72e-4 N，均为实算而非硬编码零。精确零态切线矩阵与原流式材料切线作用的相对差约 2.02e-13。

数值门槛在 [dynamic-protocol.json](dynamic-protocol.json) 中预登记。新梯度／切线检查采用 2e-5／2e-4，并保留绝对误差及三档扰动扫描，避免对近零相对误差过度苛刻；父包既有质量检查阈值未放宽。16 项测试通过，其中新增 10 项、旧 C 场景回归 6 项；这些测试不替代共同空间真实动态认证。

## 复现

在仓库根目录执行，新的 out 必须不存在。完整恢复两个质量矩阵和零态刚度约需数分钟，共享机器计时不代表独占性能。

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m benchmarks.research_c.stage2.run prepare --out docs/results/parallel-v22-stage2/C/new-run
```

该父空间预期完成证据写出后返回退出码 2，表示质量准入失败。可继续独立势能检查：

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m benchmarks.research_c.stage2.run derivatives --out docs/results/parallel-v22-stage2/C/new-run
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m unittest tests.research_c.stage2.test_dynamics tests.research_c.test_dynamics -v
```

封存包可用 `benchmarks.research_c.stage2.recompute` 检查；传入本目录、`--trusted-parent-data-root {TRUSTED}` 和 `--rebuild-mass`。`--require-dynamic` 必须被拒绝。source-snapshot.zip 包含所有父绑定源码、C 扩展源码及冻结加载器需要的 v22 源清单；父历史数据仍按 manifest 中的绝对路径只读访问，未将历史大数组复制进源码包。清洁源码副本的恢复结果见 consumer-smoke.json。

所有 q/v 都是 369×3 完整系数，方向导数不带提升；自由系数为 219×3，不能只按数组大小推断语义。材料六阶与质量五阶独立。检查点绑定源码、父包、空间、质量、材料、边界、dt、路径阶次和容差；成功步统一提交，失败步保留全部已提交值。B/E 仅提供无副作用的值提案示例，未认证真实外部对象原子性。

## 磁盘与遗留工作

当前系统盘剩余约 {disk['system_free_bytes']/1024**3:.2f} GiB，数据盘剩余约 {disk['data_free_bytes']/1024**3:.1f} GiB。C 开始迁移前，并行工作已将 pip 缓存迁到数据盘，原路径为软链接；C 复查容量后未重复迁移，也未移动历史证据。精确路径与容量见 disk-audit.json。

首次质量审计的全矩阵对角缩放遇到域外约束零行，失败记录保留于相邻 20260930T150600Z-common-dynamics 目录；本目录修正诊断实现并独立揭示自由空间秩缺陷。当前提供的是可复核阶段交付，不是完整动力学、时间收敛、连续体空间精度、Eulerian/APIC、GPU 或 B/E 联合通过声明。
'''
    (out/'REPORT_ZH.md').write_text(report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    p.add_argument('--final',action='store_true');args=p.parse_args();seal(args.out,final=args.final)


if __name__=='__main__':main()
