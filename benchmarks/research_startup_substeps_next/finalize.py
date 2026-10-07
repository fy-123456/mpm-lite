"""Close the evidence chain and report limitations without extra trajectories."""
import argparse
import numpy as np
from .provenance import *
from .lineage import default_source
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric


def prepare(run):
    run=Path(run);mutable(run);default,chain=default_source(APP,APP_SHA);owner=default.parent.parent;h=history(default);rows=h[-1]['rows']
    if len(rows)!=252 or h[-1]['state'].time!=1.6:raise ValueError('inherited scene differs')
    write(run/'S6/default-scene-decision.json',dict(status='inherited',default_case=default.name,default_case_source=dict(release_root=str(owner),release_sha256=sha(owner/'release.json'),case=default.name,identity_sha256=sha(default/'identity.json'),execution_protocol_sha256=sha(default/'execution-protocol.json')),formal_steps=252,new_formal_steps=0))
    write(run/'S6/inherited-solid-evidence.json',dict(status='inherited',source=str(default),chain=chain,steps=len(rows),end_s=h[-1]['state'].time,new_steps=0))
    unchanged={n:sha(ROOT/'benchmarks/research_boundary_reference_next'/n)==sha(ROOT/'benchmarks/research_startup_substeps_next'/n) for n in ('base_config.py','config.py','spaces.py','physics.py','run.py')}
    if not all(unchanged.values()):raise ValueError('solid core drift')
    write(run/'S6/regression-impact.json',dict(status='passed_scoped',byte_identical_solid_modules=unchanged,ancestor_sources_immutable=True,solid_equations_unchanged=True,production_q5_permission_not_extended=True,new_coupling_only_qualified_for_short_window=True))
    log=(run/'S1/tests.log').read_text()
    if 'Ran 5 tests' not in log or not log.rstrip().endswith('OK'):raise ValueError('tests missing')
    tested={k:v for k,v in all_sources().items() if k.startswith(('engine/','tests/'))}
    prior=read(run/'S1/time-protocol.json')['source_sha256']
    if any(prior.get(k)!=v for k,v in tested.items()):raise ValueError('engine/test drift since test setup')
    write(run/'S6/test-report.json',dict(status='passed_scoped',command='.venv/bin/python -B -m unittest discover -s tests/research_startup_substeps_next -v',returncode=0,distinct_tests=5,total_test_case_executions=5,suite_executions=1,log='S1/tests.log',tested_sources=tested,inherited_tests_not_rerun=True,actual_checks=['S3/operator-check.json','S3/engineering-comparison.json','S3/transaction-check.json','S4/backend-decision.json']))
    scopes={};checks=[];frames=[]
    for scope in ('legacy','boundary32'):
        folder=run/'S4'/scope
        if not (folder/'performance-decision.json').exists():
            scopes[scope]=dict(status='not_triggered',selected=False,backend='BASE-bounded-Gauss',reason='original performance or new-grid engineering gate not passed');continue
        decision=read(folder/'performance-decision.json');perf=read(folder/'paired-performance.json');op=read(folder/'operator-equivalence.json') if (folder/'operator-equivalence.json').exists() else dict(status='inherited',path=str(APP/'R4/operator-equivalence.json'),sha256=sha(APP/'R4/operator-equivalence.json'))
        for i in (0,1):
            for kind in ('A','B'):
                f=folder/f'{kind}{i}';hist=history(f);s0=hist[0]['state'];s1=hist[-1]['state'];row=hist[-1]['rows'][-1];e0=s0.child_states['fluid']['last_ledger']['total_energy_J'];balance=row['total_energy_J']-e0+row['darcy_dissipation_J']+row['numerical_dissipation_J']-row['external_work_J']-row['source_work_J']-row['reservoir_work_J']
                before=s0.child_states['fluid'];after=s1.child_states['fluid'];mass=float(np.sum(np.array(after['content_m3'])-before['content_m3'])+after['cumulative_boundary_m3']-before['cumulative_boundary_m3']-np.sum(np.array(after['cumulative_source_m3'])-before['cumulative_source_m3']))
                zero=row['source_work_J']==0 and not np.any(after['cumulative_source_m3']);passed=zero and abs(mass)<1e-10 and abs(balance)<1e-9+.01*abs(e0) and row['true_scaled_residual']<=1 and row['min_detF']>.1 and min(after['pressure_Pa'])>=0
                if not passed:raise ValueError('A/B physical balance failed')
                checks.append(dict(scope=scope,case=kind+str(i),source_zero=zero,mass_defect_m3=mass,energy_balance_J=balance,min_detF=row['min_detF'],max_residual_fraction=row['true_scaled_residual'],passed=passed))
        af=next((folder/'A1').rglob('frame.npz'));bf=next((folder/'B1').rglob('frame.npz'))
        with np.load(af) as a,np.load(bf) as b:
            if set(a.files)!=set(b.files):raise ValueError('frame keys differ')
            fc={k:metric(a[k],b[k],1e-10 if k=='face_flux_m3_s' else 1e-8,2e-5) for k in a.files}
        if not all(v['passed'] for v in fc.values()):raise ValueError('frame equivalence')
        frames.append(dict(scope=scope,checks=fc,A_sha256=sha(af),B_sha256=sha(bf)))
        scopes[scope]=dict(**decision,operators=op)
    faults=[dict(path=str(f.relative_to(run)),**read(f)) for f in (run/'S4').glob('*/fault-*/failure.json')];restarts=[dict(path=str(f.relative_to(run)),**read(f)) for f in (run/'S4').glob('*/restart.json')]
    if len(faults)!=1 or len(restarts)!=1 or not faults[0]['rollback_exact'] or not all(f['cache_cleared'] for f in faults) or not restarts[0]['same_digest']:raise ValueError('shared geometry transaction incomplete')
    # New32's transaction is inherited by legacy only for the unchanged shared cache implementation;
    # each scope still has its own state/energy equivalence. No new dynamic default is changed.
    write(run/'S4/backend-decision.json',dict(status='scoped_research_selection',scopes=scopes,shared_transaction=faults[0],restart=restarts[0],entry='benchmarks.research_startup_substeps_next.backend',explicit_A_available=True,trajectory_backend_not_mutated=True,production_default='inherited pure solid q5 certificate only'))
    write(run/'S6/physical-ledger-audit.json',dict(status='passed_scoped',AB=checks,coupled=read(run/'S3/engineering-comparison.json')['ledger'],original_residuals_unchanged=True))
    write(run/'S6/frame-equivalence.json',dict(status='passed_scoped',records=frames))
    write(run/'S6/fallback-and-restart.json',dict(status='passed_scoped',coupled=read(run/'S3/transaction-check.json'),shared_fault=faults,shared_restart=restarts,solid_q5=dict(path=str(owner/'S6/fallback-and-restart.json'),sha256=sha(owner/'S6/fallback-and-restart.json')),new_S6_dynamic_attempts=0))
    cpu=read(run/'S2/candidate-U4.json');worst=[]
    for trial in cpu['trials']:
        for category in ('raw','engineering'):
            rr=trial[category]['records'];row,key,ratio=max(((r,k,v) for r in rr for k,v in r['budget_ratios'].items()),key=lambda z:z[2]);worst.append(dict(level=trial['label'],category=category,time_s=row['time_s'],field=key,budget_ratio=ratio,metric=row['checks'][key]))
    from benchmarks.research_pressure_startup_next.coupling_review import context
    physical=read(run/'S0/physical-contract.json');top=context(physical['cuts'],physical['parameters']['storage'])['top'];directions=[]
    for label in ('h','half'):
        with np.load(run/'S2'/f'U4-{label}.npz') as z:
            j,k=np.unravel_index(np.argmax(abs(z['flux']-z['exact_flux'])),z['flux'].shape);ref=float(z['exact_flux'][j,k]);err=float(z['flux'][j,k]-ref)
            directions.append(dict(level=label,raw_start_s=float(z['times'][j]),raw_end_s=float(z['times'][j+1]),face=int(k),axis=int(top.axes[k]),boundary_sign=int(top.boundary_sign[k]),signed_flux_m3_s=float(z['flux'][j,k]),reference_flux_m3_s=ref,absolute_error_m3_s=abs(err),relative_error=abs(err/ref) if ref else None,criterion='largest raw absolute face-flow error; original vector norm gate unchanged'))
    write(run/'S5/worst-errors.json',dict(records=worst,raw_directions=directions,raw_diagnostics='S3/raw-substep-diagnostics.json',engineering='S3/engineering-comparison.json'))
    nexts=[dict(direction='y/z refinement',status='can_prepare',needs='new 3D topology and bounded geometry construction beyond current x-only limit; first operator and fixed-skeleton reference'),dict(direction='deforming spatial reference',status='needs_evidence',needs='same conservative time protocol on two spatial grids, preserve material, boundary and full M7 cross terms'),dict(direction='longer coupled cycle',status='can_prepare',needs='short staged extension after stable 200us window, monitor signed flux, work and iterative cost'),dict(direction='144 local function reallocation',status='needs_evidence',needs='trustworthy deforming spatial reference before training or replacing space'),dict(direction='coupled q5 or production C/E',status='not_this_round',needs='separate coupled tangent/work/transaction qualification')]
    write(run/'S5/next-scope-decision.json',dict(directions=nexts,new_spaces=0,training_solves=0,production_C_E_changes=0,coupled_q5=False))
    lines=['# 已保存结果的瓶颈复核','','U4将早期工程区间细分为4个完整耦合步；每个真实子步均保存。工程流量用Σh·z/ΔT，不能用末端值替代。固定骨架首个BE子步相对误差仍约12.45%，工程观察通过不代表原始子步精度通过。','','新32单元仅通过同网格28/56步的200微秒时间一致性，未取得实际可变形空间参考。无需为当前工作重训144个函数。','']
    for scope,v in scopes.items():
        if not (run/'S4'/scope/'paired-performance.json').exists():continue
        perf=read(run/'S4'/scope/'paired-performance.json');lines+=['## '+scope,'',f"性能选择：{v['backend']}；中位降幅{perf['median_gain']:.2%}；两状态极差{perf['between_state_gain_range']:.2%}。采样设备独占={perf['exclusive']}。",'']
        for rec in perf['records']:
            b=rec['B'];lines.append(f"- 输入{rec['index']}：B整步{b['advance_s']:.4f}s，几何包含项{b['geometry_inclusive_s']:.4f}s、RT0 {b['rt0_s']:.4f}s、体积梯度伴随{b['volume_gradient_adjoint_s']:.4f}s、材料{b['material_s']:.4f}s、线性求解包含项{b['linear_solve_inclusive_s']:.4f}s、场{b['field_s']:.4f}s、提交{b['commit_s']:.4f}s。嵌套计时不得相加。")
        lines+=['','下一轮根据B实际占比选择热点，优先评估仍占主要比例的几何/梯度计算；先保持当前F和压力功一致性，再考虑融合计算。当前没有证据支持优先重做切线缓存或I/O。','']
    (run/'S5/bottleneck-review.md').parent.mkdir(exist_ok=True);(run/'S5/bottleneck-review.md').write_text('\n'.join(lines))
    with PROGRESS.open('a') as f:f.write('\nS4/S5：原/新网格的性能资格分开记录，后端决定见`S4/backend-decision.json`；收支、缓存回滚及零步恢复已核对。剩余工作转向有证据的热点和空间参考，不重训144函数，也不扩展耦合q5权限。\n')
    print('PREPARED',[(k,v['selected']) for k,v in scopes.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):prepare(a.run)
