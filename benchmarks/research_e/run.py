"""Run from repository root: .venv/bin/python -m benchmarks.research_e.run."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=Path(__file__).with_name('protocol.json')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def baseline_audit():
    records={}
    for name in ['source-delivered-sha256.json','artifact-sha256.json']:
        path=ROOT/'docs/results/lite-aniso-mainline/v22'/name
        manifest=json.loads(path.read_text());bad=[]
        for rel,expected in manifest.items():
            p=ROOT/rel
            if not p.is_file() or digest(p)!=expected:bad.append(rel)
        records[name]=dict(count=len(manifest),mismatches=bad,manifest_sha256=digest(path))
    records['source_zip_sha256']=digest(ROOT/'docs/results/lite-aniso-mainline/v22/source-delivered.zip')
    records['other_parallel_sources']=[str(p.relative_to(ROOT)) for name in ['research_a','research_b','research_c','research_d'] for p in (ROOT/'engine/aniso_phase1'/name).glob('*.py')]
    return records


def export_package(out,protocol,source_hash):
    import numpy as np
    import scipy.sparse as sp
    from engine.aniso_phase1.types import AnisotropicMaterialParams
    from engine.aniso_phase1.research_e.flow import Grid,Darcy,oriented_permeability
    from engine.aniso_phase1.research_e.poro import Skeleton,Biot
    folder=out/'handoff';folder.mkdir()
    g=Grid((4,4));material=AnisotropicMaterialParams(10,20,200,[1,1,0]);s=Skeleton(g,material)
    f=Darcy(g,oriented_permeability(.01,.001,np.pi/4));b=Biot(s,f,.001)
    boundary={(d,i):('pressure',0.) for d in range(2) for i in (0,1)}
    state=b.initial();load=s.traction(0,1,[-.1,0]);M,rhs,free,frhs,src,_=b._system(state,.01,load,boundary,0.)
    for name,A in dict(solid_A=s.A,solid_G=s.G,flow_H=f.H,flow_B=f.B,storage_C=b.C,mixed_block=M).items():sp.save_npz(folder/(name+'.npz'),A)
    result=b.step(state,.01,load,boundary)
    np.savez_compressed(folder/'test_vectors.npz',rhs=rhs,solution=np.r_[result.state.u[s.free],result.state.p,result.flux[free]],
        free_u=s.free,free_flux=free,cell_reference_volume=np.full(g.nc,g.volume),centers=g.centers,permeability=f.K,
        solid_mass=np.full(g.nc,2000*(1-.35)*g.volume),fluid_mass=np.full(g.nc,1000*.35*g.volume),
        porosity=np.full(g.nc,.35),pressure=result.state.p,relative_volume_flux=g.cell_flux(result.flux),
        relative_phase_velocity=g.cell_flux(result.flux)/.35)
    meta=dict(schema_version=1,baseline_source_sha256=protocol['baseline_source_zip_sha256'],producer='research_e',producer_code_sha256=source_hash,
        input_sha256=digest(PROTOCOL),units=protocol['units'],dtype='float64',device='cpu',coordinate_system=protocol['coordinates'],
        material_parameters=dict(mu=10,lam=20,k_f=200,alpha=1,storage=.001,mu_f=1,rho_s=2000,rho_f=1000),
        boundary_conditions=dict(mechanics='x=0 clamped; x=1 traction (-0.1,0)',fluid='all sides pressure=0'),
        physical_reference_volume_convention=protocol['volume_convention'],seed=protocol['seed'],q_means='total displacement u; F=I+grad_X(u)',
        vector_order=['u[free_u]','cell_pressure','integrated_face_flux[free_flux]'],dt=.01,
        discrete_equations=['A u - G.T p = f','G delta_u + C delta_p + dt B flux = dt source_volume','H flux - B.T p = boundary_rhs'],
        exact_tangent='mixed_block.npz is exact for declared linear Biot model',pressure_schur='C + G A^-1 G.T + dt B H^-1 B.T',
        C_transaction='CouplingTransaction.begin_trial -> caller verifies other substeps -> commit; any failure -> rollback; C owns whole-step order',
        force_ownership='effective solid stress plus -alpha*p*I in quasistatic Biot; inertial drag is a separate model, never applied again on top of Darcy',
        known_solution_true_residual=result.metrics['true_residual'],integration_status='private CPU adapter ready; production/C/D/GPU/compression integration pending')
    meta['files']={p.name:digest(p) for p in folder.iterdir()};save(folder/'operator-package.json',meta)
    # Independently reload the package rather than trusting its metadata.
    A=sp.load_npz(folder/'mixed_block.npz');v=np.load(folder/'test_vectors.npz')
    residual=float(np.linalg.norm(A@v['solution']-v['rhs'],np.inf))
    return dict(passed=residual<1e-8,reloaded_block_residual=residual,status='CPU handoff validated; E10 production integration pending')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id');args=ap.parse_args()
    run_id=args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ_E_full')
    if Path(run_id).name!=run_id or run_id in ('.','..'):raise ValueError('run id must be one path component')
    out=ROOT/'docs/results/parallel-v22/E'/run_id;out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(PROTOCOL.read_text());save(out/'protocol.json',protocol)
    started=perf_counter();before=baseline_audit();save(out/'baseline-before.json',before)
    if any(v['mismatches'] for v in before.values() if isinstance(v,dict)) or before['source_zip_sha256']!=protocol['baseline_source_zip_sha256']:
        raise RuntimeError('baseline changed; explicit rebase required')
    free=shutil.disk_usage('/').free
    if free<5*1024**3:raise RuntimeError('system free disk below 5 GiB; migrate verified noncritical data before running')
    os.environ['TMPDIR']=str(out/'tmp');(out/'tmp').mkdir()
    os.environ['WARP_CACHE_PATH']=str(out/'warp-cache')
    source_paths=sorted([p for folder in ['engine/aniso_phase1/research_e','benchmarks/research_e','tests/research_e'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts])
    manifest={str(p.relative_to(ROOT)):digest(p) for p in source_paths};save(out/'source-sha256.json',manifest)
    with zipfile.ZipFile(out/'source-e.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in source_paths:z.write(p,p.relative_to(ROOT))
    source_hash=digest(out/'source-sha256.json')
    try:gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):gpu='unavailable; GPU not used'
    import numpy as np
    import scipy
    environment=dict(utc=datetime.now(timezone.utc).isoformat(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        cpu=platform.processor(),cpu_count=os.cpu_count(),gpu_inventory=gpu,device='cpu only',loadavg=os.getloadavg(),
        threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
        disk_free_before=free,data_disk_free=shutil.disk_usage('/root/autodl-tmp').free,
        memory_limit_bytes=resource.getrlimit(resource.RLIMIT_AS),cache='private E temporary/cache directories; CPU sparse factors cached by dt',
        timing_status='shared-host diagnostic only; no speedup or GPU performance claim')
    save(out/'environment.json',environment)
    from .experiments import material_audit,flow_audit,pressure_audit,drag_audit,consolidation_audit,oblique_audit,cycle_audit,state_audit,robustness_audit
    summary={}
    checks=[('E2_material',material_audit),('E3_Darcy',flow_audit),('E4_pressure',pressure_audit),('E5_drag',drag_audit),
            ('E6_consolidation',consolidation_audit),('E6_oblique',oblique_audit),('E7_cycle',cycle_audit),('E8_state',state_audit),('E9_matrix',robustness_audit)]
    for name,fn in checks:
        t=perf_counter();print(name,'START',flush=True)
        try:result=fn(out,protocol)
        except Exception as ex:
            result=dict(passed=False,error=str(ex),traceback=traceback.format_exc())
        result['wall_seconds']=perf_counter()-t;save(out/(name+'.json'),result)
        summary[name]=dict(passed=result['passed'],wall_seconds=result['wall_seconds'])
        print(name,'PASS' if result['passed'] else 'FAIL',f"{result['wall_seconds']:.3f}s",flush=True)
    summary['E10_handoff']=export_package(out,protocol,source_hash)
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/research_e','-v'],cwd=ROOT,capture_output=True,text=True)
    (out/'tests.log').write_text(test.stdout+test.stderr);summary['tests']=dict(passed=test.returncode==0)
    regression=subprocess.run([sys.executable,'-m','unittest','tests.test_aniso_phase1_material','-v'],cwd=ROOT,capture_output=True,text=True)
    (out/'baseline-material-tests.log').write_text(regression.stdout+regression.stderr);summary['baseline_material_tests']=dict(passed=regression.returncode==0)
    from .visualize import render
    try:render(out);summary['scene_artifacts']=dict(passed=True)
    except Exception as ex:summary['scene_artifacts']=dict(passed=False,error=str(ex));(out/'visualization-error.log').write_text(traceback.format_exc())
    after=baseline_audit();save(out/'baseline-after.json',after)
    summary['baseline_unchanged']=dict(passed=not any(v['mismatches'] for v in after.values() if isinstance(v,dict)))
    summary.update(independent_prototype_passed=all(r['passed'] for r in summary.values()),production_integration_completed=False,
        scope='E1-E9 bounded independent CPU prototype; E10 private handoff only; E11 report',
        run_id=run_id,total_seconds=perf_counter()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        disk_free_after=shutil.disk_usage('/').free,output_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    save(out/'acceptance.json',summary)
    write_report(out,summary)
    files={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact-sha256.json'}
    save(out/'artifact-sha256.json',files)
    print('RESULT',out,flush=True);print('INDEPENDENT_PROTOTYPE_PASSED',summary['independent_prototype_passed'],flush=True)
    return 0 if summary['independent_prototype_passed'] else 1


def write_report(out,summary):
    lines=['# E 材料与饱和流固耦合独立原型报告','',f"运行：`{summary['run_id']}`。独立原型验收：**{'通过' if summary['independent_prototype_passed'] else '未通过，见分项'}**。生产 Lite/GPU 集成未完成。",'',
        '基于当前 v22 源码；488 项原源码及 131 项原成果前后核验。E 只写独占目录。模型、单位和运行前阈值见 [protocol.json](protocol.json)。',
        '按用户要求以场景稳定、物理账本和回滚为主，保留适度解析精度及加密测试，未将有限时间步误差要求为机器零。','',
        '| 检查 | 状态 |','| --- | --- |']
    for k,v in summary.items():
        if isinstance(v,dict) and 'passed' in v:lines.append(f"| {k} | {'通过' if v['passed'] else '未通过'} |")
    lines += ['', '## 可查看的场景与原始结果','', '[场景预览](scene-preview.gif)、[场景与能量图](scene-summary.png)、[Darcy 压力和通量](darcy-pressure-flux.png)、[固结参考对照](consolidation.png)。',
        '各 E*.json 包含误差、账本、分区指标、失败地图与运行时间；NPZ 包含未经平滑的压力、位移、应力、通量和每个输出时刻的循环历史。',
        '## 接口与范围','',
        '固体使用现有 Hencky＋双边二次纤维模型作非线性材料控制；孔弹性求解采用其 F=I 线性化、固定 Q2 空间、完整 3^d 材料积分、RT0 面通量及 P0 压力。不压缩方向或流体状态。',
        '二维为单位厚度平面应变。Biot 骨架和流体均忽略惯性；两质量块的保惯性阻力另验，不能把它重复加在已消元的 Darcy 阻力上。',
        'E8 是指定有限变形、孔隙体积供液和 Piola 流量映射的状态验证，尚未求解有限变形下的全耦合力平衡。生产移动粒子、接触、GPU、压缩状态、自由液面、非饱和、毛细、塑性损伤及真实材料标定均不在通过范围。',
        '单体块使用稀疏直接求解；迭代分区使用 fixed-stress，收敛后与同一单体方程比较。一次分区残差不合格时拒绝提交。刚性阻力和后向 Euler 数值损失单列。',
        'C 总事务入口可调用 E 私有 CouplingTransaction；[D 交接包](handoff/operator-package.json) 包含矩阵、真实残差、解向量与单位/坐标/边界合同。公共入口尚未修改。',
        '## 成本与复现','',f"端到端观测耗时 {summary['total_seconds']:.2f} s，峰值 RSS {summary['max_rss_kib']/1024:.1f} MiB。共享主机计时仅作成本记录，不构成性能加速声明。",
        '资源、线程、GPU UUID 清单见 environment.json；GPU 未参与验收。源码和输入哈希、source-e.zip 及 artifact-sha256.json 用于复核。',
        '从仓库根运行：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m benchmarks.research_e.run`，每次新建 UTC 运行目录，不覆盖旧结果。','']
    (out/'REPORT_ZH.md').write_text('\n'.join(lines))


if __name__=='__main__':raise SystemExit(main())
