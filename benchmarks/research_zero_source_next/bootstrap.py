"""Register all physical and resource choices before numerical work."""
import numpy as np
from .provenance import *
from .lineage import default_source
from engine.aniso_phase1.research_pressure_startup_next.theta import schedule
from engine.aniso_phase1.research_zero_source_next.coupled import explicit_zero

def prepare():
    run=freeze();folder,chain=default_source(APP,APP_SHA)
    names=['base_config.py','config.py','spaces.py','run.py','physics.py']
    same={n:sha(ROOT/'benchmarks/research_zero_source_next'/n)==sha(ROOT/'benchmarks/research_pressure_startup_next'/n) for n in names}
    if not all(same.values()):raise ValueError('solid core changed')
    write(run/'S0/source-map.json',dict(status='passed_scoped',direct_parent=str(APP),direct_parent_sha256=APP_SHA,local_geometry_source=str(ROOT/'engine/aniso_phase1/research_stabilization_boundary_next/local_geometry.py'),default_case=str(folder),chain=chain,zero_source_cases=[str(APP/'cases'/n) for n in ('startup-zero-coarse-h','startup-zero-coarse-half')],rejected_source_history=str(APP/'S5/source-mismatch')))
    write(run/'S0/impact-and-compatibility.json',dict(status='passed_scoped',byte_identical_solid_modules=same,zero_source_wrapper_only=True,theta_equations_inherited=True,formal_default_changed=False))
    old=read(APP/'S5/selected-protocol.json');ts=np.asarray(old['times_s']);params=old['parameters'];explicit_zero(params['source_density_s_inv'])
    register(run,'S0/input-contract.json',dict(status='registered',parameters=params,cuts=old['cuts'],times_s=ts.tolist(),fine_times_s=np.sort(np.r_[ts,.5*(ts[1:]+ts[:-1])]).tolist(),methods=['backward-euler','startup'],switch_s=2.5e-5,source_required_explicit=True,source_scalar_m3_s=0.,physical_source_protocol_sha256=sha(APP/'S5/selected-protocol.json'),method_arrays={m:schedule(ts,m).tolist() for m in ('backward-euler','startup')},common_window_s=2e-4))
    register(run,'S1/full-window-protocol.json',dict(status='registered',method='backward-euler',grid='coarse',window_s=[0.,2e-4],steps=[16,32],source=0.,reuse='only after full physical identity, original generations and new mass residual budgets pass',fallback='independent16/32 from original common initial state before continuation',max_attempts=52))
    register(run,'S2/theta-protocol.json',dict(status='registered',method='startup',grid='coarse',switch_s=2.5e-5,window_s=[0.,2e-4],steps=[16,32],from_rest=True,minimum_dnum_reduction=.2,minimum_dnum_drop_J=1e-12,max_attempts=50,alternatives=1))
    (run/'S2/work-contract.md').write_bytes((APP/'S1/discrete-work-contract.md').read_bytes())
    PROGRESS.write_text(f'# MPM-lite 零体源、时间耗散与压力网格实施记录\n\n按[{PLAN.name}]({PLAN.name})顺序实施。\n\n当前S0已完成，直接基线 `{APP.name}`，发布SHA256 `{APP_SHA}`，开工完整审计31源码/31快照/913产物及祖先链通过。系统盘约5.86GiB，无需迁移；新结果和缓存位于数据盘。\n\n结果目录：`{run.relative_to(ROOT)}`。正式日常场景继续继承SOLID；本轮不修改原q5证书或正式144函数。\n')
    print('RUN',run,flush=True)

if __name__=='__main__':
    with serial_lock():prepare()
