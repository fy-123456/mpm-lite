"""Freeze the authenticated descendant without touching sealed inputs."""
import numpy as np
from .provenance import *
from .runtime import update
from benchmarks.research_phase_stress_next.time_study import history
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from engine.aniso_phase1.research_transverse_next.initial import pressure_profile
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY


def main():
    run=freeze()
    for name in ('coupled-protocol.json','grid-protocol-inherited.json'):
        (run/'S0'/name).write_bytes((APP/'S0'/name).read_bytes())
    cfg=read(run/'S0/coupled-protocol.json');cuts=read(run/'S0/grid-protocol-inherited.json')['cuts']['yz'];top=ReferenceTopology(cuts)
    p,d=pressure_profile('YZ128',top);h=history(APP/'cases/YZ128')
    register(run,'S0/physical-contract.json',dict(parameters=cfg['parameters'],cuts=cuts,pressure0=p.tolist(),initial_definition=d,mobility=(MOBILITY*1e-7).tolist(),source=0.,times=cfg['times'],full_mass_order=7,full_material_order=7,space=read(run/'selected-space.json'),original_initial_digest=h[0]['state'].digest()))
    register(run,'S0/checkpoint-sources.json',dict(records=[dict(step=i,path=str(h[i]['folder']),state_sha256=sha(h[i]['folder']/'state.json'),digest=h[i]['state'].digest()) for i in (0,8,9,16,17,18)],parent_identity_sha256=sha(APP/'cases/YZ128/identity.json')))
    register(run,'S0/observation-contract.json',dict(mode_floor_Pa=1e-5,mode_increment_rtol=.1,signal_resolution_Pa=5e-5,pressure_atol_Pa=.001,engineering_rtol=.05,flow_atol_m3_s=1e-10,content_atol_m3=1e-10,observations_s=cfg['observations_s'],flux='integrated interval mean, already m3/s',same_probe_locations=True,physical_window_max_s=175e-6,residual_window_s=200e-6))
    register(run,'S1/reference-protocol.json',dict(cases=['YZ128','Y128'],cells=128,source=0.,F='identity fixed',reference_scope='same-grid semidiscrete fixed skeleton only',independent_expm_times_s=[25e-6,175e-6],CPU_step_cap=200,mode_test='1e-5 Pa + 10% max exact increment',new_space_training=0))
    PROGRESS.write_text(f'# 横向压力参考与 DV 成本优化实施记录\n\n基于 `{APP.name}`，发布 SHA `{APP_SHA}`。执行 [{PLAN.name}]({PLAN.name})，顺序推进。\n\n结果：`{run}`。\n\nS0：最新发布全审计通过，31源码/31快照/727产物及祖先通过，物理与资源合同已冻结。系统盘{resources()["system_free_GiB"]:.2f} GiB，未触发迁移。\n')
    print(run,flush=True)


if __name__=='__main__':
    with serial_lock(None):main()
