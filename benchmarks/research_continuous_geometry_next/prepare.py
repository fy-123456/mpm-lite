"""Freeze the latest release and the exact global-index continuous branch."""
from .provenance import *
from benchmarks.research_phase_stress_next.time_study import history

def prepare():
    run=freeze();case=APP/'cases/boundary32-h';states=history(case);expected={4:'c850bc687dba393d59aa998a8f3fff1585d6a0e6a2b9be2cb2d86892217d03ab',8:'c7d5b6e99fe44cdbc06b53a2ce2708fab6b214f79d2066a695d3e2f1781dea07',16:'f615abf2f326a78052c0e750480dee6b01d715cd363b51cc870b385fc469ee19',18:'4f2fadd31c10557336adeab8b3477d5e95c89c037ae70e241fc7b966e72ad30c',28:'98ec3be1eeb66ba3c26383eb72740bd61c2fb2da839d1db7da31fd609ebce6dc'};bound={}
    for i,h in expected.items():
        f=states[i]['folder']/'state.json'
        if sha(f)!=h:raise ValueError('source checkpoint changed')
        bound[str(i)]=dict(path=str(f),sha256=h,digest=states[i]['state'].digest(),time_s=states[i]['state'].time,step=i)
    register(run,'S0/state-contract.json',dict(case=str(case),identity_sha256=sha(case/'identity.json'),states=bound))
    for name,src in [('physical-contract.json','S0/physical-contract.json'),('coupled-protocol.json','S3/coupled-protocol.json')]:write(run/'S0'/name,dict(read(APP/src),inherited_from=str(APP/src),inherited_sha256=sha(APP/src)))
    register(run,'S1/continuous-protocol.json',dict(status='registered',backend='B',source=bound['4'],start_step=4,stop_step=18,process_boundary_step=8,actual_times_s=read(APP/'S3/coupled-protocol.json')['times']['h'],observation_times_s=[i*1.25e-5 for i in range(7)],new_steps=14,fault_origin_step=16,fault_target_step=17,raw_prefix_source=str(case),original_material_q7=True,mass_M7=True,raw_substep_accuracy=False))
    PROGRESS.write_text(f'# 连续共享几何、热点与空间参考实施记录\n\n日期：2026-10-05（UTC+8）。执行[{PLAN.name}]({PLAN.name})。\n\n开工最新BASE：`{APP.name}`，SHA `{APP_SHA}`；24源码/24快照/875产物及完整祖先通过。结果：`{run.relative_to(ROOT)}`，真实路径`{run.resolve()}`。系统盘约7.9GiB，不触发迁移。\n\nS0已冻结来源、物理参数、原全局步号/时间及预算。后续顺序执行，当前没有新增动态尝试。\n')
    print('RUN',run,flush=True)

if __name__=='__main__':
    with serial_lock():prepare()
