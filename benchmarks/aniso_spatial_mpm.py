"""Advecting elastic block on rebuilt spatial grids: full-mass MPM acceptance."""
import argparse
import json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.spatial_mpm import ParticleHistory,SpatialMPM
from engine.aniso_phase1.consistent_transfer import material_response
from .aniso_history_increment import guard,DATA_ROOT
from .aniso_convergence_reference import save
from .aniso_practical_validation import device_choice

OUT=Path('docs/results/xz-mpm/v1/mpm')
H=1/16
DURATION=.012
DT=.0005


def initial(ppc=3):
    counts=np.array([4,2,2]);lo=np.array([.25,.4375,.4375]);length=H*counts
    axes=[lo[d]+(np.arange(counts[d]*ppc)+.5)*H/ppc for d in range(3)]
    X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    center=lo+length/2;relative=X-center
    x=X.copy();x[:,1]+=.004*(relative[:,0]/length[0])**2
    F=np.broadcast_to(np.eye(3),(len(X),3,3)).copy();F[:,1,0]=.008*relative[:,0]/length[0]**2
    v=np.tile([1.,.15,0.],(len(X),1))+np.cross(np.tile([.3,0.,.2],(len(X),1)),relative)
    V=np.full(len(X),np.prod(length)/len(X));A=np.zeros((len(X),3,3));A[:,0,0]=1
    return ParticleHistory(X,x,v,F,V,V,A)


def origin_at(t,switched):
    phase=int(np.floor(t/.002+1e-8))%2
    return np.array([.25,.125,-.125])*H if switched and phase else np.zeros(3)


def rms(x,w):return float(np.sqrt(w@np.sum(np.asarray(x).reshape(len(w),-1)**2,axis=1)/w.sum()))


def run(out,switched,factor,device):
    guard();state=initial();model=SpatialMPM(H,device=device);dt=DT/factor;nsteps=round(DURATION/dt)
    label=('shifted' if switched else 'fixed')+f'-dt{dt:.7f}'
    rows=[];frames=[state.x.copy()];energies=[];times=[0.]
    U0=float(state.volume@material_response(state.F,state.A,model.params)[0])
    K0=.5*float(np.sum(state.mass[:,None]*state.v**2))
    initial_momentum=(state.mass[:,None]*state.v).sum(0)
    bulk=initial_momentum/state.mass.sum()
    internal_initial=U0+.5*float(np.sum(state.mass[:,None]*(state.v-bulk)**2))
    energies.append(U0+K0);cells=np.floor(state.x/H).astype(int);crossings=0;origins=[]
    for step in range(nsteps):
        guard();origin=origin_at(step*dt,switched)
        state,info=model.step(state,dt,origin)
        nextcells=np.floor(state.x/H).astype(int)
        crossed=int(np.any(nextcells!=cells,axis=1).sum());crossings+=crossed;cells=nextcells
        rows.append(dict(step=step+1,time=(step+1)*dt,physical_cell_crossings=crossed,origin=origin.tolist(),**info))
        frames.append(state.x.copy());times.append((step+1)*dt);energies.append(info['mechanical']);origins.append(origin)
        if (step+1)%8==0:print(label,step+1,'/',nsteps,'F',info['prediction_commit_F_absolute'],'K',info['kinetic_relative'],flush=True)
    checks=dict(solve=max(r['scaled_residual'] for r in rows)<=1.01e-10,
        prediction_commit=max(r['prediction_commit_F_absolute'] for r in rows)<1e-12,
        history_rebuild=max(r['history_rebuild_F_absolute'] for r in rows)==0,
        mass=min(r['mass_min'] for r in rows)>0 and max(r['mass_error'] for r in rows)<1e-14,
        kinetic_readback=max(r['kinetic_relative'] for r in rows)<1e-11,
        roundtrip=max(r['roundtrip_relative'] for r in rows)<1e-8,
        positive_J=min(r['min_det'] for r in rows)>0,
        momentum=max(r['momentum_absolute'] for r in rows)<1e-9,
        budget=max(abs(r['energy_budget_residual']) for r in rows)<1e-14,
        physical_cell_crossings=crossings>0,
        origin_changes=(len(np.unique(origins,axis=0))>1 if switched else True))
    result=dict(config=dict(dt=dt,steps=nsteps,duration=DURATION,device=device,ppc=3,h=H,switched=switched,material_integration='particles',mass='full-consistent'),
        rows=rows,checks=checks,passed=all(checks.values()),crossings=crossings,
        projection_loss_over_initial_internal_energy=-sum(r['projection_delta'] for r in rows)/internal_initial,
        initial_internal_energy=internal_initial,initial_mechanical=U0+K0,
        final_mechanical_relative=energies[-1]/energies[0]-1,
        cumulative_momentum_relative=float(np.linalg.norm((state.mass[:,None]*state.v).sum(0)-initial_momentum)/np.linalg.norm(initial_momentum)))
    out.mkdir(parents=True,exist_ok=True);save(out/(label+'.json'),result)
    np.savez_compressed(out/(label+'.npz'),reference=state.X,positions=np.asarray(frames),times=times,mechanical=energies,
        F=state.F,v=state.v,x=state.x,mass=state.mass,volume=state.volume,A=state.A,origins=origins)
    if not result['passed']:raise RuntimeError(f'{label}: {checks}')
    return result,state


def compare(a,b,params):
    Pa=material_response(a.F,a.A,params)[1];Pb=material_response(b.F,b.A,params)[1]
    vb=b.v-(b.mass[:,None]*b.v).sum(0)/b.mass.sum()
    return dict(F=rms(a.F-b.F,b.mass)/rms(b.F-np.eye(3),b.mass),
                P=rms(Pa-Pb,b.volume)/rms(Pb,b.volume),
                v_fluctuation=rms(a.v-b.v,b.mass)/max(rms(vb,b.mass),1e-30),
                x_absolute=rms(a.x-b.x,b.mass))


def report(out):
    r=json.loads((out/'results.json').read_text())
    lines=['# 实际空间网格 MPM 短程检查','',
        '432 个粒子，Q1 空间网格，h=1/16，12 ms 自由弹性块。粒子携带 F；每步按当前位置重建网格。',
        'GPU 本构与 CPU 完整质量/解析 Newton 求解。固定原点和每 2 ms 切换原点分别运行两档时间步。','',
        f"一致性检查通过：{r['consistency_passed']}；5% 时间检查通过：{r['time_passed']}；联合验收：{r['passed']}。",'',
        '| 网格原点 | 时间差 F | 时间差 P | 时间差去平移速度 |',
        '|---|---:|---:|---:|']
    for sw,label in [('False','固定'),('True','切换')]:
        t=r['time'][sw];lines.append(f"| {label} | {t['F']:.4%} | {t['P']:.4%} | {t['v_fluctuation']:.4%} |")
    lines+=['','| 轨迹 | 步数 | 跨单元粒子事件 | max F 提交误差 | max K 相对误差 | max cond(M) | 累计投影损失 / 初始内部能 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,run in r['runs'].items():
        rows=run['rows']
        lines.append(f"| {key} | {len(rows)} | {run['crossings']} | {max(q['prediction_commit_F_absolute'] for q in rows):.3g} | {max(q['kinetic_relative'] for q in rows):.3g} | {max(q['mass_condition'] for q in rows):.3g} | {run['projection_loss_over_initial_internal_energy']:.4%} |")
    d=r['shifted_vs_fixed']
    lines+=['',f"细时间步下，切换原点相对固定原点：F {d['F']:.4%}，P {d['P']:.4%}，去平移速度 {d['v_fluctuation']:.4%}，位置 RMS {d['x_absolute']:.6g}。",'',
        '传递恒等式通过不代表动态解收敛。F 以 F-I 归一化，速度以去掉整体平移后的速度归一化。',
        '换网格后的完整质量 P2G 是投影，可能丢失速度分量；累计损失以初始弹性能加去平移动能归一化。',
        '粗网格边界质量病态和 Q1 跨单元梯度变化是后续诊断方向；本轮数据尚未分离两者因果贡献。',
        '失败的时间检查保留在 results.json，命令在保存报告后以非零状态退出。尚不能作为生产精度验收。','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--device',default='auto');p.add_argument('--factor',type=int,default=2)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.report_only:report(a.out);return
    if a.factor<2 or a.factor&(a.factor-1):p.error('factor must be a power of two >=2')
    if a.device=='auto':a.device=device_choice()
    if a.device!='cpu':
        import warp as wp
        from utils.resource_guard import prepare_warp_cache
        wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    runs={};states={}
    for switched in (False,True):
        for factor in (a.factor//2,a.factor):
            key=f'{switched}-{factor}';runs[key],states[key]=run(a.out,switched,factor,a.device)
    params=SpatialMPM().params
    times={str(sw):compare(states[f'{sw}-{a.factor//2}'],states[f'{sw}-{a.factor}'],params) for sw in (False,True)}
    grid_difference=compare(states[f'True-{a.factor}'],states[f'False-{a.factor}'],params)
    result=dict(factor=a.factor,runs=runs,time=times,time_passed=all(row[k]<.05 for row in times.values() for k in ('F','P','v_fluctuation')),
        shifted_vs_fixed=grid_difference,consistency_passed=all(r['passed'] for r in runs.values()),storage=guard())
    result['passed']=result['consistency_passed'] and result['time_passed']
    save(a.out/'results.json',result);report(a.out);print('TIME',times,flush=True);print('SHIFT EFFECT',grid_difference,flush=True)
    if not result['time_passed']:raise RuntimeError('MPM time comparison >5%; inspect before refining')

if __name__=='__main__':main()
