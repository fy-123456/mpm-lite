"""Display this turn's reference, all failed gates, and scoped paired measurements."""
import argparse,shutil
import numpy as np
from .provenance import *


def visualize(run):
    run=Path(run);mutable(run);name='daily-q5-retry';destination=run/'visualization'/name
    shutil.copytree(APP/'visualization'/name,destination,dirs_exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    for n in (32,128):
        v=np.load(run/'R1'/f'scalar-{n}.npz');x=(v['cuts'][:-1]+v['cuts'][1:])/2-v['cuts'][0]
        keep=x<300e-6
        ax[0,0].plot(x[keep]*1e6,v['pressure'][-1,keep],'-o',ms=3,label=f'RT0 {n} cells')
        ax[0,0].plot(x[keep]*1e6,v['reference_pressure'][-1,keep],'--',label=f'analytic cell means ({n})')
        ax[0,1].stairs(v['flux']*1e12,v['times']*1e6,baseline=None,label=f'RT0 {n} cells')
    ax[0,1].stairs(v['reference_flux']*1e12,v['times']*1e6,baseline=None,ls='--',color='black',label='analytic interval mean')
    ax[0,0].set(title='Independent scalar subproblem at 200 us',xlabel='cell centre distance from left boundary (um)',ylabel='cell mean pressure (Pa)')
    ax[0,1].set(title='Scalar outward flow: first interval included',xlabel='time (us)',ylabel='one-end interval flow (nL/s)')
    screen=read(run/'R2/grid-screen.json');rr=screen['records'];x=np.arange(len(rr));w=.24
    for offset,vals,label in [(-w,[max(r['reference']['max_budget_ratios'].values()) for r in rr],'64/128 reference (25% budget)'),(0,[max(r['spatial']['max_budget_ratios'].values()) for r in rr],'32/128 space (full budget)'),(w,[max(max(t['comparison']['max_budget_ratios'].values()) for t in r['time']) for r in rr],'same-grid time (full budget)')]:
        ax[0,2].bar(x+offset,vals,w,label=label)
    ax[0,2].axhline(1,color='black',ls='--');ax[0,2].set(title='Full tensor, fixed skeleton, x-refinement only',ylabel='maximum error / declared budget',yscale='log',xticks=x,xticklabels=['4um uniform','4um cubic','8um uniform','8um cubic'])
    for t in rr[0]['time']:
        records=t['comparison']['records']
        for field,ls in [('face_flux','-'),('boundary_by_side','--')]:ax[1,0].plot([r['time_s']*1e6 for r in records],[r['budget_ratios'][field] for r in records],ls,label=t['label']+' '+field)
    ax[1,0].axhline(1,color='black',ls=':');ax[1,0].axvline(25,color='gray',ls=':');ax[1,0].set(title='4um uniform family: raw time error remains',xlabel='time (us)',ylabel='error / full engineering budget',xlim=(0,200))
    perf=read(run/'R4/paired-performance.json');x=np.arange(2)
    for offset,kind,label in [(-.18,'A','A: inherited Gauss'),(.18,'B','B: shared midpoint')]:
        ax[1,1].bar(x+offset,[r[kind]['advance_s'] for r in perf['records']],.36,label=label)
        ax[1,2].bar(x+offset,[r[kind]['volume_gradient_adjoint_s'] for r in perf['records']],.36,label=label)
    ax[1,1].set(title='Actual single steps; shared GPU: NOT qualified',ylabel='measured complete step (s)',xticks=x,xticklabels=['from 12.5us','from 37.5us'])
    ax[1,2].set(title='Volume-gradient adjoint within those steps',ylabel='inclusive adjoint time (s)',xticks=x,xticklabels=['from 12.5us','from 37.5us'])
    for axis in ax.flat:axis.grid(alpha=.2);axis.legend(fontsize=7)
    fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>边界参考与共享几何</title><h1>本轮参考、限制与实步对照</h1><p>解析真解只对应独立标量子问题。全张量检查为固定骨架x向加密；首区间及失败结果全部保留。A/B状态与事务通过，GPU共享使性能资格受限。</p><img style="max-width:100%" src="diagnostics.png"><p><a href="../../implementation-report.md">实施记录</a></p></html>')
    (run/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-lite 边界参考与共享几何交付</title><style>body{{max-width:1350px;margin:2rem auto;font:18px/1.7 sans-serif;padding:0 1rem}}img{{max-width:100%}}</style><h1>边界排水参考与共享几何求值</h1><p>版本：{run.name}</p><p>新增独立解析单元均值/区间流量参考、两个边界网格族和精确Simpson体积梯度复用。实际测试为原16单元场景两对单步（4次成功）及1次故障回滚，没有新增完整耦合轨迹。</p><p>新网格参考与空间对照在均匀观察区间通过，启动时间流量仍受限，因此未启动新网格实际轨迹。几何优化的状态、离散压力功、能量及重启检查通过；共享GPU下测得的26%/38%耗时降幅仅供观察，未提升为默认后端。</p><p><a href="visualization/{name}/index.html">继承的日常固体动画</a> · <a href="visualization/diagnostics/index.html">本轮检查曲线</a> · <a href="implementation-report.md">实施记录</a> · <a href="capability-matrix.json">资格与限制</a></p><p>日常动画来源：{SOLID.name}，252步、12帧、变形显示放大10倍，本轮只做零步加载。正式144函数及原q5权限保持原范围；三维耦合连续体精度、完整周期和耦合q5未认证。</p><img src="visualization/diagnostics/diagnostics.png"></html>''')
    write(run/'S6/visualization-origin.json',dict(status='new_CPU_reference_and_AB_diagnostics_plus_inherited_daily',daily_source=str(SOLID),copied_from=str(APP/'visualization'/name),daily_steps=252,daily_frames=12,daily_deformation_scale=10,new_daily_steps=0,new_continuous_coupled_trajectories=0,new_AB_successful_steps=4,performance_shared_GPU=True))
    print('VISUALIZATION',run/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):visualize(a.run)
