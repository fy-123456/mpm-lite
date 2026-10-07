"""Portable plots of current actual trajectories and separately inherited daily frames."""
import argparse,shutil
import numpy as np
from .provenance import *
from .review import fluid,context
from .coupling import case_name

def visualize(run):
    run=Path(run);verify(run)
    if (run/'release.json').exists():raise ValueError('sealed visualization')
    name='daily-q5-retry';destination=run/'visualization'/name
    shutil.copytree(APP/'visualization'/name,destination,dirs_exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True);method=read(run/'S2/method-decision.json')['method'];grid=read(run/'S3/grid-decision.json');perf=read(run/'S4/paired-performance.json')
    p=read(run/'S0/input-contract.json');contexts={g:context(p['cuts'][g],p['parameters']['storage']) for g in ('coarse','fine')}
    fig,ax=plt.subplots(3,3,figsize=(18,13),layout='constrained')
    for m in ('backward-euler','startup'):
        for fine in (False,True):
            v,t,rows=fluid(run/'cases'/case_name('coarse',fine,m));label=m+(' 32 steps' if fine else ' 16 steps');w=contexts['coarse']['top'].V0;w=w/w.sum()
            ax[0,0].plot(t*1e6,v['pressure']@w,label=label)
            ax[1,0].plot(t[1:]*1e6,np.cumsum([r['numerical_dissipation_J'] for r in rows])*1e12,label=label)
            ax[1,1].plot(t[1:]*1e6,np.cumsum([r['darcy_dissipation_J'] for r in rows])*1e12,label=label)
    ax[0,0].set(title='Actual zero-source 16 cells: pressure',ylabel='volume mean pressure (Pa)');ax[1,0].set(title='Numerical dissipation: separate ledger',ylabel='cumulative Dnum (pJ)');ax[1,1].set(title='Physical Darcy dissipation',ylabel='cumulative Darcy energy (pJ)')
    comparisons=read(run/'S3/coupled-comparison.json')
    for axis,key,title in [(ax[0,1],'coarse_temporal','16 cells: time-step comparison'),(ax[0,2],'fine_temporal','32 cells: time-step comparison'),(ax[1,2],'spatial','16 / 32 cells: both 32 steps')]:
        for field in ('pressure','face_flux','boundary_by_side','actual_total_content'):
            rr=comparisons[key]['fluid']['records'];axis.plot([r['time_s']*1e6 for r in rr],[max(r['budget_ratios'][field],1e-14) for r in rr],label=field)
        axis.axhline(1,color='k',ls='--');axis.set(title=title,ylabel='error / engineering budget',yscale='log')
    v,t,rows=fluid(run/'cases'/case_name('coarse',True,method));groups=contexts['coarse']['groups']
    for side,label in enumerate(['-x','+x','-y','+y','-z','+z']):ax[2,0].plot(t*1e6,v['cumulative']@groups[side]*1e15,label=label)
    ax[2,0].set(title='Six-side drainage (selected method)',ylabel='signed cumulative outflow (pL)')
    for m in ('backward-euler','startup'):
        v,t,rows=fluid(run/'cases'/case_name('coarse',False,m));total=(v['flux']@groups.T).sum(axis=1)
        ax[2,1].stairs(total*1e12,t*1e6,label=m,baseline=None)
    ax[2,1].axvline(25,color='k',ls=':');ax[2,1].set(title='Original interval flow at fixed 25 us switch',ylabel='total boundary outflow (nL/s)',xlim=(0,60))
    rr=perf['records'];x=np.arange(2)
    A=[r.get('A_mean_s',r.get('A',{}).get('advance_s')) for r in rr];B=[r.get('B_mean_s',r.get('B',{}).get('advance_s')) for r in rr]
    ax[2,2].bar(x-.18,A,.36,label='A local adjoint');ax[2,2].bar(x+.18,B,.36,label='B bounded transpose');ax[2,2].set(title=f"Actual selected fixture: measured gain {100*perf['median_gain']:.1f}%",ylabel='s / measured complete step',xticks=x,xticklabels=['from12.5us','from37.5us'])
    for i,axis in enumerate(ax.flat):
        axis.grid(alpha=.2);axis.legend(fontsize=7)
        if i!=8:axis.set_xlabel('time (microseconds)')
    fig.savefig(out/'diagnostics.png',dpi=145);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>零体源θ耦合研究</title><h1>本轮实际200微秒轨迹</h1><p>零体源保留压力边界排水。全部首区间保留；数值耗散与Darcy耗散分列。两网格一致性不是连续体空间精度认证。</p><img style="max-width:100%" src="diagnostics.png"><p><a href="../../implementation-report.md">实施记录</a></p></html>')
    (run/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-lite 零体源θ耦合交付</title><style>body{{max-width:1350px;margin:2rem auto;font:18px/1.7 sans-serif;padding:0 1rem}}img{{max-width:100%}}</style><h1>零体源、低耗散时间推进与压力网格</h1><p>版本：{run.name}</p><p>本轮真实轨迹覆盖0–200微秒。选定方法：{method}；16/32压力网格结论：{grid['status']}。16单元后向欧拉复用已审核的12/24步前缀，仅续4/8步；其余新轨迹从共同初态独立计算。</p><p><a href="visualization/{name}/index.html">继承的日常固体动画</a> · <a href="visualization/diagnostics/index.html">本轮研究曲线</a> · <a href="implementation-report.md">实施记录</a> · <a href="capability-matrix.json">资格与限制</a></p><p>日常动画来自{SOLID.name}：252步、12帧，显示变形放大10倍，本轮未重跑。错误体源旧轨迹全部排除。真实耦合连续体空间精度、完整周期、耦合q5与生产C/E未认证；正式144函数保持不变。</p><img src="visualization/diagnostics/diagnostics.png"></html>''')
    write(run/'S6/visualization-origin.json',dict(status='inherited_daily_plus_new_actual_coupling_diagnostics',daily_source=str(SOLID),copied_from=str(APP/'visualization'/name),daily_steps=252,daily_frames=12,daily_deformation_scale=10,new_daily_steps=0,new_trajectories_window_us=200,BE_inherited_prefix_steps=[12,24],wrong_source_cases_excluded=True))
    print('VISUALIZATION',run/'index.html',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):visualize(a.run)
