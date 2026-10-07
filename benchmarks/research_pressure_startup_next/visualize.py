"""Static, portable views of bounded research and explicitly inherited daily data."""
from pathlib import Path
import argparse,shutil
import numpy as np
from .provenance import *
from .coupling_review import fluid,context
from .coupling import case_name

def visualize(run):
    run=Path(run);verify(run)
    if (run/'release.json').exists():raise ValueError('sealed visualization')
    name='daily-q5-retry';destination=run/'visualization'/name
    shutil.copytree(APP/'visualization'/name,destination,dirs_exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    rr=read(run/'S1/startup-screen.json')['records'];x=np.arange(len(rr))
    ax[0,0].bar(x,[r['score'] for r in rr]);ax[0,0].axhline(1,color='k',ls='--');ax[0,0].set(title='Fixed-skeleton startup screen',ylabel='maximum error / budget',xticks=x,xticklabels=[r['family']+'\n'+r['method'] for r in rr])
    params=read(run/'S5/selected-protocol.json');ct=context(params['cuts']['coarse'],params['parameters']['storage'])
    # The topology arrays are immutable; normalize into a separate vector.
    weights=ct['top'].V0/ct['top'].V0.sum()
    for fine in (False,True):
        v,t,rows=fluid(run/'cases'/case_name('coarse',fine));label='24 steps' if fine else '12 steps'
        ax[0,1].plot(t*1e6,v['pressure']@weights,label=label)
        cumulative=np.cumsum([r['numerical_dissipation_J'] for r in rows]);darcy=np.cumsum([r['darcy_dissipation_J'] for r in rows])
        ax[1,0].plot(t[1:]*1e6,cumulative*1e12,label=label+' numerical');ax[1,0].plot(t[1:]*1e6,darcy*1e12,'--',label=label+' Darcy')
    ax[0,1].set(title='Corrected ZERO source, 16 cells only',xlabel='time (microseconds)',ylabel='volume mean pressure (Pa)');ax[0,1].legend()
    ax[1,0].set(title='Separate numerical and physical dissipation',xlabel='time (microseconds)',ylabel='cumulative energy (pJ)');ax[1,0].legend(fontsize=8)
    rr=read(run/'S2/short-reference-comparison.json')['records'];t=[(v['time_s']-1.075)*1e6 for v in rr]
    for k in ('velocity','PK1'):ax[0,2].plot(t,[max(v['fields'][reg][k]['absolute']/v['fields'][reg][k]['budget'] for reg in v['fields']) for v in rr],label=k)
    ax[0,2].plot(t,[v['reaction']['absolute']/v['reaction']['budget'] for v in rr],label='reaction');ax[0,2].axhline(1,color='k',ls='--');ax[0,2].set(title='Extended solid prefix ONLY: 195 microseconds',xlabel='time since 1.075 s (microseconds)',ylabel='error / budget',yscale='log');ax[0,2].legend()
    comparison=read(run/'S5/coupled-comparison.json')['temporal']['coarse']['fluid'];rr=comparison['records']
    for key in ('pressure','face_flux','boundary_by_side','actual_total_content'):ax[1,1].plot([v['time_s']*1e6 for v in rr],[v['budget_ratios'][key] for v in rr],label=key)
    ax[1,1].axhline(1,color='k',ls='--');ax[1,1].set(title='Same-grid time check; first interval included',xlabel='time (microseconds)',ylabel='error / budget',yscale='log');ax[1,1].legend(fontsize=8)
    perf=read(run/'S4/paired-performance.json');rr=perf['records'];x=np.arange(2)
    ax[1,2].bar(x-.18,[r['A_mean_s'] for r in rr],.36,label='BASE local gradient');ax[1,2].bar(x+.18,[r['B_mean_s'] for r in rr],.36,label='bounded sparse transpose');ax[1,2].set(title=f"16-cell midpoint fixture: {100*perf['median_gain']:.1f}% lower time",ylabel='actual step wall time (s)',xticks=x,xticklabels=['state 0 (one pair)','state 1 (both pairs)']);ax[1,2].legend(fontsize=8)
    for a in ax.flat:a.grid(alpha=.2)
    fig.savefig(out/'diagnostics.png',dpi=145);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>压力启动与短窗研究</title><h1>压力启动、局部时间参照和稀疏回缩性能</h1><p>图中实际耦合仅包含修复后的零体源12/24步、16单元前缀。此前错误体源轨迹已排除。细网格及完整耦合周期尚未验证；数值耗散独立于Darcy耗散。</p><img style="max-width:100%" src="diagnostics.png"><p><a href="../../implementation-report.md">完整实施记录</a></p></html>')
    (run/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-lite 压力启动研究交付</title><style>body{{max-width:1250px;margin:2rem auto;font:18px/1.7 sans-serif;padding:0 1rem}}img{{max-width:100%}}</style><h1>MPM-lite 压力启动与短窗参照</h1><p>版本：{run.name}</p><p>局部固体参照扩展至195微秒；零源耦合验证范围为0–150微秒的16单元12/24步。稀疏回缩在两个指定中点研究状态下耗时降低18.9%，未替换正式日常默认。</p><p><a href="visualization/{name}/index.html">继承的日常固体动画</a> · <a href="visualization/diagnostics/index.html">本轮研究图表</a> · <a href="implementation-report.md">实施记录</a> · <a href="capability-matrix.json">逐项资格</a> · <a href="S5/source-mismatch/diagnosis.json">源项异常记录</a></p><p>日常动画来自{SOLID.name}：252步、12帧，变形显示放大10倍，本轮未重跑。全局空间/时间精度、细网格耦合、完整耦合周期及生产C/E仍未认证。所有实际异常与被撤回的结果保留可追溯记录。</p><img src="visualization/diagnostics/diagnostics.png"></html>''')
    write(run/'S6/visualization-origin.json',dict(status='inherited_daily_plus_new_diagnostics',daily_source=str(SOLID),copied_from=str(APP/'visualization'/name),daily_steps=252,daily_frames=12,daily_deformation_scale=10,new_daily_steps=0,excluded_source_mismatch_cases=True))
    print('VISUALIZATION',run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):visualize(a.run)
