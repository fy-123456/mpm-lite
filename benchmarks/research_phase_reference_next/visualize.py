"""View committed solid and two-cell RT0 histories; no simulation on rendering."""
from pathlib import Path
import argparse,json
import numpy as np
from .provenance import read,write,sha
from benchmarks.research_basis_allocation_next.visualize import render
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def render_pressure(run,output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=Path(output) if output else run/'visualization/pressure'
    if (run/'release.json').exists() and (out.resolve()==run.resolve() or run.resolve() in out.resolve().parents):raise ValueError('sealed output requires external destination')
    out.mkdir(parents=True,exist_ok=True);fixed=read(run/'S4/fixed-common-time.json');folder=run/'cases/coupled-rt0-small2';h=GenerationStore(folder,read(folder/'identity.json')).history();rows=h[-1]['rows']
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained');small=fixed['records'][1]
    for j in range(2):
        axs[0,0].plot([r['time'] for r in small['rows']],[r['pressure_Pa'][j] for r in small['rows']],'.-',label=f'fixed cell {j}')
        axs[0,1].plot([r['time'] for r in rows],[r['pressure_Pa'][j] for r in rows],'.-',label=f'coupled cell {j}')
    axs[1,0].plot([r['time'] for r in rows],[r['energy_balance_J'] for r in rows],'.-',label='energy closure (J)')
    z=np.asarray(rows[-1]['flux_interval_m3_s']);axs[1,1].bar(np.arange(len(z)),z);axs[1,1].set_xticks(np.arange(len(z)));axs[1,1].set_xlabel('global face (positive-axis orientation)');axs[1,1].set_ylabel('volume flux (m3/s)')
    titles=['full-tensor RT0, fixed solid, small storage','actual 3D solid coupling, eight committed steps','raw total energy balance','all 11 oriented face fluxes, final interval']
    for i,(ax,title) in enumerate(zip(axs.ravel(),titles)):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2)
        if i<3:ax.set_xlabel('actual time (s)');ax.legend(fontsize=8)
    axs[0,0].set_ylabel('pressure (Pa)');axs[0,1].set_ylabel('pressure (Pa)');fig.savefig(out/'pressure-summary.png',dpi=140);plt.close(fig)
    closure=read(run/'S4/coupled-eight-step.json')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MPM-lite 三维压力短窗</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}</style><h1>两单元三维压力检查</h1><p>八个实际积分步；完整张量渗流、11个共享面通量。压力和能量使用原始数据，没有平滑或放大。</p><p>这是有限研究夹具，尚未认证一般三维生产耦合、连续压力空间精度或压力单调性。</p><img src="pressure-summary.png" alt="压力、能量与定向通量"><pre>'+json.dumps({k:closure[k] for k in ['steps','end_s','cumulative_mass_defect_m3','max_energy_defect_J']},ensure_ascii=False,indent=2)+'</pre>')
    write(out/'metadata.json',dict(no_recomputation=True,pressure_steps=8,cells=2,faces=11,source_sha256=sha(run/'S4/coupled-eight-step.json'),end_s=rows[-1]['time']))
    return out


def bundle(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed release; render externally')
    default=read(run/'S6/final-protocol.json')['default_case'];render(run,default);render_pressure(run)
    (run/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>MPM-lite 当前成果</title><style>body{{font:18px system-ui;max-width:900px;margin:40px auto;line-height:1.7}}a{{color:#174f84}}</style><h1>MPM-lite 当前已提交结果</h1><p>当前144函数空间；主场景完整周期与采用缩放求解的两单元三维压力短窗。</p><ul><li><a href="visualization/{default}/index.html">固体交互动画与原始反力、能量曲线</a></li><li><a href="visualization/pressure/index.html">八步三维压力、定向通量与能量</a></li></ul><p>固体位移默认仅在显示时放大10倍。应力和计算数据不放大；连续空间、全周期时间精度与一般生产耦合仍以研究验收范围为限。</p>''')
    print(run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case');p.add_argument('--pressure',action='store_true');p.add_argument('--bundle',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.bundle:bundle(a.run)
    elif a.pressure:render_pressure(a.run,a.output)
    elif a.case:render(a.run,a.case,a.output)
    else:p.error('--case, --pressure or --bundle required')
