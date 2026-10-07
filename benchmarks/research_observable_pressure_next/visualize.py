"""Render saved solid/pressure fields only; no additional numerical integration."""
from pathlib import Path
import argparse,json
import numpy as np
from .provenance import read,write,sha
from benchmarks.research_basis_allocation_next.visualize import render
from benchmarks.research_sequential_next.checkpoint import GenerationStore

def pressure(run,output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=Path(output) if output else run/'visualization/pressure'
    if (run/'release.json').exists() and (out.resolve()==run.resolve() or run.resolve() in out.resolve().parents):raise ValueError('sealed output requires external destination')
    out.mkdir(parents=True,exist_ok=True);fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained');scopes=[]
    for cells in (2,4):
        p=run/f'S3/fixed-{cells}.json'
        if p.exists():
            fixed=read(p)
            for j in range(cells):axs[0,0].plot([r['time'] for r in fixed['rows']],[r['pressure_Pa'][j] for r in fixed['rows']],'.-',label=f'fixed {cells} cells / {j}')
        folder=run/'cases'/f'pressure-{cells}'
        if not (folder/'identity.json').exists():continue
        history=GenerationStore(folder,read(folder/'identity.json')).history();rows=history[-1]['rows']
        if not rows:continue
        for j in range(cells):axs[0,1].plot([r['time'] for r in rows],[r['pressure_Pa'][j] for r in rows],'.-',label=f'coupled {cells} / {j}')
        axs[1,0].plot([r['time'] for r in rows],[r['energy_balance_J'] for r in rows],'.-',label=f'{cells} cells')
        if cells==2:axs[1,1].bar(np.arange(len(rows[-1]['flux_interval_m3_s'])),rows[-1]['flux_interval_m3_s'])
        scopes.append(dict(cells=cells,steps=len(rows),end_s=rows[-1]['time'],identity_sha256=sha(folder/'identity.json')))
    for i,(ax,title) in enumerate(zip(axs.ravel(),['fixed solid at the same physical times','coupled pressure: each curve uses actual time','raw total energy balance (J)','two-cell oriented face flux (m3/s)'])):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.set_xlabel('actual time (s)' if i<3 else 'face index')
        if i<3:ax.legend(fontsize=7)
    axs[0,0].set_ylabel('pressure (Pa)');axs[0,1].set_ylabel('pressure (Pa)')
    fig.savefig(out/'pressure-summary.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MPM-lite 压力限定范围</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}</style><h1>全张量压力时间／网格检查</h1><p>固定固体比较采用相同物理时间。实际耦合曲线各按真实时刻绘制；不同终点不能直接作为网格误差。压力、通量和能量均为原始数据，未平滑。</p><p>两单元7步通过限定验收；四单元4步各自稳定，但累计边界体积网格差5.57%略超预算，保留为网格敏感诊断。未认证一般三维生产耦合、压力空间精度或耦合q5。</p><img src="pressure-summary.png"><pre>'+json.dumps(scopes,indent=2)+'</pre>')
    write(out/'metadata.json',dict(no_recomputation=True,scopes=scopes))
    return out

def bundle(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed bundle immutable')
    case=read(run/'S6/final-protocol.json')['default_case'];render(run,case);pressure(run)
    (run/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>MPM-lite 当前成果</title><style>body{{font:18px system-ui;max-width:900px;margin:40px auto;line-height:1.7}}</style><h1>MPM-lite 物理输出与压力范围优化</h1><p>正式144函数、完整M7、F45；分段时间表的局部改进通过工程检查，连续空间／完整时间精度仍未认证。</p><ul><li><a href="visualization/{case}/index.html">已提交固体动画、原始反力与能量</a></li><li><a href="visualization/pressure/index.html">全张量压力、定向通量与能量</a></li></ul><p>四单元累计排出体积比较网格敏感。F60预登记静态方向研究未通过正式空间应力预算，未替换正式主工况。位移仅在显示时放大10倍，应力、压力和计算数据不放大。</p>''')
    print(run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case');p.add_argument('--pressure',action='store_true');p.add_argument('--bundle',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.bundle:bundle(a.run)
    elif a.pressure:pressure(a.run,a.output)
    elif a.case:render(a.run,a.case,a.output)
    else:p.error('choose --case, --pressure or --bundle')
