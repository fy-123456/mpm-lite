"""Saved solid frames and exact pressure-grid diagnostics, without integration."""
from pathlib import Path
import argparse,json
import numpy as np
from .provenance import read,write,sha
from benchmarks.research_basis_allocation_next.visualize import render

def pressure(run,output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=Path(output) if output else run/'visualization/pressure'
    if (run/'release.json').exists() and (out.resolve()==run.resolve() or run.resolve() in out.resolve().parents):raise ValueError('sealed output requires external destination')
    out.mkdir(parents=True,exist_ok=True);data=read(run/'S3/fixed-grid-comparison.json');protocol=read(run/'S3/fixed-time-protocol.json');times=protocol['times_s'];fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for cells in (2,4,8):
        row=data['records'][str(cells)];p=np.asarray(row['exact_pressure']);axs[0,0].plot(times,p.mean(axis=1),label=f'{cells} cells')
        axs[0,1].plot(times,row['exact_boundary'],label=f'{cells} cells');axs[1,0].plot(times,row['content'],label=f'{cells} cells')
    for label,rows in data['comparisons'].items():axs[1,1].plot([r['time'] for r in rows],[r['boundary_volume']['absolute']/r['boundary_volume']['budget'] for r in rows],label=label)
    axs[1,1].axhline(1,color='k',linestyle='--',label='engineering budget')
    for ax,title,y in zip(axs.ravel(),['fixed solid: exact-in-time mean pressure','cumulative boundary volume from augmented ODE','stored fluid content at common physical times','boundary-volume grid difference / budget'],['pressure (Pa)','volume (m3)','content (m3)','fraction']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('actual time (s)');ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'pressure-summary.png',dpi=140);plt.close(fig)
    scope=read(run/'S3/pressure-scope-decision.json')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MPM-lite 压力网格诊断</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}</style><h1>2 / 4 / 8 单元压力网格诊断</h1><p>固定原固体空间；压力与累计边界体积都由增广矩阵指数给出同一物理时刻的值。下图使用保存结果，不增加积分，不平滑。</p><p>历史两／四单元累计排水差仍保留。八单元研究与新固体空间的生产耦合、一般三维空间精度及耦合 q5 均未认证。</p><img src="pressure-summary.png"><pre>'+json.dumps(scope,ensure_ascii=False,indent=2)+'</pre>')
    write(out/'metadata.json',dict(no_recomputation=True,source_sha256=sha(run/'S3/fixed-grid-comparison.json'),actual_end_s=times[-1],scope=scope));return out

def bundle(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed bundle immutable')
    case=read(run/'S6/final-protocol.json')['default_case'];render(run,case);pressure(run);space=read(run/'selected-space.json')['selected']
    (run/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>MPM-lite 当前成果</title><style>body{{font:18px system-ui;max-width:900px;margin:40px auto;line-height:1.7}}</style><h1>MPM-lite 跨方向空间与压力网格成果</h1><p>固体空间：{space}；144 函数，完整 M7，252 步。整体稳定性和材料压缩按实际周期验收，连续空间与完整时间精度仍未认证。</p><ul><li><a href="visualization/{case}/index.html">固体动画、原始反力与能量</a></li><li><a href="visualization/pressure/index.html">压力网格与累计排水量</a></li></ul><p>时间传播和压力研究固定在原 BASELINE，不能直接视为新空间已通过耦合。形变显示默认放大 10 倍，计算位移、应力和压力保持原值。</p>')
    print(run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case');p.add_argument('--pressure',action='store_true');p.add_argument('--bundle',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.bundle:bundle(a.run)
    elif a.pressure:pressure(a.run,a.output)
    elif a.case:render(a.run,a.case,a.output)
    else:p.error('choose --case, --pressure or --bundle')
