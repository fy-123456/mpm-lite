"""Render saved physical evidence only; never advances a simulation."""
from pathlib import Path
import argparse,json
import numpy as np
from .provenance import read,write,sha
from benchmarks.research_basis_allocation_next.visualize import render

def diagnostics(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    pressure=read(run/'S3/exact-time-reference.json');t=np.array(pressure['times_s'])
    for label,r in pressure['records'].items():
        p=np.asarray(r['exact_pressure']);w=np.array(r['V0']);axs[0,0].plot(t*1e6,p@w/w.sum(),label=f"{label}: {r['cells']} cells")
        axs[0,1].plot(t*1e6,np.sum(r['exact_boundary_by_axis'],axis=1),label=label)
    rows=read(run/'S3/grid-comparison.json')['records'];axs[1,0].plot([x['time_s']*1e6 for x in rows],[x['boundary']['absolute']/x['boundary']['budget'] for x in rows],marker='.',label='boundary difference / budget');axs[1,0].axhline(1,color='k',linestyle='--')
    phase=read(run/'S1/time-comparison.json')
    for row in phase['records']:
        records=row['comparison']['field_records'];axs[1,1].plot([x['time_s'] for x in records],[max(v['velocity']['absolute']/v['velocity']['budget'] for v in x['regions'].values()) for x in records],label=str(row['window']))
    axs[1,1].axhline(1,color='k',linestyle='--')
    for ax,title,unit in zip(axs.ravel(),['volume-weighted fixed-solid pressure','cumulative boundary volume: exact time','early pressure-grid error remains scoped','new-space h vs h/2 regional velocity'],['Pa','m3','budget fraction','budget fraction']):
        ax.set_title(title,fontsize=10);ax.set_ylabel(unit);ax.set_xlabel('microseconds' if ax is not axs[1,1] else 'seconds');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[1,0].set_xscale('log');axs[1,0].set_xlabel('microseconds (log scale)')
    fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    space=read(run/'S2/research-space-decision.json');scope=read(run/'S3/pressure-scope-decision.json')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MPM-lite研究诊断</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{width:100%}pre{white-space:pre-wrap}</style><h1>当前空间相位与早期排水</h1><p>全部曲线来自保存数据。空间候选仅取得静态研究范围；主空间仍为cross-direction-snapshot6。压力时间对照通过，早期网格差未全部通过，因此没有启动新网格实际耦合。</p><img src="diagnostics.png"><h2>空间研究范围</h2><pre>'+json.dumps(space,ensure_ascii=False,indent=2)+'</pre><h2>压力范围</h2><pre>'+json.dumps(scope,ensure_ascii=False,indent=2)+'</pre>')
    write(out/'metadata.json',dict(no_integration=True,pressure_sha256=sha(run/'S3/exact-time-reference.json'),phase_sha256=sha(run/'S1/time-comparison.json')))

def bundle(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed visualization immutable')
    final=(run/'S6/final-protocol.json').exists();case=read(run/'S6/final-protocol.json')['default_case'] if final else 'time-h';render(run,case);diagnostics(run)
    summary=read(run/'cases'/case/'summary.json');stage='最终完整场景' if final else '实施中的短窗参照'
    (run/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>MPM-lite 新空间相位与压力边界</title><style>body{{font:18px system-ui;max-width:950px;margin:40px auto;line-height:1.7}}</style><h1>MPM-lite 新空间相位与压力边界</h1><p>{stage}：{case}，{summary["steps"]}步；完整M7，未增加阻尼。空间、全局时间精度仍未全面认证。</p><ul><li><a href="visualization/{case}/index.html">固体动画、原始反力与能量</a></li><li><a href="visualization/diagnostics/index.html">时间参照、非均匀压力网格与误差范围</a></li><li><a href="S2/reserved-direction-check.json">48.75°预留方向的静态空间验收</a></li></ul><p>显示形变默认放大10倍；物理位移、应力、压力和曲线使用原始值。候选空间未替换主运行空间，压力耦合未触发。</p>')
    print(run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();bundle(a.run)
