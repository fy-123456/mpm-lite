"""Inherited daily animation and separately labelled actual research diagnostics."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .lineage import default_source
from benchmarks.research_basis_allocation_next.visualize import render


def visualize(run):
    run=Path(run);verify(run)
    if (run/'release.json').exists():raise ValueError('sealed visualization')
    folder,chain=default_source(APP,APP_SHA);owner=folder.parent.parent;name=folder.name
    render(owner,name,output=run/'visualization'/name)
    page=run/'visualization'/name/'index.html';page.write_text(page.read_text().replace('<h1>','<p style="padding:12px;background:#edf4ff">252步、12帧继承自 '+owner.name+'；本轮只重绘已有数据。</p><h1>',1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    comp=read(run/'S1/time-comparison.json')['records']
    for label,item in comp.items():
        rows=item['records'];ax[0,0].plot([r['time_s'] for r in rows],[max(v['velocity']['absolute']/v['velocity']['budget'] for v in r['fields'].values()) for r in rows],label=label.replace('candidate_',''))
    ax[0,0].axhline(1,color='k',ls='--');ax[0,0].set(title='Candidate reference remains limited',xlabel='time (s)',ylabel='velocity error / budget');ax[0,0].legend(fontsize=8)
    f=read(run/'S1/force-attribution.json')
    for k in ('material','stabilization','boundary_cross_mass'):ax[0,1].plot(f['indices'],f['physical_rms_difference_m_s2'][k],'.-',label=k)
    ax[0,1].set(title='Acceleration differences at physical probes',xlabel='reference step',ylabel='RMS acceleration difference (m/s2)',yscale='log');ax[0,1].legend(fontsize=8)
    p=read(run/'S2/paired-performance.json');x=np.arange(2)
    ax[0,2].bar(x-.18,[r['A']['advance_s'] for r in p['records']],.36,label='original');ax[0,2].bar(x+.18,[r['B']['advance_s'] for r in p['records']],.36,label='device RT0')
    ax[0,2].set(title='Actual coupled steps: ~84% less time',ylabel='advance wall time (s)',xticks=x,xticklabels=['source step 1','source step 3']);ax[0,2].legend()
    for label in ('time','grid'):
        rows=read(run/f'S3/{label}-comparison.json')['records'];ax[1,0].plot([r['time_s']*1e6 for r in rows],[r['extra']['pressure']['absolute']/r['extra']['pressure']['budget'] for r in rows],label=label)
    ax[1,0].axhline(1,color='k',ls='--');ax[1,0].set(title='Pressure comparison passes',xlabel='time (microseconds)',ylabel='pressure error / budget');ax[1,0].legend()
    d=read(run/'S3/grid-limitation-diagnostic.json')['records'];t=[r['time_s']*1e6 for r in d]
    for key,label,style in [('actual_coarse_flow_m3_s','actual 16 cells','-'),('actual_fine_flow_m3_s','actual 32 cells','-'),('fixed_coarse_flow_m3_s','fixed 16 cells','--'),('fixed_fine_flow_m3_s','fixed 32 cells','--')]:ax[1,1].plot(t,[r[key] for r in d],style,label=label)
    ax[1,1].set(title='Boundary flow remains grid sensitive',xlabel='time (microseconds)',ylabel='outward boundary flow (m3/s)');ax[1,1].legend(fontsize=8)
    r=read(run/'S4/reaction-engineering.json')['records'][-1]
    for label,s in zip(('128 steps','256 steps'),r['signals']):ax[1,2].plot(s['raw_times'],s['raw_values'],label=label)
    ax[1,2].set(title='Reaction: engineering pass; raw event limited',xlabel='interval midpoint time (s)',ylabel='reaction (N)');ax[1,2].legend()
    for a in ax.flat:a.grid(alpha=.2)
    fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>本轮研究诊断</title><h1>恢复力参照与 RT0 诊断</h1><p>研究入口真实步耗时下降约84%；候选空间参照及耦合边界网格精度仍有限。图中显示各自工程预算，未放宽数值门槛。</p><img style="max-width:100%" src="diagnostics.png"><p><a href="../../implementation-report.md">完整实施记录</a></p></html>')
    (run/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-lite 恢复力与RT0交付</title><style>body{{max-width:1200px;margin:3rem auto;font:18px/1.7 sans-serif;padding:0 1rem}}img{{width:100%}}a{{color:#1268a3}}</style><h1>MPM-lite 恢复力参照与 RT0 优化</h1><p>研究版本：{run.name}</p><p>RT0设备组装在两个真实耦合步上减少约84%耗时，修复检查点身份序列化问题。32单元短窗稳定，时间步比较通过，边界流量仍受网格分辨率限制。</p><p><a href="visualization/{name}/index.html">日常固体交互动画</a> · <a href="visualization/diagnostics/index.html">本轮诊断</a> · <a href="implementation-report.md">实施记录</a> · <a href="capability-matrix.json">适用范围</a></p><p>日常动画继承自 {owner.name}：144函数、完整质量、252步、12帧、原q5回退范围，显示变形放大10倍。本轮没有重跑完整周期，研究候选未替换正式空间，耦合尚未用于生产。</p><img src="visualization/diagnostics/diagnostics.png"></html>''')
    write(run/'S6/visualization-origin.json',dict(status='passed_scoped',source_release=str(owner),source_release_sha256=sha(owner/'release.json'),case=name,chain=chain,source_identity_sha256=sha(folder/'identity.json'),new_visualization_of_inherited_data=True,new_integration_steps=0))
    print('VISUALIZATION',run/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):visualize(a.run)
