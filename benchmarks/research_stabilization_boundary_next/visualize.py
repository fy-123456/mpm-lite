"""Keep inherited scene, local reference, failed gates and performance distinct."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .lineage import default_source
from benchmarks.research_basis_allocation_next.visualize import render

def visualize(run):
    run=Path(run);verify(run)
    if (run/'release.json').exists():raise ValueError('sealed visualization')
    folder,chain=default_source(APP,APP_SHA);owner=folder.parent.parent;name=folder.name;render(owner,name,output=run/'visualization'/name)
    page=run/'visualization'/name/'index.html';page.write_text(page.read_text().replace('<h1>','<p>继承日常252步、12帧；本轮没有重跑完整周期。显示变形放大10倍。</p><h1>',1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    rows=read(run/'S1/projection-vs-history.json')['records']
    for k in ('projection_rms','history_rms','total_rms'):ax[0,0].plot([r['index'] for r in rows],[r['components']['stabilization']['global_domain'][k] for r in rows],'.-',label=k)
    ax[0,0].set(title='Same-time projection and evolved history',xlabel='saved step',ylabel='volume-weighted RMS acceleration (m/s2)');ax[0,0].legend(fontsize=8)
    r=read(run/'S2/fixed-skeleton-reference.json')['records'];x=np.arange(len(r));ax[0,1].bar(x-.25,[max(v['reference']['max_budget_ratios'].values()) for v in r],.25,label='64/128 reference');ax[0,1].bar(x,[max(v['space']['max_budget_ratios'].values()) for v in r],.25,label='32/128 space');ax[0,1].bar(x+.25,[max(max(t['max_budget_ratios'].values()) for t in v['time'].values()) for v in r],.25,label='time')
    ax[0,1].axhline(1,color='k',ls='--');ax[0,1].set(title='Boundary startup: no new coupled trajectory',ylabel='error / respective gate',yscale='log',xticks=x,xticklabels=[str(v['ratio'])+' '+v['schedule'] for v in r]);ax[0,1].legend(fontsize=7)
    r=read(run/'S3/short-reference-comparison.json')['records'];t=[(v['time_s']-1.075)*1e6 for v in r]
    for k in ('velocity','PK1'):ax[0,2].plot(t,[max(v['fields'][reg][k]['absolute']/v['fields'][reg][k]['budget'] for reg in v['fields']) for v in r],label=k)
    ax[0,2].plot(t,[v['reaction']['absolute']/v['reaction']['budget'] for v in r],label='reaction');ax[0,2].axhline(1,color='k',ls='--');ax[0,2].set(title='Only the 98-microsecond prefix passes',xlabel='time since origin (microseconds)',ylabel='error / engineering budget',yscale='log');ax[0,2].legend()
    attribution=read(run/'S2/boundary-attribution.json')['records'];x=np.arange(6)
    for j,v in enumerate(attribution):ax[1,0].bar(x+(j-.5)*.35,np.array(v['cumulative_by_side_m3'])*1e9,.35,label='inherited '+str(16*(j+1))+' cells')
    ax[1,0].set(title='Inherited drainage, all boundary directions',ylabel='cumulative volume (1e-9 m3)',xticks=x,xticklabels=attribution[0]['side_order']);ax[1,0].legend(fontsize=8)
    spatial=read(run/'S4/space-decision.json');stat=run/'S4/static-comparison.json'
    if stat.exists():
        rr=read(stat)['records'];x=np.arange(len(rr));ax[1,1].bar(x-.18,[v['old']['interior']['fiber_PK1']['absolute'] for v in rr.values()],.36,label='prior candidate');ax[1,1].bar(x+.18,[v['errors']['interior']['fiber_PK1']['absolute'] for v in rr.values()],.36,label='origin-restoring');ax[1,1].set(xticks=x,xticklabels=list(rr))
    ax[1,1].set(title='144 functions: '+spatial['status'],ylabel='interior fiber stress error (Pa)');ax[1,1].legend(fontsize=8)
    perf=read(run/'S5/performance-decision.json');pair=run/'S5/paired-performance.json'
    if pair.exists():
        r=read(pair)['records'];x=np.arange(2);ax[1,2].bar(x-.18,[v['A']['advance_s'] for v in r],.36,label='current device RT0');ax[1,2].bar(x+.18,[v['B']['advance_s'] for v in r],.36,label='local cell gradient');ax[1,2].set(xticks=x,xticklabels=['source step 1','source step 3'])
    else:
        r=read(run/'S5/profile.json')['records'];ax[1,2].bar(np.arange(2),[v['advance_s'] for v in r],label='current device RT0')
    ax[1,2].set(title='Actual step: '+perf['status'],ylabel='advance wall time (s)');ax[1,2].legend(fontsize=8)
    for a in ax.flat:a.grid(alpha=.2)
    fig.savefig(out/'diagnostics.png',dpi=145);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>本轮诊断</title><h1>稳定化投影、启动边界层与局部参照</h1><p>原全窗口时间参照仍limited。小矩阵筛选没有授予新实际耦合资格。所有误差采用各自协议的口径。</p><img style="max-width:100%" src="diagnostics.png"><p><a href="../../implementation-report.md">实施记录</a></p></html>')
    (run/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-lite 稳定化与边界研究交付</title><style>body{{max-width:1200px;margin:3rem auto;font:18px/1.7 sans-serif;padding:0 1rem}}img{{width:100%}}</style><h1>MPM-lite 稳定化、边界层与短窗参照</h1><p>研究版本：{run.name}</p><p>局部16步参照通过；稳定化公式无错误证据。新边界网格未通过启动层筛选。新空间状态：{spatial['status']}；性能候选状态：{perf['status']}。</p><p><a href="visualization/{name}/index.html">日常固体动画</a> · <a href="visualization/diagnostics/index.html">本轮诊断</a> · <a href="implementation-report.md">实施记录</a> · <a href="capability-matrix.json">适用范围</a></p><p>日常动画继承 {owner.name}：原144函数、完整质量、252步、12帧及原q5回退范围。raw_event_status=limited（继承）；engineering_output_status=passed_scoped（继承正式固体）。本轮只认证局部前缀，原全窗口reference_limited；没有生产C/E集成或耦合q5资格。</p><img src="visualization/diagnostics/diagnostics.png"></html>''')
    write(run/'S6/visualization-origin.json',dict(status='passed_scoped',source_release=str(owner),source_release_sha256=sha(owner/'release.json'),case=name,chain=chain,identity_sha256=sha(folder/'identity.json'),new_visualization_of_inherited_data=True,new_integration_steps=0,raw_event_status='inherited_limited',engineering_output_status='inherited_formal_passed_scoped',old_full_window_time_passed=False))
    print('VISUALIZATION',run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):visualize(a.run)
