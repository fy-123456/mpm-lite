"""Render committed fields and bounded research diagnostics; never integrate."""
from pathlib import Path
import argparse,json
import numpy as np
from .provenance import read,write,sha
from benchmarks.research_basis_allocation_next.visualize import render
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def bundle(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed results are immutable')
    case=read(run/'S6/final-protocol.json')['default_case'];render(run,case);out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    fixed=read(run/'S1/full-window-check.json')['records']
    for label,rec in fixed.items():
        rows=rec['rows'];t=np.array([x['time_s'] for x in rows])*1e6;axs[0,0].plot(t,[x['min_pressure_Pa'] for x in rows],'.-',label=label)
        obs=[x for x in rows if x['physical_observation']];axs[0,1].plot([x['time_s']*1e6 for x in obs],[x['time_budget_fraction'] for x in obs],'.-',label=label)
    axs[0,1].axhline(1,color='k',ls='--',lw=.8)
    for label in ('coarse','fine'):
        folder=run/'cases'/f'coupled-{label}';h=GenerationStore(folder,read(folder/'identity.json')).history();rows=h[-1]['rows'];t=np.array([x['time'] for x in rows])*1e9
        axs[0,2].plot(t,[min(x['pressure_Pa']) for x in rows],'.-',label=label)
        axs[1,0].plot(t,[abs(x['energy_balance_J']) for x in rows],'.-',label=label)
    with np.load(h[-1]['folder']/'frame.npz') as f:
        for key in ('PK1','PK1_total'):axs[1,1].plot(f['X'][:,3,3,0],f[key][:,3,3,0,0],'.-',label=key)
    cand=read(run/'S3/candidate-time-check.json')
    for label,rows in [('candidate h vs h/2',cand.get('self_comparison',[])),('candidate vs formal h/2',cand.get('records',[{}])[-1].get('comparisons',[]))]:
        if rows:axs[1,2].plot([x['time_s'] for x in rows],[max(v['velocity']['absolute']/v['velocity']['budget'] for v in x['fields'].values()) for x in rows],label=label)
    axs[1,2].axhline(1,color='k',ls='--',lw=.8)
    if not cand.get('self_comparison'):
        axs[1,2].text(.5,.5,'Incomplete candidate window\n62 accepted steps; memory guard\nNo time-refinement qualification',ha='center',va='center',transform=axs[1,2].transAxes,fontsize=9)
    titles=['Full repaired pressure window (Pa)','Time error / allocated budget','Actual coupled minimum pressure (Pa)','Coupled absolute energy balance (J)','Fine-grid skeleton / total PK1 P11 (Pa)','Candidate velocity difference / budget']
    for i,(ax,title) in enumerate(zip(axs.ravel(),titles)):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.set_xlabel('time (microseconds)' if i<2 else 'time (nanoseconds)' if i in (2,3) else 'reference x (m)' if i==4 else 'time (s)')
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        if i<2:ax.set_xscale('log')
    fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    scope={k:read(run/p) for k,p in [('pressure','S1/pressure-scope-decision.json'),('coupling','S2/coupling-scope-decision.json'),('candidate','S3/research-space-decision.json'),('time','S4/time-decision.json')]};scope['time'].pop('times',None)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MPM-lite 完整压力窗口与实际耦合</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;line-height:1.7}img{width:100%}pre{white-space:pre-wrap}</style><h1>已提交的研究结果</h1><p>完整47步压力窗口通过；16/32单元实际耦合各19步。机械响应低于工程观测尺度，仅认证短窗稳定性、守恒与事务。应力图区分骨架PK1与包含孔压的总PK1。候选空间与相位的限制见下方原始范围记录。</p><img src="diagnostics.png"><pre>'+json.dumps(scope,ensure_ascii=False,indent=2)+'</pre>')
    write(out/'metadata.json',dict(no_integration=True,full_pressure_sha256=sha(run/'S1/full-window-check.json'),coupling_sha256=sha(run/'S2/coupled-scene-check.json'),candidate_sha256=sha(run/'S3/candidate-time-check.json')))
    (run/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>MPM-lite 完整压力窗口与实际耦合</title><style>body{{font:18px system-ui;max-width:1050px;margin:40px auto;line-height:1.8}}</style><h1>MPM-lite 本轮交付</h1><p>正式场景 {case}：144函数及原载体、完整M7、252步至1.6秒，12个显示帧。</p><ul><li><a href="visualization/{case}/index.html">交互固体场景：形变、应力、反力与能量</a></li><li><a href="visualization/{case}/cycle.gif">12帧动画</a></li><li><a href="visualization/diagnostics/index.html">完整压力窗口、实际耦合与候选诊断</a></li><li><a href="implementation-report.md">实施报告与限制</a></li></ul><p>显示形变默认放大10倍，不改变物理数据。短窗稳定不等于秒级耦合精度；未替换正式空间、未启用耦合q5或生产C/E接入。</p>''')
    print(run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();bundle(a.run)
