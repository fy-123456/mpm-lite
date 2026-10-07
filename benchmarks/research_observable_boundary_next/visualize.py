"""Display only committed solid data and explicitly limited research diagnostics."""
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
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');pressure=read(run/'S2/exact-time-reference.json');times=np.array(pressure['times_s'])
    for label,r in pressure['records'].items():
        q=np.asarray(r['exact_boundary_by_axis']).sum(axis=1);axs[0,0].plot(times[1:]*1e6,q[1:],label=f'{label} exact ({r["cells"]} cells)')
        rows=r['time_rows'];axs[0,1].plot([v['time_s']*1e6 for v in rows],[v['metrics']['boundary']['absolute']/(.25*v['metrics']['boundary']['budget']) for v in rows],'.-',label=label)
        steps=r['internal_steps'];axs[0,2].plot([v['time_s']*1e6 for v in steps],[v['min_pressure_Pa'] for v in steps],'.-',label=f'{label} numerical min')
        axs[0,2].plot(times[1:]*1e6,np.min(np.asarray(r['exact_pressure'])[1:],axis=1),'--',label=f'{label} exact min')
    for label,r in read(run/'S2/corrected-start-check.json')['records'].items():
        axs[0,2].plot([v['time_s']*1e6 for v in r['rows']],[v['min_pressure_Pa'] for v in r['rows']],':',linewidth=2,label=f'{label} repaired prefix')
    rows=read(run/'S2/grid-comparison.json')['records'];axs[1,0].plot([x['time_s']*1e6 for x in rows],[x['boundary']['absolute']/x['boundary']['budget'] for x in rows],'.-',label='16 vs 32 exact-time grids')
    for r in read(run/'S1/time-comparison.json')['records']:
        f=r['comparison']['field_records'];axs[1,1].plot([x['time_s'] for x in f],[max(v['velocity']['absolute']/v['velocity']['budget'] for v in x['regions'].values()) for x in f],label=str(r['window']))
    for r in read(run/'S3/dynamic-check.json')['records']:
        vals=r['comparisons'];axs[1,2].plot([x['time_s'] for x in vals],[max(v['velocity']['absolute']/v['velocity']['budget'] for v in x['fields'].values()) for x in vals],'.-',label=r['case'])
    for ax in (axs[0,1],axs[1,0],axs[1,1],axs[1,2]):ax.axhline(1,color='k',linestyle='--',linewidth=.8)
    axs[0,2].axhline(0,color='k',linestyle=':',linewidth=.8)
    titles=['Boundary discharge: fixed observation times','Time error / allocated budget (25%)','Negative numerical pressure is not accepted','Spatial grid error / engineering budget','Formal-space short-window velocity error','Candidate-space velocity discrepancy']
    for i,(ax,title) in enumerate(zip(axs.ravel(),titles)):
        ax.set_title(title,fontsize=10);ax.grid(alpha=.2);ax.legend(fontsize=7);ax.set_xlabel('time (microseconds)' if i in (0,1,2,3) else 'time (seconds)')
        ax.set_ylabel('volume (m3)' if i==0 else 'pressure (Pa)' if i==2 else 'budget fraction')
        if i in (0,1,2,3):ax.set_xscale('log')
    fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    scope=dict(time=read(run/'S1/time-decision.json'),pressure=read(run/'S2/pressure-scope-decision.json'),candidate=read(run/'S3/research-space-decision.json'))
    scope['time'].pop('times',None)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>MPM-lite 可观测相位与早期排水</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;line-height:1.7}img{width:100%}pre{white-space:pre-wrap}</style><h1>研究诊断与未通过项目</h1><p>所有曲线来自保存数据。正式空间与252步时间表保留；高频事件未全部通过。压力空间比较通过，但原时间推进有负压振荡，修复后最早两个观测时刻通过；完整修复时窗未认证，所以没有执行新网格耦合。候选加载短窗的速度差超预算，仍不替换正式空间。</p><img src="diagnostics.png"><pre>'+json.dumps(scope,ensure_ascii=False,indent=2)+'</pre>')
    write(out/'metadata.json',dict(no_integration=True,pressure_sha256=sha(run/'S2/exact-time-reference.json'),phase_sha256=sha(run/'S1/time-comparison.json'),candidate_sha256=sha(run/'S3/dynamic-check.json')))

def bundle(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed visualization immutable')
    case=read(run/'S6/final-protocol.json')['default_case'];render(run,case);diagnostics(run);summary=read(run/'cases'/case/'summary.json')
    (run/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>MPM-lite 可观测相位与早期排水</title><style>body{{font:18px system-ui;max-width:1000px;margin:40px auto;line-height:1.75}}</style><h1>MPM-lite 本轮场景与研究诊断</h1><p>正式场景：{case}，{summary['steps']}个接受步至1.6 s，12个显示帧，完整M7；保留cross-direction-snapshot6的144函数空间及已有GPU优化。</p><ul><li><a href="visualization/{case}/index.html">交互场景：形变、PK1应力、反力和能量</a></li><li><a href="visualization/{case}/cycle.gif">12帧动画 GIF</a></li><li><a href="visualization/diagnostics/index.html">时间相位、早期排水和候选空间的限制</a></li><li><a href="implementation-report.md">完整实施报告</a></li></ul><p>显示形变默认放大10倍，曲线和验收均使用真实物理量。原压力时间推进未通过；起步修复已通过两个观测时刻的有限复验，完整修复时窗与实际耦合未验收；候选加载短窗未通过速度比较。全局空间/时间精度与生产耦合资格尚未认证。</p>''')
    print(run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();bundle(a.run)
