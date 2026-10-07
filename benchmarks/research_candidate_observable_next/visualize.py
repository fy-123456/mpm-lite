"""Redraw inherited production frames and independently labelled new diagnostics."""
from pathlib import Path
import argparse,json,html
import numpy as np
from .provenance import *
from benchmarks.research_basis_allocation_next.visualize import render


def visualize(run):
    run=Path(run);verify(run)
    if (run/'release.json').exists():raise ValueError('sealed visualization is immutable')
    default=read(run/'S6/default-scene-decision.json')['default_case'];render(APP,default,output=run/'visualization'/default)
    inherited_page=run/'visualization'/default/'index.html'
    inherited_page.write_text(inherited_page.read_text().replace('<h1>','<p style="padding:12px;background:#edf4ff;color:#18395b">本页252步和12帧继承自父版 '+APP.name+'；本轮只重绘已有数据，没有重跑或扩大q5权限。</p><h1>',1))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    time=read(run/'S1/time-vs-space-comparison.json')['records']
    for label,record in time.items():
        x=[r['time_s'] for r in record['records']];y=[max(v['velocity']['absolute']/v['velocity']['budget'] for v in r['fields'].values()) for r in record['records']];ax[0,0].plot(x,y,label=label)
    ax[0,0].axhline(1,color='k',ls='--');ax[0,0].set(xlabel='time (s)',ylabel='velocity error / practical budget',title='Candidate: stable, reference accuracy still limited');ax[0,0].legend()
    force=read(run/'S2/physical-force-acceleration-decomposition.json')['physical_rms_difference_m_s2'];keys=['material','stabilization','boundary_cross_mass'];ax[0,1].bar(range(3),[force[k] for k in keys]);ax[0,1].set(xticks=range(3),xticklabels=['material','stabilization','boundary mass'],ylabel='physical acceleration RMS difference (m/s2)',title='Same physical probes; no damping added')
    p=run/'S3/coarse-physical-check.json'
    if p.exists():
        v=read(p)['frames'];ax[1,0].plot([x['time_s']*1e6 for x in v],[x['v_rms_m_s'] for x in v],'.-',label='actual coupled velocity');ax[1,0].axhline(3e-4,color='k',ls='--',label='registered observation threshold');ax[1,0].legend()
    else:ax[1,0].text(.1,.5,'Coupled reference incomplete; see scope report',transform=ax[1,0].transAxes)
    ax[1,0].set(xlabel='time (microseconds)',ylabel='velocity RMS (m/s)',title='Research fixture; full material and mass')
    decision=read(run/'S4/time-decision.json');local=read(run/'S4/local-phase-check.json');observed=[r for r in local.get('modal_events',[]) if r['status']!='unobserved'];text_rows=[f"Formal schedule: 252 steps (inherited)",f"New quarter window: {decision.get('new_short_steps',0)} steps",f"Fields passed: {local.get('fields_passed')}",f"Observed events passed: {local.get('observed_events_passed')}","No global time/space accuracy certificate"]
    ax[1,1].axis('off');ax[1,1].text(.02,.85,'\n'.join(text_rows),va='top',fontsize=12)
    for a in ax.flat:
        if a is not ax[1,1]:a.grid(alpha=.2)
    fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>本轮研究诊断</title><h1>候选、耦合与相位诊断</h1><p>研究图不代表正式空间或完整耦合周期已获认证。候选空间仍未推广。</p><img style="max-width:100%" src="diagnostics.png"><p><a href="../../implementation-report.md">完整实施记录</a></p>')
    (run/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-lite 本轮交付</title><style>body{{max-width:1000px;margin:3rem auto;font:18px/1.7 sans-serif;padding:0 1rem}}img{{width:100%}}a{{color:#1268a3}}</style><h1>MPM-lite 候选参照与可观测耦合</h1><p>本轮研究版：{run.name}</p><p>日常固体动画明确继承自父版 {APP.name}：144 个函数、完整质量、252 步、原 q5 回退权限。此次没有重新计算或重新认证完整周期。</p><p><a href="visualization/{default}/index.html">打开日常固体交互动画（继承父版）</a> · <a href="visualization/diagnostics/index.html">本轮研究诊断</a> · <a href="implementation-report.md">实施记录</a></p><img src="visualization/diagnostics/diagnostics.png"><p>显示变形倍率沿用 10 倍；研究候选未替换正式空间。耦合仅为已验收短窗，不等于生产物理耦合。</p></html>''')
    write(run/'S6/visualization-origin.json',dict(status='passed_scoped',source_release=str(APP),source_release_sha256=APP_SHA,case=default,source_identity_sha256=sha(APP/'cases'/default/'identity.json'),new_visualization_of_inherited_data=True,new_integration_steps=0))
    print('VISUALIZATION',run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):visualize(a.run)
