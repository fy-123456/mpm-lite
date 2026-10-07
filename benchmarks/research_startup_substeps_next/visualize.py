"""Show raw errors alongside engineering results and authentic physical fields."""
import argparse,shutil
import numpy as np
from .provenance import *
from .review import fluid,context
from engine.aniso_phase1.research_startup_substeps_next.schedule import OBSERVATIONS,aggregate

def visualize(run):
    run=Path(run);mutable(run);shutil.copytree(APP/'visualization/daily-q5-retry',run/'visualization/daily-q5-retry',dirs_exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(2,3,figsize=(18,10),layout='constrained');p=read(run/'S0/physical-contract.json');ctx=context(p['cuts'],p['parameters']['storage']);groups=ctx['groups']
    for candidate in ('U2','U4'):
        r=read(run/'S2'/f'candidate-{candidate}.json');v=r['trials'][0]
        ax[0,0].plot([x['time_s']*1e6 for x in v['engineering']['records']],[x['budget_ratios']['face_flux'] for x in v['engineering']['records']],'-o',ms=3,label=candidate+' fixed observation')
    v=read(run/'S2/candidate-U4.json')['trials'][0]
    ax[0,0].plot([x['time_s']*1e6 for x in v['raw']['records']],[x['budget_ratios']['face_flux'] for x in v['raw']['records']],':',label='U4 raw substep')
    ax[0,0].axhline(1,color='black',ls='--');ax[0,0].set(title='Fixed skeleton: original substep error retained',xlabel='time (us)',ylabel='face-flow error / engineering budget')
    with np.load(run/'S2/U4-h.npz') as v:
        signed=(groups@v['flux'].T).T;exact=(groups@v['exact_flux'].T).T;eng=(groups@v['engineering_flux'].T).T
        ax[0,1].stairs(signed[:,0]*1e12,v['times']*1e6,baseline=None,label='raw U4 left outflow');ax[0,1].stairs(exact[:,0]*1e12,v['times']*1e6,baseline=None,label='same-grid exact raw interval',ls='--');ax[0,1].stairs(eng[:,0]*1e12,OBSERVATIONS*1e6,baseline=None,label='fixed engineering interval',lw=2)
    ax[0,1].axvline(25,color='gray',ls=':');ax[0,1].set(title='Startup and theta switch: no flux interpolation',xlabel='time (us)',ylabel='left outflow (nL/s)',xlim=(0,55))
    for label in ('h','half'):
        folder=run/'cases'/('boundary32-'+label);raw,t,rows=fluid(folder);eng=aggregate(raw,t,OBSERVATIONS)
        mean=eng['pressure']@ctx['top'].V0/ctx['top'].V0.sum();ax[0,2].plot(OBSERVATIONS*1e6,mean,label=label+' volume-mean pressure')
        ax[1,0].plot(OBSERVATIONS*1e6,np.sum((groups@eng['cumulative'].T).T,axis=1)*1e12,label=label+' net cumulative boundary volume')
        if label=='h':
            tm=t[1:]*1e6;ax[1,1].plot(tm,[r['total_energy_J'] for r in rows],label='total stored E');ax[1,1].plot(tm,np.cumsum([r['darcy_dissipation_J'] for r in rows]),label='cumulative Darcy');ax[1,1].plot(tm,np.cumsum([r['numerical_dissipation_J'] for r in rows]),label='cumulative numerical');ax[1,1].plot(tm,np.cumsum([r['energy_balance_J'] for r in rows]),label='cumulative balance defect')
            for i,side in enumerate(('x-','x+','y-','y+','z-','z+')):ax[1,2].stairs((groups@eng['flux'].T)[i]*1e12,OBSERVATIONS*1e6,baseline=None,label=side)
    ax[0,2].set(title='Actual coupled boundary32: engineering nodes',xlabel='time (us)',ylabel='cell-volume mean pressure (Pa)');ax[1,0].set(title='Actual cumulative boundary exchange',xlabel='time (us)',ylabel='net outward volume (nL)');ax[1,1].set(title='Actual full ledger, including BE numerical loss',xlabel='time (us)',ylabel='energy (J)');ax[1,1].set_yscale('symlog',linthresh=1e-15);ax[1,2].set(title='Actual signed engineering flow: all six sides',xlabel='time (us)',ylabel='outward flow (nL/s)')
    for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=7)
    fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    # These are actual saved endpoint probes, not a manufactured deformation.
    with np.load(run/'cases/boundary32-h/probes.npz') as z:probes={k:z[k] for k in z.files}
    X=probes['X'];x=probes['x'][-1];sigma=probes['Cauchy_total'][-1];middle=(slice(None),X.shape[1]//2,X.shape[2]//2)
    fig=plt.figure(figsize=(15,6),layout='constrained');a=fig.add_subplot(1,2,1,projection='3d');points=x.reshape(-1,3);im=a.scatter(*points.T,c=sigma[...,0,0].ravel(),s=6,cmap='viridis');fig.colorbar(im,ax=a,label='total Cauchy xx (Pa)',shrink=.6,pad=.16);a.set(title='Actual boundary32 at 200 us; deformation scale 1',xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');a.set_box_aspect(np.ptp(points,axis=0))
    b=fig.add_subplot(1,2,2)
    for label,ls in [('h','-'),('half','--')]:
        with np.load(run/'cases'/('boundary32-'+label)/'probes.npz') as z:disp=z['x'][-1]-z['X'];b.plot(X[middle][...,0],disp[middle][...,0]*1e9,ls,label=label+' axial displacement')
    b.set(title='Actual axial displacement at centre line',xlabel='reference x (m)',ylabel='displacement (nm)');b.legend();b.grid(alpha=.2);fig.savefig(out/'actual-field.png',dpi=140);plt.close(fig)
    scopes=[s for s in ('legacy','boundary32') if (run/'S4'/s/'paired-performance.json').exists()];fig,axes=plt.subplots(1,len(scopes),figsize=(7*len(scopes),5),squeeze=False,layout='constrained')
    for a,scope in zip(axes.flat,scopes):
        perf=read(run/'S4'/scope/'paired-performance.json');t=read(run/'S4'/scope/'integration-protocol.json')['source_states'];ix=np.arange(2)
        for offset,k in [(-.18,'A'),(.18,'B')]:a.bar(ix+offset,[r[k]['advance_s'] for r in perf['records']],.36,label=k)
        a.set(title=scope+': '+perf['status']+f"; median gain {perf['median_gain']:.1%}",ylabel='synchronized actual step time (s)',xticks=ix,xticklabels=[f"from {x['time_s']*1e6:g} us" for x in t]);a.legend()
    fig.savefig(out/'performance.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>本轮诊断</title><h1>原始子步、工程区间、实际场与性能</h1><p>U4仅认证固定工程区间；原始子步误差仍保留。实际图来自新32单元200微秒终态，无位移放大。A/B是两个输入的短步测量，不是全周期加速。</p>'+''.join(f'<img style="max-width:100%" src="{n}.png">' for n in ('diagnostics','actual-field','performance'))+'<p><a href="../../implementation-report.md">实施记录</a></p></html>')
    (run/'index.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-lite 启动子步与共享几何</title><style>body{{max-width:1400px;margin:2rem auto;font:18px/1.7 sans-serif;padding:0 1rem}}img{{max-width:100%}}</style><h1>启动子步与共享几何交付</h1><p>版本：{run.name}</p><p>新增32单元实际耦合28/56步短窗到200微秒；按守恒工程区间验收。每个原始子步和能量账本保留，原始流量精度仍有限。共享几何按原/新网格分别验收，完整周期与三维空间精度尚未认证。</p><p><a href="visualization/diagnostics/index.html">本轮真实结果</a> · <a href="implementation-report.md">实施记录</a> · <a href="capability-matrix.json">资格与限制</a> · <a href="visualization/daily-q5-retry/index.html">继承的日常固体动画</a></p><p>日常动画来自pressure-window发布，252步/1.6秒/12帧，位移放大10倍；本轮仅零步加载。它不是新增耦合轨迹。</p><img src="visualization/diagnostics/diagnostics.png"><img src="visualization/diagnostics/actual-field.png"><img src="visualization/diagnostics/performance.png"></html>')
    write(run/'S6/visualization-origin.json',dict(status='actual_coupled_and_CPU_and_AB_diagnostics_plus_inherited_daily',new_coupled_steps=[28,56],coupled_time_s=.0002,physical_fields_from_saved_probes=True,physical_deformation_scale=1,daily_source=str(SOLID),daily_steps=252,daily_frames=12,daily_deformation_scale=10,new_daily_steps=0))
    print('VISUALIZATION',run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):visualize(a.run)
