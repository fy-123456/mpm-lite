"""Actual saved continuous/extension fields and separately labelled CPU references."""
import argparse,shutil
import numpy as np
from .provenance import *
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_startup_next.coupling_review import context

def main(run):
    run=Path(run);mutable(run);shutil.copytree(APP/'visualization/daily-q5-retry',run/'visualization/daily-q5-retry',dirs_exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=run/'visualization/diagnostics';out.mkdir(parents=True,exist_ok=True);backend=read(run/'S4/extension-protocol.json')['backend'];p=read(run/'S0/coupled-protocol.json');ctx=context(p['cuts'],p['parameters']['storage']);top=ctx['top'];fig,ax=plt.subplots(2,3,figsize=(18,10),layout='constrained')
    for label,path in [('A inherited',APP/'cases/boundary32-h'),('B new',run/'cases/continuous-B')]:
        h=history(path);rr=h[-1]['rows'];rr=[r for r in rr if 1.25e-5<r['time']<=7.5e-5+1e-18];t=np.array([r['time'] for r in rr])*1e6
        ax[0,0].plot(t,[r['true_scaled_residual'] for r in rr],'-o',ms=2,label=label);ax[0,1].plot(t,[top.V0@np.asarray(r['pressure_Pa'])/top.V0.sum() for r in rr],label=label)
    for a in ax[0,:2]:
        for x in (25,50):a.axvline(x,c='gray',ls=':',lw=1)
    ax[0,0].set(title='Raw true residual; switch 25us / dt change 50us',xlabel='time (us)',ylabel='fraction of frozen residual budget');ax[0,1].set(title='Continuous B: actual volume-mean pressure',xlabel='time (us)',ylabel='Pa')
    for label in ('h','half'):
        h=history(run/'cases'/('extension-'+label));rr=h[-1]['rows'];t=np.array([r['time'] for r in rr])*1e6
        ax[0,2].plot(t,[top.V0@np.array(r['pressure_Pa'])/top.V0.sum() for r in rr],'-o',ms=3,label=label)
        ax[1,0].plot(t,np.cumsum([r['energy_balance_J'] for r in rr]),label=label+' incremental defect')
        if label=='h':
            f=h[0]['state'].child_states['fluid'];E0=f['last_ledger']['total_energy_J'];ax[1,1].plot(t,[r['total_energy_J']-E0 for r in rr],label='delta stored E');ax[1,1].plot(t,np.cumsum([r['darcy_dissipation_J'] for r in rr]),label='new Darcy');ax[1,1].plot(t,np.cumsum([r['numerical_dissipation_J'] for r in rr]),label='new Dnum (old retained)')
            for i,side in enumerate(('x-','x+','y-','y+','z-','z+')):ax[1,2].stairs([ctx['groups'][i]@np.array(r['flux_interval_m3_s'])*1e12 for r in rr],np.r_[200.,t],baseline=None,label=side)
    hh=history(run/'cases/extension-h');rr=hh[-1]['rows'];initial=hh[0]['state'].child_states['fluid']['last_ledger'];tm=np.array([r['time'] for r in rr])*1e6
    for key,label in [('fluid_storage_J','delta fluid storage'),('kinetic_J','delta kinetic')]:ax[1,1].plot(tm,[r[key]-initial[key] for r in rr],label=label)
    ax[1,1].plot(tm,[r['solid_material_J']+r['stabilization_J']-initial['solid_material_J']-initial['stabilization_J'] for r in rr],label='delta elastic + stabilization');ax[1,1].set_yscale('symlog',linthresh=1e-14)
    ax[0,2].set(title='Same 200us source, new 8 /16 steps',xlabel='time (us)',ylabel='volume-mean pressure (Pa)');ax[1,0].set(title='Extension incremental energy closure',xlabel='time (us)',ylabel='J');ax[1,1].set(title='Extension energy accounts',xlabel='time (us)',ylabel='J');ax[1,2].set(title='Extension all six signed outflows',xlabel='time (us)',ylabel='nL/s')
    for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=7)
    fig.savefig(out/'diagnostics.png',dpi=140);plt.close(fig)
    with np.load(run/'cases/extension-h/probes.npz') as z:X=z['X'];x=z['x'][-1];sigma=z['Cauchy_total'][-1]
    fig=plt.figure(figsize=(15,6),layout='constrained');a=fig.add_subplot(1,2,1,projection='3d');points=x.reshape(-1,3);im=a.scatter(*points.T,c=sigma[...,0,0].ravel(),s=6,cmap='viridis');fig.colorbar(im,ax=a,label='total Cauchy xx (Pa)',shrink=.6,pad=.16);a.set(title='Actual '+backend+' at 300 us; deformation scale 1',xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');a.set_box_aspect(np.ptp(points,axis=0));b=fig.add_subplot(1,2,2);mid=(slice(None),X.shape[1]//2,X.shape[2]//2)
    for label,style in [('h','-'),('half','--')]:
        with np.load(run/'cases'/('extension-'+label)/'probes.npz') as z:b.plot(X[mid][...,0],(z['x'][-1]-z['X'])[mid][...,0]*1e9,style,label=label)
    b.set(title='Actual centre-line displacement',xlabel='reference x (m)',ylabel='axial displacement (nm)');b.grid(alpha=.2);b.legend();fig.savefig(out/'actual-field.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for key in ('base','y','z','yz'):
        with np.load(run/'S3'/('fixed-'+key+'.npz')) as z:
            axes[0].plot(z['times']*1e6,z['pressure']@z['V0']/z['V0'].sum(),label=key);axes[1].plot(z['times']*1e6,z['cumulative']@np.sum(z['B'],axis=0)*1e12,label=key)
    axes[0].set(title='Fixed skeleton only: transverse refinement',xlabel='time (us)',ylabel='mean pressure (Pa)');axes[1].set(title='Fixed skeleton only: net boundary exchange',xlabel='time (us)',ylabel='cumulative volume (nL)')
    for a in axes:a.legend();a.grid(alpha=.2)
    fig.savefig(out/'reference.png',dpi=140);plt.close(fig)
    perf=read(run/'S2/paired-performance.json');fig,a=plt.subplots(figsize=(9,5),layout='constrained');ix=np.arange(2)
    for off,key in [(-.18,'B'),(.18,'G1')]:a.bar(ix+off,[r[key]['advance_s'] for r in perf['records']],.36,label=key)
    a.set(title=f"G1 batch-four: selected={perf['selected']}; median gain={perf['median_gain']:.1%}",ylabel='actual synchronized step (s)',xticks=ix,xticklabels=['from25us','from50us']);a.legend();fig.savefig(out/'performance.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>真实短窗结果</title><h1>连续共享几何与300微秒续算</h1><p>连续B逐步对照继承A；延长两档从同一200微秒状态开始。CPU图仅为固定骨架横向参考；冻结真实位移算子也不代表细网格动态收敛。批处理候选性能单列，未达到门槛的候选不推荐。</p>'+''.join('<img style="max-width:100%" src="'+n+'.png">' for n in ('diagnostics','actual-field','reference','performance'))+'</html>')
    (run/'index.html').write_text(f'<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-lite 连续几何与空间参考</title><style>body{{max-width:1400px;margin:2rem auto;font:18px/1.7 sans-serif;padding:0 1rem}}img{{max-width:100%}}</style><h1>连续共享几何、横向参考与短窗延长</h1><p>版本：{run.name}</p><p>新增B连续12.5→75微秒，并从同一200微秒初态运行8/16步至300微秒。正式空间、q7材料和M7质量不变。原始启动子步误差、实际空间收敛及完整耦合周期仍未认证。</p><p><a href="visualization/diagnostics/index.html">本轮真实诊断</a> · <a href="implementation-report.md">实施记录</a> · <a href="capability-matrix.json">资格与限制</a> · <a href="visualization/daily-q5-retry/index.html">继承日常固体动画</a></p><p>日常动画继承252步/1.6秒/12帧，位移放大10倍；本轮仅零步加载。研究物理场图位移倍率为1。</p>'+''.join('<img src="visualization/diagnostics/'+n+'.png">' for n in ('diagnostics','actual-field','reference','performance'))+'</html>')
    write(run/'S6/visualization-origin.json',dict(status='actual_saved_fields_and_separate_CPU_reference',new_continuous_B_steps=14,new_extension_steps=[8,16],new_physical_frames=len(list(run.rglob('frame.npz'))),physical_deformation_scale=1,physical_fields_from_saved_probes=True,daily_source=str(SOLID),daily_steps=252,daily_frames=12,daily_deformation_scale=10,new_daily_steps=0));print('VISUAL',run/'index.html',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
