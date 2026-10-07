"""Display reference scope, original coupled signals and measured implementation cost."""
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .provenance import *
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_transverse_next.observables import modes
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology


def main(run):
    run=Path(run);mutable(run);out=run/'figures';out.mkdir(exist_ok=True)
    p=read(run/'S0/physical-contract.json');top=ReferenceTopology(p['cuts']);h=history(APP/'cases/YZ128');ref=np.load(run/'S1/reference-YZ128.npz');num=np.load(run/'S1/YZ128-h.npz');half=np.load(run/'S1/YZ128-half.npz')
    actual={k:np.array([modes(top,r['state'].child_states['fluid']['pressure_Pa'])[k]['value_Pa'] for r in h]) for k in ('Ay','Az')};ts=np.array([r['state'].time for r in h])
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,k in enumerate(('Ay','Az')):
        ax=axes[0,i]
        ax.plot(ref['times']*1e6,ref[k],label='fixed skeleton: spectral reference');ax.plot(num['times']*1e6,num[k],'--',label='fixed skeleton: original h');ax.plot(ts*1e6,actual[k],label='moving solid: inherited YZ128')
        ax.set(xlabel='Time (microseconds)',ylabel=f'{k} (Pa)',title='Different physical models, not total error');ax.legend(fontsize=8);ax.grid(alpha=.25)
        ax=axes[1,i]
        for z,label in ((ref,'spectral'),(num,'h'),(half,'h/2')):ax.plot(z['times']*1e6,(z[k]-z[k][0])*1e6,label=label)
        ax.set(xlabel='Time (microseconds)',ylabel=f'Change in {k} (micro-Pa)',title='Fixed-skeleton increments (small signal)');ax.legend();ax.grid(alpha=.25)
    fig.savefig(out/'pressure-modes.png',dpi=140);plt.close(fig)
    d=read(run/'S1/coupled-model-difference.json');fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,k in zip(axes,('Ay','Az')):
        names=['pressure_increment_Pa','volume_contribution_Pa','flow_contribution_Pa'];vals=[d['total_modes'][k][x]*1e6 for x in names];ax.bar(['total','solid volume','Darcy flow'],vals,color=['#176a8a','#d47d25','#4d9c68']);ax.set(ylabel='Mode change (micro-Pa)',title=f'{k}: inherited moving-solid mass ledger, 0-75 us');ax.grid(axis='y',alpha=.25)
    fig.savefig(out/'volume-attribution.png',dpi=140);plt.close(fig)
    fpath=h[-1]['folder']/'frame.npz'
    with np.load(fpath) as f:
        X=f['X'];u=f['x']-X;pp=f['pressure_Pa'];pflat=pp.reshape(-1);coords=X.reshape(-1,3);u=u.reshape(-1,3);stress=f['Cauchy_total'].reshape(-1,3,3)
        fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
        plane=np.isclose(coords[:,2],.5)
        sc=axes[0].scatter(coords[plane,0],coords[plane,1],c=pflat[plane],s=10);fig.colorbar(sc,ax=axes[0],label='Pressure (Pa)');axes[0].set(xlabel='X (m)',ylabel='Y (m)',title='Inherited 75 us probes, Z=0.5 m')
        centre=(np.isclose(coords[:,1],.5)&np.isclose(coords[:,2],.5));order=np.argsort(coords[centre,0])
        for j,name in enumerate('xyz'):axes[1].plot(coords[centre,0][order],u[centre,j][order]*1e9,label=name)
        axes[1].set(xlabel='X (m)',ylabel='Displacement (nm)',title='Raw centerline; no smoothing');axes[1].legend()
        sc=axes[2].scatter(coords[plane,0],coords[plane,1],c=stress[plane,1,1],s=10);fig.colorbar(sc,ax=axes[2],label='Total Cauchy yy (Pa)');axes[2].set(xlabel='X (m)',ylabel='Y (m)',title='Inherited 75 us stress, Z=0.5 m')
        fig.savefig(out/'inherited-physical-fields.png',dpi=140);plt.close(fig)
    groups=np.array([top.boundary_sign*(top.axes==axis)*(top.boundary_sign==sgn) for axis in range(3) for sgn in (-1,1)])
    labels=['x-','x+','y-','y+','z-','z+'];rows=h[-1]['rows'];dt=np.array([r['dt'] for r in rows]);z=np.array([r['flux_interval_m3_s'] for r in rows]);side=z@groups.T
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,label in enumerate(labels):
        axes[0,0].plot(ref['times']*1e6,(ref['cumulative']@groups.T)[:,i]*1e9,label=label)
        axes[0,1].plot(ts[1:]*1e6,side[:,i]*1e9,label=label)
    axes[0,0].set(title='Fixed skeleton: integrated boundary exchange',xlabel='Time (microseconds)',ylabel='Cumulative outward volume (1e-9 m3)')
    axes[0,1].set(title='Inherited moving solid: interval mean flow',xlabel='Interval end (microseconds)',ylabel='Outward flow (1e-9 m3/s)')
    E0=read(APP/'cases/YZ128/execution-protocol.json')['initial_energy_J'];energy=np.array([r['total_energy_J'] for r in rows])-E0
    axes[1,0].plot(ts[1:]*1e6,energy*1e12,label='E-E0')
    for key,label in [('darcy_dissipation_J','Darcy dissipation'),('numerical_dissipation_J','numerical dissipation'),('reservoir_work_J','reservoir work')]:axes[1,0].plot(ts[1:]*1e6,np.cumsum([r[key] for r in rows])*1e12,label=label)
    axes[1,0].set(title='Inherited moving-solid energy accounts',xlabel='Time (microseconds)',ylabel='Energy (pJ)')
    closure=energy+np.cumsum([r['darcy_dissipation_J']+r['numerical_dissipation_J']-r['reservoir_work_J']-r['source_work_J']-r['external_work_J'] for r in rows]);axes[1,1].plot(ts[1:]*1e6,closure,label='cumulative energy defect')
    axes[1,1].set(title='Inherited energy closure; no new trajectory',xlabel='Time (microseconds)',ylabel='Energy defect (J)')
    for ax in axes.ravel():ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.savefig(out/'flow-energy.png',dpi=140);plt.close(fig)
    perf=read(run/'S3/paired-performance.json');fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True);x=np.arange(2)
    ax.bar(x-.18,[r['DV'] for r in perf['records']],.36,label='DV');ax.bar(x+.18,[r['BD'] for r in perf['records']],.36,label='batched download candidate');ax.set(xticks=x,xticklabels=['state 8','state 16'],ylabel='Whole step incl. probes and IO (s)',title='Two paired inputs; no full-cycle speed claim');ax.legend();fig.savefig(out/'performance.png',dpi=140);plt.close(fig)
    rr=read(run/'S1/fixed-skeleton-time-review.json');r=next(x for x in rr['records'] if x['case']=='YZ128' and x['schedule']=='h');backend=read(run/'S3/backend-decision.json')
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>MPM-lite 横向压力参考与 DV 优化</title><style>body{font:17px/1.7 sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#172c38;background:#f7fafc}img{width:100%;background:white;border:1px solid #ccd9df;border-radius:8px}section{margin:28px 0}code{overflow-wrap:anywhere}.note{background:#fff0d9;padding:16px;border-radius:8px}</style><h1>横向压力参考与 DV 成本优化</h1>'''
    html+=f'<p>直接父发布：<code>{APP_SHA}</code>。研究后端：<b>{backend["backend"]}</b>。</p>'
    html+='<p class="note">本轮没有扩展耦合时间窗，没有新增物理帧。固定骨架参考用于同网格压力时间分析；运动固体曲线与 75 微秒场图来自已认证父发布，二者差异不代表总数值误差。正式空间、M7/q7 和默认纯固体场景保持。</p>'
    html+=f'<p>工程观察时间指标通过；横向参考增量仍低于信号下限。原始区间流量比较的整体范数相对差约 {r["raw"]["checks"]["face_flux"]["relative"]:.2%}，不能宣称启动峰值精度通过。两输入候选耗时降幅中位数 {perf["median_gain_fraction"]:.2%}；具体采用与否以能力清单为准。</p>'
    for title,name,caption in [('压力模式与时间参考','pressure-modes.png','上排明确区分固定骨架与运动固体；下排为固定骨架小信号增量。'),('压力变化的来源','volume-attribution.png','按已保存质量方程分解：固体体积交换是主要贡献；这是账本解释，不是连续真解。'),('继承场图','inherited-physical-fields.png','全部来自父发布 YZ128 第18步/75微秒；位移以纳米为单位，未平滑，未将其当作新物理帧。'),('实际单步成本','performance.png','相同探针和IO、空q缓存、独立进程；构造成本另记。')]:
        html+=f'<section><h2>{title}</h2><p>{caption}</p><img src="figures/{name}" alt="{title}"></section>'
    html+='<section><h2>流量与能量分账</h2><p>固定骨架六侧累计交换与继承耦合区间均值分别展示。能量图明确分开物理Darcy耗散和数值耗散；本工况源功与外功为零。场图取 Z=0.5 m 平面，压力间断面沿用原右单元归属。</p><img src="figures/flow-energy.png" alt="流量与能量分账"></section>'
    html+='<p><a href="S1/reference-self-check.json">参考自检</a> · <a href="S1/coupled-model-difference.json">体积交换分解</a> · <a href="S3/paired-performance.json">配对性能</a> · <a href="capability-matrix.json">能力范围（封存后）</a></p></html>'
    (run/'index.html').write_text(html)
    write(run/'S5/visualization-origin.json',dict(new_physical_frames=0,inherited_frame=dict(path=str(fpath),sha256=sha(fpath)),reference_paths=['S1/reference-YZ128.npz','S1/YZ128-h.npz','S1/YZ128-half.npz'],actual_curve_source=str(APP/'cases/YZ128'),displacement_unit='nm, unmodified raw values',smoothing=False,browser_interaction=False))
    write(run/'S5/visual-assets-check.json',dict(status='passed_scoped',images=[dict(path=str(p.relative_to(run)),sha256=sha(p)) for p in sorted(out.glob('*.png'))],script='no client script',new_dynamic_steps=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
