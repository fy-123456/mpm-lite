"""Six committed frames with explicit units and resolved transverse signals."""
import argparse,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .provenance import *
from .review import probes
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_startup_next.coupling_review import context

def main(run):
    run=Path(run);mutable(run);out=run/'visualization';out.mkdir(exist_ok=True);names=('Y64','Y128','YZ128')
    cuts=read(run/'S0/grid-protocol-inherited.json')['cuts'];storage=read(run/'S0/coupled-protocol.json')['parameters']['storage']
    hs={name:history(run/'cases'/name) for name in names};frames=[]
    contexts={name:context(cuts['y' if name=='Y64' else 'yz'],storage) for name in names}
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for i,name in enumerate(names):
        top=contexts[name]['top']
        for j,step in enumerate((8,18)):
            item=hs[name][step];path=item['folder']/'frame.npz'
            if not path.is_file():raise ValueError('missing committed physical frame')
            frames.append(dict(case=name,step=step,time_s=item['state'].time,path=str(path.relative_to(run)),sha256=sha(path),state_sha256=sha(item['folder']/'state.json')))
            with np.load(path) as data:
                p=np.asarray(item['state'].child_states['fluid']['pressure_Pa']).reshape(top.shape)
                # Saved frame carries same cell pressure; authenticate physical source.
                if not np.array_equal(data['cell_pressure_Pa'].ravel(),p.ravel()):raise ValueError('frame cell pressure mismatch')
                ids=top.locate(data['X'].reshape(-1,3))
                expected=p.ravel()[ids].reshape(data['X'].shape[:-1])
                if not np.array_equal(data['pressure_Pa'],expected):raise ValueError('frame probe pressure mismatch')
            pc=axes[i,j].pcolormesh(top.cuts[0]-.125,top.cuts[1],p[:,:,0].T,cmap='viridis',vmin=0,vmax=.24,shading='flat')
            axes[i,j].set(xlabel='Distance from left boundary (m)',ylabel='Y (m)',title=f'{name}: {item["state"].time*1e6:.0f} us, lower-z cell')
            axes[i,j].set_xscale('symlog',linthresh=2e-6);axes[i,j].set_xlim(0,.375)
    fig.colorbar(pc,ax=axes.ravel().tolist(),label='Pressure (Pa), shared 0..0.24 scale');fig.savefig(out/'pressure-six-frames.png',dpi=150);plt.close(fig)
    # Unequal y/z amplitudes remain visible away from the thin x boundary layer.
    fig,axes=plt.subplots(3,2,figsize=(10,10),constrained_layout=True)
    for i,name in enumerate(names):
        top=contexts[name]['top']
        for j,step in enumerate((8,18)):
            item=hs[name][step];p=np.array(item['state'].child_states['fluid']['pressure_Pa']).reshape(top.shape)
            pc=axes[i,j].pcolormesh(top.cuts[1],top.cuts[2],p[15].T,cmap='viridis',vmin=.16,vmax=.24,shading='flat')
            axes[i,j].set(xlabel='Y (m)',ylabel='Z (m)',title=f'{name}, {item["state"].time*1e6:.0f} us; x cell 15');axes[i,j].set_aspect('equal')
    fig.colorbar(pc,ax=axes.ravel().tolist(),label='Interior pressure (Pa), shared 0.16..0.24 scale');fig.savefig(out/'transverse-sections.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for name in names:
        obs=read(run/f'cases/{name}/observables.json');t=np.array([v['time_s'] for v in obs])*1e6
        axes[0,0].plot(t,[v['modes']['Ay']['value_Pa'] for v in obs],label=name)
        if obs[0]['modes']['Az']['resolved']:axes[0,1].plot(t,[v['modes']['Az']['value_Pa'] for v in obs],label=name)
        for axis,style in [('y','-'),('z','--')]:
            if obs[0]['faces']['internal'][axis]['indices']:
                axes[0,2].plot(t[1:],[v['faces']['internal'][axis]['signed_sum_m3_s'] for v in obs[1:]],style,label=name+' '+axis)
        rows=hs[name][-1]['rows'];tt=np.array([r['time'] for r in rows])*1e6
        axes[1,0].plot(tt,np.cumsum([r['energy_balance_J'] for r in rows]),label=name)
        axes[1,1].plot(tt,np.cumsum([r['darcy_dissipation_J'] for r in rows]),label=name+' Darcy')
        axes[1,1].plot(tt,np.cumsum([r['numerical_dissipation_J'] for r in rows]),'--',label=name+' numerical')
    obs=read(run/'cases/YZ128/observables.json');t=np.array([v['time_s'] for v in obs[1:]])*1e6
    for side in ('x-','x+','y-','y+','z-','z+'):axes[1,2].plot(t,[v['faces']['boundary_outward_m3_s'][side] for v in obs[1:]],label=side)
    for ax,title,unit in zip(axes.ravel(),['Resolved Ay','Resolved Az (Y64 unresolved)','Signed internal transverse flux','Cumulative energy defect','Physical and numerical dissipation','Six outward rates: YZ128'],['Pa','Pa','m3/s','J','J','m3/s']):
        ax.set(title=title,xlabel='Time (us)',ylabel=unit);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'modes-flow-energy.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    for i,name in enumerate(names):
        for step,style in ((8,'--'),(18,'-')):
            item=hs[name][step]
            with np.load(item['folder']/'frame.npz') as f:
                X=f['X'];u=f['x']-X;iy=X.shape[1]//2;iz=X.shape[2]//2
                for axis in range(3):axes[i,axis].plot(X[:,iy,iz,0],u[:,iy,iz,axis]*1e9,style,label=f'{item["state"].time*1e6:.0f} us')
        for a,ax in enumerate(axes[i]):ax.set(title=name+' u_'+ 'xyz'[a],xlabel='X (m)',ylabel='Physical displacement (nm)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(out/'component-displacement.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,10),constrained_layout=True)
    for i,name in enumerate(names):
        rr=read(run/f'S2/{name}-review.json')['regions'];t=np.array([r['time_s'] for r in rr])*1e6
        for j,key in enumerate(('Cauchy_skeleton','Cauchy_total')):
            for region in rr[0]['regions']:
                y=[float(np.linalg.norm(r['regions'][region][key]['rms'])) for r in rr]
                axes[i,j].plot(t,y,label=region)
            axes[i,j].set(title=name+' '+key,xlabel='Time (us)',ylabel='Regional RMS (Pa)');axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=7)
    fig.savefig(out/'regional-stresses.png',dpi=150);plt.close(fig)
    rh=history(run/'S2/fault');before,after=rh[0]['state'],rh[-1]['state'];fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,key,unit in [(axes[0],'pressure_Pa','Volume mean pressure (Pa)'),(axes[1],'cumulative_boundary_m3','cumulative drainage (m3)')]:
        V=contexts['YZ128']['top'].V0
        values=[float(V@np.asarray(s.child_states['fluid'][key])/V.sum()) if key=='pressure_Pa' else float(s.child_states['fluid'][key]) for s in (before,before,after)]
        ax.plot(range(3),values,'o-');ax.set_xticks(range(3),['Commit 8','Fault: exact rollback','Retry: commit 9']);ax.set(ylabel=unit);ax.grid(alpha=.2)
    fig.suptitle('Controlled resource refusal; frame-aware authenticated publication');fig.savefig(out/'recovery.png',dpi=150);plt.close(fig)
    inherited=APP/'visualization/daily-q5-retry';shutil.copytree(inherited,out/'daily-q5-retry',dirs_exist_ok=True)
    decision=read(run/'S4/backend-decision.json');scope=read(run/'S2/scope-decision.json')
    images=['pressure-six-frames.png','transverse-sections.png','component-displacement.png','modes-flow-energy.png','regional-stresses.png','recovery.png']
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MPM-lite 横向响应短窗</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f7f9fc;color:#17243b}img{width:100%;background:white;border-radius:8px;margin:15px 0}a{color:#1759a7}p{line-height:1.7}</style><h1>非均匀初压：三条短窗与事务恢复</h1>'''
    html+=f'<p>直接父版本：569eca0781cc…；Y64、Y128、YZ128各18步到75微秒，合计6个新物理帧。工程z分区比较：{scope["engineering_z_subdivision_consistency"]}。推荐研究后端：{decision["backend"]}。</p>'
    html+='<p>位移为真实计算值，以nm作单位，无几何放大。两种压力图分别共用标明的色标；Y64的Az未分辨，不画作零。应力来自相同实体探针。结果仅覆盖本短窗；不宣称完整空间、启动峰值精度或长周期耦合资格。</p>'
    for name in images:html+=f'<a href="visualization/{name}"><img src="visualization/{name}" alt="{name}"></a>'
    html+='<p><a href="S2/transverse-grid-comparison.json">方向网格比较</a> · <a href="S4/hotspot-profile.json">计时拆分</a> · <a href="S4/paired-performance.json">候选实测</a> · <a href="S2/transaction-check.json">恢复证据</a></p><p><a href="visualization/daily-q5-retry/index.html">继承纯固体动画：252步/1.6秒/12帧，位移显示10倍</a>；该长周期本轮未重跑。</p></html>'
    (run/'index.html').write_text(html)
    write(run/'S6/visualization-origin.json',dict(status='generated',new_physical_frames=6,frames=frames,physical_coordinate_scale=1,displacement_unit='nm',pressure_scales_Pa=[[0,.24],[.16,.24]],inherited_daily=dict(path=str(inherited),frames=12,displacement_scale=10),probe_source='committed frame.npz and matching generation state; stress curves from same fixed physical probes',no_new_long_cycle=True))
    print('VISUALIZATION',len(frames),'committed frames',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
