"""Six committed physical frames, scoped diagnostics, inherited daily animation."""
import argparse,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .provenance import *
from .review import probes
from .trajectory import case_name
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_startup_next.coupling_review import context
from benchmarks.research_sequential_next.compare import regions
from engine.aniso_phase1.research_startup_substeps_next.schedule import aggregate,OBSERVATIONS
from benchmarks.research_pressure_startup_next.coupling_review import fluid

def main(run):
    run=Path(run);mutable(run);out=run/'visualization';out.mkdir(exist_ok=True);cuts=read(run/'S0/grid-protocol-inherited.json')['cuts'];p=read(run/'S0/coupled-protocol.json')['parameters'];contexts={g:context(cuts[g],p['storage']) for g in ('base','yz')}
    hist={g:history(run/'cases'/case_name(g)) for g in ('base','yz')};fig,axs=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for grid,label,style in [('base','32 cells','-'),('yz','128 cells','--')]:
        h=hist[grid];rows=h[-1]['rows'];ts=np.array([x['state'].time for x in h])*1e6;fluid_states=[x['state'].child_states['fluid'] for x in h];top=contexts[grid]['top'];mean=[np.asarray(f['pressure_Pa'])@top.V0/top.V0.sum() for f in fluid_states]
        axs[0,0].plot(ts,mean,style,label=label);axs[0,1].plot(ts,[min(f['pressure_Pa']) for f in fluid_states],style,label=label)
        axs[0,2].plot(ts,[f['cumulative_boundary_m3'] for f in fluid_states],style,label=label)
        axs[1,0].plot(ts[1:],[r['min_detF'] for r in rows],style,label=label)
        axs[1,1].plot(ts[1:],np.cumsum([r['energy_balance_J'] for r in rows]),style,label=label)
    rows=hist['yz'][-1]['rows'];ts=np.array([r['time'] for r in rows])*1e6
    for key,label in [('kinetic_J','solid kinetic'),('fluid_storage_J','fluid storage'),('solid_material_J','material'),('stabilization_J','stabilization')]:axs[1,2].plot(ts,[r[key] for r in rows],label=label)
    for ax,title,unit in zip(axs.ravel(),['Mean pressure','Minimum pressure','Cumulative drainage','Minimum det F','Cumulative energy defect','Energy components (128)'],['Pa','Pa','m3','','J','J']):ax.set(title=title,xlabel='Time (microseconds)',ylabel=unit);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axs[1,0].ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for grid,label,style in [('base','32 cells','-'),('yz','128 cells','--')]:
        X,d,fields=probes(run/'cases'/case_name(grid));f=fields[max(fields)];iy=X.shape[1]//2;iz=X.shape[2]//2;x=X[:,iy,iz,0]
        axs[0,0].plot(x,(f['x']-X)[:,iy,iz,0]*1e9,style,label=label);axs[0,1].plot(x,f['Cauchy_total'][:,iy,iz,0,0],style,label=label)
        top=contexts[grid]['top'];pressure=np.asarray(hist[grid][-1]['state'].child_states['fluid']['pressure_Pa']).reshape(top.shape);xc=(top.cuts[0][1:]+top.cuts[0][:-1])/2
        axs[1,0].plot(xc[:16]-.125,pressure[:16,0,0],style,label=label)
        if grid=='yz':
            for a in range(2):
                for b in range(2):axs[1,1].plot(xc,pressure[:,a,b],label=f'y{a},z{b}')
    axs[0,0].set(title='Physical centreline displacement, 75 us',xlabel='X (m)',ylabel='u_x (nm)');axs[0,1].set(title='Total Cauchy stress, 75 us',xlabel='X (m)',ylabel='sigma_xx (Pa)')
    axs[1,0].set(title='Left pressure layer, 75 us',xlabel='Distance from left boundary (m)',ylabel='Pressure (Pa)',xscale='log');axs[1,1].set(title='Four transverse pressure columns, 128 cells',xlabel='X (m)',ylabel='Pressure (Pa)')
    for ax in axs.ravel():ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'actual-field.png',dpi=150);plt.close(fig)
    # Reuse the six existing physical observations, with identical scales.
    times=[25e-6,50e-6,75e-6];fig,axs=plt.subplots(2,3,figsize=(14,6),constrained_layout=True)
    for row,grid in enumerate(('base','yz')):
        top=contexts[grid]['top']
        for col,t in enumerate(times):
            state=min(hist[grid],key=lambda a:abs(a['state'].time-t))['state']
            pressure=np.asarray(state.child_states['fluid']['pressure_Pa']).reshape(top.shape)
            pc=axs[row,col].pcolormesh(top.cuts[0]-.125,top.cuts[1],pressure[:,:,0].T,vmin=0,vmax=.2,shading='flat',cmap='viridis')
            axs[row,col].set(xlim=(0,.02),title=f'{top.cells} cells, {t*1e6:.0f} us',xlabel='Distance from left boundary (m)',ylabel='Y (m)');axs[row,col].set_xscale('symlog',linthresh=2e-6)
    fig.colorbar(pc,ax=axs.ravel().tolist(),label='Pressure (Pa), same scale');fig.savefig(out/'pressure-sections.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True);X,d,values=probes(run/'cases'/case_name('yz'));ts=np.array(sorted(values));weights=regions(X)
    for name,w in weights.items():
        rms=[float(np.sqrt(np.sum(w[...,None,None]*values[t]['Cauchy_total']**2)/w.sum())) for t in ts]
        axs[0,0].plot(ts*1e6,rms,label=name)
    rows=hist['yz'][-1]['rows'];ts=np.array([x['time'] for x in rows]);side=np.array([contexts['yz']['groups']@np.asarray(x['flux_interval_m3_s']) for x in rows])
    for i,name in enumerate(['x-','x+','y-','y+','z-','z+']):axs[0,1].plot(ts*1e6,side[:,i],label=name)
    for key in ['darcy_dissipation_J','numerical_dissipation_J']:axs[1,0].plot(ts*1e6,np.cumsum([x[key] for x in rows]),label=key.replace('_J',''))
    # This time-refinement comparison is explicitly inherited, not a new run.
    for case,label in [('boundary32-h','inherited h'),('boundary32-half','inherited h/2')]:
        f,tt,_=fluid(TRAJECTORY/'cases'/case);v=aggregate(f,tt,OBSERVATIONS[:7]);z=v['flux']
        axs[1,1].plot(OBSERVATIONS[1:7]*1e6,np.linalg.norm(z,axis=1),label=label)
    for ax,title,unit in zip(axs.ravel(),['Regional total stress, 128 cells','Six outward side rates, 128 cells','Accumulated dissipation','Inherited 32-cell engineering flux norm'],['Pa','m3/s','J','m3/s']):ax.set(title=title,xlabel='Time (microseconds)',ylabel=unit);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'regions-flow-time.png',dpi=150);plt.close(fig)
    recovery=read(run/'S5/resource-fault-check.json');rh=history(run/'S5/resource-fault')
    if not recovery['rollback_exact']:raise ValueError('cannot display unverified restored state')
    states=[rh[0]['state'],rh[0]['state'],rh[-1]['state']];fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    axs[0].plot(range(3),[np.mean(s.child_states['fluid']['pressure_Pa']) for s in states],'o-');axs[0].set(ylabel='Cell mean pressure (Pa)')
    axs[1].plot(range(3),[s.child_states['fluid']['cumulative_boundary_m3'] for s in states],'o-');axs[1].set(ylabel='Cumulative drainage (m3)')
    for ax in axs:ax.set_xticks(range(3),['Committed step 8','Fault: unchanged digest','Retry: step 9']);ax.grid(alpha=.2)
    fig.suptitle('Controlled resource rejection: authenticated committed states');fig.savefig(out/'recovery.png',dpi=150);plt.close(fig)
    ref=read(run/'S3/solid-reference-comparison.json');perf=read(run/'S4/paired-performance.json');fig,axs=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    names=['global_domain','transition','interior'];xx=np.arange(len(names));axs[0].bar(xx-.15,[ref['formal_vs_R6'][n]['PK1']['absolute'] for n in names],width=.3,label='formal 144 vs R6');axs[0].bar(xx+.15,[ref['adjacent'][n]['PK1']['absolute'] for n in names],width=.3,label='R5 vs R6');axs[0].set_xticks(xx,names);axs[0].set(title='Independent pure-solid F45 static reference',ylabel='PK1 RMS difference (Pa)');axs[0].legend()
    for i,item in enumerate(perf['records']):axs[1].bar(i-.15,item['D3']['advance_s'],.3,color='C0',label='D3' if i==0 else None);axs[1].bar(i+.15,item['G2']['advance_s'],.3,color='C1',label='G2' if i==0 else None)
    axs[1].set_xticks([0,1],['25 us input','50 us input']);axs[1].set(title='Measured whole-step cost; setup reported separately',ylabel='seconds');axs[1].legend();fig.savefig(out/'reference-performance.png',dpi=150);plt.close(fig)
    inherited=APP/'visualization/daily-q5-retry';shutil.copytree(inherited,out/'daily-q5-retry',dirs_exist_ok=True)
    status=read(run/'S4/backend-decision.json');s=read(run/'S2/spatial-comparison.json')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MPM-lite 三维压力短窗</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px;background:#f7f9fc;color:#17243b}img{width:100%;background:white;border-radius:10px;margin:15px 0}a{color:#1759a7}p{line-height:1.7}</style><h1>MPM-lite：三维压力几何与空间参考</h1>'''
    html+=f'<p>本轮从封存版 a1780d7a… 顺序实施。32/128 单元各18步，0→75微秒，两网格工程比较：{s["status"]}。正式144函数、完整M7与q7保持。当前推荐后端：{status["backend"]}。</p>'
    html+='<p>以下为实际计算，位移以nm单位显示，未放大物理坐标。只认证指定短窗的一致性；不宣称连续空间真解、完整耦合周期或原始启动微步精度。</p>'
    for name in ['diagnostics.png','actual-field.png','pressure-sections.png','regions-flow-time.png','reference-performance.png','recovery.png']:html+=f'<a href="visualization/{name}"><img src="visualization/{name}" alt="{name}"></a>'
    html+='<p><a href="S2/spatial-comparison.json">粗细网格数据</a> · <a href="S3/solid-reference-comparison.json">固体参考数据</a> · <a href="S4/paired-performance.json">性能与设置成本</a> · <a href="S5/resource-fault-check.json">恢复证据</a></p><p><a href="visualization/daily-q5-retry/index.html">继承的日常纯固体动画：252步/1.6秒/12帧，位移放大10倍</a>。该长周期本轮未重跑。</p></html>'
    (run/'index.html').write_text(html)
    write(run/'S6/visualization-origin.json',dict(status='generated',new_physical_frames=6,new_times_us=[25,50,75],grids=[32,128],physical_coordinate_scale=1,displacement_display_unit='nm, not geometrical magnification',inherited_daily=dict(path=str(inherited),frames=12,displacement_scale=10),no_new_long_cycle=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
