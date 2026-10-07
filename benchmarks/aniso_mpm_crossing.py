"""Bounded crossing, support, smooth interpolation and velocity-retention study."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.spatial_mpm import ParticleHistory,SpatialMPM
from engine.aniso_phase1.consistent_transfer import material_response
from .aniso_spatial_mpm import initial,H,DURATION,origin_at,rms,compare
from .aniso_history_increment import guard,DATA_ROOT
from .aniso_convergence_reference import save
from .aniso_practical_validation import device_choice

OUT=Path('docs/results/mpm-crossing/v1')
VARIANTS={
    'q1':dict(h=H),
    'q1-fine':dict(h=.8*H),
    'q1-extension':dict(h=H,boundary='affine_extension'),
    'quadratic-raw':dict(h=H,kernel='quadratic'),
    'quadratic-extension':dict(h=H,kernel='quadratic',boundary='affine_extension'),
    'retained':dict(h=H,kernel='quadratic',boundary='affine_extension',velocity_enrichment=True),
    'quadratic-fine':dict(h=.8*H,kernel='quadratic',boundary='affine_extension'),
    'retained-fine':dict(h=.8*H,kernel='quadratic',boundary='affine_extension',velocity_enrichment=True),
}
STAGES={'diagnostics':['q1','q1-fine'],
        'interpolation':['q1-extension','quadratic-raw','quadratic-extension'],
        'retention':['retained'],'refine':['quadratic-fine','retained-fine']}


def key(name,shifted,dt):return f'{name}-{"shifted" if shifted else "fixed"}-dt{dt:.7f}'


def signature(name,shifted,dt,device):
    digest=hashlib.sha256()
    for file in ['engine/aniso_phase1/spatial_mpm.py','engine/aniso_phase1/spatial_basis.py']:
        digest.update(Path(file).read_bytes())
    s=initial()
    for attr in ('X','x','v','F','mass','volume','A'):digest.update(getattr(s,attr).tobytes())
    return dict(protocol='crossing-v1',engine=digest.hexdigest(),options=VARIANTS[name],shifted=shifted,dt=dt,duration=DURATION,device=device)


def checks(rows):
    if not rows:return {}
    return dict(F_commit=max(r['prediction_commit_F_absolute'] for r in rows)<1e-12,
        K_readback=max(r['kinetic_relative'] for r in rows)<1e-11,
        history=max(r['history_rebuild_F_absolute'] for r in rows)==0,
        partition=max(r['partition_error'] for r in rows)<1e-11,
        affine=max(r['affine_position_error'] for r in rows)<1e-11 and max(r['affine_gradient_error'] for r in rows)<1e-9,
        mass=min(r['mass_min'] for r in rows)>0 and max(r['mass_error'] for r in rows)<1e-13,
        solve=max(r['scaled_residual'] for r in rows)<=1.01e-10,
        momentum=max(r['momentum_absolute'] for r in rows)<1e-10,
        budget=max(abs(r['energy_budget_residual']) for r in rows)<1e-13,
        positive_J=min(r['min_det'] for r in rows)>0,
        roundtrip=max(r['roundtrip_relative'] for r in rows)<1e-8)


def load_state(path,index=-1):
    with np.load(path) as f:
        return ParticleHistory(f['X'],f['x'][index],f['v'][index],f['F'][index],f['mass'],f['volume'],f['A'])


def run(out,name,shifted,dt,device):
    tag=key(name,shifted,dt);path=out/(tag+'.json');sig=signature(name,shifted,dt,device)
    if path.exists():
        old=json.loads(path.read_text())
        if old['signature']==sig and old.get('complete'):
            print('cached',tag,flush=True);return old
    state=initial();model=SpatialMPM(device=device,**VARIANTS[name]);steps=round(DURATION/dt)
    Pold=material_response(state.F,state.A,model.params)[1];Pnorm=rms(Pold,state.volume)
    U0=float(state.volume@material_response(state.F,state.A,model.params)[0])
    bulk=(state.mass[:,None]*state.v).sum(0)/state.mass.sum()
    E0=U0+.5*float(np.sum(state.mass[:,None]*(state.v-bulk)**2))
    rows=[];xs=[state.x];vs=[state.v];Fs=[state.F];failure=None;last_origin=None
    print('start',tag,flush=True)
    for step in range(steps):
        guard();origin=origin_at(step*dt,shifted)*(model.h/H)
        oldstate=state
        try:state,info=model.step(state,dt,origin)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
            failure=dict(step=step+1,time=step*dt,error=str(exc));print('REJECTED',tag,failure,flush=True);break
        Pnew=material_response(state.F,state.A,model.params)[1]
        crossing=np.any(np.floor(state.x/model.h)!=np.floor(oldstate.x/model.h),axis=1)
        grid=model.last_grid;nodal=grid.project(state.v)
        gradient=grid.gradient(nodal)
        unresolved=oldstate.v-grid.N@grid.project(oldstate.v)
        projection_error=float(np.sum(oldstate.mass[:,None]*unresolved**2))
        info.update(projection_error_squared=projection_error,
            projection_energy_identity_error=info['projection_delta']+.5*projection_error,step=step+1,time=(step+1)*dt,origin=origin.tolist(),
            origin_changed=last_origin is not None and not np.array_equal(origin,last_origin),
            crossings=int(crossing.sum()),stress_rms=rms(Pnew,state.volume),
            stress_increment_over_initial=rms(Pnew-Pold,state.volume)/Pnorm,
            gradient_rms=rms(gradient,state.mass),
            projection_over_initial_internal=-info['projection_delta']/E0)
        info['stress_increment_crossers']=rms((Pnew-Pold)[crossing],state.volume[crossing])/Pnorm if crossing.any() else 0.
        rows.append(info);xs.append(state.x);vs.append(state.v);Fs.append(state.F);Pold=Pnew;last_origin=origin.copy()
        if (step+1)%16==0:
            print(tag,step+1,'/',steps,'cond',f"{info['mass_condition']:.3g}",flush=True)
            save(out/'status.json',dict(case=tag,step=step+1,total=steps))
    validated=checks(rows);complete=len(rows)==steps
    result=dict(signature=sig,name=name,shifted=shifted,dt=dt,initial_internal=E0,rows=rows,failure=failure,complete=complete,
        checks=validated,passed=complete and all(validated.values()))
    if rows:
        events={i for i,r in enumerate(rows) if r['crossings'] or r['origin_changed']}
        events.add(int(np.argmax([r['mass_condition'] for r in rows])))
        window=sorted({j for i in events for j in range(max(0,i-2),min(len(rows),i+3))})
        result['event_windows']=[rows[i] for i in window]
        result['projection_loss_over_initial_internal']=sum(r['projection_over_initial_internal'] for r in rows)
        result['max_mass_condition']=max(r['mass_condition'] for r in rows)
        result['crossings']=sum(r['crossings'] for r in rows)
    save(path,result)
    np.savez_compressed(out/(tag+'.npz'),X=state.X,x=xs,v=vs,F=Fs,mass=state.mass,volume=state.volume,A=state.A,times=np.arange(len(xs))*dt)
    return result


def counterfactual(out,device,dt=.000125):
    tag=key('q1',False,dt);baseline=json.loads((out/(tag+'.json')).read_text())
    indices=sorted({round(.002/dt),int(np.argmax([q['mass_condition'] for q in baseline['rows']]))})
    results=[]
    for index in indices:
        state=load_state(out/(tag+'.npz'),index);Pold=material_response(state.F,state.A,SpatialMPM().params)[1]
        branches={}
        for name in ('q1','q1-extension','quadratic-extension','retained'):
            for shifted in (False,True):
                origin=np.array([.25,.125,-.125])*H if shifted else np.zeros(3)
                model=SpatialMPM(device=device,**VARIANTS[name])
                try:
                    nxt,stats=model.step(state,dt,origin)
                    P=material_response(nxt.F,nxt.A,model.params)[1]
                    stats['stress_increment_absolute']=rms(P-Pold,state.volume)
                    branches[f'{name}-{shifted}']=stats
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:branches[f'{name}-{shifted}']=dict(error=str(exc))
        results.append(dict(step=index,time=index*dt,branches=branches))
    save(out/'same-state-branches.json',results)


def analyze(out):
    runs={}
    for path in sorted(out.glob('*-dt*.json')):
        r=json.loads(path.read_text());runs[path.stem]=r
    comparisons={};params=SpatialMPM().params
    def diff(a,b):
        if a not in runs or b not in runs or not runs[a]['passed'] or not runs[b]['passed']:return None
        return compare(load_state(out/(a+'.npz')),load_state(out/(b+'.npz')),params)
    for name in VARIANTS:
        comparisons[name]=dict(time={str(sw):diff(key(name,sw,.00025),key(name,sw,.000125)) for sw in (False,True)},
            phase=diff(key(name,True,.000125),key(name,False,.000125)))
    spatial={name:{str(sw):diff(key(name,sw,.000125),key(fine,sw,.000125)) for sw in (False,True)}
        for name,fine in [('q1','q1-fine'),('quadratic-extension','quadratic-fine'),('retained','retained-fine')]}
    summary=dict(cases={k:{field:r.get(field) for field in ('complete','passed','failure','checks','projection_loss_over_initial_internal','max_mass_condition','crossings')} for k,r in runs.items()},comparisons=comparisons,spatial=spatial,storage=guard(),production_ready=False)
    summary['time_target']=.05
    summary['time_passed']={name:(all(d[k]<.05 for d in c['time'].values() for k in ('F','P','v_fluctuation')) if all(c['time'].values()) else None) for name,c in comparisons.items()}
    summary['spatial_convergence_established']=False
    save(out/'results.json',summary)
    lines=['# 跨单元、插值与速度保留对照','',
        '固定 432 粒子、初态、材料和 12 ms 时长。主时间步 0.125 ms；空间加密 h→0.8h。原点每 2 ms 在两个位置切换。',
        '完整质量不加对角正则项，所有候选继续检查 F/K、仿射再现、历史与能量预算。','',
        '| 方案/网格/时间步 | 完成 | 一致性 | max cond(M) | 投影损失 / 初始内部能 |',
        '|---|---:|---:|---:|---:|']
    for k,r in runs.items():
        lines.append(f"| {k} | {r['complete']} | {r['passed']} | {r.get('max_mass_condition',0):.3g} | {r.get('projection_loss_over_initial_internal',0):.4%} |")
    for k,r in runs.items():
        if r['failure']:lines.append(f"\n拒绝 `{k}`：{r['failure']}\n")
    lines+=['','| 方案 | 细时间步网格原点差异 F | P | 去平移速度 |','|---|---:|---:|---:|']
    for name,c in comparisons.items():
        if c['phase']:
            d=c['phase'];lines.append(f"| {name} | {d['F']:.3%} | {d['P']:.3%} | {d['v_fluctuation']:.3%} |")
    lines+=['','空间与时间差详见 results.json；未完成轨迹不参与终态比较。',
        '每条轨迹的 event_windows 保存事件前后两步；same-state-branches.json 比较相同粒子状态下的单步网格/插值反事实。',
        '通过代数一致性并不等于空间收敛。残差基函数是独立的小场景实验选项。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    return summary


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'q1':'Q1','q1-extension':'Q1 + support','quadratic-extension':'Smooth + support','retained':'Smooth + support + residual'}
    runs={}
    for name in labels:
        for sw in (False,True):
            path=out/(key(name,sw,.000125)+'.json')
            if path.exists():runs[name,sw]=json.loads(path.read_text())
    summary=json.loads((out/'results.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for name,label in labels.items():
        if (name,False) not in runs:continue
        rows=runs[name,False]['rows'];t=np.array([r['time'] for r in rows])*1000
        axes[0,0].semilogy(t,[r['mass_min'] for r in rows],label=label)
        axes[0,1].plot(t,[r['raw_support_min'] for r in rows],label=label)
        axes[0,2].plot(t,[r['stress_increment_over_initial'] for r in rows],label=label)
        rows=runs[name,True]['rows'];t=np.array([r['time'] for r in rows])*1000
        loss=np.cumsum([max(0,r['projection_over_initial_internal']) for r in rows])*100
        axes[1,0].plot(t,loss,label=label)
    titles=['Smallest mass eigenvalue (fixed origin)','Min particle support count (raw basis)','Step stress change / initial stress (fixed)']
    for ax,title in zip(axes[0],titles):
        ax.set_title(title);ax.set_xlim(9.5,12);ax.set_xlabel('Time (ms)');ax.axvspan(10.375,10.625,color='gray',alpha=.1)
    axes[0,0].legend(fontsize=8);axes[0,1].set_ylabel('Particles supporting a node')
    axes[1,0].set_title('Cumulative P2G loss (switched origin)');axes[1,0].set_ylabel('% of initial internal energy');axes[1,0].set_xlabel('Time (ms)');axes[1,0].legend(fontsize=8)
    names=[n for n in labels if summary['comparisons'][n]['phase']]
    axes[1,1].bar(np.arange(len(names)),[100*summary['comparisons'][n]['phase']['P'] for n in names])
    axes[1,1].set_xticks(np.arange(len(names)),[labels[n] for n in names],rotation=20,ha='right');axes[1,1].set_title('Stress sensitivity to grid phase (%)')
    names=[n for n in labels if all(summary['comparisons'][n]['time'].values())]
    for off,sw,label in [(-.18,'False','fixed'),(.18,'True','switched')]:
        axes[1,2].bar(np.arange(len(names))+off,[100*summary['comparisons'][n]['time'][sw]['P'] for n in names],width=.35,label=label)
    axes[1,2].set_xticks(np.arange(len(names)),[labels[n] for n in names],rotation=20,ha='right');axes[1,2].set_title('Stress time-step difference (%)');axes[1,2].axhline(5,color='gray',ls='--');axes[1,2].legend()
    fig.savefig(out/'overview.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--stage',choices=tuple(STAGES)+('time','branches','analyze','plot','all'),default='diagnostics')
    p.add_argument('--device',default='auto');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);guard()
    if a.stage=='analyze':analyze(a.out);return
    if a.stage=='plot':plot(a.out);return
    if a.device=='auto':a.device=device_choice()
    if a.device!='cpu':
        import warp as wp
        from utils.resource_guard import prepare_warp_cache
        wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    stages=list(STAGES)+['time','branches'] if a.stage=='all' else [a.stage]
    for stage in stages:
        if stage=='branches':counterfactual(a.out,a.device);continue
        names=['quadratic-extension','retained'] if stage=='time' else STAGES[stage]
        dt=.00025 if stage=='time' else .000125
        for name in names:
            for shifted in (False,True):run(a.out,name,shifted,dt,a.device)
        analyze(a.out)
    analyze(a.out)

if __name__=='__main__':main()
