"""Render published saved frames only; no simulation or GPU is required."""
import argparse
import json
from pathlib import Path
import numpy as np
from .checkpoint import GenerationStore
from .provenance import read,write


def render(run,case,output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    run=Path(run);folder=run/'cases'/case;identity=read(folder/'identity.json')
    history=GenerationStore(folder,identity).history()
    if not history:raise ValueError('no committed trajectory')
    frames=[]
    for item in history:
        path=item['folder']/'frame.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:frames.append({k:np.array(z[k]) for k in z.files})
    if not frames:raise ValueError('no saved display frames')
    out=Path(output) if output else run/'visualization'/case
    out.mkdir(parents=True,exist_ok=True)
    rows=history[-1]['rows'];times=np.array([float(f['time']) for f in frames]);last=history[-1]['state']
    cfg=read(folder/'execution-protocol.json');implementation=cfg.get('implementation',{})
    operator=implementation.get('operator','original');cache=implementation.get('linearization_cache',False)
    X=frames[0]['X'];mid=X.shape[2]//2;plane=X[:,:,mid,:2]
    positions=np.array([f['x'][:,:,mid,:2] for f in frames]);stress=np.array([f['PK1'][:,:,mid,0,0] for f in frames])
    lo,hi=float(stress.min()),float(stress.max());hi=max(hi,lo+1e-12)
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    dots=axs[0,0].scatter(*((plane+10*(positions[0]-plane)).reshape(-1,2).T),
        c=stress[0].ravel(),vmin=lo,vmax=hi,cmap='coolwarm',s=16)
    axs[0,0].set(xlim=(float(X[...,0].min())-.01,float(X[...,0].max())+.08),
        ylim=(float(X[...,1].min())-.03,float(X[...,1].max())+.03),xlabel='x (m)',ylabel='y (m)')
    axs[0,0].set_aspect('equal');fig.colorbar(dots,ax=axs[0,0],label='PK1 P11 (Pa)')
    t=[r['time'] for r in rows]
    axs[0,1].plot(t,[r['reaction_N'] for r in rows],'.-');axs[0,1].set(xlabel='time (s)',ylabel='right-grip interval reaction (N)')
    for key,label in [('material_J','material'),('stabilization_J','stabilization'),('kinetic_J','kinetic')]:
        axs[1,0].plot(t,[r[key] for r in rows],label=label)
    axs[1,0].set(xlabel='time (s)',ylabel='energy (J)');axs[1,0].legend()
    axs[1,1].plot(t,[r['min_detF'] for r in rows],'.-');axs[1,1].set(xlabel='time (s)',ylabel='minimum path/endpoint det(F)')
    for ax in (axs[0,1],axs[1,0],axs[1,1]):ax.grid(alpha=.2)
    def update(i):
        dots.set_offsets((plane+10*(positions[i]-plane)).reshape(-1,2));dots.set_array(stress[i].ravel())
        axs[0,0].set_title(f't={times[i]:.4f} s; display displacement x10')
        fig.suptitle(f'{case}: {len(rows)} accepted steps, {len(frames)} saved frames\n'
            f'{cfg["device"]}, {operator}, material q{cfg["material_order"]}; time/space accuracy uncertified',fontsize=11)
        return [dots]
    peak=int(np.argmax(np.max(abs(positions-plane),axis=(1,2,3))));update(peak)
    fig.savefig(out/'scene-summary.png',dpi=140)
    animation=FuncAnimation(fig,update,frames=len(frames),interval=450,blit=False)
    animation.save(out/'cycle.gif',writer=PillowWriter(fps=2));plt.close(fig)
    payload=dict(times=times.tolist(),X=plane.reshape(-1,2).tolist(),x=positions.reshape(len(frames),-1,2).tolist(),
        stress=stress.reshape(len(frames),-1).tolist(),lo=lo,hi=hi,case=case,steps=len(rows),
        committed_time=last.time,device=cfg['device'],order=cfg['material_order'],operator=operator,cache=cache)
    html='''<!doctype html><meta charset="utf-8"><title>MPM-lite 顺序优化场景</title>
<style>body{font:16px system-ui;max-width:1080px;margin:30px auto;color:#183448;background:#f3f6f9}canvas,img{width:100%;background:white}label{display:inline-block;margin:12px}small{color:#536373}</style>
<h1>MPM-lite 已提交场景</h1><p id="status"></p><p id="title"></p>
<label>显示帧 <input id="frame" type="range" min="0" value="0"></label>
<label>形变显示倍率 <input id="mag" type="range" min="1" max="20" value="10"><span id="factor"></span></label>
<button id="play">播放 / 暂停</button><canvas id="scene" width="1050" height="350"></canvas>
<p><small>颜色为截面 PK1 P11。显示倍率不改变计算数据。显示帧与内部积分步分开；时间和空间精度以独立验收报告为准。</small></p>
<img src="scene-summary.png" alt="场景与全部积分步的反力、能量和最小Jacobian">
<script>const data=PAYLOAD,frame=document.querySelector('#frame'),mag=document.querySelector('#mag'),ctx=document.querySelector('#scene').getContext('2d');frame.max=data.times.length-1;
document.querySelector('#status').textContent=data.case+' | '+data.device+' | '+data.operator+' | 材料 q'+data.order+' | 接受步 '+data.steps+' | 显示帧 '+data.times.length+' | 已提交至 '+data.committed_time+' s';
const xmin=Math.min(...data.X.map(x=>x[0]))-.01,xmax=Math.max(...data.X.map(x=>x[0]))+.10,ymin=Math.min(...data.X.map(x=>x[1]))-.04,ymax=Math.max(...data.X.map(x=>x[1]))+.04;
function draw(){const k=+frame.value,m=+mag.value;document.querySelector('#title').textContent='实际物理时刻 t='+data.times[k].toFixed(4)+' s';document.querySelector('#factor').textContent=m+'×';ctx.clearRect(0,0,1050,350);
for(let j=0;j<data.X.length;j++){const X=data.X[j],x=data.x[k][j],s=(data.stress[k][j]-data.lo)/(data.hi-data.lo);ctx.fillStyle='hsl('+(240*(1-s))+',70%,50%)';ctx.beginPath();ctx.arc(30+(X[0]+m*(x[0]-X[0])-xmin)/(xmax-xmin)*990,320-(X[1]+m*(x[1]-X[1])-ymin)/(ymax-ymin)*290,3,0,2*Math.PI);ctx.fill();}ctx.fillStyle='#183448';ctx.fillText('PK1 P11: '+data.lo.toFixed(4)+' ... '+data.hi.toFixed(4)+' Pa',30,22);}
frame.oninput=draw;mag.oninput=draw;let playing=false;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){frame.value=(+frame.value+1)%data.times.length;draw();}},600);draw();</script>'''
    (out/'index.html').write_text(html.replace('PAYLOAD',json.dumps(payload,allow_nan=False)))
    write(out/'metadata.json',dict(case=case,accepted_steps=len(rows),saved_frames=len(frames),
        committed_time_s=last.time,protocol_sha256=identity['protocol_sha256'],no_recomputation=True,
        device=cfg['device'],operator=operator,linearization_cache=cache,material_order=cfg['material_order'],mass_order=cfg['mass_order']))
    print(out/'index.html',flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--case',required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    render(a.run,a.case,a.output)
