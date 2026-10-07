"""Render saved trajectories without recomputing simulation or using a GPU."""
import argparse
import json
from pathlib import Path
import numpy as np


def render(run, case='cpu-q7', output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    run=Path(run);out=Path(output) if output else run/'visualization'/case
    out.mkdir(parents=True,exist_ok=True)
    with np.load(run/case/'trajectory.npz',allow_pickle=False) as z:
        t=z['times'];X=z['X'];x=z['x'];P=z['PK1']
    rows=json.loads((run/case/'ledger.json').read_text())
    if len(t)!=len(rows)+1:raise ValueError('trajectory and ledger frame count disagree')
    if not all(np.isfinite(a).all() for a in (t,X,x,P)):raise ValueError('nonfinite visualization data')
    mid=X.shape[2]//2
    plane=X[:,:,mid,:2];positions=x[:,:,:,mid,:2];stress=P[:,:,:,mid,0,0]
    lo,hi=float(stress.min()),float(stress.max())
    if hi-lo<1e-12:hi=lo+1.
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax=axs[0,0];m=10.
    draw=plane+m*(positions[0]-plane)
    dots=ax.scatter(draw[...,0],draw[...,1],c=stress[0],vmin=lo,vmax=hi,cmap='coolwarm',s=16)
    ax.set(xlim=(float(X[...,0].min())-.01,float(X[...,0].max())+.07),
        ylim=(float(X[...,1].min())-.03,float(X[...,1].max())+.03),xlabel='x (m)',ylabel='y (m)')
    ax.set_aspect('equal');fig.colorbar(dots,ax=ax,label='PK1 P11 (Pa)')
    axs[0,1].plot(t[1:],[r['reaction_N'] for r in rows],'o-',label='total reaction')
    axs[0,1].set(xlabel='time (s)',ylabel='right grip reaction (N)');axs[0,1].grid(alpha=.2)
    for key,label in [('material_J','material'),('stabilization_J','stabilization'),('kinetic_J','kinetic')]:
        axs[1,0].plot(t[1:],[r[key] for r in rows],'o-',label=label)
    axs[1,0].set(xlabel='time (s)',ylabel='energy (J)');axs[1,0].legend();axs[1,0].grid(alpha=.2)
    axs[1,1].plot(t[1:],[r['min_detF'] for r in rows],'o-',label='min det(F)')
    axs[1,1].set(xlabel='time (s)',ylabel='minimum det(F)');axs[1,1].grid(alpha=.2)
    cursors=[a.axvline(0,color='#888888',ls='--') for a in (axs[0,1],axs[1,0],axs[1,1])]
    def update(i):
        draw=plane+m*(positions[i]-plane);dots.set_offsets(draw.reshape(-1,2));dots.set_array(stress[i].ravel())
        ax.set_title(f't={t[i]:.3f} s; deformation x10')
        for line in cursors:line.set_xdata([t[i],t[i]])
        fig.suptitle(f'{case}: 7-step practical scene check\nSpatial and transient accuracy remain uncertified',fontsize=12)
        return [dots,*cursors]
    update(int(np.argmax([np.max(abs(p-plane)) for p in positions])))
    fig.savefig(out/'scene-summary.png',dpi=140)
    animation=FuncAnimation(fig,update,frames=len(t),interval=450,blit=False)
    animation.save(out/'cycle.gif',writer=PillowWriter(fps=2));plt.close(fig)
    payload=dict(times=t.tolist(),X=plane.reshape(-1,2).tolist(),x=positions.reshape(len(t),-1,2).tolist(),
        stress=stress.reshape(len(t),-1).tolist(),minimum=lo,maximum=hi,case=case)
    html='''<!doctype html><meta charset="utf-8"><title>MPM-lite 场景检查</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1050px;background:#f5f7fa;color:#152a3a}canvas,img{width:100%;background:white;border-radius:8px}label{display:inline-block;margin:12px}input{vertical-align:middle}small{color:#536373}</style>
<h1>MPM-lite 场景检查</h1><p id="title"></p>
<label>帧 <input id="frame" type="range" min="0" value="0" step="1"></label>
<label>形变放大 <input id="mag" type="range" min="1" max="20" value="10" step="1"> <span id="factor"></span></label>
<button id="play">播放 / 暂停</button><canvas id="scene" width="1050" height="350"></canvas>
<p><small>颜色为截面 PK1 P11 应力。放大只用于显示。七步粗时间采样用于运行与稳定性检查，不代表空间或瞬态精度认证。</small></p>
<img src="scene-summary.png" alt="Reaction, energy and minimum deformation Jacobian">
<script>const data=PAYLOAD;const frame=document.querySelector('#frame'),mag=document.querySelector('#mag'),ctx=document.querySelector('#scene').getContext('2d');frame.max=data.times.length-1;
const xmin=Math.min(...data.X.map(x=>x[0]))-.01,xmax=Math.max(...data.X.map(x=>x[0]))+.12,ymin=Math.min(...data.X.map(x=>x[1]))-.04,ymax=Math.max(...data.X.map(x=>x[1]))+.04;
function draw(){const k=+frame.value,m=+mag.value;document.querySelector('#title').textContent=data.case+' | t = '+data.times[k].toFixed(3)+' s';document.querySelector('#factor').textContent=m+'×';ctx.clearRect(0,0,1050,350);
for(let j=0;j<data.X.length;j++){const X=data.X[j],x=data.x[k][j],s=(data.stress[k][j]-data.minimum)/(data.maximum-data.minimum);ctx.fillStyle='hsl('+(240*(1-s))+',70%,50%)';ctx.beginPath();ctx.arc(30+(X[0]+m*(x[0]-X[0])-xmin)/(xmax-xmin)*990,320-(X[1]+m*(x[1]-X[1])-ymin)/(ymax-ymin)*290,3,0,2*Math.PI);ctx.fill();}ctx.fillStyle='#152a3a';ctx.fillText('PK1 P11: '+data.minimum.toFixed(3)+' ... '+data.maximum.toFixed(3)+' Pa',30,22);}
frame.oninput=draw;mag.oninput=draw;let playing=false;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){frame.value=(+frame.value+1)%data.times.length;draw();}},600);draw();</script>'''
    (out/'index.html').write_text(html.replace('PAYLOAD',json.dumps(payload,allow_nan=False)))
    print(out/'index.html',flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path)
    p.add_argument('--case',default='cpu-q7',choices=['cpu-q7','gpu-q7','gpu-q5']);p.add_argument('--output',type=Path)
    a=p.parse_args();render(a.run,a.case,a.output)


if __name__=='__main__':main()
