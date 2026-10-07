"""Saved physical frames, optional display exaggeration, standalone HTML/PNG."""
import argparse,json
from pathlib import Path
import numpy as np
from .run import write

def render(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    case=run/'S3/cycle24'
    with np.load(case/'frames.npz') as f:data={k:f[k].copy() for k in f.files}
    metrics=json.loads((case/'metrics.json').read_text());summary=json.loads((case/'summary.json').read_text())
    X=data['X'];disp=data['x']-X;times=data['time']
    from engine.aniso_phase1.research_common_kinematics_next.model import Space
    Q,w,c=Space().rule();reactions=[]
    for i,m in enumerate(metrics):
        dp=w@(data['qvel'][i+1]-data['qvel'][i])-np.array(m['prepare']['momentum_projection'])
        h=times[i+1]-times[i];t=(times[i+1]+times[i])/2/.06;ramp=max(0.,min(t/.25,1.,(1-t)/.25))
        body=np.array([0.,-ramp*np.dot(w,np.exp(-((Q[:,0]-.7)/.2)**2)),0.])
        reactions.append((dp/h-body).tolist())
    write(run/'S3/reaction-resultant.json',dict(time=times[1:].tolist(),reaction_N=reactions,
        definition='net clamp reaction from quadrature momentum increment minus separately reported transfer projection impulse and body force',
        scope='resultant only; no clamp traction distribution; stationary clamp in the small modal model'))
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    axs[0,0].plot(times,np.max(np.linalg.norm(disp,axis=2),axis=1)*1e3,label='max |u| [mm]')
    axs[0,0].plot(times,np.sqrt(np.mean(data['qvel']**2,axis=(1,2)))*1e3,label='RMS v [mm/s]')
    axs[0,0].set_xlabel('time [s]');axs[0,0].legend();axs[0,0].grid(alpha=.3)
    axs[0,1].plot(times,data['pressure']);axs[0,1].set_ylabel('gauge pressure [Pa]');axs[0,1].set_xlabel('time [s]');axs[0,1].grid(alpha=.3)
    d=np.array([m['darcy_dissipation'] for m in metrics]);project=np.array([m['prepare']['kinetic_projection'] for m in metrics])
    axs[1,0].plot(times[1:],np.cumsum(d),label='Darcy dissipation [J]');axs[1,0].plot(times[1:],np.cumsum(project),label='projection work [J]');axs[1,0].legend();axs[1,0].grid(alpha=.3)
    displayed=X+100*disp[-1];im=axs[1,1].scatter(displayed[:,0],displayed[:,1],c=np.linalg.norm(disp[-1],axis=1)*1e3,cmap='viridis',s=20)
    axs[1,1].set_title('Final x-y projection; displacement display x100');axs[1,1].set_xlabel('x [m]');axs[1,1].set_ylabel('y [m]');fig.colorbar(im,ax=axs[1,1],label='physical |u| [mm]')
    fig.savefig(run/'diagnostics.png',dpi=150);plt.close(fig)
    payload=dict(X=X.tolist(),x=data['x'].tolist(),J=np.linalg.det(data['F']).tolist(),time=times.tolist(),pressure=data['pressure'].tolist())
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-Lite 有界备用历史</title>
<style>body{font:17px system-ui;max-width:1100px;margin:24px auto;background:#f4f6fa;color:#192333;line-height:1.6}canvas{width:100%;background:white;border:1px solid #cad3df}button,select,input{margin:8px;font:inherit}img{width:100%}.note{background:#e7edf7;padding:14px}pre{white-space:pre-wrap}</style>
<h1>有界备用历史：实际24步研究周期</h1><p class="note">8个 SH 模态载体 + 2个材料局部函数，材料q4有128点，备用q6有432点，固定q4几何与质量；两个材料压力控制体。默认周期无回退，另有q2真实拒绝记录。不是正式144空间或原生产隐式求解器。帧来自实际计算；放大仅用于显示。时间／空间全局精度尚未认证。</p>
<button id="play">播放／暂停</button><label>帧<input id="frame" type="range" min="0" max="24" value="0"></label>
<label>显示放大<select id="scale"><option value="1">1×</option><option value="25">25×</option><option value="100" selected>100×</option></select></label>
<label>观察平面<select id="plane"><option value="1">x-y</option><option value="2">x-z</option></select></label>
<label>颜色<select id="color"><option value="u">位移 [mm]</option><option value="J">体积比 J</option><option value="p">压力 [Pa]</option></select></label>
<p id="status"></p><canvas id="view" width="1100" height="450"></canvas><p>两压力值按粒子的材料单元身份显示；粒子仅128个。投影可能重叠。</p>
<img src="diagnostics.png" alt="实际位移、速度、压力和能量曲线">
<p><a href="implementation-report.md">实施记录</a> · <a href="requirement-audit.json">逐项状态</a> · <a href="S1-formal/report.json">正式空间静态检查与稳定化限制</a></p>
<script>const data=PAYLOAD;const view=document.getElementById('view'),ctx=view.getContext('2d'),slider=document.getElementById('frame');let playing=false;
function draw(){const n=Number(slider.value),amp=Number(document.getElementById('scale').value),axis=Number(document.getElementById('plane').value),mode=document.getElementById('color').value;const values=data.X.map((X,i)=>mode==='J'?data.J[n][i]:mode==='p'?data.pressure[n][X[0]<.5?0:1]:1000*Math.hypot(...X.map((v,k)=>data.x[n][i][k]-v)));let lo=Math.min(...values),hi=Math.max(...values);ctx.clearRect(0,0,1100,450);ctx.fillStyle='#526171';ctx.font='16px system-ui';ctx.fillText('x →；参考域：[0,1] × [0,0.25] m',35,30);ctx.strokeStyle='#ccd5df';ctx.strokeRect(60,85,970,275);data.X.forEach((X,i)=>{let p=X.map((v,k)=>v+amp*(data.x[n][i][k]-v));let t=(values[i]-lo)/Math.max(hi-lo,1e-12);ctx.fillStyle=`hsl(${240-240*t},75%,45%)`;ctx.beginPath();ctx.arc(60+970*p[0],360-1100*p[axis],4,0,7);ctx.fill();});document.getElementById('status').textContent=`帧 ${n}/24 | t=${data.time[n].toFixed(5)} s | 显示放大 ${amp}× | 颜色范围 ${lo.toPrecision(5)} — ${hi.toPrecision(5)}`;}
for(const id of ['frame','scale','plane','color'])document.getElementById(id).addEventListener('input',draw);document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(Number(slider.value)+1)%data.time.length;draw();}},200);draw();</script></html>'''
    (run/'index.html').write_text(page.replace('PAYLOAD',json.dumps(payload,separators=(',',':'))))
    print(run/'index.html')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();render(a.run)
