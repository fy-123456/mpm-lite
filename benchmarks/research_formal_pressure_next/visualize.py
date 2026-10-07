import argparse,json
from pathlib import Path
import numpy as np

def render(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=run/'S4/closed';rows=json.loads((root/'metrics.json').read_text())
    with np.load(root/'frames.npz') as f:d={k:f[k].copy() for k in f.files}
    fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True);t=d['time']
    ax[0,0].plot(t,d['p'][:,0],'o-',label='left material volume');ax[0,0].plot(t,d['p'][:,1],'o-',label='right material volume');ax[0,0].set_ylabel('pressure [Pa]');ax[0,0].legend()
    ax[0,1].plot(t[1:],[x['reaction_right_x'] for x in rows],'o-');ax[0,1].set_ylabel('right grip reaction [N]')
    ax[1,0].plot(t[1:],np.cumsum([x['boundary_work'] for x in rows]),label='boundary work');ax[1,0].plot(t[1:],[x['kinetic']+x['material_U']+x['stabilization_U']+x['storage_U'] for x in rows],label='total energy');ax[1,0].plot(t[1:],np.cumsum([x['darcy_dissipation'] for x in rows]),label='Darcy dissipation');ax[1,0].legend();ax[1,0].set_ylabel('energy [J]')
    ax[1,1].plot(t[1:],[x['minJ'] for x in rows],'o-');ax[1,1].set_ylabel('minimum J on full material samples')
    for a in ax.flat:a.grid(alpha=.3);a.set_xlabel('time [s]')
    fig.savefig(run/'diagnostics.png',dpi=140);plt.close(fig)
    payload=dict(X=d['X'].tolist(),x=d['x'].tolist(),J=np.linalg.det(d['F']).tolist(),p=d['p'].tolist(),time=t.tolist())
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-Lite 正式材料空间压力耦合</title>
<style>body{font:17px system-ui;line-height:1.6;max-width:1100px;margin:24px auto;background:#f3f6fa;color:#192333}canvas{width:100%;background:white}img{width:100%}button,select,input{font:inherit;margin:8px}.note{background:#e1eaf4;padding:16px}</style>
<h1>正式材料空间：短窗固液压力交换</h1><p class="note">225载体 + 144局部函数，648个自由向量分量。原完整质量与q7材料积分，两个固定材料压力控制体，常量参考构形流阻网络，中点积分。本页播放实际封闭边界结果，仅显示165个材料探针。尚未接入生产GPU，没有独立空间精度证书。</p>
<button id="play">播放／暂停</button><label>帧<input id="frame" type="range" min="0" max="MAXFRAME" value="0"></label>
<label>显示放大<select id="scale"><option value="1">1×</option><option value="1000" selected>1000×</option><option value="10000">10000×</option></select></label>
<label>颜色<select id="color"><option value="p">控制体压力 [Pa]</option><option value="u">位移 [μm]</option><option value="J">探针 J</option></select></label>
<p id="status"></p><canvas id="view" width="1100" height="450"></canvas>
<p>压力是控制体常数，着色不表示已解析的细尺度压力场。显示放大不改变物理数据；完整材料规则最小J见图表。</p>
<img src="diagnostics.png" alt="压力、夹具反力、能量与J"><p><a href="implementation-report.md">实施记录</a> · <a href="requirement-audit.json">逐项验收</a> · <a href="S4/closed/metrics.json">原始记录</a></p>
<script>const data=PAYLOAD;const ctx=document.getElementById('view').getContext('2d'),slider=document.getElementById('frame');let playing=false;
function draw(){let n=Number(slider.value),amp=Number(document.getElementById('scale').value),mode=document.getElementById('color').value;let vals=data.X.map((X,i)=>mode==='p'?data.p[n][X[0]<.5?0:1]:mode==='J'?data.J[n][i]:1e6*Math.hypot(...X.map((v,k)=>data.x[n][i][k]-v)));let lo=Math.min(...vals),hi=Math.max(...vals);ctx.clearRect(0,0,1100,450);ctx.fillStyle='#e0e5ed';ctx.fillRect(60,85,970/6,275);ctx.fillRect(60+970*5/6,85,970/6,275);ctx.strokeStyle='#ccd5df';ctx.strokeRect(60,85,970,275);data.X.forEach((X,i)=>{let p=X.map((v,k)=>v+amp*(data.x[n][i][k]-v)),c=(vals[i]-lo)/Math.max(hi-lo,1e-12);ctx.fillStyle=`hsl(${240-240*c},75%,45%)`;ctx.beginPath();ctx.arc(60+970*(p[0]-.125)/.75,360-1100*(p[1]-.375),4,0,7);ctx.fill();});document.getElementById('status').textContent=`帧 ${n}/${data.time.length-1} | t=${data.time[n].toFixed(4)} s | ${amp}× | 颜色范围 ${lo.toPrecision(5)} — ${hi.toPrecision(5)}`;}
for(const id of ['frame','scale','color'])document.getElementById(id).addEventListener('input',draw);document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(Number(slider.value)+1)%data.time.length;draw();}},350);draw();</script></html>'''
    (run/'index.html').write_text(page.replace('MAXFRAME',str(len(t)-1)).replace('PAYLOAD',json.dumps(payload,separators=(',',':'))))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);render(a.parse_args().run)
