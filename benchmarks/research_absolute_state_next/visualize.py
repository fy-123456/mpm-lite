"""Actual four-step dry-solid results, explicitly separated from coupled demos."""
import argparse,json
from pathlib import Path
import numpy as np

def render(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    case=run/'S2/nonlinear4'
    with np.load(case/'frames.npz') as d:data={k:d[k].copy() for k in d.files}
    metrics=json.loads((case/'metrics.json').read_text());summary=json.loads((case/'summary.json').read_text())
    t=data['time'];disp=data['x']-data['X'];fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    ax[0,0].plot(t,1e6*np.max(np.linalg.norm(disp,axis=2),axis=1),'o-');ax[0,0].set_ylabel('max probe displacement [micrometre]')
    ax[0,1].plot(t[1:],[r['true_free_force_residual'] for r in metrics],label='true residual');ax[0,1].plot(t[1:],[r['force_tolerance'] for r in metrics],label='declared tolerance');ax[0,1].set_yscale('log');ax[0,1].legend()
    ax[1,0].plot(t[1:],[r['kinetic'] for r in metrics],label='kinetic');ax[1,0].plot(t[1:],[r['potential'] for r in metrics],label='potential');ax[1,0].plot(t[1:],np.cumsum([r['boundary_work'] for r in metrics]),label='boundary work');ax[1,0].legend();ax[1,0].set_ylabel('energy [J]')
    ax[1,1].plot(t[1:],[r['minJ'] for r in metrics],'o-');ax[1,1].set_ylabel('min det F on FULL q7 rule')
    for a in ax.flat:a.set_xlabel('time [s]');a.grid(alpha=.3)
    fig.savefig(run/'diagnostics.png',dpi=140);plt.close(fig)
    payload=dict(X=data['X'].tolist(),x=data['x'].tolist(),J=np.linalg.det(data['F']).tolist(),time=t.tolist())
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-Lite 原材料绝对状态</title>
<style>body{font:17px system-ui;max-width:1100px;margin:24px auto;background:#f4f6fa;color:#192333;line-height:1.6}canvas{width:100%;background:white;border:1px solid #cad3df}button,select,input{margin:8px;font:inherit}img{width:100%}.note{background:#e7edf7;padding:14px}</style>
<h1>原材料绝对状态：4步非线性干固体短窗</h1><p class="note">原225载体+144局部函数，648自由向量分量，完整q7材料积分。左夹具固定，右夹具小幅加速。没有压力耦合，没有生产GPU接入。使用中点积分及真实非线性残差检查；不是原AVF全周期。这里只显示165个材料探针。</p>
<button id="play">播放／暂停</button><label>帧<input id="frame" type="range" min="0" max="4" value="0"></label>
<label>显示放大<select id="scale"><option value="1">1×</option><option value="1000">1000×</option><option value="10000" selected>10000×</option></select></label>
<label>颜色<select id="color"><option value="u">位移 [μm]</option><option value="J">探针体积比 J</option></select></label>
<p id="status"></p><canvas id="view" width="1100" height="450"></canvas>
<p>灰色区域是参考夹持体。显示放大不改变物理数据；图表中的最小J来自完整材料规则，可能低于稀疏探针的最小J。</p>
<img src="diagnostics.png" alt="实际位移、残差、能量和完整规则最小J">
<p><a href="implementation-report.md">实施记录</a> · <a href="requirement-audit.json">逐项状态</a> · <a href="S1/closure.json">状态闭合证据</a></p>
<script>const data=PAYLOAD;const view=document.getElementById('view'),ctx=view.getContext('2d'),slider=document.getElementById('frame');let playing=false;
function draw(){const n=Number(slider.value),amp=Number(document.getElementById('scale').value),mode=document.getElementById('color').value;const values=data.X.map((X,i)=>mode==='J'?data.J[n][i]:1e6*Math.hypot(...X.map((v,k)=>data.x[n][i][k]-v)));let lo=Math.min(...values),hi=Math.max(...values);ctx.clearRect(0,0,1100,450);ctx.fillStyle='#e0e5ed';ctx.fillRect(60,85,970/6,275);ctx.fillRect(60+970*5/6,85,970/6,275);ctx.fillStyle='#526171';ctx.font='16px system-ui';ctx.fillText('x →；参考域：[0.125,0.875] × [0.375,0.625] m',35,30);ctx.strokeStyle='#ccd5df';ctx.strokeRect(60,85,970,275);data.X.forEach((X,i)=>{let p=X.map((v,k)=>v+amp*(data.x[n][i][k]-v));let z=(values[i]-lo)/Math.max(hi-lo,1e-12);ctx.fillStyle=`hsl(${240-240*z},75%,45%)`;ctx.beginPath();ctx.arc(60+970*(p[0]-.125)/.75,360-1100*(p[1]-.375),4,0,7);ctx.fill();});document.getElementById('status').textContent=`帧 ${n}/4 | t=${data.time[n].toFixed(4)} s | 显示放大 ${amp}× | 颜色范围 ${lo.toPrecision(5)} — ${hi.toPrecision(5)}`;}
for(const id of ['frame','scale','color'])document.getElementById(id).addEventListener('input',draw);document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){slider.value=(Number(slider.value)+1)%data.time.length;draw();}},350);draw();</script></html>'''
    (run/'index.html').write_text(page.replace('PAYLOAD',json.dumps(payload,separators=(',',':'))));print(run/'index.html')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);render(p.parse_args().run)
