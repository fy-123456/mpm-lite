"""Actual scoped-cycle curves and interactive XY frames, no fabricated data."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build(root):
    if (root/'release.json').exists():raise ValueError('sealed result is read only')
    folder=root/'S3/director-cycle';rows=json.loads((folder/'rows.json').read_text());frames=json.loads((folder/'frames.json').read_text())
    with np.load(folder/'terminal.npz') as d:X=d['X'].copy()
    t=[r['time'] for r in rows];fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    ax=axes[0,0];ax.plot(t,[r['max_particle_displacement']*1000 for r in rows],'o-');ax.set(xlabel='Time [s]',ylabel='Sampled displacement [mm]',title='12-step material-coordinate cycle')
    ax=axes[0,1];ax.plot(t,[r['p_min'] for r in rows],label='min');ax.plot(t,[r['p_max'] for r in rows],label='max');ax.axhline(0,color='grey',lw=.6);ax.legend();ax.set(xlabel='Time [s]',ylabel='Gauge pressure [Pa]',title='Unclipped pressure')
    ax=axes[1,0]
    for key,label in [('energy_final','Stored + kinetic'),('darcy_dissipation','Cumulative Darcy'),('external_work','Cumulative work')]:
        y=np.array([r[key] for r in rows]);ax.plot(t,y if key=='energy_final' else np.cumsum(y),label=label)
    ax.legend(fontsize=8);ax.set(xlabel='Time [s]',ylabel='Energy [J]',title='Actual unsmoothed ledger')
    from engine.aniso_phase1.research_unified_lite_poro.space import Space
    s=Space();line=np.c_[np.linspace(0,1,100),np.full(100,.125),np.full(100,.125)];N,_=s.basis(line);ax=axes[1,1]
    for case,label in [('director-short-ppc2','PPC2 dt=.005'),('director-short-ppc4','PPC4 dt=.005'),('director-quarterdt','PPC4 dt=.00125')]:
        with np.load(root/'S3'/case/'terminal.npz') as d:ax.plot(line[:,0],1000*(N@d['q'])[:,1],label=label)
    ax.legend(fontsize=8);ax.set(xlabel='Reference x [m]',ylabel='Displacement y [mm]',title='Common line at t=.02 s; phase not certified')
    fig.suptitle('Bounded director reconstruction | CPU material bridge | Eulerian integration pending')
    fig.savefig(root/'summary.png',dpi=140);plt.close(fig)
    payload=json.dumps(dict(X=X.tolist(),frames=frames),allow_nan=False)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>MPM-Lite 本轮优化结果</title>
<style>body{font:16px system-ui;color:#182635;max-width:1120px;margin:30px auto;padding:0 20px;background:#f6f8fa}.note{background:#fff2cc;padding:16px}canvas{width:100%;background:white}img{max-width:100%}label{margin-right:24px}a{color:#1458a2}</style>
<h1>方向重建与原 Lite 接口验证</h1><p class="note">本图展示 CPU 材料坐标小模型，54 粒子、128 材料点、12 步至 0.06 s。Eulerian 耦合集成未通过；时间相位仍敏感。局部平滑单方向候选不代表任意方向分布或正式 144 空间。</p>
<p>旧诊断状态材料力差：13.44% → 0.182%。这是同空间积分参考下的材料误差，不是连续空间真解精度。</p>
<label>时间帧 <input id="frame" type="range" min="0" max="3" value="3"></label><label>形变显示倍数 <input id="scale" type="range" min="1" max="30" value="10"></label><span id="info"></span>
<canvas id="view" width="1050" height="400"></canvas><p>灰点为参考位置，蓝点为实际位移的 XY 投影。倍数仅改变显示；没有插值生成模拟帧。</p><img src="summary.png" alt="位移、压力、能量和时间步对照">
<p><a href="S1/legacy-state-recheck.json">材料误差</a> · <a href="S2/original-kernel-probes.json">原内核跨格与接口问题</a> · <a href="S3/common-site-comparisons.json">时间步和粒子采样</a> · <a href="requirement-audit.json">逐项实施状态</a> · <a href="implementation-report.md">实施记录</a></p>
<script>const data=PAYLOAD;const slider=document.getElementById('frame'),scale=document.getElementById('scale'),canvas=document.getElementById('view'),ctx=canvas.getContext('2d');slider.max=data.frames.length-1;slider.value=slider.max;function draw(){const f=data.frames[+slider.value],s=+scale.value;ctx.clearRect(0,0,1050,400);const pt=x=>[55+x[0]*940,340-x[1]*1000];data.X.forEach((x,i)=>{const a=pt(x),b=pt(x.map((v,k)=>v+s*(f.x[i][k]-v)));ctx.fillStyle='#b8bec5';ctx.beginPath();ctx.arc(...a,3,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#c1cedc';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.fillStyle='#2464be';ctx.beginPath();ctx.arc(...b,4,0,Math.PI*2);ctx.fill()});document.getElementById('info').textContent=`t=${f.time.toFixed(3)} s; ×${s}; p=${f.pressure.map(v=>v.toFixed(4)).join(', ')} Pa`};slider.oninput=draw;scale.oninput=draw;draw();</script></html>'''.replace('PAYLOAD',payload)
    (root/'index.html').write_text(html)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run)
