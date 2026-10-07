"""Standalone raw-curve PNG + small interactive physical-frame viewer."""
import json,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    root=Path(root)
    if (root/'release.json').exists():raise ValueError('sealed view cannot be overwritten')
    folder=root/'S3/fixed-cycle-ppc3';rows=json.loads((folder/'rows.json').read_text());frames=json.loads((folder/'frames.json').read_text())
    state=np.load(folder/'terminal.npz');X=state['X'];t=np.array([r['time'] for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    ax=axes[0,0]
    ax.plot(t,np.array([r['max_particle_displacement'] for r in rows])*1000,'o-');ax.set(xlabel='Time [s]',ylabel='Max sampled displacement [mm]',title='Particle motion (same PPC trajectory)')
    ax=axes[0,1]
    ax.plot(t,[r['p_min'] for r in rows],'o-',label='min');ax.plot(t,[r['p_max'] for r in rows],'o-',label='max');ax.axhline(0,color='grey',lw=.7);ax.legend();ax.set(xlabel='Time [s]',ylabel='Gauge pressure [Pa]',title='Pressure: signed raw values')
    ax=axes[1,0]
    ax.plot(t,[r['energy_final'] for r in rows],label='mechanical + storage');ax.plot(t,np.cumsum([r['darcy_dissipation'] for r in rows]),label='cumulative Darcy loss');ax.plot(t,np.cumsum([r['external_work'] for r in rows]),label='cumulative external work');ax.legend(fontsize=8);ax.set(xlabel='Time [s]',ylabel='Energy [J]',title='Unsmoothed energy terms')
    ax=axes[1,1]
    a=np.load(root/'S3/fixed-short-ppc2/terminal.npz');b=np.load(root/'S3/fixed-short-ppc4/terminal.npz')
    from engine.aniso_phase1.research_unified_lite_poro.space import Space
    space=Space();line=np.c_[np.linspace(0,1,100),np.full(100,.125),np.full(100,.125)];N,_=space.basis(line)
    ax.plot(line[:,0],1000*(N@a['q'])[:,1],label='16 particles');ax.plot(line[:,0],1000*(N@b['q'])[:,1],'--',label='128 particles');ax.legend();ax.set(xlabel='Reference x [m]',ylabel='y displacement [mm]',title='Same physical line, t = 0.02 s')
    fig.suptitle('Scoped CPU particle / enrichment / Darcy bridge | no Eulerian or spatial-accuracy certification')
    fig.savefig(root/'summary.png',dpi=150);plt.close(fig)
    data=json.dumps(dict(X=X.tolist(),frames=frames),allow_nan=False)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>统一粒子耦合研究验证</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;color:#182635;background:#f6f8fa}canvas{width:100%;background:white;border:1px solid #ccd7df}img{max-width:100%}.note{padding:16px;background:#fff4d4}label{margin-right:24px}code{background:#eef;padding:3px}</style>
<h1>粒子—富集—渗流：CPU 工程验证</h1>
<p class="note">固定材料坐标、完整笛卡尔参考粒子模板。实际更新粒子与下一步历史；尚未接入原 Lite 的 Eulerian 网格重定位，未认证连续空间精度，未替换正式 144 函数空间或 BD。</p>
<p>54 粒子，128 固定材料积分点，12 个实际接受步，物理时间 0.06 s。压力为有符号表压。</p>
<label>时间帧 <input id="frame" type="range" min="0" max="3" value="3"></label>
<label>形变显示倍数 <input id="scale" type="range" min="1" max="30" value="10"></label><span id="info"></span>
<canvas id="view" width="1050" height="400"></canvas>
<p>灰点为参考位置，蓝点为当前位置的 XY 投影。显示倍数只作用于形变，不改变物理数据。</p>
<img src="summary.png" alt="原始位移、压力、能量与共同位置对照曲线">
<p><a href="capability-matrix.json">能力与限制</a> · <a href="requirement-audit.json">28 步实施状态</a> · <a href="S6/common-site-ppc.json">共同位置对照</a> · <a href="S3/fixed-cycle-ppc3/rows.json">逐步原始记录</a></p>
<script>const data=DATA;const slider=document.getElementById('frame'),scale=document.getElementById('scale'),canvas=document.getElementById('view'),ctx=canvas.getContext('2d');
slider.max=data.frames.length-1;slider.value=slider.max;
function draw(){let f=data.frames[+slider.value],s=+scale.value;ctx.clearRect(0,0,1050,400);function pt(x){return [55+x[0]*940,340-x[1]*1000]};data.X.forEach((x,i)=>{let a=pt(x),b=pt(x.map((v,k)=>v+s*(f.x[i][k]-v)));ctx.fillStyle='#b8bec5';ctx.beginPath();ctx.arc(...a,3,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#bac8dd';ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.fillStyle='#2464be';ctx.beginPath();ctx.arc(...b,4,0,Math.PI*2);ctx.fill()});document.getElementById('info').textContent=`t=${f.time.toFixed(3)} s; 显示 ×${s}; p=${f.pressure.map(v=>v.toFixed(4)).join(', ')} Pa`};slider.oninput=draw;scale.oninput=draw;draw();</script></html>'''.replace('DATA',data)
    (root/'index.html').write_text(html)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);main(p.parse_args().run)
