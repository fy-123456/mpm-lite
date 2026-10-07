"""Show diagnostic reference and measured costs with explicit inherited fields."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .provenance import *

def main(run):
    run=Path(run);mutable(run);out=run/'figures';out.mkdir(exist_ok=True)
    a=dict(np.load(run/'S1/reference-r12.npz'));b=dict(np.load(run/'S1/reference-r8.npz'));fixed=dict(np.load(APP/'S1/reference-YZ128.npz'));ts=a['times']*1e6
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,k in zip(axes,('Ay','Az')):
        ax.plot(ts,(a[k]-a[k][0])*1e6,'o-',label='coupled affine r12 (diagnostic)')
        ax.plot(ts,(b[k]-b[k][0])*1e6,'x--',label='coupled affine r8 (diagnostic)')
        ax.plot(ts,(a['actual_'+k]-a['actual_'+k][0])*1e6,'s:',label='inherited nonlinear states')
        ids=fixed['times']<=a['times'][-1]+1e-15
        ax.plot(fixed['times'][ids]*1e6,(fixed[k][ids]-fixed[k][0])*1e6,label='fixed skeleton (different model)')
        ax.set(xlabel='Time (microseconds)',ylabel=f'Change in {k} (micro-Pa)',title='Moving-volume coupling explains mode evolution');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(out/'coupled-reference.png',dpi=140);plt.close(fig)
    valid=read(run/'S1/local-validity.json');fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for rec in valid['records']:
        axes[0].plot(ts,np.array(rec['displacement_error_m'])*1e9,'o-',label=f'r{rec["rank"]}')
        axes[1].semilogy([r['time_s']*1e6 for r in rec['projection_residual']],[r['acceleration_projection_relative'] for r in rec['projection_residual']],'o-',label=f'r{rec["rank"]}')
    axes[0].set(xlabel='Time (microseconds)',ylabel='Max probe displacement difference (nm)',title='Difference includes inherited time error')
    axes[1].set(xlabel='Time (microseconds)',ylabel='Mass-norm acceleration projection residual',title='Sampled residual; not a global error bound')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'reference-limits.png',dpi=140);plt.close(fig)
    perf=read(run/'S3/paired-performance.json');d=read(run/'S3/backend-decision.json');fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True);x=np.arange(2)
    ax.bar(x-.18,[r['BD_s'] for r in perf['records']],.36,label='BD (retained baseline)');ax.bar(x+.18,[r['FBD_s'] for r in perf['records']],.36,label='FBD (one candidate)');ax.set(xticks=x,xticklabels=['input state 8','input state 16'],ylabel='Whole step + probes + IO (s)',title='Two paired inputs; adoption follows registered gates');ax.legend();fig.savefig(out/'performance.png',dpi=140);plt.close(fig)
    inherited=[]
    for name in ('inherited-physical-fields.png','flow-energy.png'):
        p=APP/'figures'/name;(out/name).write_bytes(p.read_bytes());inherited.append(dict(path=str(p),sha256=sha(p),new_path='figures/'+name))
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>MPM-lite 运动骨架耦合参考与 BD 成本</title><style>body{font:17px/1.7 sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;color:#173140;background:#f7fafc}img{width:100%;background:white;border:1px solid #ccd9df;border-radius:8px}section{margin:28px 0}code{overflow-wrap:anywhere}.note{background:#fff0d9;padding:16px;border-radius:8px}</style><h1>运动骨架耦合参考与 BD 成本</h1>'''
    html+=f'<p>基线发布：<code>{APP_SHA}</code>。实际保留后端：<b>{d["backend"]}</b>。</p>'
    html+='<p class="note">新参考含固体质量与惯性、体积交换、压力几何项和Darcy几何项，采用8/12维固体诊断子空间，压力保留128格。它通过局部工程检查，但不代表完整非线性或连续空间精度。没有扩窗、没有新增物理帧，也没有更改正式144函数空间。</p>'
    html+=f'<p>FBD算子等价通过；两组整步耗时减少 {perf["records"][0]["gain_fraction"]:.2%} / {perf["records"][1]["gain_fraction"]:.2%}，未满足稳定采用条件，继续使用BD。不将候选局部收益外推为完整周期加速。</p>'
    for title,name,caption in [('耦合压力模式','coupled-reference.png','标记点为已有时刻，连线仅辅助阅读；固定骨架是不同物理问题。'),('参考的局部误差与边界','reference-limits.png','误差包含旧轨迹时间误差。位移以纳米表示；采样残差小不是连续空间误差证明。'),('实际整步成本','performance.png','同输入、同预热、独立进程，构造成本单独记录。'),('继承实体场','inherited-physical-fields.png','来自父发布75微秒场；压力、位移与总应力未平滑，新增物理帧为0。'),('继承流量与能量账本','flow-energy.png','图中固定骨架与运动固体分开标注；Darcy物理耗散与启动数值耗散分别展示。')]:
        html+=f'<section><h2>{title}</h2><p>{caption}</p><img src="figures/{name}" alt="{title}"></section>'
    html+='<p><a href="S1/reference-self-check.json">参考自检</a> · <a href="S1/holdout-review.json">留出状态检查</a> · <a href="S3/paired-performance.json">配对性能</a> · <a href="S5/physical-ledger-audit.json">实际步账本</a> · <a href="capability-matrix.json">能力范围</a></p></html>'
    (run/'index.html').write_text(html)
    write(run/'S5/visualization-origin.json',dict(new_physical_frames=0,inherited=inherited,new_reference=['S1/reference-r8.npz','S1/reference-r12.npz'],smoothing=False,browser_interaction=False))
    write(run/'S5/visual-assets-check.json',dict(status='passed_scoped',images=[dict(path=str(p.relative_to(run)),sha256=sha(p)) for p in sorted(out.glob('*.png'))]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
