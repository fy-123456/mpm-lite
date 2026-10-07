"""Plot archived stabilization and moment audits without rerunning simulation."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('docs/results/stabilization-moments')
    load=lambda n:json.loads((root/(n+'.json')).read_text())
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    static=load('static-beam')
    for mode in ('none','supplemental','hourglass'):
        r=[x for x in static if x['stabilization']==mode]
        ax[0,0].plot([x['grid'] for x in r],[x['soft_modes'] for x in r],'o-',label=mode)
    ax[0,0].set(title='Clamped beam: near-zero stiffness modes',xlabel='Grid resolution',ylabel='Mode count',xticks=[17,33])
    for grid,dt in ((17,.001),(17,.0005),(33,.001)):
        with (root/f'beam-g{grid}-dt{dt}-supplemental.csv').open() as f:rows=list(csv.DictReader(f))
        E0=float(rows[0]['mechanical'])
        ax[0,1].plot([float(x['time']) for x in rows],[100*(float(x['mechanical'])/E0-1) for x in rows],label=f'g{grid}, dt={dt}')
    ax[0,1].set(title='Short beam release: supplemental stabilization',xlabel='Time (s)',ylabel='Mechanical energy change (%)')
    moments=load('moments');fields=['uniform','crossed','smooth'];x=np.arange(3)
    for i,key in enumerate(('energy','stress','tangent')):
        y=[next(r for r in moments if r['field']==f and r['direction_model']=='mean_tensor')[key+'_relative_error']*100 for f in fields]
        ax[1,0].bar(x+(i-1)*.24,y,.24,label=key)
    ax[1,0].set(xticks=x,xticklabels=fields,title='Mean tensor error (whole material response)',ylabel='Relative error (%)')
    perf=load('performance')
    for i,(kind,model,label) in enumerate([('center','mean_tensor','Center, mean tensor'),('center','fourth_moment','Center, fourth moment'),('particle','mean_tensor','Particle quadrature')]):
        r=[r for r in perf if r['quadrature']==kind and r['direction_model']==model]
        ax[1,1].bar(np.arange(2)+(i-1)*.24,[v['median_step_ms'] for v in r],.24,label=label)
    ax[1,1].set(xticks=[0,1],xticklabels=['512 particles','4096 particles'],title='Same volume, mass, time and stabilization',ylabel='Median step time (ms)')
    for a in ax.flat:a.legend(fontsize=8);a.grid(alpha=.2)
    fig.savefig(root/'summary.png',dpi=160);fig.savefig(root/'summary.svg');plt.close(fig)


if __name__=='__main__':main()
