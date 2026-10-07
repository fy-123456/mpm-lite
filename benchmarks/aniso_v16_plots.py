"""Export shareable plots of the sealed-data comparisons."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_v16_experiments import OUT,load
from benchmarks.aniso_v16_analysis import rows


def main():
    a=load(OUT/'summary.json');b=load(OUT/'avf/summary.json');colors=['#607d8b','#9c755f','#d97732','#147d92']
    names=['Original split','Common-gradient split','Joint backward Euler','Joint AVF']
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');x=np.arange(2)
    for j,mode in enumerate(('legacy_split','common_split','joint','avf')):
        summary=b if mode=='avf' else a
        values=[100*summary['refinement'][f'{label}-moving-{mode}'][-1]['P_relative'] for label in ('early_hold','late_hold')]
        bars=ax.bar(x+(j-1.5)*.2,values,.19,label=names[j],color=colors[j])
        ax.bar_label(bars,fmt='%.2f',fontsize=9)
    ax.axhline(2,color='#b33b3b',ls='--',lw=1,label='2% gate');ax.set_xticks(x,['1.10–1.15 s','1.60–1.65 s']);ax.set_ylabel('Stress RMS difference (%)')
    ax.set_title('Moving particles: last timestep pair (0.00025 / 0.000125 s)');ax.legend(ncol=2,fontsize=9);ax.set_ylim(0,20)
    for ext in ('png','pdf'):fig.savefig(OUT/f'time-comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,label in enumerate(('early_hold','late_hold')):
        start=1.1 if col==0 else 1.6
        for j,mode in [(0,'legacy_split'),(2,'joint'),(3,'avf')]:
            root=OUT/'avf' if mode=='avf' else OUT
            rr=rows(root/'cases'/f'{label}-frozen-{mode}-fourth/steps.jsonl');t=np.array([r['time'] for r in rr])-start
            total=np.array([r['total_J'] for r in rr]);initial=total[0]-rr[0]['delta_total_J']
            axes[0,col].plot(t,[r['stress_rms_Pa'] for r in rr],label=names[j],color=colors[j])
            axes[1,col].plot(t,100*(total/initial-1),label=names[j],color=colors[j])
        axes[0,col].set_title(f'Fixed geometry: start at {start:.2f} s')
        axes[0,col].set_ylabel('Stress RMS (Pa)');axes[1,col].set_ylabel('Total energy change (%)');axes[1,col].set_xlabel('Elapsed time (s)')
        axes[0,col].legend(fontsize=8);axes[1,col].axhline(0,color='black',lw=.5)
        for row in range(2):axes[row,col].grid(alpha=.2)
    for ext in ('png','pdf'):fig.savefig(OUT/f'energy-response.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for j in range(2):
        be=a['modal_controls'][j]['time_comparison'];avf=b['exact_linear_controls'][j]['records']
        axes[j].loglog([r['dt'] for r in be],[100*r['nonlinear_vs_linear_exact_relative'] for r in be],'o-',color=colors[2],label='Nonlinear joint BE')
        axes[j].loglog([r['dt'] for r in avf],[100*r['nonlinear_vs_linear_exact_relative'] for r in avf],'s-',color=colors[3],label='Nonlinear joint AVF')
        axes[j].axhline(2,color='#b33b3b',ls='--',lw=1);axes[j].set_title(f'Fixed geometry: start {a["modal_controls"][j]["time"]:.2f} s')
        axes[j].set_xticks([.000125,.00025,.0005,.001], ['1.25e-4','2.5e-4','5e-4','1e-3']);axes[j].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter());axes[j].set_xlabel('Timestep (s)');axes[j].set_ylabel('Difference from exact linear-time control (%)');axes[j].grid(alpha=.2,which='both');axes[j].legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(OUT/f'linear-time-control.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':main()
