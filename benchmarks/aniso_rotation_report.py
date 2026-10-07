"""Plot archived results without rerunning GPU simulations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('docs/results/rotation-dissipation')
    read=lambda name:np.genfromtxt(root/(name+'.csv'),delimiter=',',names=True)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    colors={'none':'#3282ce','supplemental':'#d74b32','hourglass':'#299d61'}
    for ax,grid in zip(axes[0,:2],(17,33)):
        for mode,c in colors.items():
            name=f'rotation-g{grid}-dt0.01-{mode}';a=read(name);E0=read(name+'-budget')['elastic'][0]
            ax.plot(a['stabilization_sampling_angle_degrees'],100*((a['center_material_energy']+a['stabilization_energy'])/E0-1),'-o',color=c,label=mode,ms=3)
        ax.set(title=f'Exact bent rotation, grid {grid}',xlabel='Step-start sampling angle (deg)',ylabel='Elastic change (%)');ax.legend(fontsize=8);ax.grid(alpha=.3)
    release=json.loads((root/'release.json').read_text());ax=axes[0,2]
    for j,mode in enumerate(('none','supplemental')):
        group=[r for r in release if r['stabilization']==mode]
        ax.bar(np.arange(4)+(j-.5)*.35,[100*r['relative_energy_change'] for r in group],.35,label=mode,color=colors[mode])
    ax.set_xticks(range(4),['dt=.001\nf=.9','dt=.0005\nf=.9','dt=.001\nf=0','dt=.001\nf=1']);ax.set(title='Release at T=.008',ylabel='Mechanical change (%)');ax.legend(fontsize=8)
    ax=axes[1,0];groups=['solve','velocity_transfer','history_reconstruction','boundary']
    for j,r in enumerate([r for r in release if r['stabilization']=='supplemental'][:2]):
        ax.bar(np.arange(4)+(j-.5)*.35,[100*r['budget_relative'][k] for k in groups],.35,label=f"dt={r['dt']}")
    ax.set_xticks(range(4),['solve','transfer','history','boundary']);ax.set(title='Supplemental release budget',ylabel='Contribution / initial E (%)');ax.legend(fontsize=8)
    for kind,style in [('center','-'),('particle','--')]:
        a=read(f'tensile-{kind}-supplemental-T0.16')[1:]
        axes[1,1].plot(1000*a['displacement'],a['right_force'],style,label=kind)
    axes[1,1].set(title='Crossed fibers, slow two cycles',xlabel='Prescribed displacement (x 0.001)',ylabel='Actuator force');axes[1,1].legend(fontsize=8);axes[1,1].grid(alpha=.3)
    for T in (.08,.16):
        a=read(f'tensile-center-none-T{T}')[1:]
        axes[1,2].plot(a['time']/T,a['right_force'],label=f'loading time {T}')
    axes[1,2].set(title='Loading-rate effect (center M4)',xlabel='Time / loading time',ylabel='Actuator force');axes[1,2].legend(fontsize=8);axes[1,2].grid(alpha=.3)
    fig.savefig(root/'summary.png',dpi=160);fig.savefig(root/'summary.svg');plt.close(fig)


if __name__=='__main__':main()
