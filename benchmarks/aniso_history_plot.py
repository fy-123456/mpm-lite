"""Render archived audits; no GPU simulation is rerun."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('docs/results/history-reference')
    def rows(name):
        with (root/(name+'.csv')).open() as f:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    fig,ax=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for policy,label in [('grid_locked','Grid-locked history'),('particle_resample','Particle resampling')]:
        data=rows('translate-g65-dt0.05-'+policy+'-exact')
        ax[0,0].plot([r['time'] for r in data],[100*r['elastic_relative_error'] for r in data],'o-',label=label)
    ax[0,0].set(title='Prescribed translation: elastic energy error',xlabel='Time (s)',ylabel='Error (%)')
    for dt,rule in [(.05,'exact'),(.05,'euler'),(.025,'euler')]:
        data=rows(f'rotate-g65-dt{dt}-particle_resample-{rule}')
        ax[0,1].plot([r['time'] for r in data],[100*r['integration_energy_relative'] for r in data],label=f'{rule}, dt={dt}')
    ax[0,1].set(title='Rotation: error from the prescribed time update',xlabel='Time (s)',ylabel='Elastic energy change (%)')
    for T in (.04,.08,.16):
        data=rows(f'cycles-g9-T{T}-dt0.005')
        ax[1,0].plot([1000*r['displacement'] for r in data],[r['right_force'] for r in data],label=f'Tload={T} s')
    ax[1,0].set(title='Two loading cycles, fixed dt=0.005 s',xlabel='Commanded displacement (mm)',ylabel='Total grip reaction (N)')
    records=json.loads((root/'direction-mixtures.json').read_text())
    records=[r for r in records if r['deformation']=='stretch'];x=np.arange(len(records))
    for j,(key,label) in enumerate([('energy','Energy'),('stress','PK1 stress'),('tangent','Tangent')]):
        ax[1,1].bar(x+(j-1)*.24,[100*r[key+'_mean_A_relative_error'] for r in records],.24,label=label)
    ax[1,1].set(xticks=x,xticklabels=[r['direction_field'] for r in records],ylabel='Relative error (%)',title='Mean direction tensor: fiber-only errors')
    for a in ax.flat:a.legend(fontsize=8);a.grid(alpha=.2)
    fig.savefig(root/'summary.png',dpi=160);fig.savefig(root/'summary.svg');plt.close(fig)


if __name__=='__main__':main()
