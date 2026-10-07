"""Standalone plots for v15 attribution and branch/cycle comparisons."""
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from benchmarks.aniso_compatible_diagnosis import BASE
OUT=BASE/'v15'


def main():
    d=json.loads((OUT/'diagnosis/summary.json').read_text());r=[q for q in d['records'] if q['level']=='fourth']
    t=np.array([q['time'] for q in r]);fig,ax=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    bottom=np.zeros(len(r))
    for key,label,color in [('exterior','Exterior virtual support','#d67738'),('grip_transition','Grip / transition','#9a68a8'),('interior','Interior','#338cb8')]:
        y=np.array([q['regions'][key]['stabilization_row_J']/q['stabilization_J'] for q in r])*100
        ax[0].bar(np.arange(len(r)),y,bottom=bottom,label=label,color=color);bottom+=y
    ax[0].set_xticks(np.arange(len(r)),[f'{v:g}' for v in t]);ax[0].set(xlabel='Time (s)',ylabel='Patch-row energy share (%)',title='Allocation convention, not material density');ax[0].legend(fontsize=8)
    ax[1].plot(t,[q['strain_rms'] for q in r],'-o',label='Particle strain RMS')
    ax[1].plot(t,[q['history_F_rms'] for q in r],'-o',label='F - G0 Y RMS')
    ax[1].set(yscale='log',xlabel='Time (s)',ylabel='Dimensionless RMS',title='Discrete history mismatch');ax[1].legend(fontsize=8);ax[1].grid(alpha=.2)
    z=np.load(OUT/'diagnosis/fourth-1.6.npz');xy=z['marker_X'][:,:2];xyu,inv=np.unique(xy,axis=0,return_inverse=True);E=np.bincount(inv,weights=z['marker_energy_J'])*1e6
    p=ax[2].scatter(xyu[:,0],xyu[:,1],c=E,s=120,norm=LogNorm(vmin=max(E.min(),1e-6),vmax=E.max()),cmap='magma')
    ax[2].add_patch(Rectangle((.125,.375),.75,.25,fill=False,color='#168164',linewidth=2));ax[2].axvline(.25,ls='--',color='.5');ax[2].axvline(.75,ls='--',color='.5')
    ax[2].set(xlabel='Reference x (m)',ylabel='Reference y (m)',title='t=1.6 s: sum over z; green = material');fig.colorbar(p,ax=ax[2],label='Allocated energy (microjoule)')
    fig.savefig(OUT/'energy-location.png',dpi=170);fig.savefig(OUT/'energy-location.pdf');plt.close(fig)
    if not (OUT/'summary.json').exists():return
    s=json.loads((OUT/'summary.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained');colors=['#d67738','#338cb8']
    for i,mode in enumerate(('baseline','compatible')):
        phases=('unload','early_hold','late_hold');values=[100*s['branch_refinement'][p][mode][-1]['P_relative'] for p in phases]
        ax[0].bar(np.arange(3)+(i-.5)*.32,values,width=.32,color=colors[i],label=mode)
        phases2=('ramp','loaded_hold','unload','final_hold');values=[100*s['two_level_full_cycle_refinement'][mode][p]['P_relative'] for p in phases2]
        ax[1].bar(np.arange(4)+(i-.5)*.32,values,width=.32,color=colors[i],label=mode)
    ax[0].set_xticks(np.arange(3),['Unload','Early hold','Late hold']);ax[0].set(title='Same-state 0.05 s branches: last dt pair',ylabel='Stress RMS time difference (%)')
    ax[1].set_xticks(np.arange(4),['Load','Loaded hold','Unload','Final hold']);ax[1].set(title='Fresh full cycles: coarse/fine pair only',ylabel='Stress RMS time difference (%)')
    for a in ax:a.legend();a.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'time-comparison.png',dpi=170);fig.savefig(OUT/'time-comparison.pdf');plt.close(fig)

if __name__=='__main__':main()
