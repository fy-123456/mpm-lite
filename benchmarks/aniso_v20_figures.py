"""Export raw numerical evidence; no temporal smoothing."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_v20_analysis import rows

def main():
    dest=OUT/'figures';dest.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.22,'savefig.dpi':180})
    clusters=load(OUT/'reaction-clusters.json')['records']['1.40-sampled']['groups'][:8];fig,ax=plt.subplots(1,2,figsize=(10,3.5));names=[','.join(map(str,r['modes'])) for r in clusters];pos=np.arange(len(names));ax[0].barh(pos,[r['reaction_rms_N'] for r in clusters]);ax[0].set_yticks(pos,names);ax[0].invert_yaxis();ax[0].set_xlabel('Coherent group reaction RMS [N]');ax[0].set_ylabel('Local mode group at t = 1.4 s');ax[1].scatter([r['omega_rad_s'][0] for r in clusters],pos,c=[r['min_stabilization_fraction'] for r in clusters],cmap='viridis',vmin=0,vmax=1);ax[1].set_xscale('log');ax[1].set_yticks(pos,names);ax[1].invert_yaxis();ax[1].set_xlabel('Angular frequency [rad/s]');fig.suptitle('Reaction-important modes; cross terms retained within each group');fig.tight_layout();fig.savefig(dest/'reaction-modes.png');plt.close(fig)
    a=load(OUT/'local-space.json')['records'];b=load(OUT/'local-relaxation.json')['records'];data=[a[f'F45-r32-e{i}'] for i in (0,1,2,3)]+[b[f'r32-w{w}'] for w in (.125,.1875,.25)];labels=['Base','6 local','54 local','150 local','FE strip 0.125','FE strip 0.1875','Full interior'];fig,ax=plt.subplots(1,2,figsize=(11,3.8));x=np.arange(len(data));ax[0].plot(x,[r['reaction_relative']*100 for r in data],'o-',label='reaction');ax[0].axhline(2,color='red',linestyle='--',label='2% target');ax[0].set_ylabel('Relative difference [%]');ax[0].legend()
    for region in ('global','grip','interior'):ax[1].plot(x,[r['stress_relative'][region]*100 for r in data],'o-',label=region)
    ax[1].axhline(2,color='red',linestyle='--');ax[1].legend();ax[1].set_ylabel('Stress difference [%]')
    for a in ax:a.set_xticks(x,labels,rotation=28,ha='right')
    fig.suptitle('F45 vs local Q3: original patch coefficient retained; Q3 stress not certified');fig.tight_layout();fig.savefig(dest/'local-space.png');plt.close(fig)
    paths={n:ROOT/r['path'] for n,r in load(OUT/'completed-case-paths.json')['cases'].items()};fig,ax=plt.subplots(2,2,figsize=(11,6));colors=['#805ad5','#dd6b20','#3182ce','#238b45']
    for i in range(4):
        name=f'gauss3-condensed-L{i}';r=rows(paths[name]);t=np.array([v['time'] for v in r]);R=np.array([v['reaction_N'] for v in r]);P=np.array([v['stress_rms_Pa'] for v in r]);label=f"dt = {r[0]['dt']*1e6:g} us";ax[0,0].plot(t,R,lw=.8,c=colors[i],label=label);mask=t>1.1;ax[0,1].plot(t[mask],R[mask],lw=.7,c=colors[i],label=label);ax[1,0].plot(t,P,lw=.8,c=colors[i]);mask=(t>1.38)&(t<1.42);ax[1,1].plot(t[mask],R[mask],lw=.8,c=colors[i])
    for a in ax.ravel():a.set_xlabel('Time [s]')
    ax[0,0].set_ylabel('Raw reaction [N]');ax[0,1].set_ylabel('Final hold raw reaction [N]');ax[1,0].set_ylabel('Material stress RMS [Pa]');ax[1,1].set_ylabel('Raw reaction zoom [N]');ax[0,0].legend();fig.suptitle('Complete moving cycle: all raw samples, no smoothing');fig.tight_layout();fig.savefig(dest/'raw-cycle.png');plt.close(fig)
    r=rows(paths['gauss3-condensed-L3']);t=np.array([v['time'] for v in r]);fig,ax=plt.subplots(1,2,figsize=(11,3.5))
    for k,label in [('material_J','material'),('stabilization_J','stabilization'),('kinetic_J','kinetic'),('total_J','total')]:ax[0].plot(t,[v[k] for v in r],label=label,lw=1)
    for k,label in [('boundary_work_J','boundary work'),('metric_change_J','metric change'),('constraint_kinetic_loss_J','constraint loss')]:ax[1].plot(t,np.cumsum([v[k] for v in r]),label=label,lw=1)
    ax[1].set_yscale('symlog',linthresh=1e-15)
    for a in ax:a.set_xlabel('Time [s]');a.set_ylabel('Energy [J]');a.legend()
    fig.suptitle('Finest moving cycle: separate energy channels and cumulative work');fig.tight_layout();fig.savefig(dest/'energy-ledger.png');plt.close(fig)
    acceptance=load(OUT/'cycle-acceptance.json');pairs=acceptance['pairs'];dt=np.array([250.,125.,62.5]);fig,ax=plt.subplots(1,2,figsize=(11,3.8))
    values={'raw reaction / full cycle':[r['raw_reaction_relative'] for r in pairs],'raw reaction / final hold':[r['stages']['final_hold']['raw_reaction_relative'] for r in pairs],'stress / full cycle':[r['relative'] for r in pairs],'stress / terminal':[r['terminal_relative'] for r in pairs]}
    for label,v in values.items():ax[0].loglog(dt,100*np.array(v),'o-',label=label)
    ax[0].axhline(2.,color='red',ls='--',label='2% target');ax[0].set_xlabel('Fine dt of adjacent pair [us]');ax[0].set_ylabel('Relative difference [%]');ax[0].legend(fontsize=7)
    losses=[acceptance['cases'][f'gauss3-condensed-L{i}']['sums_J']['constraint_kinetic_loss_J'] for i in range(4)];ax[1].semilogx([500,250,125,62.5],losses,'o-');ax[1].set_ylim(0,max(losses)*1.12);ax[1].set_xlabel('dt [us]');ax[1].set_ylabel('Accumulated constraint loss [J]');fig.suptitle('Time convergence and the separately recorded constraint loss');fig.tight_layout();fig.savefig(dest/'time-convergence.png');plt.close(fig)

if __name__=='__main__':main()
