"""Standalone figures for the material-reference energy audit."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_material_history import OUT,BASE,load
from benchmarks.aniso_dynamic_space import stress

def main():
    s=load(OUT/'summary.json');p=load(OUT/'protocol.json');fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for mode in ('baseline','material'):
        name=mode+'-fourth';cfg=p['configs'][name]
        with np.load(OUT/'cases'/name/'frames.npz') as z:t=z['time'];P=stress(z['F'].reshape(-1,3,3),cfg).reshape(z['F'].shape)
        norm=np.sqrt(np.mean(np.sum(P*P,axis=(2,3)),axis=1));rows=[__import__('json').loads(l) for l in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()];tr=[r['time'] for r in rows]
        axs[0,0].plot(t,norm,label=mode);axs[0,1].plot(tr,[r['right_force'] for r in rows],label=mode)
        axs[1,0].plot(tr,np.array([r['mechanical'] for r in rows])*1e6,label=mode)
        mask=t>=1.1-1e-10;axs[1,1].plot(t[mask],norm[mask]/norm[np.argmin(abs(t-.5))],label=mode)
    for ax,title,y in zip(axs.flat,['F45 stress, dt=0.000125 s','Actuator reaction','Mechanical energy','Post-unload recovery'],['PK1 RMS (Pa)','R (N)','E (microjoule)','P RMS / loaded P RMS']):
        ax.set(title=title,xlabel='physical time (s)',ylabel=y);ax.grid(alpha=.25);ax.legend()
        for t0 in (.5,.6,1.1):
            if ax is not axs[1,1]:ax.axvline(t0,color='gray',lw=.7,ls=':')
    fig.savefig(OUT/'cycle-recovery.png',dpi=160);fig.savefig(OUT/'cycle-recovery.pdf');plt.close(fig)
    phases=['ramp','hold','unload','early_hold','late_hold','final_hold'];fig,axs=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True);xx=np.arange(len(phases))
    for mode,shift in [('baseline',-.18),('material',.18)]:
        axs[0].bar(xx+shift,[100*s['refinement'][mode][phase]['pairs'][-1]['P_relative'] for phase in phases],.36,label=mode)
        for key,ls in [('P','-'),('F','--')]:axs[1].plot([.0005,.00025,.000125],[100*r[key+'_relative'] for r in s['refinement'][mode]['whole']['pairs']],ls+'o',label=mode+' '+key)
    axs[0].set_xticks(xx,phases,rotation=25);axs[0].set(ylabel='last adjacent stress difference (%)',title='Phase errors');axs[1].set(xlabel='finer dt (s)',ylabel='whole-cycle adjacent difference (%)',title='Four-step refinement');axs[1].set_xscale('log')
    for ax in axs:ax.axhline(2,color='k',ls=':',label='2% gate');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'time-errors.png',dpi=160);plt.close(fig)
    space=load(BASE/'v14-space/summary.json')['records'];fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for quad,order in [('midpoint',2),('midpoint',4),('gauss',2),('gauss',4)]:
        rr=[r for r in space if r['quadrature']==quad and r['order']==order];h=[1/(r['grid']-1) for r in rr]
        for ax,key in zip(axs,('interior','near_grip')):ax.plot(h,[100*r['regions'][key]['P_relative'] for r in rr],'-o',label=f'{quad} {order}/axis');ax.set(title=key,xlabel='h (m)',ylabel='stress difference vs local Q3 (%)')
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(OUT/'spatial-stress.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
