"""Standalone figures and an unsmoothed field animation from sealed raw data."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter


def render(out):
    with np.load(out/'cycle_dt_0.005.npz') as z:
        t=z['time'];p=z['pressure'];u=z['displacement'];stress=z['stress_mean'];nodes=z['nodes'];ledger=z['ledger'];cols=z['ledger_columns'].tolist()
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for index,label in [(9,'load'),(19,'hold'),(29,'unload'),(39,'final')]:
        axes[0,0].plot(np.arange(64),p[index],label=label)
    axes[0,0].set(title='Cell pressure at four phase ends',xlabel='Cell',ylabel='Pa');axes[0,0].legend()
    axes[0,1].plot(t,stress[:,:,0,0].mean(axis=1),label='mean total xx')
    axes[0,1].plot(t,np.sqrt(np.mean(stress**2,axis=(1,2,3))),label='total tensor RMS');axes[0,1].set(xlabel='s',ylabel='Pa',title='Corrected cell-mean stress');axes[0,1].legend()
    for k in ('physical_dissipation','numerical_loss','external_work'):
        axes[1,0].plot(ledger[:,cols.index('time')],np.cumsum(ledger[:,cols.index(k)]),label=k)
    axes[1,0].set(xlabel='s',ylabel='J',title='Separate physical/numerical energy ledger');axes[1,0].legend(fontsize=8)
    for k in ('local_mass_defect','energy_residual','residual_darcy'):
        axes[1,1].semilogy(ledger[:,cols.index('time')],np.maximum(abs(ledger[:,cols.index(k)]),1e-20),label=k)
    axes[1,1].set(xlabel='s',title='Per-step defects (declared units)');axes[1,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'scene-summary.png',dpi=160);plt.close(fig)
    result=json.loads((out/'E5_manufactured.json').read_text());rows=result['rows']
    fig,ax=plt.subplots(figsize=(8,5))
    for region in ('global','boundary','interior'):
        for kind in ('total','effective'):
            ax.loglog([r['n'] for r in rows],[r['stress'][region][kind+'_full_field']['error'] for r in rows],'-o',label=region+' '+kind)
    ax.axhline(.04,color='k',ls='--',label='4% target');ax.set(xlabel='Cells per axis',ylabel='Weighted L2 stress error',title='Same-point manufactured stress convergence');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    im=axes[0].imshow(p[0].reshape(8,8).T,origin='lower',extent=(0,1,0,1),vmin=p.min(),vmax=p.max());fig.colorbar(im,ax=axes[0],label='Pa')
    axes[0].set_title('P0 pressure')
    grid=nodes.reshape(17,17,2);lines=[]
    for j in range(17):
        lines+=axes[1].plot(grid[j,:,0],grid[j,:,1],color='tab:blue',lw=.5)
        lines+=axes[1].plot(grid[:,j,0],grid[:,j,1],color='tab:blue',lw=.5)
    axes[1].set(xlim=(-.2,1.2),ylim=(-.2,1.2),aspect='equal',title='Q2 displacement (40x display)')
    writer=PillowWriter(fps=10)
    with writer.saving(fig,str(out/'scene-preview.gif'),100):
        for i in range(len(t)):
            im.set_data(p[i].reshape(8,8).T);deformed=(nodes+40*u[i].reshape(-1,2)).reshape(17,17,2)
            for j in range(17):
                lines[2*j].set_data(deformed[j,:,0],deformed[j,:,1]);lines[2*j+1].set_data(deformed[:,j,0],deformed[:,j,1])
            fig.suptitle(f'Fixed-space quasistatic Biot, t={t[i]:.2f} s');writer.grab_frame()
    plt.close(fig)
