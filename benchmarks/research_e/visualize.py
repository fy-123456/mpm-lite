"""Standalone plots and animated scene preview from unmodified raw arrays."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from .reference import consolidation


def render(out):
    d=np.load(out/'darcy_fields.npz');n=int(d['shape'][0]);xy=d['centers'];q=d['flux']
    fig,ax=plt.subplots(figsize=(7,6));im=ax.pcolormesh(xy[:,0].reshape(n,n),xy[:,1].reshape(n,n),d['pressure'].reshape(n,n),shading='nearest',cmap='viridis')
    pick=np.arange(len(xy)).reshape(n,n)[::3,::3].ravel()
    ax.quiver(xy[pick,0],xy[pick,1],q[pick,0],q[pick,1],color='white')
    ax.set(title='45 degree anisotropic Darcy | K ratio 10',xlabel='X (m)',ylabel='Y (m)',aspect='equal')
    fig.colorbar(im,ax=ax,label='Pressure (Pa)');fig.tight_layout();fig.savefig(out/'darcy-pressure-flux.png',dpi=150);plt.close(fig)
    c=np.load(out/'consolidation_raw.npz');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for n in [8,16,32]:
        key=f'space_{n}_0.0005';x=(np.arange(n)+.5)/n
        axes[0].plot(x,c[key+'_pressure'],'.-',label=f'{n} cells')
    x=np.linspace(0,1,400);p,u=consolidation(x,.2);axes[0].plot(x,p,'k--',label='Fourier reference')
    trace=c['space_32_0.0005_trace'];tip=np.array([consolidation(1.,t)[1] for t in trace[:,0]])
    axes[1].plot(trace[:,0],trace[:,2],label='Biot tip displacement');axes[1].plot(trace[:,0],tip,'k--',label='Fourier reference')
    axes[0].set(xlabel='X (m)',ylabel='Pressure (Pa)',title='Consolidation at t=0.2 s')
    axes[1].set(xlabel='Time (s)',ylabel='Tip displacement (m)',title='Transient consolidation')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'consolidation.png',dpi=150);plt.close(fig)
    raw=np.load(out/'cycle_raw.npz');prefix='dt_0.005_';p=raw[prefix+'pressure'];u=raw[prefix+'displacement'];q=raw[prefix+'flux'];trace=raw[prefix+'trace']
    n=int(raw['shape'][0]);nodes=raw['nodes'];centers=raw['centers']
    fig,axes=plt.subplots(2,2,figsize=(11,8));ax=axes[0,0]
    frame=9;im=ax.imshow(p[frame].reshape(n,n).T,origin='lower',extent=[0,1,0,1],cmap='coolwarm',vmin=p.min(),vmax=p.max())
    ax.quiver(centers[:,0],centers[:,1],q[frame,:,0],q[frame,:,1],color='black');ax.set_title('Pressure + relative Darcy flux at t=0.4 s');fig.colorbar(im,ax=ax,label='Pa')
    deformed=nodes+30*u[frame].reshape(-1,2);grid=deformed.reshape(2*n+1,2*n+1,2)
    for i in range(0,2*n+1,2):
        axes[0,1].plot(grid[i,:,0],grid[i,:,1],'C0-',lw=.8);axes[0,1].plot(grid[:,i,0],grid[:,i,1],'C0-',lw=.8)
    axes[0,1].set(title='Skeleton displacement shown at 30x',aspect='equal',xlim=(-.1,1.1),ylim=(-.1,1.1))
    axes[1,0].plot(trace[:,0],trace[:,1:4],label=['Mean p','Minimum p','Maximum p']);axes[1,0].set(xlabel='Time (s)',ylabel='Pressure (Pa)',title='Load / hold / unload / hold')
    for i,label in [(6,'Elastic'),(7,'Storage'),(8,'Darcy loss'),(9,'Numerical loss'),(10,'External work')]:axes[1,1].plot(trace[:,0],trace[:,i],label=label)
    axes[1,1].set(xlabel='Time (s)',ylabel='Energy (J)',title='Separate physical and numerical budgets')
    for ax in axes[1]:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'scene-summary.png',dpi=140);plt.close(fig)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4))
    im=ax1.imshow(p[0].reshape(n,n).T,origin='lower',extent=[0,1,0,1],cmap='coolwarm',vmin=p.min(),vmax=p.max())
    fig.colorbar(im,ax=ax1,label='Pressure (Pa)');ax1.set(xlabel='X (m)',ylabel='Y (m)')
    lines=[]
    for _ in range(2*(n+1)):lines.append(ax2.plot([],[],'C0-',lw=1)[0])
    ax2.set(title='Skeleton deformation (30x)',xlim=(-.1,1.1),ylim=(-.1,1.1),aspect='equal')
    title=fig.suptitle('')
    def animate(i):
        im.set_data(p[i].reshape(n,n).T);grid=(nodes+30*u[i].reshape(-1,2)).reshape(2*n+1,2*n+1,2)
        for k,j in enumerate(range(0,2*n+1,2)):
            lines[2*k].set_data(grid[j,:,0],grid[j,:,1]);lines[2*k+1].set_data(grid[:,j,0],grid[:,j,1])
        t=trace[i,0];stage=['loading','holding','unloading','final hold'][min(int((t-1e-8)/.4),3)]
        title.set_text(f'Saturated Biot | {stage} | t={t:.2f} s | full material integration')
        return [im,title]+lines
    fig.tight_layout(rect=(0,0,1,.94));ani=FuncAnimation(fig,animate,frames=len(trace),interval=100,blit=False)
    ani.save(out/'scene-preview.gif',writer=PillowWriter(fps=10));plt.close(fig)
