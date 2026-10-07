"""Static spatial evidence with shared references and explicit regional metrics."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from benchmarks.aniso_v21_common import *

def main():
    folder=OUT/'figures';folder.mkdir(exist_ok=True);plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2,'savefig.dpi':170})
    single=load(OUT/'space-validation.json')['cases'];adaptive=load(OUT/'adaptive-validation.json')['cases'];gain=load(OUT/'gain-validation.json')['cases'];families={}
    families['Initial stress selection']=[single[f'graded/F45-stress-{k:02d}'] for k in (2,6,12,18,24,40)]
    for label,prefix,data in [('Updated geometric','geometric',adaptive),('Updated raw error','stress',adaptive),('Updated correctable error','gain',gain)]:families[label]=[data[f'{prefix}/round{k}'] for k in range(1,7)]
    fig,ax=plt.subplots(1,2,figsize=(11,3.8))
    for label,values in families.items():
        dof=[r['scalar_local_dofs'] for r in values]
        for a,key in zip(ax,('stress_relative','fiber_strain_relative')):a.plot(dof,[100*r['regions']['global'][key] for r in values],'o-',label=label,lw=1.2)
    for a in ax:a.axhline(2,color='red',ls='--',lw=.8);a.set_xlabel('Added scalar basis functions (3 displacement DOFs each)');a.legend(fontsize=7)
    ax[0].set_ylabel('Global stress difference [%]');ax[1].set_ylabel('Fiber axial strain difference [%]');fig.suptitle('Equal budgets: held-out local3 Q4 reference, not used to select bases');fig.tight_layout();fig.savefig(folder/'space-budget.png');plt.close(fig)
    ref=load(OUT/'reference-self-checks.json');labels=['Global','Grip','Interior','Deep interior'];regions_=['global','grip','interior','deep_interior'];fig,ax=plt.subplots(1,2,figsize=(11,3.7));x=np.arange(4)
    for j,(name,v) in enumerate(ref['pairs'].items()):
        for a,key in zip(ax,('stress_relative','fiber_strain_relative')):a.bar(x+(j-1)*.24,[100*v['regions'][r][key] for r in regions_],width=.24,label=name)
    for a in ax:a.axhline(2,color='red',ls='--');a.set_xticks(x,labels);a.legend(fontsize=8)
    ax[0].set_ylabel('Reference stress difference [%]');ax[1].set_ylabel('Reference fiber strain difference [%]');fig.suptitle('Final reference: independent h and p comparisons; no corner exclusion');fig.tight_layout();fig.savefig(folder/'reference-checks.png');plt.close(fig)
    acceptance=load(OUT/'final-spatial-acceptance.json');reference=read_field(ROOT/acceptance['reference']);paths=load(OUT/'final-candidate-selection.json')['cases'];base=read_field(ROOT/paths['graded/baseline']);new=read_field(OUT/'adaptive/gain/round6.npz');xx=np.linspace(.25001,.74999,240);yy=np.linspace(.37501,.62499,120);X,Y=np.meshgrid(xx,yy,indexing='xy');points=np.column_stack((X.ravel(),Y.ravel(),np.full(X.size,.5)))
    strain=[]
    for field in (reference,base,new):L=gradient(points,*field);strain.append(np.einsum('i,pij,j->p',FIBER,L,FIBER).reshape(X.shape))
    fig,ax=plt.subplots(1,3,figsize=(12,3.2));im=ax[0].pcolormesh(X,Y,strain[0],shading='auto',cmap='coolwarm');fig.colorbar(im,ax=ax[0],label='Fiber strain');maximum=max(np.max(abs(strain[k]-strain[0])) for k in (1,2));norm=LogNorm(vmin=max(maximum*1e-5,1e-9),vmax=maximum)
    for a,k in zip(ax[1:],(1,2)):im=a.pcolormesh(X,Y,np.maximum(abs(strain[k]-strain[0]),1e-12),shading='auto',cmap='magma',norm=norm);fig.colorbar(im,ax=a,label='Absolute fiber strain difference')
    for a,title in zip(ax,['Final Q4 reference','Base reconstruction','144 correctable-error functions']):a.set_title(title);a.set_xlabel('Reference x');a.set_ylabel('Reference y');a.set_aspect('equal');a.grid(False)
    fig.suptitle('F45 mid-plane z = 0.5; plots are diagnostic, acceptance uses volume integrals');fig.tight_layout();fig.savefig(folder/'fiber-strain.png');plt.close(fig)
if __name__=='__main__':main()
