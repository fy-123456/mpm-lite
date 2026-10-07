"""Shareable static figures for v17 diagnostics (no external data)."""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_v17_time import OUT,load
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=170);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)

def modes():
    data=load(OUT/'modal-snapshots.json')['records'];fig,axs=plt.subplots(2,2,figsize=(12,8))
    for row,r in enumerate(data):
        records=r['modes'];w=np.array([m['omega_rad_s'] for m in records]);score=np.array([m['stress_self_power_fraction'] for m in records]);ks=np.array([m['stabilization_fraction'] for m in records])
        ax=axs[row,0];sc=ax.scatter(w,score*100,c=ks,cmap='viridis',vmin=0,vmax=1,s=18)
        ax.set(xscale='log',yscale='log',ylim=(1e-12,200),xlabel='Angular frequency [rad/s]',ylabel='Stress self-power share [%]',title=r['label']+' hold: stress importance')
        fig.colorbar(sc,ax=ax,label='Stabilization stiffness fraction',shrink=.85,pad=.03)
        ax.text(.02,.02,'Shares below 1e-12% omitted',transform=ax.transAxes,fontsize=8)
        i=r['stress_ranking'][0];ax.annotate(f'top: {w[i]:.1f} rad/s',(w[i],score[i]*100),xytext=(16,-23),textcoords='offset points')
        ids=r['stress_ranking'][:10];rank=np.arange(1,11);ax=axs[row,1]
        ax.bar(rank,[records[i]['stabilization_fraction']*100 for i in ids],label='Stabilization stiffness',color='#db8742')
        ax.bar(rank,[(1-records[i]['stabilization_fraction'])*100 for i in ids],bottom=[records[i]['stabilization_fraction']*100 for i in ids],label='Material stiffness',color='#4676ab')
        ax.plot(rank,[records[i]['kinetic_C_fraction']*100 for i in ids],'ko-',label='Affine C kinetic share',ms=3)
        ax.set(xlabel='Rank by stress self-power',ylabel='Share [%]',ylim=(0,105),xticks=rank,title=r['label']+' hold: stiffness / kinetic split')
        if row==0:ax.legend(fontsize=8,loc='lower left')
    save(fig,'stress-ranked-modes')
    scans=load(OUT/'spatial-mode-scan.json')['records'];fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ns in (4,6,8):
        rows=[r for r in scans if r['ns']==ns and r['amplitude']==0]
        h=[r['h'] for r in rows];w=[r['modes'][r['stress_ranking'][0]]['omega_rad_s'] for r in rows]
        axs[0].plot(h,w,'o-',label=f'{3*ns} x {ns} x {ns} particles')
        axs[1].plot(h,[r['zero_inertia_modes'] for r in rows],'o-',label=f'ns={ns}')
    axs[0].set(xlabel='Grid spacing h',ylabel='Top stress mode [rad/s]',title='Common smooth velocity; rest geometry');axs[0].legend(fontsize=8)
    axs[1].set(xlabel='Grid spacing h',ylabel='Strict zero-inertia directions',title='Positive static stiffness, singular inertia',yticks=[9,15]);axs[1].legend()
    save(fig,'spatial-mode-scan')
    generic=load(OUT/'generic-geometry-scan.json')['records'];rows=sorted([r for r in generic if r['h']==.125 and r['ns']==4],key=lambda r:r['amplitude'])
    fig,axs=plt.subplots(1,2,figsize=(11,4));amps=np.array([r['amplitude'] for r in rows])
    axs[0].loglog(amps,[r['smallest_kinetic_singular'] for r in rows],'o-');axs[0].set(xlabel='Deformation amplitude',ylabel='Smallest weighted-J singular value',title='Observable inertia shrinks toward rest')
    vals=np.array([[d['rayleigh_omega_rad_s'] for d in r['reference_rest_null_directions']] for r in rows])
    axs[1].loglog(amps,vals,'o-',alpha=.6);axs[1].set(xlabel='Deformation amplitude',ylabel='Rayleigh angular frequency [rad/s]',title='Former null directions (not eigenfrequencies)')
    save(fig,'geometry-inertia-scaling')

def timeplots():
    data=load(OUT/'time-summary.json');fig,axs=plt.subplots(2,2,figsize=(11,8))
    for col,label in enumerate(('early','late')):
        rows=data['refinement'][label];x=np.array([r['dt_fine'] for r in rows]);ax=axs[0,col]
        ax.loglog(x,[r['relative']*100 for r in rows],'o-',label='Full tensor, all shared steps')
        ax.loglog(x,[r['terminal_relative']*100 for r in rows],'s-',label='Terminal tensor')
        ax.axhline(2,color='#b44',ls='--',label='2% threshold');ax.invert_xaxis();ax.set(xlabel='Finer dt [s]',ylabel='Adjacent-dt difference [%]',title=label+' fixed hold');ax.legend(fontsize=8)
        cases=sorted([(k,r) for k,r in data['cases'].items() if k.startswith(label)],key=lambda v:v[1]['dt'],reverse=True)
        dt=[r['dt'] for k,r in cases];ax=axs[1,col]
        ax.loglog(dt,[r['nonlinear_vs_exact_linear']['relative']*100 for k,r in cases],'o-',label='AVF vs exact linear time')
        ax.loglog(dt,[r['nonlinear_vs_discrete_linear']['relative']*100 for k,r in cases],'s-',label='AVF vs same-dt linear midpoint')
        ax.invert_xaxis();ax.set(xlabel='dt [s]',ylabel='Full-history stress difference [%]',title='Time error and local linearization gap');ax.legend(fontsize=8)
    save(fig,'time-convergence')
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for col,label in enumerate(('early','late')):
        ref=np.load(OUT/f'modal-{label}.npz');i=211;w=ref['omega'][i];eq=ref['eq'][i];v0=ref['v0'][i];init=-eq-1j*v0/w
        for level in (1,3,5):
            with np.load(OUT/'cases'/f'{label}-L{level}'/'series.npz') as f:
                t=f['time'];ph=(f['modal_q'][:,i]-eq)-1j*f['modal_v'][:,i]/w;ratio=ph/init
                axs[0,col].plot(t,(np.abs(ratio)-1)*100,label=f'L{level}, dt={data["cases"][f"{label}-L{level}"]["dt"]:g}')
                axs[1,col].plot(t,np.unwrap(np.angle(ratio))-w*t)
        axs[0,col].set(title=f'{label}: time-error mode ({w:.1f} rad/s)',ylabel='Amplitude error vs local linear [%]');axs[0,col].legend(fontsize=8)
        axs[1,col].set(xlabel='Continuation time [s]',ylabel='Phase error vs local linear [rad]')
    save(fig,'amplitude-phase')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['modes','time']);a=p.parse_args();modes() if a.action=='modes' else timeplots()
