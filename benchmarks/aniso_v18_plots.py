"""Exportable figures for moving AVF and independent spatial checks."""
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_v18_runs import OUT,load
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=170);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
def moving():
    d=load(OUT/'moving-acceptance.json');fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,label in zip(axs,('early','late')):
        rows=d['refinement'][label];dt=np.array([r['dt_fine'] for r in rows])*1e6
        ax.loglog(dt,[100*r['relative'] for r in rows],'o-',label='Full time / tensor')
        ax.loglog(dt,[100*r['terminal_relative'] for r in rows],'s-',label='Terminal tensor');ax.axhline(2,color='#aa4444',ls='--',label='2% gate');ax.invert_xaxis();ax.set(title=label+' moving hold',xlabel='Finer dt [microseconds]',ylabel='Adjacent-dt stress difference [%]');ax.legend(fontsize=8)
    save(fig,'moving-time-convergence')
def space():
    d=load(OUT/'fixed-mode-quadrature.json');labels=['sampled192','gauss2','gauss3','gauss4'];rows=[next(r for r in d['records'] if r['label']=='sampled-target211' and r['scheme']==s) for s in labels];fig,axs=plt.subplots(1,3,figsize=(14,4));x=np.arange(4)
    for ax,keys,title in [(axs[0],('material_stiffness','stabilization_stiffness'),'Same displacement: stiffness'),(axs[1],('translation_inertia','affine_inertia'),'Same displacement: inertia')]:
        base=sum(rows[0][k] for k in keys);a=np.array([r[keys[0]]/base for r in rows]);b=np.array([r[keys[1]]/base for r in rows]);ax.bar(x,a,label=keys[0].replace('_',' '));ax.bar(x,b,bottom=a,label=keys[1].replace('_',' '));ax.set(xticks=x,xticklabels=labels,title=title,ylabel='Ratio to original total');ax.tick_params(axis='x',rotation=20);ax.legend(fontsize=7)
    axs[2].plot(x,[r['rayleigh_rad_s'] for r in rows],'o-');axs[2].set(xticks=x,xticklabels=labels,ylabel='Rayleigh angular frequency [rad/s]',title='Identical carrier field, h=1/8');axs[2].tick_params(axis='x',rotation=20);save(fig,'same-mode-quadrature')
    q=load(OUT/'space-quadrature.json');fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,target in zip(axs,(139,211)):
        rows=[r for r in q['records'] if r['order']==3];match=[next(m for m in r['matches'] if m['target']==target) for r in rows];sc=ax.scatter([r['h'] for r in rows],[m['omega_rad_s'] for m in match],c=[m['stress_MAC'] for m in match],cmap='viridis',vmin=0,vmax=1,s=90)
        for r,m in zip(rows,match):ax.annotate(f'MAC {m["stress_MAC"]:.3f}',(r['h'],m['omega_rad_s']),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
        ax.margins(x=.2,y=.25);ax.set(title=f'Best stress-shape match to target {target}',xlabel='Grid spacing h',ylabel='Matched angular frequency [rad/s]');fig.colorbar(sc,ax=ax,label='Stress MAC')
    save(fig,'space-mode-matching')
    if not (OUT/'compatible-reference.json').exists():return
    c=load(OUT/'compatible-reference.json');rows=c['comparisons'][c['finest']];fig,axs=plt.subplots(1,2,figsize=(12,4));x=np.arange(4);width=.24
    for j,key in enumerate(('global','grip','interior')):axs[0].bar(x+(j-1)*width,[100*r['regions'][key]['stress_mismatch_relative'] for r in rows],width,label=key)
    axs[0].set(xticks=x,xticklabels=[r['label'].replace('-target','\nmode ') for r in rows],ylabel='Stress mismatch [%]',title='Lite target vs compatible physical projection');axs[0].legend(fontsize=8)
    names=['q1-n64','q2-n32','q2-n48','q2-local1'];
    for j,lab in enumerate([r['label'] for r in rows]):axs[1].plot(np.arange(len(names)),[100*c['comparisons'][name][j]['stress_vs_finest_relative'] for name in names],'o-',label=lab)
    axs[1].axhline(2,color='#aa4444',ls='--');axs[1].set(xticks=np.arange(len(names)),xticklabels=names,ylabel='Stress vs finest reference [%]',title='Reference discretization remains separate');axs[1].legend(fontsize=7);save(fig,'compatible-stress-reference')
def cycle():
    d=load(OUT/'cycle-acceptance.json');fig,axs=plt.subplots(2,2,figsize=(12,8))
    for name,case in d['cases'].items():
        rows=[json.loads(s) for s in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()];t=np.array([r['time'] for r in rows]);every=max(1,len(rows)//3000);sl=slice(None,None,every);label=f'{name}, dt={rows[0]["dt"]:g}'
        axs[0,0].plot(t[sl],[r['stress_rms_Pa'] for r in rows[sl]],label=label,lw=.8);axs[0,1].plot(t[sl],[r['reaction_N'] for r in rows[sl]],label=label,lw=.8)
    name='cycle-L3';rows=[json.loads(s) for s in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()];t=np.array([r['time'] for r in rows]);sl=slice(None,None,max(1,len(rows)//3000))
    for key in ('material_J','stabilization_J','kinetic_J'):axs[1,0].plot(t[sl],[r[key] for r in rows[sl]],label=key)
    cum={key:np.cumsum([r[key] for r in rows]) for key in ('boundary_work_J','metric_change_J','kinetic_force_work_defect_J','potential_quadrature_error_J')}
    axs[1,1].plot(t[sl],np.array([r['total_J'] for r in rows])[sl]-cum['boundary_work_J'][sl],label='Total energy - external work')
    axs[1,1].plot(t[sl],cum['metric_change_J'][sl],ls='--',label='Accumulated metric change')
    for ax in axs.ravel():
        for a,b in ((.5,.6),(1.1,1.6)):ax.axvspan(a,b,color='#999999',alpha=.12)
        ax.set_xlabel('Time [s]');ax.legend(fontsize=7)
    axs[0,0].set(ylabel='Full stress tensor RMS [Pa]',title='Full prescribed-displacement cycle');axs[0,1].set(ylabel='Right-grip reaction [N]',title='Midpoint reaction')
    axs[1,0].set(ylabel='Energy [J]',title='Finest run: stored and kinetic energy');axs[1,1].set(ylabel='Energy [J]',title='Finest run: work and metric budget');save(fig,'full-cycle-response')

def reaction():
    d=load(OUT/'reaction-diagnosis.json');fig,axs=plt.subplots(2,2,figsize=(12,8))
    for level in range(4):
        name=f'cycle-L{level}';rows=[json.loads(s) for s in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()];dt=rows[0]['dt'];R=np.array([r['reaction_N'] for r in rows]);t=dt*(np.arange(len(R))+.5);mask=t>1.596
        axs[0,0].plot(t[mask],R[mask]*1000,'.-',label=f'{dt*1e6:g} us',ms=2,lw=.7)
        w=round(.0025/dt);tm=t.reshape(-1,w).mean(1);mean=R.reshape(-1,w).mean(1);mask=tm>1.1;axs[0,1].plot(tm[mask],mean[mask]*1000,label=f'{dt*1e6:g} us',lw=.8)
    dt=np.array([r['dt'] for r in d['records']])*1e6;hold=[r['phases']['final_hold'] for r in d['records']]
    axs[1,0].loglog(dt,[1000*r['raw_rms_N'] for r in hold],'o-',label='Raw RMS');axs[1,0].loglog(dt,[1000*r['alternating_rms_N'] for r in hold],'s-',label='Adjacent-step alternating part');axs[1,0].invert_xaxis()
    terminal=[r for r in d['independent_component_audits'] if abs(r['time']-1.6)<1e-10]
    for key in ('inertia_N','material_N','stabilization_N'):axs[1,1].plot(dt,[1000*r[key] for r in terminal],'o-',label=key)
    axs[1,1].invert_xaxis()
    axs[0,0].set(title='Raw midpoint reaction: unresolved alternation',xlabel='Time [s]',ylabel='Reaction [mN]')
    axs[0,1].set(title='2.5 ms impulse average: diagnostic only',xlabel='Time [s]',ylabel='Averaged reaction [mN]')
    axs[1,0].set(title='Final hold: raw force grows with refinement',xlabel='dt [microseconds]',ylabel='RMS force [mN]')
    axs[1,1].set(title='Independent terminal force decomposition',xlabel='dt [microseconds]',ylabel='Force [mN]')
    for ax in axs.ravel():ax.legend(fontsize=8)
    save(fig,'reaction-failure-diagnosis')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['moving','space','cycle','reaction']);a=p.parse_args();globals()[a.action]()
