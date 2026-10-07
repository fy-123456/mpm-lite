"""v12 full factorial attribution, regional sensitivity, and scientific plots."""
import json
from pathlib import Path
import numpy as np
from benchmarks.aniso_time_compare import compare,LEVELS
from benchmarks.aniso_mainline import write_json
from benchmarks.aniso_dynamic_space import stress
from benchmarks.aniso_apic_frequency import Oracle
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/results/lite-aniso-mainline';OUT=BASE/'v12'


def load(p):return json.loads(p.read_text())
def records(folder,name):return [json.loads(l) for l in (folder/'cases'/name/'steps.jsonl').read_text().splitlines()]


def controls():
    rows={};checks=[]
    for key,folder,prefix in [('baseline',BASE/'v11','F45'),('velocity_only',BASE/'v12-transfer-controls','velocity_only'),('affine_only',BASE/'v12-transfer-controls','affine_only'),('both',BASE/'v12-exploration','F45')]:
        p=load(folder/'protocol.json');r=compare(folder,p['configs'],prefix)
        rows[key]=r
        if key=='baseline':continue
        for level in LEVELS:
            name=prefix+'-'+level;cfg=p['configs'][name];bv=cfg['flip_ratio'];bc=p.get('affine_beta',{}).get(name,bv)
            for file in sorted((folder/'cases'/name).glob('audit-*.npz')):
                with np.load(file) as z:
                    o=Oracle(z['particle_x_before'],z['particle_mass'],1/(cfg['grid']-1));raw,new=z['grid_velocity_raw'],z['grid_velocity_new'];dt=cfg['dt']
                    raw_check=o.apply(z['particle_velocity_before'],z['particle_C_before'],0.)
                    L=np.einsum('pc,cnj,ni->pij',o.S,o.D,new);Lraw=np.einsum('pc,cnj,ni->pij',o.S,o.D,raw)
                    expected=dict(particle_x_after=z['particle_x_before']+dt*(o.S@o.H@new),particle_F_after=(np.eye(3)+dt*L)@z['particle_F_before'],particle_L_after=L,particle_C_after=L+bc*(z['particle_C_before']-Lraw),particle_velocity_after=bv*(z['particle_velocity_before']+o.S@o.H@(new-raw))+(1-bv)*(o.S@o.H@new),grid_velocity_raw=raw_check['grid_v'])
                    errors={k:float(np.max(abs(a-z[k]))) for k,a in expected.items()}
                    assert max(errors.values())<1e-11,(key,name,file,errors)
                    checks.append(dict(mode=key,case=name,snapshot=file.name,errors=errors))
    E={k:r['pairs'][-1]['P_relative'] for k,r in rows.items()}
    effect=dict(remove_velocity_only=E['velocity_only']-E['baseline'],remove_affine_only=E['affine_only']-E['baseline'],interaction=E['both']-E['velocity_only']-E['affine_only']+E['baseline'])
    result=dict(completed=True,modes=rows,stress_error_factorial_effects=effect,units='relative error, not physical stress; interactions of nonlinear trajectory error norms',checks=checks,independent_snapshots=len(checks),max_oracle_error=max(max(r['errors'].values()) for r in checks),source_of_improvement='affine PIC relaxation removal dominates; velocity-only changes reduce last-pair space-time error slightly but worsen terminal error',stabilization='same energy and reconstruction in all four factorial modes; downstream reconstructed energies may differ because the states differ')
    write_json(OUT/'factorial-attribution.json',result)
    for k,r in rows.items():print(k,{x:r['pairs'][-1][x] for x in ('P_relative','P_terminal_relative','reaction_relative')},'physical',r['physical_passed'])


def fields(folder,prefix='F45'):
    p=load(folder/'protocol.json');allF=[];allP=[]
    for level in LEVELS:
        name=prefix+'-'+level
        with np.load(folder/'cases'/name/'frames.npz') as z:t=z['time'].copy();F=z['F'].copy();X=z['x'][0].copy()
        keep=t>=.05-1e-12;t=t[keep];F=F[keep];P=stress(F.reshape(-1,3,3),p['configs'][name]).reshape(F.shape);allF.append(F);allP.append(P)
    return t,X,allF,allP


def final():
    # Public API and the isolated pre-interface experiment must coincide.
    matches=[]
    for level in LEVELS:
        new=OUT/'cases'/('F45-'+level);old=BASE/'v12-transfer-controls/cases'/('affine_only-'+level)
        with np.load(new/'frames.npz') as a,np.load(old/'frames.npz') as b:
            err=max(float(np.max(abs(a[k]-b[k]))) for k in ('time','x','F'))
        nr=records(OUT,'F45-'+level);br=records(BASE/'v12-transfer-controls','affine_only-'+level)
        for key in ('right_force','left_force','elastic','stabilization_energy','stabilization_rebuild_delta'):
            err=max(err,max(abs(a[key]-b[key]) for a,b in zip(nr,br)))
        assert len(nr)==len(br) and err<1e-12
        matches.append(dict(level=level,steps=len(nr),max_difference=err))
    write_json(OUT/'public-api-equivalence.json',dict(passed=True,cases=matches,max_difference=max(r['max_difference'] for r in matches)))
    regional=[]
    for label,folder in [('v11',BASE/'v11'),('v12',OUT)]:
        t,X,F,P=fields(folder);near=np.minimum(abs(X[:,0]-.25),abs(X[:,0]-.75))<.0625
        masks=dict(whole=np.ones(len(X),bool),near_grip=near,interior=(X[:,0]>.3125)&(X[:,0]<.6875),deep_interior=(X[:,0]>.375)&(X[:,0]<.625))
        norm=lambda a:float(np.sqrt(np.trapezoid(np.mean(np.sum(a*a,axis=(2,3)),axis=1),t)/(t[-1]-t[0])))
        for key,m in masks.items():
            a,b=P[-2][:,m],P[-1][:,m];regional.append(dict(version=label,region=key,particles=int(m.sum()),P_space_time_relative=norm(a-b)/norm(b),P_terminal_relative=float(np.linalg.norm(a[-1]-b[-1])/np.linalg.norm(b[-1]))))
    write_json(OUT/'regional-time.json',dict(records=regional,scope='original-material-coordinate regions; equal reference-volume particle sampling; time sensitivity, not independent spatial accuracy'))
    plots()


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    modes=load(OUT/'factorial-attribution.json')['modes'];colors={'baseline':'#c44e52','velocity_only':'#8172b2','affine_only':'#4c72b0','both':'#55a868'}
    labels={'baseline':'v11 baseline','velocity_only':'Remove v smoothing','affine_only':'v12: remove C smoothing','both':'Remove both'}
    dt=np.array([.001,.0005,.00025])
    for key,r in modes.items():
        ax[0,0].plot(dt,100*np.array([x['P_relative'] for x in r['pairs']]),'o-',color=colors[key],label=labels[key])
    ax[0,0].set_xscale('log',base=2);ax[0,0].invert_xaxis();ax[0,0].set_xticks(dt,labels=['0.001','0.0005','0.00025']);ax[0,0].axhline(2.,color='k',ls=':',lw=1);ax[0,0].set(xlabel='Coarser dt of adjacent pair (s)',ylabel='Space-time stress difference (%)',title='A. Controlled transfer changes');ax[0,0].legend(fontsize=8)
    for label,folder,color in [('v11',BASE/'v11','#c44e52'),('v12',OUT,'#4c72b0')]:
        t,X,F,P=fields(folder);err=np.linalg.norm((P[-2]-P[-1]).reshape(len(t),-1),axis=1)/np.linalg.norm(P[-1].reshape(len(t),-1),axis=1)
        ax[0,1].plot(t,100*err,'o-',ms=3,label=label,color=color)
        for level,style in [('finest','--'),('fourth','-')]:
            r=records(folder,'F45-'+level);ax[1,0].plot([a['time'] for a in r],[a['right_force'] for a in r],style,lw=1.5,label=f'{label} {level}',color=color)
        r=compare(folder,load(folder/'protocol.json')['configs']);ax[1,1].plot([.001,.0005,.00025,.000125],[100*x['rebuild_over_elastic'] for x in r['energy']],'o-',label=label,color=color)
    ax[0,1].axhline(2.,color='k',ls=':',lw=1);ax[0,1].set(xlabel='Physical time (s)',ylabel='Stress difference (%)',title='B. Finest pair throughout loading');ax[0,1].legend()
    ax[1,0].set(xlabel='Physical time (s)',ylabel='Right reaction (N)',title='C. Same loading and energy functional');ax[1,0].legend(fontsize=8)
    ax[1,1].set_xscale('log',base=2);ax[1,1].invert_xaxis();ax[1,1].set_xticks([.001,.0005,.00025,.000125],labels=['0.001','0.0005','0.00025','0.000125']);ax[1,1].set(xlabel='dt (s)',ylabel='Net rebuild change / final elastic (%)',title='D. Reference rebuild remains active');ax[1,1].legend()
    for a in ax.flat:a.grid(alpha=.2)
    fig.savefig(OUT/'F45-time-attribution.png',dpi=160);fig.savefig(OUT/'F45-time-attribution.pdf');plt.close(fig)

if __name__=='__main__':
    import sys
    controls() if len(sys.argv)>1 and sys.argv[1]=='controls' else final()
