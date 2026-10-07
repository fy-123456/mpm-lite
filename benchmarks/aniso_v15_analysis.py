"""Independent v15 history/force replay and honestly bounded accuracy summary."""
import hashlib,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from benchmarks.aniso_compatible_diagnosis import BASE,ROOT,maps,gradient,write
from benchmarks.aniso_dynamic_check import pk1,interpolation,CORNERS
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.types import AnisotropicMaterialParams
OUT=BASE/'v15';LEVELS=('coarse','fine','finest','fourth')


def load(path):return json.loads(path.read_text())
def rows(path):return [json.loads(l) for l in path.read_text().splitlines()]
def arrays(path):
    with np.load(path) as f:return {k:f[k].copy() for k in f.files}
def norm(a):return float(np.sqrt(np.mean(np.sum(a*a,axis=(-2,-1)))))


def independent_N(Y,nodes,h):
    lo,hi=nodes.min(0),nodes.max(0);lookup={tuple(n):i for i,n in enumerate(nodes)};N=np.zeros((len(Y),len(nodes)))
    for j,y in enumerate(Y):
        base=np.clip(np.floor(y/h).astype(int),lo,hi-1);frac=y/h-base
        for c in CORNERS:N[j,lookup[tuple(base+c)]]+=np.prod(np.where(c,frac,1-frac))
    return sp.csr_matrix(N)


def replay(path,cfg,row):
    path=Path(path).resolve()
    z=arrays(path);h=1/(cfg['grid']-1);dt=cfg['dt'];V=z['particle_V'];nodes=z['grid_nodes'];F0=z['particle_F_before'];F=z['particle_F'];Y=z['Y'];v=z['grid_v']
    T,G=maps(z['particle_x_before'],nodes,h);N=independent_N(z['Y_before'],nodes,h)
    _,G0=maps(z['particle_reference_x'],np.rint(z['X']/h).astype(int),h)
    if cfg['stabilization']=='compatible_patch':
        R=z['compatible_R'];B=[sum((G0[j]@N).multiply(R[:,j,k,None]) for j in range(3)).tocsr() for k in range(3)]
    else:B=[sum(G[j].multiply(F0[:,j,k,None]) for j in range(3)).tocsr() for k in range(3)]
    Fnew=F0+dt*gradient(B,v);Ynew=z['Y_before']+dt*(N@v)
    S=interpolation(z['particle_x_before'],z['center_coords'],h);cv=S.T@V;W=sp.diags(1/cv)@S.T@sp.diags(V)
    e=dict(F=float(np.max(abs(Fnew-F))),Y=float(np.max(abs(Ynew-Y))),x=float(np.max(abs(z['particle_x_before']+dt*(T@v)-z['particle_x']))),
           center_F=float(np.max(abs((W@F.reshape(len(F),9)).reshape(-1,3,3)-z['center_trial']))))
    r=np.einsum('cij,cja->cia',z['P'],Y[z['ids']]);Us=float(.5*np.sum(z['weights'][:,None,None]*r*r))
    params=AnisotropicMaterialParams(10,20,cfg['kf']);Um=float(V@energy_density(F,z['particle_A'],params))
    e['energy_J']=abs(Us+Um-row['elastic']);e['stabilization_J']=abs(Us-row['stabilization_energy'])
    if cfg['stabilization']=='compatible_patch':e['common_history']=float(np.max(abs(gradient(G0,Y)@z['compatible_R']-F)))
    P=pk1(F,z['particle_A'],cfg['kf']);fm=sum(b.T@(V[:,None]*P[:,:,k]) for k,b in enumerate(B))
    cf=np.zeros_like(Y);local=z['weights'][:,None,None]*np.einsum('cji,cja->cia',z['P'],r);np.add.at(cf,z['ids'].ravel(),local.reshape(-1,3));fs=N.T@cf
    mass=T.T@(z['particle_V']*cfg['density']);fi=mass[:,None]*(v-z['grid_raw'])/dt
    for name,m in [('left',nodes[:,0]*h<=.25),('right',nodes[:,0]*h>=.75)]:e[name+'_reaction_N']=abs(float((fm+fs+fi)[m,0].sum())-row[name+'_force'])
    free=(nodes[:,0]*h>.25)&(nodes[:,0]*h<.75)
    e['free_force_norm_N']=abs(float(np.linalg.norm((fm+fs+fi)[free]))-row['free_force_residual_norm'])
    for key,value in e.items():
        tolerance=1e-12 if key.endswith('_J') else 1e-9 if key.endswith('_N') else 1e-10
        assert value<tolerance,(str(path),key,value,tolerance)
    result=dict(file=str(path.relative_to(ROOT)),errors=e,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if path.parent.name.startswith('cycle-'):
        from benchmarks.aniso_compatible_controls import stiffness
        from benchmarks.aniso_residual_gate import rank_gate
        ids=z['ids'];m=ids.shape[1];vals=z['weights'][:,None,None]*(z['P'].swapaxes(1,2)@z['P'])
        Ky=sp.csr_matrix((vals.ravel(),(np.repeat(ids,m,axis=1).ravel(),np.tile(ids,(1,m)).ravel())),shape=(len(Y),len(Y)))
        Ks=N.T@Ky@N;Km,_=stiffness(F,z['particle_A'],V,B,cfg['kf']);Km2,_=stiffness(F,z['particle_A'],V,B,cfg['kf'],5e-7)
        result['massless']=rank_gate({'free':np.flatnonzero(np.tile(free,3))},Km+sp.block_diag([Ks]*3,format='csr'),True)
        result['hessian_fd_refinement']=float(abs(Km-Km2).max()/max(abs(Km).max(),1e-30))
        assert result['massless']['passed'] and result['hessian_fd_refinement']<1e-6,result
    return result


def field_stress(frames,kf=200.):
    F=frames['F'];n=np.prod(F.shape[:-2]);a=np.array([1.,1.,0.])/np.sqrt(2);A=np.broadcast_to(np.outer(a,a),(n,3,3))
    return pk1(F.reshape(-1,3,3),A,kf).reshape(F.shape)


def pair_metrics(fa,fb,cfg,start=None,end=None):
    np.testing.assert_allclose(fa['time'],fb['time'],atol=1e-10,rtol=0)
    t=fa['time'];keep=np.ones(len(t),bool)
    if start is not None:keep&=t>=start-1e-10
    if end is not None:keep&=t<=end+1e-10
    t=t[keep];Pa,Pb=field_stress(fa,cfg['kf'])[keep],field_stress(fb,cfg['kf'])[keep]
    rms=lambda q:float(np.sqrt(np.trapezoid(np.mean(np.sum(q*q,axis=(2,3)),axis=1),t)/(t[-1]-t[0])))
    absolute=rms(Pa-Pb);relative=absolute/max(rms(Pb),1e-30)
    return dict(P_absolute_Pa=absolute,P_relative=relative,P_terminal_absolute_Pa=norm(Pa[-1]-Pb[-1]),P_terminal_relative=norm(Pa[-1]-Pb[-1])/max(norm(Pb[-1]),1e-30))


def main():
    p=load(OUT/'protocol.json');batch=load(OUT/'batch.json');assert batch['completed']
    data={};frames={};metrics={};audits=[]
    for name,spec in p['cases'].items():
        folder=OUT/'cases'/name;status=load(folder/'status.json');assert status['completed']
        r=rows(folder/'steps.jsonl');z=arrays(folder/'frames.npz');data[name]=r;frames[name]=z
        assert np.isfinite(z['F']).all() and np.min(np.linalg.det(z['F']))>0
        if spec['start']:
            original=arrays(folder/'initial.npz')
            for k in ('x','F','v','C','Y'):np.testing.assert_array_equal(z[k][0],original[k])
        else:np.testing.assert_array_equal(z['F'][0],np.broadcast_to(np.eye(3),z['F'][0].shape))
        checks={k:max(abs(a[k]) for a in r) for k in ('stage_budget_error','stabilization_rebuild_delta','particle_trial_commit_max','carrier_commit_max','center_trial_commit_max','common_history_max','particle_momentum_balance_error_norm','free_force_residual_norm')}
        last=r[-1];P=field_stress(z,spec['config']['kf']);metrics[name]=dict(steps=len(r),checks=checks,
            terminal_P_rms_Pa=norm(P[-1]),terminal_elastic_J=last['elastic'],terminal_stabilization_J=last['stabilization_energy'],terminal_kinetic_J=last['kinetic'],terminal_history_rms=last['branch_history_rms'],
            max_history_rms=max(a['branch_history_rms'] for a in r),terminal_position_rms_m=last['branch_position_rms'])
        if spec['start']:
            ini=load(folder/'initial.json');metrics[name].update(mechanical_change_J=last['mechanical']-(ini['material_J']+ini['stabilization_J']+ini['kinetic_J']),
                initial_P_rms_Pa=norm(P[0]),initial_elastic_J=ini['material_J']+ini['stabilization_J'])
        for path in sorted(folder.glob('audit-*.npz')):
            step=int(path.stem.split('-')[-1]);audit=replay(path,spec['config'],r[step-1]);snap=arrays(path)
            frame_idx=int(np.argmin(abs(z['time']-r[step-1]['time'])))
            assert abs(z['time'][frame_idx]-r[step-1]['time'])<1e-10
            frame_errors={k:float(np.max(abs(z[k][frame_idx]-snap[target]))) for k,target in [('x','particle_x'),('F','particle_F'),('Y','Y')]}
            assert max(frame_errors.values())==0.,frame_errors
            audit['saved_frame_errors']=frame_errors;audits.append(audit)
    initial_checks=[];branch_refinement={};intervention={}
    for label in ('unload','early_hold','late_hold'):
        names=[f'{label}-{mode}-{level}' for mode in ('baseline','compatible') for level in LEVELS]
        anchor=arrays(OUT/'cases'/names[0]/'initial.npz')
        err=max(float(np.max(abs(arrays(OUT/'cases'/n/'initial.npz')[k]-anchor[k]))) for n in names for k in ('x','F','v','C','Y'))
        assert err==0.;initial_checks.append(dict(phase=label,cases=len(names),max_input_state_difference=err))
        branch_refinement[label]={}
        for mode in ('baseline','compatible'):
            pairs=[]
            for a,b in zip(LEVELS[:-1],LEVELS[1:]):
                na,nb=f'{label}-{mode}-{a}',f'{label}-{mode}-{b}'
                m=pair_metrics(frames[na],frames[nb],p['cases'][nb]['config']);m['pair']=[a,b]
                if pairs:m['order']=-float(np.log2(m['P_absolute_Pa']/pairs[-1]['P_absolute_Pa']))
                pairs.append(m)
            branch_refinement[label][mode]=pairs
        a,b=f'{label}-baseline-fourth',f'{label}-compatible-fourth'
        intervention[label]=dict(baseline=metrics[a],compatible=metrics[b],same_dt_field_difference=pair_metrics(frames[a],frames[b],p['cases'][b]['config']))
    # Archived baseline continuation check validates restoration without using
    # changes in loading phase or roundoff as a claimed new-method effect.
    anchors=[];old=arrays(BASE/'v14/cases/material-fourth/frames.npz')
    for label in ('unload','early_hold'):
        f=frames[f'{label}-baseline-fourth'];t=f['time'][-1];j=np.argmin(abs(old['time']-t));assert abs(old['time'][j]-t)<1e-10
        e={k:float(np.max(abs(f[k][-1]-old[k][j]))) for k in ('x','F','v','C')};assert max(e.values())<1e-9,e
        anchors.append(dict(phase=label,time=float(t),errors=e))
    cycles={};phase_intervals={'ramp':(.05,.5),'loaded_hold':(.5,.6),'unload':(.6,1.1),'final_hold':(1.1,1.6)}
    for mode in ('baseline','compatible'):
        fa=arrays(BASE/'v14/cases/material-coarse/frames.npz') if mode=='baseline' else frames['cycle-compatible-coarse']
        fb=arrays(BASE/'v14/cases/material-fine/frames.npz') if mode=='baseline' else frames['cycle-compatible-fine']
        cycles[mode]={key:pair_metrics(fa,fb,p['cases']['cycle-compatible-fine']['config'],*interval) for key,interval in phase_intervals.items()}
    result=dict(completed=True,trajectories=len(metrics),total_steps=sum(r['steps'] for r in metrics.values()),tests=load(OUT/'tests.json')['tests_run'],
        cases=metrics,independent_snapshots=audits,input_state_checks=initial_checks,baseline_archive_anchors=anchors,branch_refinement=branch_refinement,intervention=intervention,
        two_level_full_cycle_refinement=cycles,full_four_level_global_acceptance=False,spatial_accuracy_revalidated=False,overall_accuracy_accepted=False,default_changed=False)
    write(OUT/'summary.json',result)
    print('Completed',result['total_steps'],'steps;',len(audits),'audits')
    for label in branch_refinement:
        print(label,{m:v[-1]['P_relative'] for m,v in branch_refinement[label].items()}, {m:intervention[label][m]['terminal_P_rms_Pa'] for m in ('baseline','compatible')})
    print('cycle refinement',cycles)

if __name__=='__main__':main()
