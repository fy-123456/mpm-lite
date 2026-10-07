"""Independent NumPy check of v9 raw updates AND the variational grip force."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmarks.aniso_apic_frequency import Oracle
from benchmarks.aniso_mainline import write_json


def stress(F,A,mu,lam,kf):
    U,s,Vt=np.linalg.svd(F);logs=np.log(s)
    principal=(2*mu*logs+lam*np.sum(logs,axis=1)[:,None])/s
    FA=F@A;invariant=np.sum(F*FA,axis=(1,2))
    return (U*principal[:,None,:])@Vt+2*kf*(invariant-1)[:,None,None]*FA


def check(out):
    protocol=json.loads((out/'protocol.json').read_text());records=[];energy={}
    for name,cfg in protocol['configs'].items():
        folder=out/'cases'/name;status=json.loads((folder/'status.json').read_text())
        rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
        assert status['run_completed'] and len(rows)==round(.5/cfg['dt'])
        assert abs(rows[-1]['time']-.5)<1e-12
        with np.load(folder/'frames.npz') as z:
            assert len(z['time'])==21
            np.testing.assert_array_equal(z['F'][0],np.broadcast_to(np.eye(3),z['F'][0].shape))
        paths=sorted(folder.glob('audit-*.npz'));assert len(paths)==4
        for path in paths:
            with np.load(path) as z:
                dt=cfg['dt'];beta=cfg['flip_ratio'];o=Oracle(z['particle_x_before'],z['particle_mass'],1/(cfg['grid']-1))
                lookup={tuple(n):i for i,n in enumerate(z['grid_nodes'])};ids=[lookup[tuple(n)] for n in o.nodes]
                raw=z['grid_velocity_raw'][ids];new=z['grid_velocity_new'][ids]
                L=np.einsum('pc,cnj,ni->pij',o.S,o.D,new)
                Lraw=np.einsum('pc,cnj,ni->pij',o.S,o.D,raw)
                C=L+beta*(z['particle_C_before']-Lraw)
                v=beta*(z['particle_velocity_before']+o.S@o.H@(new-raw))+(1-beta)*(o.S@o.H@new)
                F=(np.eye(3)+dt*L)@z['particle_F_before']
                x=z['particle_x_before']+dt*(o.S@o.H@new)
                errors={k:float(np.max(abs(value-z[target]))) for k,value,target in
                    [('C',C,'particle_C_after'),('L',L,'particle_L_after'),('v',v,'particle_velocity_after'),
                     ('F',F,'particle_F_after'),('x',x,'particle_x_after')]}
                V=o.S.T@z['particle_volume'];W=o.S.T*z['particle_volume'][None,:]/V[:,None]
                Fn=np.einsum('cp,pij->cij',W,z['particle_F_before'])
                GP=np.einsum('pc,cnj->pnj',o.S,o.D)
                if cfg['history_consistency']=='projected_center':
                    Fc=np.einsum('cp,pij->cij',W,F)
                    material_map=np.einsum('cp,pak,pna->cnk',W,z['particle_F_before'],GP)
                else:
                    Lc=np.einsum('cnj,ni->cij',o.D,new)
                    Fc=(np.eye(3)+dt*Lc)@Fn
                    material_map=np.einsum('cak,cna->cnk',Fn,o.D)
                clookup={tuple(c):i for i,c in enumerate(z['coords'])};ci=[clookup[tuple(c)] for c in o.c]
                errors['center_F']=float(np.max(abs(Fc-z['center_F_committed'][ci])))
                A=np.einsum('cp,pij->cij',W,z['particle_A0'])
                P=stress(Fc,A,10.,20.,cfg['kf'])
                internal=np.einsum('c,cij,cnj->ni',V,P,material_map)
                inertia=o.mn[:,None]*(new-raw)/dt
                step=int(path.stem.split('-')[-1]);row=rows[step-1]
                for side,mask in [('left',o.xn[:,0]<=.25),('right',o.xn[:,0]>=.75)]:
                    errors[side+'_elastic_force']=abs(float(internal[mask,0].sum())-row[side+'_elastic_force'])
                    errors[side+'_force']=abs(float((internal+inertia)[mask,0].sum())-row[side+'_force'])
                assert max(errors.values())<=1e-12,(name,path.name,errors)
                records.append(dict(case=name,snapshot=path.name,errors=errors,internal_resultant=float(np.linalg.norm(internal.sum(axis=0)))))
        energy[name]={key:sum(r[key] for r in rows) for key in ('p2c_delta','c2g_delta','g2p_delta','state_transport_delta','volume_remap_delta')}
        energy[name].update(loading_work_J=rows[-1]['loading_work'],final_mechanical_J=rows[-1]['mechanical'],
                            transfer_kinetic_net_J=sum(energy[name][key] for key in ('p2c_delta','c2g_delta','g2p_delta')))
    result=dict(passed=len(records)==20,snapshots=records,signed_energy=energy,
        max_oracle_error=max(max(r['errors'].values()) for r in records),
        method='independent dense NumPy W/S/D, F prediction, SVD stress and unprojected variational grip forces')
    write_json(out/'artifact-check.json',result)
    print('verified',len(records),'raw snapshots; maximum state/force error',result['max_oracle_error'])


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=Path('docs/results/lite-aniso-mainline/v9'))
    check(p.parse_args().output)
