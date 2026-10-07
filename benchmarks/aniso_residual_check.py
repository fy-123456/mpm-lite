"""Independent replay of local material forces and stabilized grip reactions."""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from benchmarks.aniso_dynamic_check import interpolation, CORNERS, pk1

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/results/lite-aniso-mainline/v10'


def stabilization_energy(z,velocity,dt,kf):
    F=z['stabilizer_F0']+dt*np.einsum('cna,cqnb->cqab',velocity[z['stabilizer_ids']],z['stabilizer_B'])
    U,_,Vt=np.linalg.svd(F[:,0]);R=U@Vt
    D=np.swapaxes(R,1,2)[:,None]@(F[:,1:]-F[:,0,None])
    A=z['native_center_A'][:,None]
    S=10*(D+np.swapaxes(D,2,3))+20*np.trace(D,axis1=2,axis2=3)[:,:,None,None]*np.eye(3)
    S+=4*kf*np.sum(A*D,axis=(2,3))[:,:,None,None]*A
    return float(.5*np.sum(z['native_center_V'][:,None]*np.sum(D*S,axis=(2,3)))/8)


def check():
    protocol=json.loads((OUT/'protocol.json').read_text());records=[]
    for name,cfg in protocol['configs'].items():
        folder=OUT/'cases'/name;rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(rows)==round(.05/cfg['dt']) and abs(rows[-1]['time']-.05)<1e-12
        assert abs(rows[-1]['displacement']-.005)<1e-12
        with np.load(folder/'frames.npz') as f:
            np.testing.assert_array_equal(f['F'][0],np.broadcast_to(np.eye(3),f['F'][0].shape))
            assert np.linalg.det(f['F']).min()>0
        paths=sorted(folder.glob('audit-*.npz'));assert len(paths)==4
        for path in paths:
            with np.load(path) as z:
                dt=cfg['dt'];dx=1/(cfg['grid']-1);beta=cfg['flip_ratio'];centers=z['coords'];nodes=z['grid_nodes']
                lookup={tuple(n):i for i,n in enumerate(nodes)}
                ids=np.array([[lookup[tuple(c+o)] for o in CORNERS] for c in centers])
                ci=np.repeat(np.arange(len(centers)),8);shape=(len(centers),len(nodes))
                H=sp.csr_matrix((np.full(len(ci),.125),(ci,ids.ravel())),shape=shape)
                D=[sp.csr_matrix((np.tile((2*CORNERS[:,k]-1)/(4*dx),len(centers)),(ci,ids.ravel())),shape=shape) for k in range(3)]
                S=interpolation(z['particle_x_before'],centers,dx);G=[S@d for d in D]
                V=S.T@z['particle_volume'];W=sp.diags(1/V)@S.T@sp.diags(z['particle_volume'])
                raw,new=z['grid_velocity_raw'],z['grid_velocity_new']
                L=np.stack([g@new for g in G],axis=2);Lraw=np.stack([g@raw for g in G],axis=2)
                F=(np.eye(3)+dt*L)@z['particle_F_before'];Fc=(W@F.reshape(len(F),9)).reshape(-1,3,3)
                v=beta*(z['particle_velocity_before']+S@(H@(new-raw)))+(1-beta)*(S@(H@new))
                C=L+beta*(z['particle_C_before']-Lraw);x=z['particle_x_before']+dt*(S@(H@new))
                errors={k:float(np.max(abs(value-z[target]))) for k,value,target in (
                    ('F',F,'particle_F_after'),('Fc',Fc,'center_F_committed'),('L',L,'particle_L_after'),
                    ('C',C,'particle_C_after'),('v',v,'particle_velocity_after'),('x',x,'particle_x_after'))}
                P=pk1(F,z['particle_A0'],cfg['kf']);tau=P@np.swapaxes(z['particle_F_before'],1,2)
                force=sum(g.T@(z['particle_volume'][:,None]*tau[:,:,k]) for k,g in enumerate(G))
                mass=H.T@(S.T@z['particle_mass']);inertia=mass[:,None]*(new-raw)/dt
                row=rows[int(path.stem.split('-')[-1])-1]
                E=stabilization_energy(z,z['native_velocity'],dt,cfg['kf'])
                energy_error=abs(E-row['stabilization_energy'])
                assert energy_error<=1e-12
                force_errors={}
                for side in ('left','right'):
                    mask=nodes[:,0]*dx<=.25 if side=='left' else nodes[:,0]*dx>=.75
                    native=z['native_nodes'];nm=native[:,0]*dx<=.25 if side=='left' else native[:,0]*dx>=.75
                    dv=np.zeros_like(z['native_velocity']);dv[nm,0]=1.;eps=1e-4
                    ep=stabilization_energy(z,z['native_velocity']+eps*dv,dt,cfg['kf'])
                    em=stabilization_energy(z,z['native_velocity']-eps*dv,dt,cfg['kf'])
                    hg_force=(ep-em)/(2*eps*dt)
                    elastic=float(force[mask,0].sum())+hg_force
                    force_errors[side+'_elastic']=abs(elastic-row[side+'_elastic_force'])
                    force_errors[side+'_total']=abs(elastic+float(inertia[mask,0].sum())-row[side+'_force'])
                assert max(errors.values())<1e-11,(name,path,errors)
                assert max(force_errors.values())<1e-8,(name,path,force_errors)
                records.append(dict(case=name,snapshot=path.name,state_errors=errors,force_errors_N=force_errors,
                                    stabilization_energy_error_J=energy_error))
    result=dict(passed=len(records)==32,snapshots=records,max_state_error=max(max(r['state_errors'].values()) for r in records),
        max_force_error_N=max(max(r['force_errors_N'].values()) for r in records),
        max_stabilization_energy_error_J=max(r['stabilization_energy_error_J'] for r in records),
        method='independent sparse transfers, particle SVD stress, polar stabilization energy and grip-energy finite differences')
    (OUT/'artifact-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='snapshots'})


if __name__=='__main__':check()
