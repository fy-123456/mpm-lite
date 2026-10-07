"""Independent v13 replay using sparse history maps and a Gram eigensolve."""
import json
import numpy as np
import scipy.sparse as sp
from benchmarks.aniso_material_history import OUT,LEVELS,load,write
from benchmarks.aniso_unresolved_history import displacement
from benchmarks.aniso_dynamic_check import interpolation,CORNERS,pk1
from benchmarks.aniso_selective_check import check_patch_reference,stabilization_energy as old_energy
from benchmarks.aniso_apic_frequency import Oracle


def material_N(z):
    nodes=z['native_nodes'];Y=z['patch_origin']
    # Native coordinates are integer grid indices. The actual spacing is stored by caller.
    dx=material_N.dx;lo=nodes.min(0);hi=nodes.max(0);lookup={tuple(n):i for i,n in enumerate(nodes)};N=np.zeros((len(Y),len(nodes)))
    for p,y in enumerate(Y):
        cell=np.clip(np.floor(y/dx).astype(int),lo,hi-1);f=y/dx-cell
        for c in CORNERS:N[p,lookup[tuple(cell+c)]]+=np.prod(np.where(c,f,1-f))
    return N

def stabilization_energy(z,v,dt,kf):
    if 'marker_after' not in z:return old_energy(z,v,dt,kf)
    Y=z['patch_origin']+dt*(material_N(z)@v);r=np.einsum('cij,cja->cia',z['patch_P'],Y[z['patch_ids']])
    return float(.5*np.sum(z['patch_weight'][:,None,None]*r*r))


def filtered(z,cfg):
    v=z['particle_velocity_unfiltered'];C=z['particle_C_unfiltered'];mode=cfg['velocity_dissipation']
    if mode=='none':return v,C
    x=z['particle_x_after'];m=z['particle_mass'];h=1/(cfg['grid']-1);o=Oracle(x,m,h);T=o.S@o.H;r=o.xn[:,None,:]-x[None,:,:]
    A=np.concatenate([T.T*m[None,:]]+[T.T*m[None,:]*r[:,:,k] for k in range(3)],axis=1)
    D=np.einsum('pn,npj,npk->pjk',T,r,r).diagonal(axis1=1,axis2=2)
    q=np.concatenate([m]+[m*D[:,k] for k in range(3)]);W=A/np.sqrt(o.mn[:,None]*q[None,:]);n=len(x)
    affine=np.zeros((4*n,4));affine[:n,0]=1;affine[:n,1:]=x
    for k in range(3):affine[(k+1)*n:(k+2)*n,k+1]=1
    Q=np.linalg.qr(np.sqrt(q[:,None])*affine,mode='reduced')[0];B=W-(W@Q)@Q.T
    lam,U=np.linalg.eigh(B@B.T);keep=lam>1e-12;lam=lam[keep];R=(B.T@U[:,keep]/np.sqrt(lam)[None,:]).T
    y=np.sqrt(q[:,None])*np.concatenate([v]+[C[:,:,k] for k in range(3)]);ya=Q@(Q.T@y);c=R@(y-ya);null=y-ya-R.T@c
    if mode=='null':a=np.ones(len(lam));an=0.
    else:
        rate=np.sqrt(10/cfg['density'])/h;a=np.exp(-rate*cfg['dt']*np.maximum(0,1-lam/.1)**2);an=np.exp(-rate*cfg['dt'])
    new=(ya+R.T@(a[:,None]*c)+an*null)/np.sqrt(q[:,None])
    return new[:n],np.stack([new[(k+1)*n:(k+2)*n] for k in range(3)],axis=2)


def check():
    p=load(OUT/'protocol.json');records=[]
    for name,cfg in p['configs'].items():
        folder=OUT/'cases'/name;rows=[json.loads(l) for l in (folder/'steps.jsonl').read_text().splitlines()]
        assert len(rows)==round(p['duration']/cfg['dt']) and abs(rows[-1]['displacement'])<1e-12
        for path in sorted(folder.glob('audit-*.npz')):
            with np.load(path) as z:
                dt=cfg['dt'];dx=1/(cfg['grid']-1);beta=cfg['flip_ratio'];centers=z['coords'];nodes=z['grid_nodes'];lookup={tuple(n):i for i,n in enumerate(nodes)}
                ids=np.array([[lookup[tuple(c+o)] for o in CORNERS] for c in centers]);ci=np.repeat(np.arange(len(centers)),8);shape=(len(centers),len(nodes))
                H=sp.csr_matrix((np.full(len(ci),.125),(ci,ids.ravel())),shape=shape)
                D=[sp.csr_matrix((np.tile((2*CORNERS[:,k]-1)/(4*dx),len(centers)),(ci,ids.ravel())),shape=shape) for k in range(3)]
                S=interpolation(z['particle_x_before'],centers,dx);G=[S@d for d in D];V=S.T@z['particle_volume'];W=sp.diags(1/V)@S.T@sp.diags(z['particle_volume'])
                raw,new=z['grid_velocity_raw'],z['grid_velocity_new'];L=np.stack([g@new for g in G],axis=2);Lraw=np.stack([g@raw for g in G],axis=2)
                F=(np.eye(3)+dt*L)@z['particle_F_before'];Fc=(W@F.reshape(len(F),9)).reshape(-1,3,3)
                v=beta*(z['particle_velocity_before']+S@(H@(new-raw)))+(1-beta)*(S@(H@new));C=L+cfg['affine_flip_ratio']*(z['particle_C_before']-Lraw);x=z['particle_x_before']+dt*(S@(H@new))
                vf,Cf=filtered(z,cfg)
                errors={k:float(np.max(abs(value-z[target]))) for k,value,target in [('F',F,'particle_F_after'),('Fc',Fc,'center_F_committed'),('L',L,'particle_L_after'),('x',x,'particle_x_after'),('unfiltered_v',v,'particle_velocity_unfiltered'),('unfiltered_C',C,'particle_C_unfiltered'),('v',vf,'particle_velocity_after'),('C',Cf,'particle_C_after')]}
                P=pk1(F,z['particle_A0'],cfg['kf']);tau=P@np.swapaxes(z['particle_F_before'],1,2);force=sum(g.T@(z['particle_volume'][:,None]*tau[:,:,k]) for k,g in enumerate(G));mass=H.T@(S.T@z['particle_mass']);inertia=mass[:,None]*(new-raw)/dt
                row=rows[int(path.stem.split('-')[-1])-1];material_N.dx=dx
                if 'marker_after' in z:
                    N=material_N(z);errors['carrier_commit']=float(np.max(abs(z['patch_origin']+dt*(N@z['native_velocity'])-z['marker_after'])))
                    reference_error=float(np.max(abs(N@(z['native_nodes']*dx)-z['patch_origin'])))
                    from engine.aniso_phase1.selective_patch import polynomial
                    projector_error=float(np.max(abs(z['patch_P']@polynomial(z['patch_X'])[z['patch_ids']])))
                else:reference_error,projector_error=check_patch_reference(z,S@H,mass,nodes,dx)
                E=stabilization_energy(z,z['native_velocity'],dt,cfg['kf']);energy_error=abs(E-row['stabilization_energy']);force_errors={}
                for side in ('left','right'):
                    mask=nodes[:,0]*dx<=.25 if side=='left' else nodes[:,0]*dx>=.75;native=z['native_nodes'];nm=native[:,0]*dx<=.25 if side=='left' else native[:,0]*dx>=.75;dv=np.zeros_like(z['native_velocity']);dv[nm,0]=1;eps=1e-4
                    hg=(stabilization_energy(z,z['native_velocity']+eps*dv,dt,cfg['kf'])-stabilization_energy(z,z['native_velocity']-eps*dv,dt,cfg['kf']))/(2*eps*dt)
                    elastic=float(force[mask,0].sum())+hg;force_errors[side]=abs(elastic+float(inertia[mask,0].sum())-row[side+'_force'])
                assert max(errors.values())<1e-10,(name,path,errors)
                assert max(force_errors.values())<1e-8 and energy_error<1e-12
                assert reference_error<1e-12 and projector_error<1e-12
                records.append(dict(case=name,snapshot=path.name,state_errors=errors,force_errors=force_errors,reference_error=reference_error,projector_error=projector_error,stabilization_energy_error=energy_error))
    r=dict(passed=len(records)==len(p['configs'])*len(p['snapshot_times']),snapshots=records,max_state_error=max(max(r['state_errors'].values()) for r in records),max_force_error=max(max(r['force_errors'].values()) for r in records),max_reference_error=max(r['reference_error'] for r in records),max_stabilization_energy_error=max(r['stabilization_energy_error'] for r in records),method='independent sparse material/transfer maps, Gram velocity eigensolve, independently assembled carrier Q1 map, committed-history and patch-force finite differences')
    write(OUT/'artifact-check.json',r);print({k:v for k,v in r.items() if k!='snapshots'})

if __name__=='__main__':check()
